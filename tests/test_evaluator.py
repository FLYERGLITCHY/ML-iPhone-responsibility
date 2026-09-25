from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from iphone_valuator import evaluator
from iphone_valuator.domain import Condition, require_model
from iphone_valuator.evaluator import (
    InvalidListingError,
    ListingQuery,
    Valuator,
    Verdict,
    classify,
    compute_delta_pct,
    format_report,
    parse_battery_input,
    parse_condition_input,
    parse_model_input,
    parse_price_input,
    parse_storage_input,
    prompt_query,
    run_interactive,
)
from iphone_valuator.synthetic import expected_price
from iphone_valuator.train import TrainingResult
from tests.helpers import SCRAPED_AT, StubEstimator, make_stub_bundle

TODAY = date(2026, 9, 1)


def query(**overrides: object) -> ListingQuery:
    fields: dict[str, object] = {
        "model": "iPhone 13 Pro",
        "storage_gb": 256,
        "condition": Condition.USED,
        "asking_price": 65_000,
        "battery_health": 78,
    }
    fields.update(overrides)
    return ListingQuery(**fields)


def stub_valuator(price: float = 55_000.0, threshold: float = 10.0) -> Valuator:
    return Valuator(make_stub_bundle(price), threshold_pct=threshold, today=TODAY)


def test_compute_delta_pct() -> None:
    assert compute_delta_pct(65_000, 55_000) == pytest.approx(18.1818, rel=1e-4)
    assert compute_delta_pct(49_500, 55_000) == pytest.approx(-10.0)
    with pytest.raises(ValueError, match="positive"):
        compute_delta_pct(1, 0)


@pytest.mark.parametrize(
    ("delta", "threshold", "verdict"),
    [
        (-10.01, 10, Verdict.GREAT_DEAL),
        (-10.0, 10, Verdict.FAIR),
        (0.0, 10, Verdict.FAIR),
        (10.0, 10, Verdict.FAIR),
        (10.01, 10, Verdict.OVERPRICED),
        (-6.0, 5, Verdict.GREAT_DEAL),
        (6.0, 5, Verdict.OVERPRICED),
    ],
)
def test_classify_boundaries(delta: float, threshold: float, verdict: Verdict) -> None:
    assert classify(delta, threshold) is verdict


def test_overpriced_summary_matches_the_spec_example() -> None:
    valuation = stub_valuator().evaluate(query())
    assert valuation.verdict is Verdict.OVERPRICED
    assert valuation.fair_price == pytest.approx(55_000)
    assert valuation.delta_pct == pytest.approx(18.18, abs=0.01)
    assert valuation.fair_range == pytest.approx((49_500, 60_500))
    assert valuation.summary == (
        "Fair price is ~55,000 RUB, but this seller is asking 65,000 RUB (+18%) for a used "
        "iPhone 13 Pro 256GB with only 78% battery health — overpriced."
    )


def test_great_deal_summary_warns_about_scams() -> None:
    valuation = stub_valuator().evaluate(query(asking_price=45_000, battery_health=95))
    assert valuation.verdict is Verdict.GREAT_DEAL
    assert "only 45,000 RUB (-18%)" in valuation.summary
    assert "a healthy 95% battery" in valuation.summary
    assert "scam" in valuation.summary


@pytest.mark.parametrize(
    ("asking", "tail"),
    [
        (58_500, "slightly above market, so there is room to negotiate"),
        (51_500, "slightly below market, a solid offer"),
        (55_500, "right in line with the market"),
    ],
)
def test_fair_summary_nuances(asking: int, tail: str) -> None:
    valuation = stub_valuator().evaluate(query(asking_price=asking, battery_health=85))
    assert valuation.verdict is Verdict.FAIR
    assert "85% battery health" in valuation.summary
    assert valuation.summary.endswith(f"— {tail}.")


def test_extras_are_mentioned() -> None:
    summary = stub_valuator().evaluate(query(has_box=True, has_receipt=True)).summary
    assert "(original box and receipt included)" in summary


def test_feature_row_passed_to_the_model() -> None:
    valuator = stub_valuator()
    valuator.evaluate(query(region="г. Санкт-Петербург", has_box=True))
    bundle_estimator = valuator._bundle.estimator
    assert isinstance(bundle_estimator, StubEstimator)
    row = bundle_estimator.frames[-1].iloc[0]
    assert row["model"] == "iPhone 13 Pro"
    assert row["model_tier"] == "pro"
    assert row["storage"] == "256GB"
    assert row["region"] == "Санкт-Петербург"
    assert row["battery_health"] == 78
    assert row["battery_known"] == 1
    assert row["phone_age_months"] == 59
    assert row["has_box"] == 1
    assert row["has_receipt"] == 0


def test_unknown_battery_is_imputed_from_training_statistics() -> None:
    valuator = stub_valuator()
    valuation = valuator.evaluate(query(battery_health=None))
    assert valuation.battery_assumed == 86
    assert valuation.region == "Москва"
    assert "unknown battery health" in valuation.summary
    row = valuator._bundle.estimator.frames[-1].iloc[0]
    assert row["battery_health"] == 86
    assert row["battery_known"] == 0
    older = stub_valuator().evaluate(query(model="iPhone 11", storage_gb=64, battery_health=None))
    assert older.battery_assumed == 88


def test_new_phone_without_battery_input_assumes_full_health() -> None:
    valuation = stub_valuator().evaluate(query(condition=Condition.NEW, battery_health=None))
    assert valuation.battery_assumed == 100
    assert "brand-new iPhone 13 Pro 256GB with a new battery" in valuation.summary


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model": "iPhone 99"}, "Unknown iPhone model"),
        ({"model": "iPhone 13", "storage_gb": 64}, "never sold with 64GB"),
        ({"condition": Condition.FOR_PARTS}, "For-parts"),
        ({"battery_health": 0}, "between 1 and 100"),
        ({"battery_health": 101}, "between 1 and 100"),
        ({"asking_price": 0}, "must be positive"),
    ],
)
def test_invalid_queries_are_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(InvalidListingError, match=message):
        stub_valuator().evaluate(query(**overrides))


def test_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError, match="threshold"):
        Valuator(make_stub_bundle(), threshold_pct=0)


def test_custom_threshold_changes_verdict() -> None:
    valuator = stub_valuator(threshold=5.0)
    assert valuator.threshold_pct == 5.0
    assert valuator.evaluate(query(asking_price=58_500)).verdict is Verdict.OVERPRICED
    assert valuator.predict_fair_price(query()) == pytest.approx(55_000)


def test_valuation_to_dict() -> None:
    payload = stub_valuator().evaluate(query()).to_dict()
    assert payload["verdict"] == "Overpriced / Not Worth It"
    assert payload["storage"] == "256GB"
    assert payload["fair_price"] == 55_000
    assert payload["fair_range"] == [49_500, 60_500]
    assert payload["delta_pct"] == 18.18
    json.dumps(payload, ensure_ascii=False)


def test_format_report() -> None:
    report = format_report(stub_valuator().evaluate(query(battery_health=None)))
    assert "battery unknown (assumed ~86%)" in report
    assert "Fair market value : ~55,000 RUB (fair range 49,500 RUB – 60,500 RUB)" in report
    assert "Verdict           : Overpriced / Not Worth It" in report
    assert "typical error ±6.0%" in report


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("14 pro max", "iPhone 14 Pro Max"),
        ("Айфон 12 мини", "iPhone 12 mini"),
        ("XR", "iPhone XR"),
    ],
)
def test_parse_model_input(text: str, expected: str) -> None:
    assert parse_model_input(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("256", 256), ("256GB", 256), ("1TB", 1024), ("1024", 1024), ("2 тб", 2048)],
)
def test_parse_storage_input(text: str, expected: int) -> None:
    assert parse_storage_input(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("used", Condition.USED),
        ("Б/у", Condition.USED),
        ("новый", Condition.NEW),
        ("NEW", Condition.NEW),
        ("восстановленный", Condition.REFURBISHED),
        ("refurbished", Condition.REFURBISHED),
        ("broken", Condition.FOR_PARTS),
    ],
)
def test_parse_condition_input(text: str, expected: Condition) -> None:
    assert parse_condition_input(text) is expected


@pytest.mark.parametrize(("text", "expected"), [("87", 87), ("87%", 87), (" ", None), ("", None)])
def test_parse_battery_input(text: str, expected: int | None) -> None:
    assert parse_battery_input(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("65000", 65_000),
        ("65 000 ₽", 65_000),
        ("65k", 65_000),
        ("65,5к", 65_500),
        ("70 тыс", 70_000),
    ],
)
def test_parse_price_input(text: str, expected: int) -> None:
    assert parse_price_input(text) == expected


@pytest.mark.parametrize(
    ("parser", "text"),
    [
        (parse_model_input, "Galaxy S24"),
        (parse_storage_input, "100"),
        (parse_storage_input, "big"),
        (parse_condition_input, "whatever"),
        (parse_battery_input, "150"),
        (parse_battery_input, "abc"),
        (parse_price_input, "0"),
        (parse_price_input, "free"),
    ],
)
def test_input_parsers_reject_garbage(parser: object, text: str) -> None:
    assert callable(parser)
    with pytest.raises(InvalidListingError):
        parser(text)


def test_prompt_query_reprompts_until_valid() -> None:
    answers = iter(
        [
            "Nokia 3310",
            "13 pro",
            "64",
            "256",
            "for parts",
            "used",
            "150",
            "78",
            "free",
            "65 000",
        ]
    )
    messages: list[str] = []
    result = prompt_query(lambda _prompt: next(answers), messages.append)
    assert result == query(region=None)
    assert len(messages) == 5
    assert any("iPhone 13 Pro comes with 128GB, 256GB, 512GB, 1TB" in m for m in messages)


def test_run_interactive_evaluates_until_user_stops() -> None:
    answers = iter(["iPhone 13 Pro", "256", "used", "78", "65000", "n"])
    output: list[str] = []
    assert run_interactive(stub_valuator(), lambda _prompt: next(answers), output.append) == 0
    assert any("Overpriced / Not Worth It" in line for line in output)


def test_run_interactive_handles_end_of_input() -> None:
    def closed(_prompt: str) -> str:
        raise EOFError

    assert run_interactive(stub_valuator(), closed, lambda _line: None) == 0


def test_trained_model_verdicts_track_the_true_market(training_result: TrainingResult) -> None:
    valuator = Valuator(training_result.bundle, today=SCRAPED_AT.date())
    model = require_model("iPhone 13 Pro")
    true_price = expected_price(model, 256, Condition.USED, 88, False, False, 1.04)
    listing = query(battery_health=88, region="Москва")
    fair = valuator.evaluate(replace_price(listing, true_price))
    assert fair.verdict is Verdict.FAIR
    assert abs(fair.delta_pct) < 10
    assert (
        valuator.evaluate(replace_price(listing, true_price * 1.5)).verdict is Verdict.OVERPRICED
    )
    assert (
        valuator.evaluate(replace_price(listing, true_price * 0.6)).verdict is Verdict.GREAT_DEAL
    )


def replace_price(listing: ListingQuery, price: float) -> ListingQuery:
    return ListingQuery(
        model=listing.model,
        storage_gb=listing.storage_gb,
        condition=listing.condition,
        asking_price=round(price),
        battery_health=listing.battery_health,
        region=listing.region,
    )


def test_cli_text_output(model_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = evaluator.main(
        [
            "--model-path",
            str(model_path),
            "-m",
            "iPhone 13 Pro",
            "-s",
            "256",
            "-c",
            "used",
            "-b",
            "78",
            "-p",
            "150000",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Verdict           : Overpriced / Not Worth It" in output
    assert "iPhone 13 Pro 256GB | used | battery 78%" in output


def test_cli_json_output(model_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = evaluator.main(
        [
            "--model-path",
            str(model_path),
            "-m",
            "14 pro max",
            "-s",
            "1tb",
            "-c",
            "new",
            "-p",
            "10k",
            "--box",
            "--receipt",
            "--region",
            "Казань",
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "iPhone 14 Pro Max"
    assert payload["verdict"] == "Great Deal / Underpriced"
    assert payload["battery_assumed"] == 100
    assert payload["region"] == "Казань"
    assert payload["has_box"] is True


def test_cli_invalid_listing_exit_code(
    model_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = evaluator.main(
        ["--model-path", str(model_path), "-m", "iPhone 13", "-s", "64", "-c", "used", "-p", "1"]
    )
    assert exit_code == 2
    assert "never sold with 64GB" in capsys.readouterr().err


def test_cli_missing_model_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = evaluator.main(
        [
            "--model-path",
            str(tmp_path / "absent.joblib"),
            "-m",
            "13",
            "-s",
            "128",
            "-c",
            "used",
            "-p",
            "1000",
        ]
    )
    assert exit_code == 1
    assert "Train one first" in capsys.readouterr().err


def test_cli_partial_arguments_are_rejected(model_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        evaluator.main(["--model-path", str(model_path), "-m", "iPhone 13"])
    assert excinfo.value.code == 2


def test_cli_interactive_mode(
    model_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = iter(["iPhone 12", "128", "used", "", "30000", "no"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert evaluator.main(["--model-path", str(model_path)]) == 0
    assert "Verdict" in capsys.readouterr().out
