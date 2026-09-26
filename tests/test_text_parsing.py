from __future__ import annotations

import numpy as np
import pytest

from iphone_valuator.config import UNKNOWN_REGION
from iphone_valuator.domain import Condition
from iphone_valuator.text_parsing import (
    JunkReason,
    detect_junk,
    find_param,
    has_original_box,
    has_receipt,
    mentions_installment,
    normalize_region,
    normalize_text,
    parse_battery_health,
    parse_condition,
    parse_model,
    parse_price,
    parse_storage,
)


def test_normalize_text_unifies_case_yo_and_spaces() -> None:
    assert normalize_text("  Ёмкость\u00a0АКБ   87% \n\n Коробка ") == "емкость акб 87%\nкоробка"
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("iPhone 13 Pro, 256 ГБ", "iPhone 13 Pro"),
        ("Смартфон Apple iPhone 15 Pro Max 256GB", "iPhone 15 Pro Max"),
        ("Айфон 12 мини 64гб", "iPhone 12 mini"),
        ("айфон 11 про макс", "iPhone 11 Pro Max"),
        ("IPHONE XS MAX 256", "iPhone XS Max"),
        ("iPhone ХР 128", "iPhone XR"),
        ("iPhone Хs", "iPhone XS"),
        ("iPhone X 64gb", "iPhone X"),
        ("iphone 10", "iPhone X"),
        ("iPhone 14+ 128", "iPhone 14 Plus"),
        ("iPhone 14 plus", "iPhone 14 Plus"),
        ("iPhone 16e 128 ГБ", "iPhone 16e"),
        ("iPhone Air 256", "iPhone Air"),
        ("iPhone 17 Air", "iPhone Air"),
        ("iPhone 17 Pro Max 2 ТБ", "iPhone 17 Pro Max"),
        ("iPhone SE 2020", "iPhone SE 2020"),
        ("iPhone SE (2-го поколения)", "iPhone SE 2020"),
        ("iPhone SE 3", "iPhone SE 2022"),
        ("iPhone SE 2022 64", "iPhone SE 2022"),
        ("iPhone 11Pro", "iPhone 11 Pro"),
        ("iPhone 11 ProMax", "iPhone 11 Pro Max"),
        ("iPhone 13 Pro-Max", "iPhone 13 Pro Max"),
        ("i-Phone 13", "iPhone 13"),
        ("i phone 8 plus", "iPhone 8 Plus"),
        ("iPhone 12 max", "iPhone 12 Pro Max"),
        ("iPhone 13 \u0420ro", "iPhone 13 Pro"),
        ("Продаю iPhone 13 (не 14)", "iPhone 13"),
    ],
)
def test_parse_model_recognises_variants(text: str, expected: str) -> None:
    assert parse_model(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "iPhone SE",
        "iPhone 9",
        "iPhone 14 mini",
        "iPhone 7 32gb",
        "Samsung Galaxy S23",
        "13 pro max",
        "",
        None,
        float("nan"),
    ],
)
def test_parse_model_rejects_unknown_or_ambiguous(text: object) -> None:
    assert parse_model(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("14 pro max", "iPhone 14 Pro Max"),
        ("xr", "iPhone XR"),
        ("13", "iPhone 13"),
        ("se 2022", "iPhone SE 2022"),
        ("iPhone 15", "iPhone 15"),
        ("256", None),
        ("pro", None),
    ],
)
def test_parse_model_without_prefix(text: str, expected: str | None) -> None:
    assert parse_model(text, require_prefix=False) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("128 ГБ", 128),
        ("256Gb", 256),
        ("64гб", 64),
        ("512 GB", 512),
        ("iPhone 13, 128 гигабайт", 128),
        ("256G", 256),
        ("1 ТБ", 1024),
        ("1TB", 1024),
        ("2 тб", 2048),
        ("1024 ГБ", 1024),
        ("память 256 гб, оперативка 6 гб", 256),
    ],
)
def test_parse_storage_with_units(text: str, expected: int) -> None:
    assert parse_storage(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "32 ГБ",
        "гарантия 1 г.",
        "скидка 2 т.р.",
        "оперативная память 6 GB",
        "iPhone 13 Pro 256",
        "",
        None,
    ],
)
def test_parse_storage_rejects_invalid(text: object) -> None:
    assert parse_storage(text) is None


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("iPhone 13 Pro 256", 256),
        ("iPhone 12 64", 64),
        ("iPhone 13 128 000 руб", None),
        ("iPhone 13 за 128 тыс", None),
        ("iPhone 13 A2482", None),
    ],
)
def test_parse_storage_bare_numbers_in_titles(title: str, expected: int | None) -> None:
    assert parse_storage(title, allow_bare=True) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("АКБ 87%", 87),
        ("акб 87", 87),
        ("Акб-91%", 91),
        ("Аккумулятор: 90 %", 90),
        ("Ёмкость аккумулятора 86%", 86),
        ("Макс. ёмкость 84%", 84),
        ("88% акб", 88),
        ("Battery health 91%", 91),
        ("батарея 100 процентов", 100),
        ("аккумулятор родной, 89%", 89),
        ("Состояние батареи 95%", 95),
        ("100% оригинал, акб 87%", 87),
    ],
)
def test_parse_battery_health(text: str, expected: int) -> None:
    assert parse_battery_health(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "100% оригинал",
        "аккумулятор 85 циклов",
        "АКБ 3000 mAh",
        "батарея держит 12 часов",
        "АКБ 30%",
        "акб 120%",
        "АКБ родной. 100% оригинал",
        "",
        None,
    ],
)
def test_parse_battery_health_rejects_unanchored_or_implausible(text: object) -> None:
    assert parse_battery_health(text) is None


@pytest.mark.parametrize(
    ("title", "description", "reason"),
    [
        ("iPhone 12", "Заблокирован на iCloud, пароль не помню", JunkReason.LOCKED),
        ("iPhone 12", "Телефон на айклауде", JunkReason.LOCKED),
        ("iPhone 12", "Activation lock, продаю как есть", JunkReason.LOCKED),
        ("iPhone 12", "Забыл пароль", JunkReason.LOCKED),
        ("iPhone XR", "Работает через r-sim", JunkReason.LOCKED),
        ("iPhone 11", "На запчасти", JunkReason.FOR_PARTS),
        ("iPhone 11", "Отдам на детали", JunkReason.FOR_PARTS),
        ("iPhone 11", "Не включается после обновления", JunkReason.FOR_PARTS),
        ("iPhone 11", "Face ID не работает", JunkReason.FOR_PARTS),
        ("iPhone 11", "Требует ремонта", JunkReason.FOR_PARTS),
        ("iPhone 13", "Экран разбит, остальное работает", JunkReason.BROKEN_SCREEN),
        ("iPhone 13", "Трещина на экране", JunkReason.BROKEN_SCREEN),
        ("iPhone 13", "Без царапин, но экран разбит", JunkReason.BROKEN_SCREEN),
        ("iPhone 13", "Утопленник", JunkReason.WATER_DAMAGE),
        ("iPhone 13", "Попадал в воду, сушили", JunkReason.WATER_DAMAGE),
        ("iPhone 14 Pro", "Реплика, точная копия", JunkReason.FAKE),
        ("Чехол для iPhone 13", "Силиконовый", JunkReason.ACCESSORY),
        ("Стекло на iPhone 14", "Защитное", JunkReason.ACCESSORY),
        ("iPhone 13 + Apple Watch", "Комплектом", JunkReason.ACCESSORY),
        ("Куплю iPhone 13", "Дорого", JunkReason.NOT_FOR_SALE),
        ("Ремонт iPhone", "Замена экрана за час", JunkReason.NOT_FOR_SALE),
    ],
)
def test_detect_junk_flags_bad_listings(title: str, description: str, reason: JunkReason) -> None:
    assert detect_junk(title, description) is reason


@pytest.mark.parametrize(
    "description",
    [
        "Не битый, не утопленник",
        "Без сколов, царапин и трещин",
        "Трещин нет, сколов нет",
        "Никогда не был в воде",
        "iCloud отвязан, готов к активации",
        "Привязан к моему iCloud, отвяжу при продаже",
        "С перекупами не работаю",
        "Все запчасти оригинальные",
        "Неисправностей нет",
        "Есть копия чека",
        "Оригинал, не реплика",
        "Влагозащита IP68",
        "Не заблокирован",
        "В комплекте чехол и стекло",
        "Неверлок, работает с любой симкой",
        "Никогда не вскрывался, не ремонтировался",
        "Приезжайте на детальный осмотр",
    ],
)
def test_detect_junk_respects_negations_and_harmless_mentions(description: str) -> None:
    assert detect_junk("iPhone 13 Pro, 256 ГБ", description) is None


@pytest.mark.parametrize(
    ("param", "text", "expected"),
    [
        ("Новое", "", Condition.NEW),
        ("Б/у", "", Condition.USED),
        ("Отличное", "", Condition.USED),
        ("Удовлетворительное", "", Condition.USED),
        ("Восстановленное", "", Condition.REFURBISHED),
        ("Требует ремонта", "", Condition.FOR_PARTS),
        ("used", "", Condition.USED),
        ("for_parts", "", Condition.FOR_PARTS),
        ("Хорошее", "Не включается", Condition.FOR_PARTS),
        ("Новое", "Восстановленный, как новый", Condition.REFURBISHED),
        ("Новое", "Пользовался месяц", Condition.USED),
        ("Новое", "Новый, не пользовался", Condition.NEW),
        (None, "Запечатан, не активирован", Condition.NEW),
        (None, "Телефон не активирован", Condition.NEW),
        (None, "Телефон б/у, в чехле", Condition.USED),
        (None, "Не восстановленный, родные детали", None),
        (None, "Просто текст", None),
    ],
)
def test_parse_condition(param: str | None, text: str, expected: Condition | None) -> None:
    assert parse_condition(param, text) is expected


@pytest.mark.parametrize(
    ("text", "box", "receipt"),
    [
        ("Полный комплект: коробка, чек.", True, True),
        ("Коробка есть, чека нет.", True, False),
        ("Без коробки, но есть чек.", False, True),
        ("Коробки и чека нет.", False, False),
        ("Нет коробки, есть документы", False, True),
        ("Только телефон.", False, False),
        ("Full set, receipt included", True, True),
    ],
)
def test_box_and_receipt_flags(text: str, box: bool, receipt: bool) -> None:
    assert has_original_box(text) is box
    assert has_receipt(text) is receipt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Первый взнос 30%, остальное в рассрочку", True),
        ("Возможна рассрочка", True),
        ("Trade-in принимаем", True),
        ("Сдам в аренду", True),
        ("Без предоплаты", False),
        ("Отличное состояние", False),
    ],
)
def test_mentions_installment(text: str, expected: bool) -> None:
    assert mentions_installment(text) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (54990, 54990),
        (54990.0, 54990),
        (np.int64(1000), 1000),
        ("54 990 ₽", 54990),
        ("54\u00a0990 ₽", 54990),
        ("54990.0", 54990),
        ("от 50 000 ₽", 50000),
        ("Цена не указана", None),
        (None, None),
        (float("nan"), None),
        (0, None),
        (-5, None),
        ("0", None),
        (True, None),
        ([1, 2], None),
    ],
)
def test_parse_price(value: object, expected: int | None) -> None:
    assert parse_price(value) == expected


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("Москва, ул. Тверская, 7", "Москва"),
        ("г. Москва", "Москва"),
        ("СПб, Невский пр-т", "Санкт-Петербург"),
        ("мск", "Москва"),
        ("Московская область, Химки", "Московская область"),
        ("ростов-на-дону", "Ростов-на-дону"),
        ("м. Тверская", UNKNOWN_REGION),
        ("ул. Ленина, 5", UNKNOWN_REGION),
        ("", UNKNOWN_REGION),
        (None, UNKNOWN_REGION),
        (float("nan"), UNKNOWN_REGION),
    ],
)
def test_normalize_region(location: object, expected: str) -> None:
    assert normalize_region(location) == expected


def test_find_param_prefers_exact_keys_then_substrings() -> None:
    params = {
        "Состояние аккумулятора": "87%",
        "Состояние": "Отличное",
        "Объём встроенной памяти": "256 ГБ",
    }
    assert find_param(params, "состояние") == "Отличное"
    assert find_param(params, "встроенная память", "встроен") == "256 ГБ"
    assert find_param(params, "цвет") is None
    assert find_param({}, "модель") is None
