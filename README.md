# ML iPhone Valuator for Avito

An end-to-end machine-learning pipeline that answers one question: **is this iPhone listing on
[Avito](https://www.avito.ru) worth its asking price?**

```
scraper.py ──► data/raw/iphones_raw.csv ──► cleaner.py ──► data/processed/iphones_clean.csv
                                                                          │
evaluator.py ◄── artifacts/model.joblib (+ metrics.json) ◄── train.py ◄───┘
```

| Stage | Module | What it does |
|---|---|---|
| Scrape | `iphone_valuator/scraper.py`, `browser.py` | Playwright + BeautifulSoup crawl of Avito search and listing pages |
| Clean | `iphone_valuator/cleaner.py` | Junk filters, deduplication, outlier removal, battery imputation |
| Train | `iphone_valuator/train.py` | CatBoost vs LightGBM with K-fold CV (RMSE / MAE / MAPE), saves the best model |
| Evaluate | `iphone_valuator/evaluator.py` | CLI and interactive valuation: fair price, price delta, verdict, explanation |

Shared modules: `text_parsing.py` (regex extraction and junk detection), `domain.py` (iPhone
catalog with release dates and storage options), `features.py` (identical features for training and
inference), `modeling.py` (estimators and the persisted model bundle), `synthetic.py` (offline demo
data).

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt           # or: pip install -e ".[dev]" for tests, linters, CLI entry points
python -m playwright install chromium     # only needed for scraping
```

Tested with Python 3.12, pandas 3.0, scikit-learn 1.9, CatBoost 1.2.10, LightGBM 4.7 and
Playwright 1.63.

## Quick start (offline, synthetic data)

Avito blocks most datacenter IPs, so the quickest way to exercise the whole pipeline is the bundled
generator of **synthetic** Avito-like listings. They include realistic Russian text, missing fields,
duplicates and junk, but the prices are made up.

```bash
python -m iphone_valuator.synthetic --rows 5000          # -> data/raw/iphones_raw.csv
python -m iphone_valuator.cleaner --outlier-method both   # -> data/processed/iphones_clean.csv
python -m iphone_valuator.train                           # -> artifacts/model.joblib, artifacts/metrics.json
python -m iphone_valuator.evaluator -m "iPhone 13 Pro" -s 256 -c used -b 78 -p 65000
```

```
iPhone 13 Pro 256GB | used | battery 78% | Москва | asking 65,000 RUB
Fair market value : ~48,000 RUB (fair range 43,200 RUB – 52,600 RUB)
Price delta       : +35.7%
Verdict           : Overpriced / Not Worth It
Summary           : Fair price is ~48,000 RUB, but this seller is asking 65,000 RUB (+36%) for a used iPhone 13 Pro 256GB with only 78% battery health — overpriced.
Model uncertainty : typical error ±5.1% (cross-validated)
```

With `pip install -e .` the same commands are available as `iphone-synthetic`, `iphone-scrape`,
`iphone-clean`, `iphone-train` and `iphone-evaluate`.

## 1. Scraping Avito (`scraper.py`)

```bash
python -m iphone_valuator.scraper --region moskva --query iphone --max-pages 10
python -m iphone_valuator.scraper \
    --search-url "https://www.avito.ru/sankt-peterburg/telefony/mobilnye_telefony?q=iphone+15" \
    --max-pages 5 --headful --manual-captcha
```

Each row of `data/raw/iphones_raw.csv` holds `item_id, listing_url, title, model, storage_gb,
condition, battery_health, price, description, location, params, scraped_at`. Structured Avito
attributes (`Модель`, `Встроенная память`, `Состояние`) take priority over the title, and the title
over the description.

How it copes with Avito:

- **Pagination**: follows the `p=` parameter and stops at the last page, at a repeated page, or at
  `--max-pages` / `--max-listings`.
- **Dynamic content**: a real Chromium renders JavaScript, waits for listing selectors and scrolls
  like a human to trigger lazy loading.
- **Anti-bot measures**: automation flags and `navigator.webdriver` hidden; user agent matched to the
  real engine without the `Headless` marker; `ru-RU` locale and Moscow timezone; random viewport;
  mouse movement; randomized delays with periodic long pauses; images and fonts blocked; cookies
  persisted between runs (`--state-file`); exponential backoff on network errors.
- **Blocks**: the firewall page (HTTP 429, "Доступ ограничен: проблема с IP") is detected. The
  scraper cools down, or rotates proxies (`--proxy`, `--proxy-file`). With `--headful
  --manual-captcha` it waits for you to solve the captcha in the browser window. If Avito keeps
  blocking, the run stops and keeps everything scraped so far.
- **Resumable**: rows are appended as they are scraped, and saved IDs are skipped on the next run.
- **Markup changes**: all selectors live at the top of `scraper.py` (Avito `data-marker` attributes
  with fallbacks, plus JSON-LD for listing pages).

Please respect Avito's terms of use and applicable law: keep the default delays and only collect
what you need.

## 2. Cleaning and anti-garbage filtering (`cleaner.py`)

```bash
python -m iphone_valuator.cleaner --outlier-method iqr|isolation_forest|both|none
```

The cleaner re-parses every row, so it also accepts raw CSVs from other sources. Rules run in this
order and each one reports how many rows it removed:

| Rule | What it removes |
|---|---|
| `missing_price`, `unknown_model`, `missing_storage` | Rows without a price, a catalog iPhone model or a storage size |
| `storage_not_offered` | Impossible combinations such as an iPhone 13 with 64GB |
| `junk_locked` | iCloud / activation lock, forgotten passcode, lost mode, MDM, R-SIM or carrier lock |
| `junk_for_parts` | "на запчасти", does not turn on or charge, Face ID not working, no network |
| `junk_broken_screen` | Broken or cracked screen or glass |
| `junk_water_damage` | Water damage ("утопленник", "попадал в воду") |
| `junk_fake` | Replicas, dummies, copies |
| `junk_accessory` | Titles that sell cases, glass, parts, boxes, or bundles with other Apple devices |
| `junk_not_for_sale` | "Куплю", repair services, rentals, exchange-only listings |
| `condition_for_parts` | Avito condition "Требует ремонта" / for parts |
| `price_below_minimum`, `price_above_maximum` | Prices below 5,000 or above 400,000 RUB |
| `duplicate_item_id`, `duplicate_listing_url`, `duplicate_content` | Exact duplicates and re-posted ads (same model, storage, price and description) |
| `suspiciously_cheap` | Below 30% of the median price for the same model and storage, or below 50% combined with installment, down-payment or trade-in wording |
| `price_outlier_iqr` | Outside `Q1 − 2·IQR … Q3 + 2·IQR` of log-price within (model, storage, condition), falling back to coarser groups when a group has fewer than 10 listings |
| `price_outlier_isolation_forest` | Multivariate anomalies: price deviation vs. battery, condition, storage, age (2% contamination) |

Junk detection understands negations, so "не битый, не утопленник", "трещин нет", "без сколов,
царапин и трещин" and "с перекупами не работаю" are not flagged.

Missing battery health gets a `battery_known = 0` flag and is imputed with the median battery of
phones from the same release year (100% for new phones). Output: `data/processed/iphones_clean.csv`.

Key regexes (`text_parsing.py`):

- Storage: `(64|128|256|512|1024|1|2)\s*(ГБ|GB|гиг|G|ТБ|TB)`; bare numbers such as
  "iPhone 13 Pro 256" are accepted in titles only.
- Battery: anchored to a keyword such as `АКБ 87%`, `87% акб`, `аккумулятор: 90` or
  `Battery health 91%`. Bare percentages like "100% оригинал" and cycle counts ("85 циклов") are
  ignored on purpose.
- Model: English, Cyrillic and mixed spellings: "Айфон 13 про макс", "iPhone ХР", "14+", "11ProMax",
  "SE 2-го поколения", "16e", "iPhone Air".

## 3. Training (`train.py`)

```bash
python -m iphone_valuator.train --folds 5 --select-by mape
```

- **Target**: `price` in RUB, modelled as `log(price)`, so errors are relative and predictions are
  always positive.
- **Categorical features**: `model`, `model_tier` (base/mini/plus/pro/pro_max/…), `storage`,
  `condition`, `region` (regions with fewer than 10 listings are grouped).
- **Numeric features**: `storage_gb`, `battery_health`, `battery_known`, `phone_age_months`,
  `has_box`, `has_receipt`. The box and receipt flags come from negation-aware keyword matching in
  the listing text.
- **Models**: CatBoost with native categorical handling, and LightGBM with pandas categoricals and
  monotonic constraints (better battery, more storage or a box never lower the price). CatBoost gets
  no monotonic constraints because they break its categorical handling and double the error.
- **Evaluation**: K-fold CV with RMSE, MAE and MAPE on the RUB scale. The winner is refit on all
  data and saved with training statistics (battery medians, default region, typical error) in
  `artifacts/model.joblib`. `artifacts/metrics.json` holds the CV table and feature importances.

On 5,000 synthetic rows:

```
model               RMSE, RUB           MAE, RUB        MAPE, %
---------------------------------------------------------------
catboost       4,264 ± 174        2,860 ± 77       5.08 ± 0.10   <- best
lightgbm       4,393 ± 178        2,950 ± 91       5.24 ± 0.10
```

## 4. Valuation (`evaluator.py`)

Inputs: model, storage, condition, battery health (optional) and asking price, plus optional
`--region`, `--box`, `--receipt`.

1. Predict the fair market value ŷ.
2. Compute Δ = (asking price − ŷ) / ŷ × 100%.
3. Classify Δ: below −10% is **Great Deal / Underpriced**, −10% to +10% is **Fair Market Price**,
   above +10% is **Overpriced / Not Worth It**. Change the band with `--threshold`. Inside the fair
   band, a Δ beyond ±5% is described as slightly above or below market.
4. Explain the verdict in one sentence.

```bash
python -m iphone_valuator.evaluator -m "14" -s 128 -c "б/у" -b 88 -p 40000 --region "Санкт-Петербург" --json
python -m iphone_valuator.evaluator --interactive
```

Example `--json` output (abridged):

```json
{
  "model": "iPhone 14",
  "storage": "128GB",
  "condition": "used",
  "battery_health": 88,
  "fair_price": 42800,
  "delta_pct": -6.54,
  "verdict": "Fair Market Price",
  "summary": "Fair price is ~43,000 RUB and this seller is asking 40,000 RUB (-7%) for a used iPhone 14 128GB with 88% battery health — slightly below market, a solid offer."
}
```

Inputs are forgiving: `"14 pro max"`, `"Айфон 12 мини"`, `1TB`, `б/у`, `65k`. Impossible
combinations (for example iPhone 13 with 64GB) are rejected. For-parts phones are rejected too,
because they are excluded from the training data. When battery health is unknown, the training-set
median for that release year is assumed and shown.

From Python:

```python
from pathlib import Path
from iphone_valuator.domain import Condition
from iphone_valuator.evaluator import ListingQuery, Valuator

valuator = Valuator.from_path(Path("artifacts/model.joblib"))
result = valuator.evaluate(
    ListingQuery(model="iPhone 13 Pro", storage_gb=256, condition=Condition.USED,
                 asking_price=65_000, battery_health=78)
)
print(result.verdict, result.fair_price, result.summary)
```

## Development

```bash
pip install -e ".[dev]"
pytest              # 376 tests, no network or browser needed
ruff check . && ruff format --check .
mypy iphone_valuator
```

## Extending and limitations

- The catalog in `domain.py` covers iPhone 8 through the iPhone 17 series (including 16e and Air).
  Listings of newer models are dropped as `unknown_model` until you add them to `IPHONE_MODELS`.
- Avito's markup changes over time. If the scraper finds no listings, update the selectors at the
  top of `scraper.py`.
- The model learns asking prices, not sale prices, so "fair" means typical for Avito right now.
  Retrain regularly, because iPhone prices drift quickly.
- `model.joblib` is a pickle: only load model files you trust.
