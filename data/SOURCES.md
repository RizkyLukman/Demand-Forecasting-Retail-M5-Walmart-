# Data sources

Every number below comes from `python -m retailfc.verify_m5 --config config.yaml`
(full output: `reports/m5_verification.json`). Raw files live in `data/raw/` and are
not tracked by git.

---

## 1. M5 Forecasting – Accuracy (Walmart)

| Field | Value |
|---|---|
| URL | https://www.kaggle.com/competitions/m5-forecasting-accuracy/data |
| Download | `kaggle competitions download -c m5-forecasting-accuracy` (Kaggle CLI 2.2.4) |
| Access date | 2026-10-22 |
| License / terms | **Not verified yet.** Governed by the competition rules accepted on Kaggle (Late Submission). The "Competition Data" section of https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules must be read while logged in and quoted here. |
| Location | `data/raw/m5/` |

### Files and checksums

| File | Bytes | SHA-256 |
|---|---:|---|
| `m5-forecasting-accuracy.zip` | 48,009,163 | `0349ba38a2efd30d0f5acc6394c1110e140e1a990c650d7b5ca44c5b25dd12f5` |
| `calendar.csv` | 103,469 | `d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5` |
| `sales_train_evaluation.csv` | 121,736,518 | `4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0` |
| `sales_train_validation.csv` | 120,007,726 | `f368e66ed1dbecb48b2cc8fc589bf68b3deddbbb36bf5c88b4d6d0a09b9b6724` |
| `sample_submission.csv` | 5,228,786 | `eb7597afbec367475ef25dc6dbbc009b8faa47fa2107252c72a029e3cdf72343` |
| `sell_prices.csv` | 203,395,785 | `9da3ad1f8b8ccacdbdc70612191dd375ec24a4ac6625c24b75b3bc60b0bed2ef` |

### Schema check (documented vs actual)

All three schemas in PROJECT_PLAN.md §2.1 **match the files exactly** (same columns, same order,
nothing missing, nothing extra):

- `calendar.csv`: `date, wm_yr_wk, weekday, wday, month, year, d, event_name_1, event_type_1, event_name_2, event_type_2, snap_CA, snap_TX, snap_WI`
- `sell_prices.csv`: `store_id, item_id, wm_yr_wk, sell_price`
- `sales_train_evaluation.csv`: `id, item_id, dept_id, cat_id, store_id, state_id, d_1 … d_1941`

### Verified facts

**Calendar**
- 1,969 rows, `d_1` … `d_1969`, 2011-01-29 → 2016-06-19, no gaps in dates or `d`.
- The calendar runs 28 days beyond the sales data (`d_1942` … `d_1969` = the
  competition's private evaluation window, no labels).
- 162 event days; 5 days carry a second event (`event_name_2`).
- Event types: Cultural, National, Religious, Sporting.
- **Easter** (from the file): d_86 2011-04-24, d_436 2012-04-08, d_793 2013-03-31,
  d_1178 2014-04-20, d_1528 2015-04-05, d_1885 2016-03-27.
  In 2011 Easter is in `event_name_2` (with OrthodoxEaster in `event_name_1`), so event
  features must read **both** event columns.

**Sales (`sales_train_evaluation.csv`)**
- 30,490 series (unique `id`), 3,049 items, 10 stores (CA_1–4, TX_1–3, WI_1–3),
  7 departments, 3 categories (FOODS, HOBBIES, HOUSEHOLD), 3 states.
- `d_1` … `d_1941` contiguous; all present in `calendar.csv`.
- Values 0 … 763 → fit in `int16`. In memory as `int16`: 115.6 MB.
- **Rows after melt: 59,181,090** (30,490 × 1,941). The estimate in PROJECT_PLAN.md is correct.
- Zero share, all cells: 68.0%.
- Cells before each series' first sale: 12,384,870. Every series has at least one sale.
  Removing them (Phase 1) leaves **46,796,220** rows, with a zero share of 59.5%.

**Prices (`sell_prices.csv`)**
- 6,841,121 rows, 10 stores, 3,049 items, `wm_yr_wk` 11101 … 11621.
- Price range 0.01 … 107.32, no NaN, no duplicate (store, item, week) keys.

**Validation vs evaluation file**
- `sales_train_validation.csv` ends at `d_1913`; same series order as the evaluation
  file, and days `d_1` … `d_1913` are **identical**. The evaluation file is a strict
  superset, so only it is used.

**Resource use of the check:** peak RAM 0.57 GB, 7.4 s.

### Fold boundaries (dates from `calendar.csv`)

| Fold | Cutoff | Horizon | Dates of horizon |
|---|---|---|---|
| CV 1 | d_1801 | d_1802–d_1829 | 2016-01-04 → 2016-01-31 |
| CV 2 | d_1829 | d_1830–d_1857 | 2016-02-01 → 2016-02-28 |
| CV 3 | d_1857 | d_1858–d_1885 | 2016-02-29 → 2016-03-27 |
| CV 4 | d_1885 | d_1886–d_1913 | 2016-03-28 → 2016-04-24 |
| Holdout | d_1913 | d_1914–d_1941 | 2016-04-25 → 2016-05-22 |

Note for P4: Easter 2016 (d_1885) is the **last day of CV 3's horizon**, and CV 4
starts the day after. No single fold covers the window around Easter 2016. The Easter
ablation needs extra folds around earlier Easters (e.g., 2015-04-05 = d_1528).

---

## 2. Retail sales dataset of a pharmacy in Indonesia

Not downloaded yet (Phase 8). DOI `10.17632/2ym7v78wtd.1`, CC BY 4.0.
