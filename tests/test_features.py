"""Tes fitur Fase 4: anti-leakage untuk setiap fungsi fitur, dan kebenaran nilainya.

Memakai potongan data asli (40 item pertama CA_1 + kalender M5); dilewati jika data
tidak ada.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retailfc.calendar_features import (
    calendar_features, calendar_table, days_since_event, days_to_event, event_days,
)
from retailfc.features import (
    check_layout, fold_features, price_features, sales_features, stockout_features,
    to_model_frame, zero_run_length,
)
from retailfc.folds import closed_day_mask
from retailfc.io import load_calendar, load_config
from retailfc.leakage import assert_no_leakage

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
RAW = ROOT / CFG["paths"]["raw_m5"]
PART = ROOT / CFG["paths"]["interim"] / "sales_long" / "store_id=CA_1" / "part-0.parquet"
F = CFG["features"]
# Cutoff fold yang benar-benar dipakai: dua fold Easter, akhir tahun, cv1 dan cv4.
CUTOFFS = [1164, 1514, 1773, 1801, 1885]

pytestmark = pytest.mark.skipif(not PART.exists() or not (RAW / "calendar.csv").exists(),
                                reason="data M5 / output Fase 1 tidak ada")


@pytest.fixture(scope="module")
def cal():
    return load_calendar(RAW)


@pytest.fixture(scope="module")
def df():
    long = pd.read_parquet(PART)
    items = long["item_id"].cat.categories[:40]
    return long[long["item_id"].isin(items)].reset_index(drop=True)


# --- bentuk data ------------------------------------------------------------------

def test_layout_ok_on_real_data(df):
    check_layout(df)


def test_layout_rejects_gap(df):
    with pytest.raises(ValueError, match="contiguous"):
        check_layout(df.drop(index=df.index[10]).reset_index(drop=True))


# --- anti-leakage: setiap fungsi fitur, beberapa cutoff ----------------------------

@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_sales_features_no_leakage(df, cal, cutoff):
    closed = closed_day_mask(df["d"].to_numpy(), cal, CFG)
    fn = lambda d, c: sales_features(d, c, closed, F["lags"], F["rolling_windows"])
    assert_no_leakage(fn, df, cutoff)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_stockout_features_no_leakage(df, cal, cutoff):
    closed = closed_day_mask(df["d"].to_numpy(), cal, CFG)
    assert_no_leakage(lambda d, c: stockout_features(d, c, closed), df, cutoff)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_price_features_no_leakage(df, cutoff):
    fn = lambda d, c: price_features(d, c, F["price_max_window"], F["discount_threshold"])
    assert_no_leakage(fn, df, cutoff)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_calendar_features_no_leakage(df, cal, cutoff):
    ctab = calendar_table(cal, "CA", F["moving_events"])
    assert_no_leakage(lambda d, c: calendar_features(d, c, ctab), df, cutoff)


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_fold_features_no_leakage(df, cal, cutoff):
    closed = closed_day_mask(df["d"].to_numpy(), cal, CFG)
    assert_no_leakage(lambda d, c: fold_features(d, c, closed), df, cutoff)


# --- kebenaran nilai ----------------------------------------------------------------

def test_lags_match_direct_lookup(df, cal):
    closed = closed_day_mask(df["d"].to_numpy(), cal, CFG)
    out = sales_features(df, 0, closed, F["lags"], F["rolling_windows"])
    sales = df["sales"].astype(float).where(~closed)
    lookup = pd.Series(sales.to_numpy(), index=pd.MultiIndex.from_arrays([df["item_id"].astype(str), df["d"]]))
    for k in (28, 364):
        keys = pd.MultiIndex.from_arrays([df["item_id"].astype(str), df["d"] - k])
        expected = lookup.reindex(keys).to_numpy()
        assert np.allclose(out[f"lag_{k}"].to_numpy(), expected, equal_nan=True), k


def test_rolling_mean_28_matches_manual(df, cal):
    closed = closed_day_mask(df["d"].to_numpy(), cal, CFG)
    out = sales_features(df, 0, closed, F["lags"], F["rolling_windows"])
    item = df["item_id"].cat.categories[0]
    m = (df["item_id"] == item).to_numpy()
    lag28 = out.loc[m, "lag_28"].to_numpy()
    i = 400   # satu baris di tengah deret
    assert out.loc[m, "rmean_28"].to_numpy()[i] == pytest.approx(np.nanmean(lag28[i - 27:i + 1]), rel=1e-5)


def test_christmas_is_masked_in_lags(df, cal):
    closed = closed_day_mask(df["d"].to_numpy(), cal, CFG)
    out = sales_features(df, 0, closed, F["lags"], F["rolling_windows"])
    xmas_plus_28 = df["d"].isin([1792 + 28]).to_numpy()   # 28 hari setelah Natal 2015
    assert xmas_plus_28.any()
    assert out.loc[xmas_plus_28, "lag_28"].isna().all()


def test_zero_run_matches_manual_loop(df, cal):
    closed = closed_day_mask(df["d"].to_numpy(), cal, CFG)
    run = zero_run_length(df, closed)
    for item in df["item_id"].cat.categories[:10]:
        m = (df["item_id"] == item).to_numpy()
        expected, r = [], 0
        for s, c in zip(df.loc[m, "sales"].to_numpy(), closed[m]):
            if c:
                pass                       # hari tutup: tidak menambah, tidak memutus
            elif s > 0:
                r = 0
            else:
                r += 1
            expected.append(r)
        assert run[m].tolist() == expected, item


def test_zero_run_lag28_is_run_28_days_earlier(df, cal):
    closed = closed_day_mask(df["d"].to_numpy(), cal, CFG)
    run = pd.Series(zero_run_length(df, closed), index=df.index)
    out = stockout_features(df, 0, closed)
    lookup = pd.Series(run.to_numpy(), index=pd.MultiIndex.from_arrays([df["item_id"].astype(str), df["d"]]))
    keys = pd.MultiIndex.from_arrays([df["item_id"].astype(str), df["d"] - 28])
    assert np.allclose(out["zero_run_lag28"].to_numpy(), lookup.reindex(keys).to_numpy(), equal_nan=True)


def test_wday_convention(cal):
    ctab = calendar_table(cal, "CA", {})
    assert cal.loc[cal["d_num"] == 1, "weekday"].iloc[0] == "Saturday"
    assert ctab.loc[1, "wday"] == 1


def test_easter_features(cal):
    ev = event_days(cal, "Easter")
    assert ev.tolist() == [86, 436, 793, 1178, 1528, 1885]   # 2011 ada di event_name_2
    d = np.array([1521, 1526, 1528, 1529, 1535, 1600])
    assert days_to_event(d, ev, 7).tolist() == [7, 2, 0, 7, 7, 7]
    assert days_since_event(d, ev, 7).tolist() == [7, 7, 0, 1, 7, 7]


def test_snap_follows_store_state(cal):
    for state in ("CA", "TX", "WI"):
        ctab = calendar_table(cal, state, {})
        assert (ctab["snap"].to_numpy() == cal[f"snap_{state}"].to_numpy()).all()


def test_price_rel_dept_averages_to_one(df):
    out = price_features(df, 0, F["price_max_window"], F["discount_threshold"])
    avg = out["price_rel_dept"].groupby([df["dept_id"], df["wm_yr_wk"]], observed=True).mean()
    assert np.allclose(avg.to_numpy(), 1.0, atol=1e-4)


def test_fold_features_use_train_rows_only(df, cal):
    closed = closed_day_mask(df["d"].to_numpy(), cal, CFG)
    cutoff = 1801
    out = fold_features(df, cutoff, closed)
    item = df["item_id"].cat.categories[0]
    m = ((df["item_id"] == item) & (df["d"] <= cutoff)).to_numpy() & ~closed
    assert out.loc[df["item_id"] == item, "te_item_mean"].iloc[0] == pytest.approx(df.loc[m, "sales"].mean(), rel=1e-5)


def test_item_encoding_options(df):
    X = df[["item_id", "sell_price"]]
    assert to_model_frame(X, ["item_id", "sell_price"], "label")["item_id"].dtype == np.int16
    assert isinstance(to_model_frame(X, ["item_id", "sell_price"], "category")["item_id"].dtype,
                      pd.CategoricalDtype)
