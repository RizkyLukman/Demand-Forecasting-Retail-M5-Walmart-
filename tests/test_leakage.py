"""Pengecekan leakage harus menangkap fitur yang bocor dan meloloskan fitur yang aman.

Berjalan pada potongan kecil data asli (30 item pertama CA_1); dilewati jika output
Fase 1 tidak ada.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retailfc.io import load_config
from retailfc.leakage import assert_no_leakage, leaking_columns

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
PART = ROOT / CFG["paths"]["interim"] / "sales_long" / "store_id=CA_1" / "part-0.parquet"
CUTOFF = 1857

pytestmark = pytest.mark.skipif(not PART.exists(), reason="run `python -m retailfc.io` first")


@pytest.fixture(scope="module")
def df():
    long = pd.read_parquet(PART)
    items = long["item_id"].cat.categories[:30]
    return long[long["item_id"].isin(items) & (long["d"] <= CUTOFF + 28)].reset_index(drop=True)


def by_item(df, col="sales"):
    return df.groupby("item_id", observed=True, sort=False)[col]


# --- fitur aman --------------------------------------------------------------

def safe_lags(df, cutoff):
    g = by_item(df)
    out = pd.DataFrame(index=df.index)
    for k in (28, 35, 364):
        out[f"lag_{k}"] = g.shift(k)
    lag28 = out["lag_28"]
    out["rmean_28"] = lag28.groupby(df["item_id"], observed=True).transform(
        lambda x: x.rolling(28, min_periods=1).mean())
    return out


def safe_item_mean(df, cutoff):
    """Target encoding yang dihitung hanya dari baris train."""
    m = df[df["d"] <= cutoff].groupby("item_id", observed=True)["sales"].mean()
    return pd.DataFrame({"item_mean": df["item_id"].map(m).astype(float)}, index=df.index)


def safe_price(df, cutoff):
    """Harga masa depan sudah diketahui di M5, jadi boleh dipakai."""
    return pd.DataFrame({"price_rel": df["sell_price"] / by_item(df, "sell_price").transform("mean")},
                        index=df.index)


# --- fitur bocor -------------------------------------------------------------

def leaky_short_lag(df, cutoff):
    return pd.DataFrame({"lag_7": by_item(df).shift(7)}, index=df.index)


def leaky_item_mean(df, cutoff):
    return pd.DataFrame({"item_mean_all": by_item(df).transform("mean")}, index=df.index)


def leaky_centred_rolling(df, cutoff):
    return pd.DataFrame({"rmean_c": by_item(df).transform(
        lambda x: x.rolling(7, center=True, min_periods=1).mean())}, index=df.index)


@pytest.mark.parametrize("fn", [safe_lags, safe_item_mean, safe_price])
def test_safe_features_pass(df, fn):
    assert_no_leakage(fn, df, CUTOFF)


@pytest.mark.parametrize("fn,col", [(leaky_short_lag, "lag_7"), (leaky_item_mean, "item_mean_all"),
                                    (leaky_centred_rolling, "rmean_c")])
def test_leaky_features_are_caught(df, fn, col):
    assert leaking_columns(fn, df, CUTOFF) == [col]


def test_lag_28_is_safe_only_up_to_horizon_28(df):
    """Lag 28 aman untuk horizon 28 hari, tetapi tidak untuk horizon yang lebih panjang."""
    def lag28(d, c):
        return pd.DataFrame({"lag_28": by_item(d).shift(28)}, index=d.index)
    assert leaking_columns(lag28, df, CUTOFF - 1, horizon=28) == []
    assert leaking_columns(lag28, df, CUTOFF - 1, horizon=29) == ["lag_28"]


def test_check_refuses_data_without_future_rows(df):
    with pytest.raises(ValueError, match="vacuous"):
        leaking_columns(safe_lags, df[df["d"] <= CUTOFF], CUTOFF)
