"""Checks the Phase 1 output against the raw M5 files.

Uses the real data in data/raw/m5 and data/interim; skipped when it is not present
(raw data is not tracked by git).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retailfc.io import (
    CAT_COLS, load_calendar, load_config, load_prices, load_sales_wide, store_to_long,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
RAW = ROOT / CFG["paths"]["raw_m5"]
LONG = ROOT / CFG["paths"]["interim"] / "sales_long"

pytestmark = pytest.mark.skipif(
    not (RAW / "sales_train_evaluation.csv").exists(), reason="raw M5 data not present"
)


@pytest.fixture(scope="module")
def raw():
    wide = load_sales_wide(RAW / CFG["m5"]["sales_file"])
    return {
        "wide": wide,
        "cal": load_calendar(RAW),
        "prices": load_prices(RAW),
        "d_cols": [c for c in wide.columns if c.startswith("d_")],
        "categories": {c: pd.Index(sorted(wide[c].astype(str).unique())) for c in CAT_COLS},
    }


def test_small_real_slice(raw):
    """First 20 series of CA_1, first 120 days: drop rule, values and price join."""
    wide = raw["wide"]
    d_cols = raw["d_cols"][:120]
    ws = wide[wide["store_id"] == "CA_1"].head(20)
    long, stats = store_to_long(ws, raw["prices"], raw["cal"], d_cols, raw["categories"])

    values = ws[d_cols].to_numpy()
    for i, (_, row) in enumerate(ws.iterrows()):
        sub = long[long["item_id"] == row["item_id"]]
        nz = np.flatnonzero(values[i] > 0)
        if len(nz) == 0:
            assert sub.empty
            continue
        # Starts exactly at the first sale and keeps every later day.
        assert sub["d"].iloc[0] == nz[0] + 1
        assert sub["d"].tolist() == list(range(nz[0] + 1, len(d_cols) + 1))
        assert np.array_equal(sub["sales"].to_numpy(), values[i, nz[0]:])

    # Prices equal a direct lookup in sell_prices.csv.
    p = raw["prices"]
    lookup = p[p["store_id"] == "CA_1"].astype({"item_id": str}).set_index(["item_id", "wm_yr_wk"])["sell_price"]
    keys = list(zip(long["item_id"].astype(str), long["wm_yr_wk"]))
    expected = lookup.reindex(keys).to_numpy()
    assert np.allclose(long["sell_price"].to_numpy(), expected, equal_nan=True)
    # Dropped cells are all zero, so no units are lost.
    assert stats["sales_total_wide"] == stats["sales_total_long"] == int(values.sum())


@pytest.mark.skipif(not LONG.exists(), reason="run `python -m retailfc.io` first")
@pytest.mark.parametrize("store", ["CA_1", "TX_3", "WI_2"])
def test_parquet_matches_wide(raw, store):
    long = pd.read_parquet(LONG / f"store_id={store}" / "part-0.parquet")
    ws = raw["wide"][raw["wide"]["store_id"] == store]
    values = ws[raw["d_cols"]].to_numpy()

    # Dtypes as specified in PROJECT_PLAN.md (Phase 1).
    assert long["sales"].dtype == np.int16
    assert long["d"].dtype == np.int16
    assert long["wm_yr_wk"].dtype == np.int16
    assert long["sell_price"].dtype == np.float32
    for c in ["item_id", "dept_id", "cat_id", "state_id"]:
        assert isinstance(long[c].dtype, pd.CategoricalDtype)

    # Totals and row count.
    first = (values > 0).argmax(axis=1)
    assert len(long) == int((values.shape[1] - first).sum())
    assert int(long["sales"].sum()) == int(values.sum(dtype=np.int64))
    assert not long.duplicated(["item_id", "d"]).any()
    assert long["sell_price"].notna().all()

    # Random sample of series: exact values from first sale onward.
    rng = np.random.default_rng(0)
    for i in rng.choice(len(ws), size=25, replace=False):
        item = ws["item_id"].iloc[i]
        sub = long[long["item_id"] == item]
        assert np.array_equal(sub["sales"].to_numpy(), values[i, first[i]:])
