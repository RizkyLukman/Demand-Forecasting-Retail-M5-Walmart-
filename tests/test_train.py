"""Tes logika pendukung training (Fase 5): deteksi censored demand dan pemetaan
prediksi ke matriks lebar. Memakai potongan data asli (40 item pertama CA_1)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retailfc.features import zero_run_length
from retailfc.folds import Fold, closed_day_mask
from retailfc.io import load_calendar, load_config
from retailfc.train import censored_mask, to_wide

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
RAW = ROOT / CFG["paths"]["raw_m5"]
PART = ROOT / CFG["paths"]["interim"] / "sales_long" / "store_id=CA_1" / "part-0.parquet"

pytestmark = pytest.mark.skipif(not PART.exists() or not (RAW / "calendar.csv").exists(),
                                reason="data M5 / output Fase 1 tidak ada")


@pytest.fixture(scope="module")
def df():
    long = pd.read_parquet(PART)
    items = long["item_id"].cat.categories[:40]
    return long[long["item_id"].isin(items)].reset_index(drop=True)


@pytest.fixture(scope="module")
def closed(df):
    return closed_day_mask(df["d"].to_numpy(), load_calendar(RAW), CFG)


@pytest.mark.parametrize("cutoff", [1514, 1801, 1885])
def test_censored_mask_ignores_data_after_cutoff(df, closed, cutoff):
    a = censored_mask(df, cutoff, closed, 28, 0.5)
    pert = df.copy()
    after = pert["d"].to_numpy() > cutoff
    pert.loc[after, "sales"] = (pert.loc[after, "sales"] + 7).astype(pert["sales"].dtype)
    b = censored_mask(pert, cutoff, closed, 28, 0.5)
    assert np.array_equal(a, b)
    assert not a[after].any()                 # tidak pernah menandai baris setelah cutoff


def test_censored_rows_are_zero_days_of_busy_items_in_long_runs(df, closed):
    cutoff = 1885
    m = censored_mask(df, cutoff, closed, 28, 0.5)
    assert m.any()
    assert (df.loc[m, "sales"] == 0).all()
    upto = (df["d"] <= cutoff).to_numpy()
    sub = df[upto]
    share = 1 - (sub["sales"] == 0).groupby(sub["item_id"], observed=True).mean()
    flagged_items = df.loc[m, "item_id"].unique()
    # Hari tutup ikut di penyebut di sini, jadi bisa sedikit di bawah 0,5; beri toleransi kecil.
    assert (share.reindex(flagged_items) >= 0.49).all()
    run = zero_run_length(df[upto], closed[upto])
    assert run[m[upto]].max() >= 28


def test_to_wide_places_predictions_by_series_and_day():
    ids = pd.DataFrame({"item_id": ["A", "B", "A"], "store_id": ["S1", "S1", "S2"]})
    fold = Fold("t", "cv", 100, 28, 28)
    long = pd.DataFrame({"item_id": ["B", "A"], "d": [101, 128], "pred": [2.5, 7.0]})
    out = np.zeros((3, 28), dtype=np.float32)
    to_wide(long, ids, "S1", fold, out)
    assert out[1, 0] == 2.5 and out[0, 27] == 7.0 and out.sum() == 9.5
    with pytest.raises(ValueError):
        to_wide(pd.DataFrame({"item_id": ["Z"], "d": [101], "pred": [1.0]}), ids, "S1", fold, out)
