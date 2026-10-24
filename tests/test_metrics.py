"""Verifikasi WRMSSE (Fase 5).

1. Bobot untuk cutoff d_1913 dibandingkan dengan `weights_validation.csv` resmi M5
   (dihitung penyelenggara dari d_1886–d_1913, jadi tidak memakai holdout).
2. Skor dibandingkan dengan implementasi kedua yang ditulis terpisah di file ini,
   mengikuti logika `datasetsforecast.M5Evaluation` (groupby pandas, skala per baris),
   pada fold cv4 (cutoff d_1885, horizon d_1886–d_1913).

Semua data dimuat hanya sampai d_1913: holdout (d_1914–d_1941) tidak dibaca.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retailfc.io import load_config, load_wide_matrices
from retailfc.metrics import LEVELS, WRMSSE, basic_errors, naive_scale, pinball_loss

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
RAW = ROOT / CFG["paths"]["raw_m5"]
OFFICIAL = ROOT / "data" / "raw" / "m5_reference" / "weights_validation.csv"
MAX_D = CFG["validation"]["holdout_cutoff"]          # 1913: batas data yang boleh dibaca

needs_data = pytest.mark.skipif(not (RAW / "sales_train_evaluation.csv").exists(),
                                reason="data M5 tidak ada")


@pytest.fixture(scope="module")
def wide():
    ids, sales, prices = load_wide_matrices(CFG, MAX_D)
    assert sales.shape[1] == MAX_D
    return ids, sales, prices


# --- sifat dasar (tanpa data) ---------------------------------------------------

def test_naive_scale_starts_at_first_nonzero():
    x = np.array([[0, 0, 2, 4, 4, 1], [0, 0, 0, 0, 0, 3]], dtype=float)
    s = naive_scale(x)
    assert s[0] == pytest.approx(((4 - 2) ** 2 + 0 + (1 - 4) ** 2) / 3)
    assert np.isnan(s[1])                     # tidak ada selisih setelah nilai bukan-nol pertama


def test_pinball_and_basic_errors():
    y = np.array([0.0, 2.0, 5.0])
    assert pinball_loss(y, y, 0.9) == 0
    assert pinball_loss(y, y + 1, 0.9) == pytest.approx(0.1)
    assert pinball_loss(y, y - 1, 0.9) == pytest.approx(0.9)
    e = basic_errors(y, y + 1)
    assert (e["mae"], e["rmse"], e["bias"]) == (1.0, 1.0, 1.0)


# --- verifikasi terhadap data asli ------------------------------------------------

@needs_data
@pytest.mark.skipif(not OFFICIAL.exists(), reason="weights_validation.csv resmi tidak ada")
def test_weights_match_official_m5(wide):
    ids, sales, prices = wide
    ev = WRMSSE(ids, sales, prices, cutoff=1913)
    assert ev.n_excluded_series == 0          # di cutoff resmi tidak ada deret yang dikeluarkan
    ours = ev.weights_frame()
    official = pd.read_csv(OFFICIAL)
    m = official.merge(ours, on=["Level_id", "Agg_Level_1", "Agg_Level_2"], how="outer",
                       suffixes=("_off", "_ours"), indicator=True)
    assert (m["_merge"] == "both").all(), m[m["_merge"] != "both"].head()
    assert len(m) == 42840
    # Harga disimpan float32, jadi penjumlahan jutaan baris memberi selisih relatif ~1e-7
    # (terukur: maks 3,8e-7). Toleransi relatif dipakai, bukan absolut.
    assert np.allclose(m["Dollar_Sales_ours"], m["Dollar_Sales_off"], rtol=1e-6, atol=0.005)
    assert np.abs(m["weight_off"] - m["weight_ours"]).max() < 1e-8
    assert ((m["Dollar_Sales_off"] == 0) == (m["Dollar_Sales_ours"] == 0)).all()


def independent_wrmsse(ids, hist, actual, pred, weights):
    """Implementasi kedua: logika M5Evaluation (groupby pandas, skala per baris)."""
    def aggregate(mat):
        df = pd.DataFrame(mat).join(ids.astype(str)).assign(total="Total")
        out = []
        for level, keys in LEVELS:
            k = keys or ["total"]
            g = df.groupby(k).sum(numeric_only=True).reset_index()
            g = g.rename(columns=dict(zip(k, ["Agg_Level_1", "Agg_Level_2"])))
            out.append(g.assign(Level_id=level))
        a = pd.concat(out).fillna("X")
        return a.set_index(["Level_id", "Agg_Level_1", "Agg_Level_2"]).sort_index()

    def scale(x):
        x = x.values
        x = x[np.argmax(x != 0):]
        return ((x[1:] - x[:-1]) ** 2).mean()

    scales = aggregate(hist).agg(scale, axis=1).rename("scale")
    mse = ((aggregate(actual) - aggregate(pred)) ** 2).mean(axis=1).rename("mse")
    s = pd.concat([mse, scales], axis=1).join(
        weights.set_index(["Level_id", "Agg_Level_1", "Agg_Level_2"])["weight"])
    s["contrib"] = np.sqrt(s["mse"] / s["scale"]).where(s["weight"] > 0, 0) * s["weight"]
    per_level = s.groupby(level="Level_id")["contrib"].sum()
    return float(per_level.mean()), per_level


@needs_data
def test_score_matches_independent_implementation_on_cv4(wide):
    ids, sales, prices = wide
    cutoff = 1885
    hist = sales[:, :cutoff].astype(float)
    actual = sales[:, cutoff:cutoff + 28].astype(float)
    # Forecast uji: seasonal naive (7 hari terakhir diulang).
    pred = np.tile(hist[:, -7:], 4)
    ev = WRMSSE(ids, sales, prices, cutoff=cutoff)
    ours = ev.score(actual, pred)
    ref_total, ref_levels = independent_wrmsse(ids[["item_id", "dept_id", "cat_id", "store_id", "state_id"]],
                                               hist, actual, pred, ev.weights_frame())
    assert ours["wrmsse"] == pytest.approx(ref_total, rel=1e-9)
    for level, v in ours["per_level"].items():
        assert v == pytest.approx(ref_levels[level], rel=1e-6, abs=1e-9), level


@needs_data
def test_perfect_forecast_scores_zero(wide):
    ids, sales, prices = wide
    ev = WRMSSE(ids, sales, prices, cutoff=1857)
    actual = sales[:, 1857:1885].astype(float)
    assert ev.score(actual, actual)["wrmsse"] == 0.0
