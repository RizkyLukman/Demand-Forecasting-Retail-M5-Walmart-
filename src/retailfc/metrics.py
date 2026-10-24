"""Fase 5: metrik evaluasi. WRMSSE (metrik resmi M5), RMSSE, MAE, bias, pinball.

WRMSSE mengikuti panduan kompetisi M5:
- 42.840 deret di 12 level hierarki (total, state, store, cat, dept, state×cat,
  state×dept, store×cat, store×dept, item, item×state, item×store).
- Skala per deret: rata-rata kuadrat selisih naive satu langkah pada data latih,
  dihitung mulai dari penjualan bukan-nol pertama deret itu.
- Bobot per deret: penjualan dalam dolar selama 28 hari terakhir sebelum cutoff,
  dinormalisasi agar setiap level berjumlah 1.
- WRMSSE = (1/12) · Σ_level Σ_deret bobot · RMSSE.

Implementasi ini diverifikasi di `tests/test_metrics.py`:
- bobot untuk cutoff d_1913 dibandingkan dengan `weights_validation.csv` resmi M5;
- skor dibandingkan dengan implementasi kedua yang ditulis terpisah (logika groupby
  seperti `datasetsforecast.M5Evaluation`), pada fold CV (holdout tidak dipakai).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Urutan dan kunci level sama dengan file bobot resmi M5.
LEVELS: list[tuple[str, list[str]]] = [
    ("Level1", []),
    ("Level2", ["state_id"]),
    ("Level3", ["store_id"]),
    ("Level4", ["cat_id"]),
    ("Level5", ["dept_id"]),
    ("Level6", ["state_id", "cat_id"]),
    ("Level7", ["state_id", "dept_id"]),
    ("Level8", ["store_id", "cat_id"]),
    ("Level9", ["store_id", "dept_id"]),
    ("Level10", ["item_id"]),
    ("Level11", ["state_id", "item_id"]),
    ("Level12", ["item_id", "store_id"]),
]


def build_aggregation(ids: pd.DataFrame) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Matriks penjumlahan S (42.840 × n_bottom) dan label tiap baris agregat.

    `ids` = kolom ID deret paling bawah, satu baris per deret item-toko.
    Label memakai format file bobot resmi: Level_id, Agg_Level_1, Agg_Level_2.
    """
    n = len(ids)
    blocks, labels = [], []
    for level, keys in LEVELS:
        if not keys:
            codes = np.zeros(n, dtype=np.int64)
            lab = pd.DataFrame({"Agg_Level_1": ["Total"], "Agg_Level_2": ["X"]})
        else:
            key_df = ids[keys].astype(str)
            uniq = key_df.drop_duplicates().sort_values(keys).reset_index(drop=True)
            codes = key_df.merge(uniq.reset_index(), on=keys, how="left")["index"].to_numpy()
            lab = pd.DataFrame({"Agg_Level_1": uniq[keys[0]],
                                "Agg_Level_2": uniq[keys[1]] if len(keys) > 1 else "X"})
        m = len(lab)
        blocks.append(sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(m, n)))
        labels.append(lab.assign(Level_id=level))
    S = sparse.vstack(blocks).tocsr()
    lab = pd.concat(labels, ignore_index=True)[["Level_id", "Agg_Level_1", "Agg_Level_2"]]
    return S, lab


def naive_scale(history: np.ndarray) -> np.ndarray:
    """Rata-rata kuadrat selisih satu langkah, mulai dari nilai bukan-nol pertama tiap baris."""
    nz = history != 0
    first = np.where(nz.any(axis=1), nz.argmax(axis=1), history.shape[1])
    diff2 = np.diff(history, axis=1) ** 2                      # selisih ke-j = x[j+1] - x[j]
    valid = np.arange(diff2.shape[1])[None, :] >= first[:, None]
    count = valid.sum(axis=1)
    total = np.where(valid, diff2, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count


class WRMSSE:
    """Evaluator WRMSSE untuk satu cutoff.

    `sales`: matriks penjualan lebar (n_bottom × hari), kolom ke-j = d_(j+1).
    `prices`: matriks harga harian yang sejajar dengan `sales` (NaN jika tidak ada harga).
    """

    def __init__(self, ids: pd.DataFrame, sales: np.ndarray, prices: np.ndarray,
                 cutoff: int, horizon: int = 28):
        if sales.shape[1] < cutoff:
            raise ValueError("sales must cover d_1..d_cutoff")
        self.ids = ids.reset_index(drop=True)
        self.cutoff, self.horizon = cutoff, horizon
        self.S, self.labels = build_aggregation(self.ids)

        hist = sales[:, :cutoff].astype(np.float64)              # d_1..d_cutoff
        self.scale = naive_scale(self.S @ hist)

        last = slice(cutoff - 28, cutoff)                        # 28 hari terakhir <= cutoff
        dollars = np.nan_to_num(sales[:, last] * prices[:, last]).sum(axis=1)
        agg_dollars = self.S @ dollars
        # Deret yang baru laku di hari-hari terakhir sebelum cutoff bisa punya bobot positif
        # tetapi skala tak terdefinisi (tidak ada selisih naive) atau nol. Di cutoff resmi M5
        # (d_1913) ini tidak terjadi; di cutoff lain deret seperti itu dikeluarkan dari bobot
        # (bobot level dinormalisasi ulang) dan jumlahnya dicatat.
        invalid = (agg_dollars > 0) & (~np.isfinite(self.scale) | (self.scale == 0))
        self.n_excluded_series = int(invalid.sum())
        self.excluded_labels = self.labels[invalid].copy()
        agg_dollars = np.where(invalid, 0.0, agg_dollars)
        level = self.labels["Level_id"].to_numpy()
        level_total = pd.Series(agg_dollars).groupby(level).transform("sum").to_numpy()
        self.dollars = agg_dollars
        self.weight = agg_dollars / level_total

    def weights_frame(self) -> pd.DataFrame:
        return self.labels.assign(Dollar_Sales=self.dollars, weight=self.weight)

    def score(self, actual: np.ndarray, pred: np.ndarray) -> dict:
        """`actual`, `pred`: (n_bottom × horizon), urutan baris sama dengan `ids`."""
        if actual.shape != pred.shape or actual.shape[0] != len(self.ids):
            raise ValueError("actual/pred must be (n_bottom, horizon)")
        err = self.S @ (actual.astype(np.float64) - pred.astype(np.float64))
        mse = (err ** 2).mean(axis=1)
        use = self.weight > 0
        if np.isnan(self.scale[use]).any() or (self.scale[use] == 0).any():
            raise ValueError("a series with positive weight has no valid scale")
        rmsse = np.zeros_like(mse)
        rmsse[use] = np.sqrt(mse[use] / self.scale[use])
        contrib = self.weight * rmsse
        per_level = pd.Series(contrib).groupby(self.labels["Level_id"].to_numpy()).sum()
        per_level = per_level.reindex([lv for lv, _ in LEVELS])
        return {"wrmsse": float(per_level.mean()), "per_level": per_level.round(6).to_dict(),
                "n_excluded_series": self.n_excluded_series}

    def rmsse_bottom(self, actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
        """RMSSE per deret item-toko (Level12), urutan sama dengan `ids`."""
        lv12 = (self.labels["Level_id"] == "Level12").to_numpy()
        scale12 = self.scale[lv12]
        # Baris Level12 diurutkan per (item, store); petakan kembali ke urutan `ids`.
        lab12 = self.labels[lv12].reset_index(drop=True)
        key = pd.MultiIndex.from_frame(self.ids[["item_id", "store_id"]].astype(str))
        pos = pd.MultiIndex.from_frame(lab12[["Agg_Level_1", "Agg_Level_2"]]).get_indexer(key)
        mse = ((actual.astype(np.float64) - pred) ** 2).mean(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(mse / scale12[pos])


def pinball_loss(actual: np.ndarray, pred_q: np.ndarray, alpha: float) -> float:
    """Pinball (quantile) loss rata-rata untuk kuantil `alpha`."""
    diff = actual - pred_q
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def basic_errors(actual: np.ndarray, pred: np.ndarray) -> dict:
    """MAE, RMSE, dan bias (rata-rata prediksi - aktual) di level item-toko-hari."""
    e = pred.astype(np.float64) - actual
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "bias": float(e.mean())}
