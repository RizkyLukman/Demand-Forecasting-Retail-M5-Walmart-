"""Klasifikasi pola permintaan Syntetos–Boylan (ADI & CV²).

- ADI (average demand interval) = jumlah hari / jumlah hari dengan penjualan > 0.
- CV² = (simpangan baku / rata-rata)² dari ukuran penjualan pada hari yang laku.
- Ambang: ADI = 1,32 dan CV² = 0,49.

| ADI < 1,32 | CV² < 0,49 → smooth       | CV² ≥ 0,49 → erratic |
| ADI ≥ 1,32 | CV² < 0,49 → intermittent | CV² ≥ 0,49 → lumpy   |

Dihitung per fold dari data train saja (hari <= cutoff), mulai dari penjualan pertama
tiap deret, tanpa hari tutup. Deret tanpa penjualan sampai cutoff = "no_history".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
SEGMENTS = ["smooth", "erratic", "intermittent", "lumpy", "no_history"]


def classify(hist: np.ndarray, closed_cols: np.ndarray | None = None) -> pd.DataFrame:
    """`hist`: (n_deret × hari) sampai cutoff. `closed_cols`: bool per kolom (hari tutup)."""
    x = hist.astype(np.float64)
    n, t = x.shape
    closed_cols = np.zeros(t, dtype=bool) if closed_cols is None else closed_cols
    sold = x > 0
    has = sold.any(axis=1)
    first = np.where(has, sold.argmax(axis=1), t)
    valid = (np.arange(t)[None, :] >= first[:, None]) & ~closed_cols[None, :]
    nz = sold & valid
    n_days = valid.sum(axis=1)
    n_nz = nz.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        adi = n_days / n_nz
        mean = (x * nz).sum(axis=1) / n_nz
        var = (x ** 2 * nz).sum(axis=1) / n_nz - mean ** 2
        cv2 = np.maximum(var, 0) / mean ** 2
    seg = np.select(
        [~has, (adi < ADI_THRESHOLD) & (cv2 < CV2_THRESHOLD), adi < ADI_THRESHOLD, cv2 < CV2_THRESHOLD],
        ["no_history", "smooth", "erratic", "intermittent"], default="lumpy")
    return pd.DataFrame({"adi": adi, "cv2": cv2, "segment": pd.Categorical(seg, categories=SEGMENTS)})
