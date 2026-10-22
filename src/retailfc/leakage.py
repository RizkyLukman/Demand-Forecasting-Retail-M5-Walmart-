"""Pengecekan anti-leakage untuk fungsi fitur (Fase 3).

Fungsi fitur punya signature `fn(df, cutoff) -> DataFrame` dan mengembalikan satu
baris per baris input (index sama). Fungsi itu bebas leakage untuk suatu cutoff jika
tidak ada nilai fitur di baris d <= cutoff + horizon yang berubah ketika semua nilai
target setelah cutoff diganti.

Hanya target yang diacak. Harga dan kalender untuk hari mendatang sudah diketahui
di M5 (sell_prices.csv mencakup minggu-minggu forecast), jadi fitur boleh memakainya.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

FeatureFn = Callable[[pd.DataFrame, int], pd.DataFrame]


def leaking_columns(
    fn: FeatureFn,
    df: pd.DataFrame,
    cutoff: int,
    horizon: int = 28,
    target: str = "sales",
    seed: int = 0,
) -> list[str]:
    """Nama kolom fitur yang nilainya bergantung pada target setelah `cutoff`."""
    future = df["d"].to_numpy() > cutoff
    if not future.any():
        raise ValueError("df has no rows after the cutoff; the check would be vacuous")

    perturbed = df.copy()
    rng = np.random.default_rng(seed)
    noise = rng.integers(1, 500, size=int(future.sum()))
    vals = perturbed[target].to_numpy().copy()
    vals[future] = (vals[future].astype(np.int64) + noise).astype(vals.dtype)
    perturbed[target] = vals

    a = fn(df.copy(), cutoff)
    b = fn(perturbed, cutoff)
    if not a.index.equals(df.index) or not b.index.equals(df.index):
        raise ValueError("feature function must return one row per input row, same index")

    rows = df["d"].to_numpy() <= cutoff + horizon
    leaks = []
    for col in a.columns:
        x, y = a.loc[rows, col], b.loc[rows, col]
        if isinstance(x.dtype, pd.CategoricalDtype) or x.dtype == object:
            same = x.astype(str).equals(y.astype(str))
        else:
            same = np.allclose(x.to_numpy(np.float64), y.to_numpy(np.float64), equal_nan=True)
        if not same:
            leaks.append(col)
    return leaks


def assert_no_leakage(fn: FeatureFn, df: pd.DataFrame, cutoff: int, **kw) -> None:
    leaks = leaking_columns(fn, df, cutoff, **kw)
    if leaks:
        raise AssertionError(f"features use targets after d_{cutoff}: {leaks}")
