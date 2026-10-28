"""Fase 4: fitur lag/rolling, harga, kalender, dan target encoding.

Dua lapis:
1. **Fitur dasar** (tidak bergantung cutoff), dihitung sekali per toko dan disimpan ke
   data/processed/base/store_id=<id>/part-0.parquet:
   lag >= 28, rolling dari lag 28, panjang deretan nol per 28 hari lalu (indikasi stok
   habis), kalender + SNAP + hari libur bergerak, dan fitur harga yang hanya membaca
   harga (harga masa depan sudah diketahui di M5).
   Untuk horizon <= 28 hari, fitur ini hanya membaca penjualan <= cutoff. Ini dibuktikan
   oleh tes anti-leakage untuk beberapa cutoff, bukan diasumsikan.
2. **Fitur per fold** (bergantung cutoff), dihitung saat training:
   target encoding item/dept dan harga relatif terhadap rata-rata harga item,
   semuanya hanya dari baris d <= cutoff.

Hari tutup (config `closed_days`) diberi sales = NaN sebelum lag/rolling dihitung,
supaya tidak ikut ke statistik rolling, dan tidak ikut dihitung di target encoding.

Setiap fungsi fitur bertanda tangan `fn(df, cutoff, ...) -> DataFrame` dengan index
sama seperti `df`. `df` harus satu toko, terurut per item lalu d, dengan hari berurutan.

Cara pakai:
    python -m retailfc.features --config config.yaml [--stores CA_1 TX_2]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retailfc.calendar_features import calendar_features, calendar_table
from retailfc.folds import closed_day_mask
from retailfc.io import load_calendar, load_config
from retailfc.profiling import PeakRSS

ID_COLS = ["item_id", "dept_id", "cat_id"]


# --- pemeriksaan bentuk data ---------------------------------------------------

def check_layout(df: pd.DataFrame) -> None:
    """Pastikan satu baris per item per hari, terurut, tanpa hari yang bolong.

    Lag dihitung dengan `shift(k)` per item, yang hanya sama dengan "k hari lalu" jika
    hari-harinya berurutan tanpa celah.
    """
    item = df["item_id"].cat.codes.to_numpy()
    d = df["d"].to_numpy().astype(np.int32)
    same_item = item[1:] == item[:-1]
    if not (np.diff(d)[same_item] == 1).all():
        raise ValueError("days per item must be contiguous and sorted")
    starts = np.flatnonzero(~same_item) + 1
    if len(np.unique(item[np.r_[0, starts]])) != len(starts) + 1:
        raise ValueError("rows of one item must be in one block")


# --- fitur dasar (tidak bergantung cutoff) ------------------------------------

def sales_features(df: pd.DataFrame, cutoff: int, closed: np.ndarray,
                   lags: list[int], windows: list[int]) -> pd.DataFrame:
    """Lag penjualan dan statistik rolling dari lag 28. `cutoff` tidak dipakai: semua lag
    >= 28 hari, jadi untuk horizon 28 hari hanya membaca data <= cutoff."""
    if min(lags) < 28:
        raise ValueError("lags < 28 need a recursive pipeline (see PROJECT_PLAN.md, Fase 4)")
    sales = df["sales"].astype(np.float32).to_numpy().copy()
    sales[closed] = np.nan                    # hari tutup bukan permintaan
    s = pd.Series(sales, index=df.index)
    key = df["item_id"]
    g = s.groupby(key, observed=True, sort=False)
    out = pd.DataFrame(index=df.index)
    for k in lags:
        out[f"lag_{k}"] = g.shift(k).astype(np.float32)
    g28 = out["lag_28"].groupby(key, observed=True, sort=False)
    for w in windows:
        r = g28.rolling(w, min_periods=1)
        out[f"rmean_{w}"] = r.mean().reset_index(level=0, drop=True).astype(np.float32)
        r = g28.rolling(w, min_periods=2)
        out[f"rstd_{w}"] = r.std().reset_index(level=0, drop=True).astype(np.float32)
    return out


def zero_run_length(df: pd.DataFrame, closed: np.ndarray) -> np.ndarray:
    """Panjang deretan hari tanpa penjualan yang berakhir di hari itu (0 jika hari itu laku).

    Hari tutup tidak menambah dan tidak memutus deretan. Nilai di hari d hanya bergantung
    pada penjualan <= d.
    """
    sales = df["sales"].to_numpy()
    item = df["item_id"].cat.codes.to_numpy()
    new_item = np.r_[True, item[1:] != item[:-1]]
    reset = ((sales > 0) & ~closed) | new_item
    inc = ((sales == 0) & ~closed).astype(np.int32)
    return pd.Series(inc).groupby(np.cumsum(reset)).cumsum().to_numpy()


def stockout_features(df: pd.DataFrame, cutoff: int, closed: np.ndarray) -> pd.DataFrame:
    """Indikasi stok habis / produk ditarik: panjang deretan nol per 28 hari lalu.

    Item yang biasanya laku tetapi tidak terjual berminggu-minggu hampir pasti sedang
    kosong di rak (censored demand), bukan tanpa pembeli. `cutoff` tidak dipakai:
    nilainya digeser 28 hari, jadi untuk horizon 28 hari hanya membaca data <= cutoff.
    """
    run = pd.Series(zero_run_length(df, closed).astype(np.float32), index=df.index)
    out = pd.DataFrame(index=df.index)
    out["zero_run_lag28"] = run.groupby(df["item_id"], observed=True, sort=False).shift(28).astype(np.float32)
    return out


def price_features(df: pd.DataFrame, cutoff: int, max_window: int, discount_threshold: float) -> pd.DataFrame:
    """Fitur harga yang hanya membaca harga. `cutoff` tidak dipakai (harga masa depan
    diketahui di M5)."""
    p = df["sell_price"].astype(np.float32)
    gp = p.groupby(df["item_id"], observed=True, sort=False)
    out = pd.DataFrame(index=df.index)
    out["sell_price"] = p
    out["price_change_7d"] = (p / gp.shift(7) - 1).astype(np.float32)
    dept_week = p.groupby([df["dept_id"], df["wm_yr_wk"]], observed=True).transform("mean")
    out["price_rel_dept"] = (p / dept_week).astype(np.float32)
    pmax = gp.rolling(max_window, min_periods=1).max().reset_index(level=0, drop=True)
    out["price_rel_max"] = (p / pmax).astype(np.float32)
    out["is_discount"] = (out["price_rel_max"] < discount_threshold).astype(np.int8)
    return out


# --- fitur per fold (bergantung cutoff) ---------------------------------------

def fold_features(df: pd.DataFrame, cutoff: int, closed: np.ndarray,
                  recent_windows: tuple[int, ...] = ()) -> pd.DataFrame:
    """Target encoding dan harga relatif rata-rata item, dari baris d <= cutoff saja.

    `recent_windows`: untuk tiap W, tambahkan `te_item_mean_W` = rata-rata penjualan item di
    W hari terakhir sampai cutoff (hari tutup dikeluarkan). Default kosong: perilaku lama.
    """
    d = df["d"].to_numpy()
    train = (d <= cutoff) & ~closed
    tr = df.loc[train]
    sales = tr["sales"].astype(np.float64)
    item_mean = sales.groupby(tr["item_id"], observed=True).mean()
    item_zero = (sales == 0).groupby(tr["item_id"], observed=True).mean()
    dept_mean = sales.groupby(tr["dept_id"], observed=True).mean()
    price_mean = tr["sell_price"].astype(np.float64).groupby(tr["item_id"], observed=True).mean()

    out = pd.DataFrame(index=df.index)
    out["te_item_mean"] = df["item_id"].map(item_mean).astype(np.float32)
    out["te_item_zero_share"] = df["item_id"].map(item_zero).astype(np.float32)
    out["te_dept_mean"] = df["dept_id"].map(dept_mean).astype(np.float32)
    out["price_rel_item"] = (df["sell_price"] / df["item_id"].map(price_mean).astype(np.float32)).astype(np.float32)
    for w in recent_windows:
        m = train & (d > cutoff - w)
        recent_mean = df.loc[m, "sales"].astype(np.float64).groupby(df.loc[m, "item_id"], observed=True).mean()
        out[f"te_item_mean_{w}"] = df["item_id"].map(recent_mean).astype(np.float32)
    return out


# --- perakitan -----------------------------------------------------------------

def build_base(long: pd.DataFrame, cal: pd.DataFrame, cfg: dict, state: str) -> pd.DataFrame:
    """Tabel fitur dasar untuk satu toko (semua hari yang ada di `long`)."""
    check_layout(long)
    f = cfg["features"]
    closed = closed_day_mask(long["d"].to_numpy(), cal, cfg)
    ctab = calendar_table(cal, state, f["moving_events"])
    parts = [
        long[ID_COLS + ["d", "wm_yr_wk", "sales"]],
        pd.DataFrame({"is_closed": closed.astype(np.int8)}, index=long.index),
        sales_features(long, 0, closed, f["lags"], f["rolling_windows"]),
        stockout_features(long, 0, closed),
        price_features(long, 0, f["price_max_window"], f["discount_threshold"]),
        calendar_features(long, 0, ctab),
    ]
    return pd.concat(parts, axis=1)


def add_fold_features(base: pd.DataFrame, cutoff: int, horizon: int,
                      recent_windows: tuple[int, ...] = ()) -> pd.DataFrame:
    """Potong `base` ke d <= cutoff + horizon lalu tambahkan fitur per fold."""
    df = base[base["d"] <= cutoff + horizon]
    closed = df["is_closed"].to_numpy().astype(bool)
    return pd.concat([df, fold_features(df, cutoff, closed, recent_windows)], axis=1)


NON_FEATURES = {"d", "wm_yr_wk", "sales", "is_closed"}
FOLD_FEATURE_NAMES = ["te_item_mean", "te_item_zero_share", "te_dept_mean", "price_rel_item"]


def feature_names(df: pd.DataFrame, drop: tuple[str, ...] = ()) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES and c not in drop]


def all_feature_names(base: pd.DataFrame, drop: tuple[str, ...] = ()) -> list[str]:
    """Fitur dasar dari tabel `base` + fitur per fold (yang baru dibuat saat training)."""
    return [c for c in feature_names(base) + FOLD_FEATURE_NAMES if c not in drop]


def to_model_frame(df: pd.DataFrame, feats: list[str], item_encoding: str = "label") -> pd.DataFrame:
    """Matriks fitur untuk XGBoost. `item_encoding`: "label" (kode integer) atau
    "category" (fitur kategorikal bawaan). Fase 1 menunjukkan "category" membuat model
    ~40x lebih besar; perbandingan finalnya di Fase 5."""
    X = df[feats]
    if item_encoding == "label" and "item_id" in feats:
        X = X.assign(item_id=X["item_id"].cat.codes.astype(np.int16))
    elif item_encoding != "category":
        raise ValueError(f"unknown item_encoding: {item_encoding}")
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stores", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cal = load_calendar(Path(cfg["paths"]["raw_m5"]))
    src = Path(cfg["paths"]["interim"]) / "sales_long"
    dst = Path(cfg["paths"]["processed"]) / "base"
    stores = args.stores or sorted(p.name.split("=", 1)[1] for p in src.glob("store_id=*"))

    report = {"stores": []}
    with PeakRSS() as mem:
        for store in stores:
            t = time.perf_counter()
            long = pd.read_parquet(src / f"store_id={store}" / "part-0.parquet")
            base = build_base(long, cal, cfg, state=store.split("_")[0])
            out = dst / f"store_id={store}"
            out.mkdir(parents=True, exist_ok=True)
            base.to_parquet(out / "part-0.parquet", index=False)
            row = {"store_id": store, "rows": len(base), "columns": base.shape[1],
                   "memory_mb": round(base.memory_usage(deep=True).sum() / 2**20, 1),
                   "parquet_mb": round((out / "part-0.parquet").stat().st_size / 2**20, 1),
                   "seconds": round(time.perf_counter() - t, 1),
                   "peak_rss_gb_so_far": round(mem.peak_gb, 2)}
            report["stores"].append(row)
            print(row)
            del long, base
    report["feature_names_base"] = feature_names(pd.read_parquet(dst / f"store_id={stores[0]}" / "part-0.parquet"))
    report["peak_rss_gb"] = round(mem.peak_gb, 2)
    Path(cfg["paths"]["reports"], "data", "phase4_features.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"peak RSS {report['peak_rss_gb']} GB, {len(report['feature_names_base'])} base features")


if __name__ == "__main__":
    main()
