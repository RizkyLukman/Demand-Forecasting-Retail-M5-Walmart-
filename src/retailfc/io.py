"""Fase 1: muat M5, downcast, ubah ke format panjang per toko, simpan ke Parquet.

Output: data/interim/sales_long/store_id=<id>/part-0.parquet
Kolom: item_id, dept_id, cat_id, state_id (category), d (int16),
       wm_yr_wk (int16), sell_price (float32, NaN jika tidak ada harga), sales (int16).

Baris sebelum penjualan pertama tiap item-toko dibuang: produk belum dijual saat
itu, jadi nol-nya bukan permintaan. Ada tidaknya harga di periode itu dicek dan
dilaporkan, bukan diasumsikan.

Cara pakai:
    python -m retailfc.io --config config.yaml [--stores CA_1 TX_2]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from retailfc.profiling import PeakRSS, system_ram

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
INT16_MAX = np.iinfo(np.int16).max


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_calendar(raw: Path) -> pd.DataFrame:
    cal = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])
    cal["d_num"] = cal["d"].str.slice(2).astype(np.int16)
    assert cal["wm_yr_wk"].max() <= INT16_MAX
    cal["wm_yr_wk"] = cal["wm_yr_wk"].astype(np.int16)
    return cal


def load_sales_wide(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: np.int16 for c in header if c.startswith("d_")}
    dtypes.update({c: "category" for c in CAT_COLS})
    return pd.read_csv(path, dtype=dtypes)


def load_prices(raw: Path) -> pd.DataFrame:
    pr = pd.read_csv(
        raw / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": np.int32,
               "sell_price": np.float32},
    )
    assert pr["wm_yr_wk"].max() <= INT16_MAX
    pr["wm_yr_wk"] = pr["wm_yr_wk"].astype(np.int16)
    return pr


def store_to_long(
    wide: pd.DataFrame,
    prices: pd.DataFrame,
    cal: pd.DataFrame,
    d_cols: list[str],
    categories: dict[str, pd.Index],
) -> tuple[pd.DataFrame, dict]:
    """Tabel format panjang untuk baris-baris satu toko dari file penjualan lebar.

    `wide` harus berisi tepat satu toko. `prices` boleh berisi semua toko.
    """
    store = wide["store_id"].iloc[0]
    assert (wide["store_id"] == store).all()
    values = wide[d_cols].to_numpy()                       # (jumlah deret, jumlah hari) int16
    n_series, n_days = values.shape
    d_first = int(d_cols[0][2:])
    d_nums = np.arange(d_first, d_first + n_days, dtype=np.int16)

    # Minggu kalender (wm_yr_wk) untuk setiap kolom hari.
    week_of_d = cal.set_index("d_num")["wm_yr_wk"].reindex(d_nums).to_numpy()
    assert not pd.isna(week_of_d).any(), "sales day missing from calendar"
    weeks = np.unique(week_of_d)
    week_pos_of_day = np.searchsorted(weeks, week_of_d)

    # Matriks harga: deret x minggu (NaN = tidak ada harga di minggu itu).
    item_codes = pd.Categorical(wide["item_id"], categories=categories["item_id"]).codes
    row_of_item = np.full(len(categories["item_id"]), -1, dtype=np.int64)
    row_of_item[item_codes] = np.arange(n_series)
    ps = prices[prices["store_id"] == store]
    p_rows = row_of_item[pd.Categorical(ps["item_id"], categories=categories["item_id"]).codes]
    p_week_pos = np.searchsorted(weeks, ps["wm_yr_wk"].to_numpy())
    in_range = (p_rows >= 0) & (p_week_pos < len(weeks))
    in_range &= weeks[np.minimum(p_week_pos, len(weeks) - 1)] == ps["wm_yr_wk"].to_numpy()
    price_mat = np.full((n_series, len(weeks)), np.nan, dtype=np.float32)
    price_mat[p_rows[in_range], p_week_pos[in_range]] = ps["sell_price"].to_numpy()[in_range]
    has_price = ~np.isnan(price_mat[:, week_pos_of_day])  # (jumlah deret, jumlah hari) bool

    # Penjualan pertama per deret; simpan hari >= penjualan pertama.
    nonzero = values > 0
    any_sale = nonzero.any(axis=1)
    first_sale = np.where(any_sale, nonzero.argmax(axis=1), n_days)
    keep = np.arange(n_days)[None, :] >= first_sale[:, None]

    has_any_price = has_price.any(axis=1)
    first_price = np.where(has_any_price, has_price.argmax(axis=1), n_days)
    # Harga dicatat per minggu, jadi harga beberapa hari sebelum penjualan pertama di
    # minggu yang sama itu wajar. Yang penting: harga di minggu yang LEBIH AWAL.
    first_price_wk = week_pos_of_day[np.minimum(first_price, n_days - 1)]
    first_sale_wk = week_pos_of_day[np.minimum(first_sale, n_days - 1)]
    price_weeks_early = np.where(any_sale & has_any_price, first_sale_wk - first_price_wk, 0)

    stats = {
        "store_id": str(store),
        "n_series": int(n_series),
        "rows_wide_cells": int(values.size),
        "rows_dropped_before_first_sale": int((~keep).sum()),
        "rows_kept": int(keep.sum()),
        "series_without_any_sale": int((~any_sale).sum()),
        "dropped_rows_with_price": int((has_price & ~keep).sum()),
        "kept_rows_without_price": int((~has_price & keep).sum()),
        "series_price_before_first_sale": int((first_price < first_sale).sum()),
        "series_first_sale_before_first_price": int((first_sale < first_price).sum()),
        "series_price_in_earlier_week_than_first_sale": int((price_weeks_early > 0).sum()),
        "max_weeks_price_before_first_sale": int(price_weeks_early.max()),
        "sales_total_wide": int(values.sum(dtype=np.int64)),
    }

    # Bangun tabel panjang langsung dari sel yang disimpan (urut per deret, lalu hari).
    rows, cols = np.nonzero(keep)
    long = pd.DataFrame({
        "item_id": pd.Categorical.from_codes(item_codes[rows], categories["item_id"]),
        "dept_id": pd.Categorical(wide["dept_id"].to_numpy()[rows], categories=categories["dept_id"]),
        "cat_id": pd.Categorical(wide["cat_id"].to_numpy()[rows], categories=categories["cat_id"]),
        "state_id": pd.Categorical(wide["state_id"].to_numpy()[rows], categories=categories["state_id"]),
        "d": d_nums[cols],
        "wm_yr_wk": week_of_d[cols].astype(np.int16),
        "sell_price": price_mat[rows, week_pos_of_day[cols]],
        "sales": values[rows, cols],
    })
    stats["sales_total_long"] = int(long["sales"].to_numpy().sum(dtype=np.int64))
    stats["memory_mb"] = round(long.memory_usage(deep=True).sum() / 2**20, 1)
    return long, stats


def run(cfg: dict, stores: list[str] | None = None) -> dict:
    raw = Path(cfg["paths"]["raw_m5"])
    out_root = Path(cfg["paths"]["interim"]) / "sales_long"
    report: dict = {"started": datetime.now().isoformat(timespec="seconds"),
                    "system_ram_at_start": system_ram(), "steps": {}, "stores": []}

    with PeakRSS() as mem:
        t = time.perf_counter()
        cal = load_calendar(raw)
        wide = load_sales_wide(raw / cfg["m5"]["sales_file"])
        prices = load_prices(raw)
        report["steps"]["load"] = {"seconds": round(time.perf_counter() - t, 1),
                                   "peak_rss_gb_so_far": round(mem.peak / 2**30, 2)}

        d_cols = [c for c in wide.columns if c.startswith("d_")]
        categories = {c: pd.Index(sorted(wide[c].astype(str).unique())) for c in CAT_COLS}
        todo = stores or list(categories["store_id"])

        for store in todo:
            t = time.perf_counter()
            ws = wide[wide["store_id"] == store]
            long, stats = store_to_long(ws, prices, cal, d_cols, categories)
            out_dir = out_root / f"store_id={store}"
            out_dir.mkdir(parents=True, exist_ok=True)
            long.to_parquet(out_dir / "part-0.parquet", index=False)
            stats["seconds"] = round(time.perf_counter() - t, 1)
            stats["peak_rss_gb_so_far"] = round(mem.peak / 2**30, 2)
            stats["parquet_mb"] = round((out_dir / "part-0.parquet").stat().st_size / 2**20, 1)
            report["stores"].append(stats)
            print(f"{store}: kept {stats['rows_kept']:,} rows, "
                  f"dropped {stats['rows_dropped_before_first_sale']:,}, "
                  f"{stats['seconds']} s, peak RSS {stats['peak_rss_gb_so_far']} GB")
            del long, ws

    s = report["stores"]
    report["totals"] = {
        k: sum(x[k] for x in s)
        for k in ["rows_wide_cells", "rows_dropped_before_first_sale", "rows_kept",
                  "dropped_rows_with_price", "kept_rows_without_price",
                  "series_price_before_first_sale", "series_first_sale_before_first_price",
                  "series_price_in_earlier_week_than_first_sale",
                  "sales_total_wide", "sales_total_long"]
    }
    report["totals"]["max_weeks_price_before_first_sale"] = max(
        x["max_weeks_price_before_first_sale"] for x in s)
    report["totals"]["parquet_mb"] = round(sum(x["parquet_mb"] for x in s), 1)
    report["totals"]["memory_mb_long_all_stores"] = round(sum(x["memory_mb"] for x in s), 1)
    report["peak_rss_gb"] = round(mem.peak_gb, 2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stores", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    report = run(cfg, args.stores)
    out = Path(cfg["paths"]["reports"]) / "phase1_io.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["totals", "peak_rss_gb", "steps"]}, indent=2))
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
