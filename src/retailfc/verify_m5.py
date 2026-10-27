"""Fase 0: verifikasi file mentah M5 terhadap skema yang terdokumentasi.

Hanya membaca file asli di `paths.raw_m5`, mencetak laporan, lalu menyimpannya ke
`reports/data/m5_verification.json`. Tidak ada yang diasumsikan; setiap angka berasal
dari file.

Cara pakai:
    python -m retailfc.verify_m5 --config config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPECTED_COLUMNS = {
    "calendar.csv": [
        "date", "wm_yr_wk", "weekday", "wday", "month", "year", "d",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ],
    "sell_prices.csv": ["store_id", "item_id", "wm_yr_wk", "sell_price"],
}
SALES_ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
EXPECTED_FILES = [
    "calendar.csv",
    "sell_prices.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sample_submission.csv",
]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def compare_columns(actual: list[str], expected: list[str]) -> dict:
    return {
        "match": actual == expected,
        "missing": [c for c in expected if c not in actual],
        "unexpected": [c for c in actual if c not in expected],
    }


def d_index(cols) -> list[int]:
    return [int(c[2:]) for c in cols if re.fullmatch(r"d_\d+", c)]


def file_inventory(raw: Path) -> dict:
    inv = {}
    for p in sorted(raw.iterdir()):
        if p.is_file():
            inv[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    inv["_missing_expected"] = [f for f in EXPECTED_FILES if not (raw / f).exists()]
    return inv


def check_calendar(raw: Path) -> tuple[dict, pd.DataFrame]:
    cal = pd.read_csv(raw / "calendar.csv", parse_dates=["date"])
    days = cal["date"].diff().dropna().dt.days
    d_nums = cal["d"].str.slice(2).astype(int)
    out = {
        "columns": compare_columns(list(cal.columns), EXPECTED_COLUMNS["calendar.csv"]),
        "rows": len(cal),
        "date_min": str(cal["date"].min().date()),
        "date_max": str(cal["date"].max().date()),
        "dates_contiguous": bool((days == 1).all()),
        "d_min": int(d_nums.min()),
        "d_max": int(d_nums.max()),
        "d_contiguous": bool((d_nums.diff().dropna() == 1).all()),
        "d1_date": str(cal.loc[cal["d"] == "d_1", "date"].iloc[0].date()),
        "n_event_days": int(cal["event_name_1"].notna().sum()),
        "n_event_days_with_second_event": int(cal["event_name_2"].notna().sum()),
        "event_types": sorted(cal["event_type_1"].dropna().unique().tolist()),
    }
    return out, cal


def check_prices(raw: Path) -> dict:
    pr = pd.read_csv(
        raw / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32",
               "sell_price": "float32"},
    )
    return {
        "columns": compare_columns(list(pr.columns), EXPECTED_COLUMNS["sell_prices.csv"]),
        "rows": len(pr),
        "n_stores": int(pr["store_id"].nunique()),
        "n_items": int(pr["item_id"].nunique()),
        "wm_yr_wk_min": int(pr["wm_yr_wk"].min()),
        "wm_yr_wk_max": int(pr["wm_yr_wk"].max()),
        "price_min": float(pr["sell_price"].min()),
        "price_max": float(pr["sell_price"].max()),
        "price_nan": int(pr["sell_price"].isna().sum()),
        "duplicate_keys": int(pr.duplicated(["store_id", "item_id", "wm_yr_wk"]).sum()),
    }


def read_sales(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: "int16" for c in header if c.startswith("d_")}
    dtypes.update({c: "category" for c in SALES_ID_COLUMNS if c != "id"})
    return pd.read_csv(path, dtype=dtypes)


def check_sales(path: Path, cal: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    df = read_sales(path)
    id_cols = [c for c in df.columns if not c.startswith("d_")]
    d_nums = d_index(df.columns)
    values = df[[f"d_{i}" for i in d_nums]].to_numpy()

    # Nol di awal = hari sebelum penjualan pertama yang tercatat untuk tiap item-toko.
    nonzero = values > 0
    has_sale = nonzero.any(axis=1)
    first_sale = np.where(has_sale, nonzero.argmax(axis=1), values.shape[1])
    cal_d = set(cal["d"])

    out = {
        "id_columns": compare_columns(id_cols, SALES_ID_COLUMNS),
        "rows": len(df),
        "id_unique": bool(df["id"].is_unique),
        "d_min": min(d_nums),
        "d_max": max(d_nums),
        "n_d_columns": len(d_nums),
        "d_contiguous": d_nums == list(range(min(d_nums), max(d_nums) + 1)),
        "all_d_in_calendar": all(f"d_{i}" in cal_d for i in d_nums),
        "n_items": int(df["item_id"].nunique()),
        "n_stores": int(df["store_id"].nunique()),
        "n_depts": int(df["dept_id"].nunique()),
        "n_cats": int(df["cat_id"].nunique()),
        "n_states": int(df["state_id"].nunique()),
        "stores": sorted(df["store_id"].astype(str).unique().tolist()),
        "cats": sorted(df["cat_id"].astype(str).unique().tolist()),
        "depts": sorted(df["dept_id"].astype(str).unique().tolist()),
        "states": sorted(df["state_id"].astype(str).unique().tolist()),
        "min_value": int(values.min()),
        "max_value": int(values.max()),
        "fits_int16": bool(values.max() <= np.iinfo(np.int16).max),
        "long_format_rows": int(values.size),
        "zero_share_all_cells": float((values == 0).mean()),
        "series_without_any_sale": int((~has_sale).sum()),
        "leading_zero_cells": int(first_sale.sum()),
        "zero_share_after_first_sale": float(
            ((values == 0).sum() - first_sale.sum()) / (values.size - first_sale.sum())
        ),
        "memory_mb_int16": round(df.memory_usage(deep=True).sum() / 2**20, 1),
    }
    return out, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    raw = Path(cfg["paths"]["raw_m5"])

    report: dict = {"files": file_inventory(raw)}
    report["calendar"], cal = check_calendar(raw)
    report["sell_prices"] = check_prices(raw)
    report["sales_train_evaluation"], ev = check_sales(raw / cfg["m5"]["sales_file"], cal)

    # File validation seharusnya sama dengan 1913 hari pertama file evaluation.
    va = read_sales(raw / "sales_train_validation.csv")
    va_d = [c for c in va.columns if c.startswith("d_")]
    ev_aligned = ev.set_index("id").loc[va["id"].str.replace("_validation", "_evaluation")]
    report["validation_vs_evaluation"] = {
        "validation_d_max": max(d_index(va.columns)),
        "same_series_order": bool(
            (va["id"].str.replace("_validation", "_evaluation").values == ev["id"].values).all()
        ),
        "overlap_identical": bool((ev_aligned[va_d].to_numpy() == va[va_d].to_numpy()).all()),
    }

    out_path = Path(cfg["paths"]["reports"]) / "data" / "m5_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
