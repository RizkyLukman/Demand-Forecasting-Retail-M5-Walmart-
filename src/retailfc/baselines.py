"""Fase 5: baseline wajib, untuk semua 30.490 deret dan semua fold (kecuali holdout).

| Nama | Definisi |
|---|---|
| snaive | 7 hari terakhir sebelum cutoff, diulang sepanjang horizon |
| ma28 | rata-rata 28 hari terakhir sebelum cutoff, datar |
| croston | statsforecast CrostonOptimized, semua deret |
| tsb | statsforecast TSB(alpha_d=0,1, alpha_p=0,1), semua deret |
| classical | AutoETS(season_length=7) untuk deret smooth/erratic dengan riwayat >= 56 hari, TSB untuk deret lain |

ETS tidak dilaporkan sendiri: ia hanya dijalankan untuk sebagian deret, jadi WRMSSE-nya
atas semua deret akan menyesatkan. Prediksinya disimpan (`ets.npy`, NaN di luar
cakupan) untuk analisis per segmen di Fase 6. Riwayat minimal 56 hari dibutuhkan karena
AutoETS menolak deret yang terlalu pendek ("tiny datasets").

Aturan yang sama untuk semua model (termasuk XGBoost):
- riwayat tiap deret dimulai dari penjualan pertamanya;
- deret yang belum pernah laku sampai cutoff diberi forecast 0 (tanggal mulai jual
  adalah informasi masa depan);
- hari tutup di riwayat diganti nilai 7 hari sebelumnya; hari tutup di horizon = 0;
- forecast negatif dipotong ke 0.

Output: prediksi per fold ke data/processed/preds/<fold>/<metode>.npy (n_deret × 28),
segmen per fold ke data/processed/preds/<fold>/segments.parquet, metrik ke
reports/baselines.json.

Cara pakai:
    python -m retailfc.baselines --config config.yaml [--folds cv1 cv2]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from retailfc.folds import build_folds, closed_day_mask
from retailfc.io import load_calendar, load_config, load_wide_matrices
from retailfc.metrics import WRMSSE, basic_errors
from retailfc.segments import classify

METHODS = ["snaive", "ma28", "croston", "tsb", "classical"]
ETS_MIN_HISTORY = 56


def fill_closed(hist: np.ndarray, closed_cols: np.ndarray) -> np.ndarray:
    """Ganti hari tutup dengan nilai 7 hari sebelumnya (berurutan, jadi aman berantai)."""
    out = hist.astype(np.float64).copy()
    for j in np.flatnonzero(closed_cols):
        if j >= 7:
            out[:, j] = out[:, j - 7]
    return out


def seasonal_naive(hist: np.ndarray, h: int) -> np.ndarray:
    reps = int(np.ceil(h / 7))
    return np.tile(hist[:, -7:], reps)[:, :h]


def moving_average(hist: np.ndarray, h: int, window: int = 28) -> np.ndarray:
    return np.repeat(hist[:, -window:].mean(axis=1, keepdims=True), h, axis=1)


def statsforecast_predict(hist: np.ndarray, rows: np.ndarray, first: np.ndarray,
                          models: list, h: int) -> dict[str, np.ndarray]:
    """Jalankan model statsforecast untuk `rows`; hasil (n_deret × h), NaN di luar `rows`."""
    from statsforecast import StatsForecast

    parts = []
    for i in rows:
        y = hist[i, first[i]:]
        parts.append(pd.DataFrame({"unique_id": np.int32(i), "ds": np.arange(len(y), dtype=np.int32),
                                   "y": y.astype(np.float32)}))
    df = pd.concat(parts, ignore_index=True)
    # n_jobs dibatasi: tiap proses membawa salinan data, RAM laptop terbatas.
    fc = StatsForecast(models=models, freq=1, n_jobs=6).forecast(df=df, h=h)
    fc = fc.sort_values(["unique_id", "ds"])
    out = {}
    for col in [c for c in fc.columns if c not in ("unique_id", "ds")]:
        mat = np.full((hist.shape[0], h), np.nan)
        mat[fc["unique_id"].unique()] = fc[col].to_numpy().reshape(-1, h)
        out[col] = mat
    return out


def run_fold(fold, sales, ids, prices, cal, cfg, out_dir: Path) -> dict:
    from statsforecast.models import TSB, AutoETS, CrostonOptimized

    c, h = fold.cutoff, fold.horizon
    t0 = time.perf_counter()
    closed_all = closed_day_mask(np.arange(1, c + h + 1), cal, cfg)
    hist = fill_closed(sales[:, :c], closed_all[:c])
    sold = hist > 0
    has = sold.any(axis=1)
    first = np.where(has, sold.argmax(axis=1), c)
    seg = classify(sales[:, :c], closed_all[:c])

    preds = {"snaive": seasonal_naive(hist, h), "ma28": moving_average(hist, h)}
    rows_all = np.flatnonzero(has)
    sf = statsforecast_predict(hist, rows_all, first,
                               [CrostonOptimized(), TSB(alpha_d=0.1, alpha_p=0.1)], h)
    preds["croston"], preds["tsb"] = sf["CrostonOptimized"], sf["TSB"]
    long_enough = (c - first) >= ETS_MIN_HISTORY
    rows_ets = np.flatnonzero(seg["segment"].isin(["smooth", "erratic"]).to_numpy() & long_enough)
    ets = statsforecast_predict(hist, rows_ets, first, [AutoETS(season_length=7)], h)["AutoETS"]
    preds["classical"] = np.where(np.isnan(ets), preds["tsb"], ets)

    closed_h = closed_all[c:c + h]
    actual = sales[:, c:c + h].astype(np.float64)
    ev = WRMSSE(ids, sales, prices, cutoff=c, horizon=h)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg.assign(id=ids["id"].to_numpy()).to_parquet(out_dir / "segments.parquet", index=False)
    np.save(out_dir / "ets.npy", ets.astype(np.float32))      # NaN di luar cakupan ETS

    result = {"fold": fold.name, "role": fold.role, "cutoff": c,
              "segment_counts": seg["segment"].value_counts().reindex(seg["segment"].cat.categories).to_dict(),
              "ets_series": int(len(rows_ets)), "methods": {}}
    for name in METHODS:
        p = np.clip(np.nan_to_num(preds[name], nan=0.0), 0, None)
        p[:, closed_h] = 0.0
        p[~has] = 0.0
        np.save(out_dir / f"{name}.npy", p.astype(np.float32))
        sc = ev.score(actual, p)
        result["methods"][name] = {"wrmsse": round(sc["wrmsse"], 5), "per_level": sc["per_level"],
                                   **{k: round(v, 5) for k, v in basic_errors(actual, p).items()}}
    result["seconds"] = round(time.perf_counter() - t0, 1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--folds", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    folds = [f for f in build_folds(cfg) if f.role != "holdout"]
    if args.folds:
        folds = [f for f in folds if f.name in args.folds]
    max_d = cfg["validation"]["holdout_cutoff"]          # holdout tidak dibaca
    ids, sales, prices = load_wide_matrices(cfg, max_d)
    cal = load_calendar(Path(cfg["paths"]["raw_m5"]))
    preds_root = Path(cfg["paths"]["processed"]) / "preds"

    report_path = Path(cfg["paths"]["reports"]) / "baselines.json"
    results = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for fold in folds:
        r = run_fold(fold, sales, ids, prices, cal, cfg, preds_root / fold.name)
        results[fold.name] = r
        print(f"{fold.name}: " + ", ".join(f"{m}={v['wrmsse']:.4f}" for m, v in r["methods"].items())
              + f" ({r['seconds']} s)")
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
