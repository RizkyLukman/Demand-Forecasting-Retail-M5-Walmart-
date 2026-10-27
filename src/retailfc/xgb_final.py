"""Fase 5: run final XGBoost untuk semua 10 toko × semua fold non-holdout.

Konfigurasi dibaca dari `config.yaml → model` (dipilih lewat `xgb_experiments.py`).
Output:
- prediksi lebar per fold: data/processed/preds/<fold>/<nama_model>.npy (30.490 × 28);
- folder run: reports/runs/<timestamp>_<nama_model>/ berisi config.json, metrics.json
  (WRMSSE per fold + per level, MAE/RMSE/bias), infos.json (best_iteration dll.);
- ringkasan pembanding dengan baseline: reports/data/model_comparison.json.

Holdout (d_1914–d_1941) TIDAK dipakai di sini.

Cara pakai:
    python -m retailfc.xgb_final --config config.yaml [--name xgb_point] [--folds cv1 cv2]
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from retailfc.baselines import METHODS as BASELINE_METHODS
from retailfc.features import all_feature_names
from retailfc.folds import build_folds
from retailfc.io import load_config, load_wide_matrices
from retailfc.metrics import WRMSSE, basic_errors
from retailfc.profiling import PeakRSS
from retailfc.train import TrainOptions, fit_predict, to_wide


def model_params(mcfg: dict) -> tuple[dict, TrainOptions]:
    params = {"device": "cuda", "tree_method": "hist", "eval_metric": "rmse", **mcfg["params"]}
    if "quantile_alpha" in params:
        params["quantile_alpha"] = np.array(params["quantile_alpha"])
    return params, TrainOptions(**mcfg.get("options", {}))


def compare(cfg: dict, fold_names: list[str], model_names: list[str]) -> dict:
    """WRMSSE semua model (XGBoost + baseline) per fold, dari prediksi yang tersimpan."""
    max_d = cfg["validation"]["holdout_cutoff"]
    ids, sales, prices = load_wide_matrices(cfg, max_d)
    folds = {f.name: f for f in build_folds(cfg)}
    root = Path(cfg["paths"]["processed"]) / "preds"
    table: dict = {}
    for fname in fold_names:
        f = folds[fname]
        ev = WRMSSE(ids, sales, prices, cutoff=f.cutoff, horizon=f.horizon)
        actual = sales[:, f.cutoff:f.cutoff + f.horizon].astype(np.float64)
        table[fname] = {}
        for m in model_names:
            path = root / fname / f"{m}.npy"
            if path.exists():
                p = np.load(path).astype(np.float64)
                sc = ev.score(actual, p)
                table[fname][m] = {"wrmsse": round(sc["wrmsse"], 5), "per_level": sc["per_level"],
                                   **{k: round(v, 5) for k, v in basic_errors(actual, p).items()}}
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--name", default="xgb_point")
    ap.add_argument("--folds", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"][args.name]
    params, opt = model_params(mcfg)
    folds = [f for f in build_folds(cfg) if f.role != "holdout"]
    if args.folds:
        folds = [f for f in folds if f.name in args.folds]

    max_d = cfg["validation"]["holdout_cutoff"]
    ids, _, _ = load_wide_matrices(cfg, max_d)
    stores = sorted(ids["store_id"].astype(str).unique())
    base_root = Path(cfg["paths"]["processed"]) / "base"
    preds_root = Path(cfg["paths"]["processed"]) / "preds"
    preds = {f.name: np.zeros((len(ids), f.horizon), dtype=np.float32) for f in folds}
    infos = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["paths"]["reports"]) / "runs" / f"{stamp}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with PeakRSS() as mem:
        for store in stores:
            base = pd.read_parquet(base_root / f"store_id={store}" / "part-0.parquet")
            feats = all_feature_names(base, drop=tuple(mcfg.get("drop_features", [])))
            for f in folds:
                p, info = fit_predict(base, f, feats, params, options=opt)
                to_wide(p, ids, store, f, preds[f.name])
                infos.append(info | {"store_id": store})
                print(f"{store} {f.name:13s} best={info['best_iteration']:4d} {info['seconds']:5.1f}s", flush=True)
            del base

    for f in folds:
        (preds_root / f.name).mkdir(parents=True, exist_ok=True)
        np.save(preds_root / f.name / f"{args.name}.npy", preds[f.name])

    table = compare(cfg, [f.name for f in folds], [args.name] + BASELINE_METHODS)
    cv = [f.name for f in folds if f.role == "cv"]
    summary = {}
    for m in [args.name] + BASELINE_METHODS:
        w = [table[fn][m]["wrmsse"] for fn in cv if m in table[fn]]
        if len(w) == len(cv) and w:
            summary[m] = {"cv_wrmsse_mean": round(float(np.mean(w)), 5),
                          "cv_wrmsse_std": round(float(np.std(w, ddof=1)), 5) if len(w) > 1 else None,
                          "per_fold": dict(zip(cv, w))}
    (run_dir / "config.json").write_text(json.dumps(
        {"name": args.name, "model": mcfg, "params_used": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                           for k, v in params.items()},
         "options": opt.__dict__, "folds": [f.name for f in folds], "xgboost": xgb.__version__,
         "python": platform.python_version(), "seed": params.get("seed")}, indent=2), encoding="utf-8")
    (run_dir / "infos.json").write_text(json.dumps(infos, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({"per_fold": table, "cv_summary": summary,
                                                      "peak_rss_gb": round(mem.peak_gb, 2)}, indent=2),
                                          encoding="utf-8")
    comp_path = Path(cfg["paths"]["reports"]) / "data" / "model_comparison.json"
    comp_path.write_text(json.dumps({"run_dir": str(run_dir), "per_fold": table, "cv_summary": summary},
                                    indent=2), encoding="utf-8")
    for m, s in sorted(summary.items(), key=lambda kv: kv[1]["cv_wrmsse_mean"]):
        print(f"{m:14s} CV WRMSSE {s['cv_wrmsse_mean']:.4f} ± {s['cv_wrmsse_std']}")
    print(f"run saved to {run_dir}")


if __name__ == "__main__":
    main()
