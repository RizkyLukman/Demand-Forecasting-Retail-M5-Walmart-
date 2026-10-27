"""Fase 5: eksperimen konfigurasi XGBoost di 2 toko × 4 fold CV.

Rancangan (ditetapkan sebelum melihat hasil):

| # | Pertanyaan | Varian | Aturan memilih |
|---|---|---|---|
| E1 | Objective | tweedie p=1,1/1,2/1,3/1,5, poisson, squarederror | WRMSSE rata-rata terendah |
| –  | Noise | konfigurasi dasar dengan seed 43 | dipakai sebagai ukuran noise |
| E2 | Encoding item_id | label vs category | jika selisih < noise, pilih label |
| E3 | Censored demand | biasa vs keluarkan deretan nol >= 28 hari item laris | dilaporkan; Fase 7 pakai versi tanpa deretan nol |
| E4 | Latih ulang setelah early stopping | tidak vs ya | pilih "ya" hanya jika lebih baik melebihi noise |

Toko: CA_3 dan WI_2 (berbeda dari toko ablation Easter). Metrik: WRMSSE atas hierarki
kedua toko itu, per fold, lalu rata-rata dan sebaran antar fold.

Cara pakai:
    python -m retailfc.xgb_experiments --config config.yaml --stage 1
    python -m retailfc.xgb_experiments --config config.yaml --stage 2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from retailfc.features import all_feature_names
from retailfc.folds import build_folds
from retailfc.io import load_config, load_wide_matrices
from retailfc.metrics import WRMSSE, basic_errors
from retailfc.profiling import PeakRSS
from retailfc.train import TrainOptions, fit_predict, to_wide

STORES = ["CA_3", "WI_2"]
FOLDS = ["cv1", "cv2", "cv3", "cv4"]


def base_params(seed: int = 42) -> dict:
    return {"device": "cuda", "tree_method": "hist", "max_bin": 256, "eval_metric": "rmse",
            "eta": 0.05, "max_depth": 8, "min_child_weight": 10,
            "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed}


OBJECTIVES = {
    "tweedie_1.1": {"objective": "reg:tweedie", "tweedie_variance_power": 1.1},
    "tweedie_1.2": {"objective": "reg:tweedie", "tweedie_variance_power": 1.2},
    "tweedie_1.3": {"objective": "reg:tweedie", "tweedie_variance_power": 1.3},
    "tweedie_1.5": {"objective": "reg:tweedie", "tweedie_variance_power": 1.5},
    "poisson": {"objective": "count:poisson"},
    "squarederror": {"objective": "reg:squarederror"},
}


def stage_configs(stage: int, best_obj: str | None) -> dict[str, tuple[dict, TrainOptions]]:
    if stage == 1:
        cfgs = {name: ({**base_params(), **obj}, TrainOptions()) for name, obj in OBJECTIVES.items()}
        cfgs["tweedie_1.2_seed43"] = ({**base_params(43), **OBJECTIVES["tweedie_1.2"]}, TrainOptions())
        return cfgs
    obj = OBJECTIVES[best_obj]
    if stage == 3:
        # Ditambahkan setelah tahap 2: latih ulang belum pernah diuji bersama pengecualian
        # deretan nol. Dipakai di final hanya jika mengalahkan "censored_excluded" saja
        # dengan selisih rata-rata melebihi noise seed.
        return {f"{best_obj}_censored_excluded_refit": (
            {**base_params(), **obj}, TrainOptions(exclude_censored=True, refit=True))}
    return {
        f"{best_obj}_category": ({**base_params(), **obj}, TrainOptions(item_encoding="category")),
        f"{best_obj}_censored_excluded": ({**base_params(), **obj}, TrainOptions(exclude_censored=True)),
        f"{best_obj}_refit": ({**base_params(), **obj}, TrainOptions(refit=True)),
    }


def run(cfg: dict, configs: dict, out_path: Path) -> dict:
    folds = {f.name: f for f in build_folds(cfg)}
    max_d = cfg["validation"]["holdout_cutoff"]
    ids_all, sales_all, prices_all = load_wide_matrices(cfg, max_d)
    sub = ids_all["store_id"].astype(str).isin(STORES).to_numpy()
    ids = ids_all[sub].reset_index(drop=True)
    sales, prices = sales_all[sub], prices_all[sub]
    del sales_all, prices_all
    base_root = Path(cfg["paths"]["processed"]) / "base"

    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"runs": {}}
    results["stores"], results["folds"] = STORES, FOLDS
    results["xgboost_version"] = xgb.__version__
    preds = {(name, f): np.zeros((len(ids), 28), dtype=np.float32) for name in configs for f in FOLDS}
    infos: dict = {name: [] for name in configs}

    with PeakRSS() as mem:
        for store in STORES:
            base = pd.read_parquet(base_root / f"store_id={store}" / "part-0.parquet")
            feats = all_feature_names(base)
            for fname in FOLDS:
                for name, (params, opt) in configs.items():
                    p, info = fit_predict(base, folds[fname], feats, params, options=opt)
                    to_wide(p, ids, store, folds[fname], preds[(name, fname)])
                    infos[name].append(info | {"store_id": store})
                    print(f"{store} {fname} {name:32s} best={info['best_iteration']:4d} "
                          f"{info['seconds']:5.1f}s model={info['model_mb']}MB", flush=True)
            del base

    for name, (params, opt) in configs.items():
        per_fold = {}
        for fname in FOLDS:
            f = folds[fname]
            actual = sales[:, f.cutoff:f.cutoff + f.horizon].astype(np.float64)
            ev = WRMSSE(ids, sales, prices, cutoff=f.cutoff, horizon=f.horizon)
            sc = ev.score(actual, preds[(name, fname)])
            per_fold[fname] = {"wrmsse": round(sc["wrmsse"], 5),
                               **{k: round(v, 5) for k, v in basic_errors(actual, preds[(name, fname)]).items()}}
        w = np.array([per_fold[f]["wrmsse"] for f in FOLDS])
        results["runs"][name] = {
            "params": {k: v for k, v in params.items()}, "options": opt.__dict__,
            "per_fold": per_fold, "wrmsse_mean": round(float(w.mean()), 5),
            "wrmsse_std": round(float(w.std(ddof=1)), 5), "infos": infos[name],
        }
    results["peak_rss_gb_last_stage"] = round(mem.peak_gb, 2)
    results["updated"] = datetime.now().isoformat(timespec="seconds")
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_path = Path(cfg["paths"]["reports"]) / "data" / "xgb_experiments.json"
    best = None
    if args.stage in (2, 3):
        prev = json.loads(out_path.read_text(encoding="utf-8"))["runs"]
        best = min(OBJECTIVES, key=lambda k: prev[k]["wrmsse_mean"])
        print(f"objective terbaik dari tahap 1: {best}")
    res = run(cfg, stage_configs(args.stage, best), out_path)
    for name, r in sorted(res["runs"].items(), key=lambda kv: kv[1]["wrmsse_mean"]):
        print(f"{name:34s} WRMSSE {r['wrmsse_mean']:.4f} ± {r['wrmsse_std']:.4f}  "
              + " ".join(f"{f}={r['per_fold'][f]['wrmsse']:.4f}" for f in FOLDS))


if __name__ == "__main__":
    main()
