"""Tuning parameter pohon XGBoost point (sisa Fase 5), di 2 toko × 4 fold CV.

Rancangan (ditetapkan sebelum melihat hasil):

- Toko CA_3 dan WI_2, fold cv1-cv4: sama dengan eksperimen Fase 5
  (`reports/data/xgb_experiments.json`), jadi angkanya bisa dibandingkan.
- Titik awal = konfigurasi final saat ini (`config.yaml → model.xgb_point`).
- Noise = selisih WRMSSE rata-rata titik awal antara seed 42 dan seed 43.
  Ambang tau = max(noise terukur, 0,004). 0,004 adalah noise seed yang terukur di Fase 5.
- Tahap berurutan, satu kelompok parameter per tahap. Tiap tahap mulai dari konfigurasi
  terbaik tahap sebelumnya:

  | Tahap | Parameter | Kandidat (selain nilai saat ini) |
  |---|---|---|
  | T1 | max_depth | 6, 10, 12 |
  | T2 | min_child_weight | 1, 50, 200 |
  | T3 | subsample / colsample_bytree | 0,6/0,8; 1,0/0,8; 0,8/0,5; 0,8/1,0 |
  | T4 | reg_lambda (default 1) | 5, 20, 100 |
  | T5 | max_bin | 128, 512 |
  | T6 | eta | 0,03; 0,1 |

- Aturan menerima: kandidat dengan WRMSSE rata-rata terendah di tahap itu diterima hanya jika
  (a) lebih baik dari konfigurasi saat ini lebih dari tau, dan (b) lebih baik di >= 3 dari 4
  fold. Kalau tidak, nilai saat ini dipertahankan.
- Hasil tuning baru dipakai jika lolos konfirmasi di 10 toko × 4 fold (langkah terpisah,
  `xgb_final.py`) dengan aturan yang sama terhadap WRMSSE 10 toko saat ini.
- Holdout tidak disentuh.

Hasil disimpan setelah setiap konfigurasi selesai, jadi run yang terhenti bisa dilanjutkan.

Cara pakai:
    python -m retailfc.xgb_tuning --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from retailfc.features import all_feature_names
from retailfc.folds import build_folds
from retailfc.io import load_config, load_wide_matrices
from retailfc.metrics import WRMSSE
from retailfc.profiling import PeakRSS
from retailfc.train import fit_predict, to_wide
from retailfc.xgb_final import model_params

STORES = ["CA_3", "WI_2"]
FOLDS = ["cv1", "cv2", "cv3", "cv4"]
MIN_TAU = 0.004
STAGES: list[tuple[str, list[dict]]] = [
    ("T1_max_depth", [{"max_depth": v} for v in (6, 10, 12)]),
    ("T2_min_child_weight", [{"min_child_weight": v} for v in (1, 50, 200)]),
    ("T3_sampling", [{"subsample": a, "colsample_bytree": b}
                     for a, b in ((0.6, 0.8), (1.0, 0.8), (0.8, 0.5), (0.8, 1.0))]),
    ("T4_reg_lambda", [{"reg_lambda": v} for v in (5, 20, 100)]),
    ("T5_max_bin", [{"max_bin": v} for v in (128, 512)]),
    ("T6_eta", [{"eta": v} for v in (0.03, 0.1)]),
]


def config_name(params: dict, base: dict) -> str:
    diff = {k: v for k, v in params.items() if base.get(k) != v}
    return "base" if not diff else "_".join(f"{k}={v}" for k, v in sorted(diff.items()))


def accept(cand: dict, cur: dict, tau: float) -> bool:
    better_folds = sum(cand["per_fold"][f] < cur["per_fold"][f] for f in FOLDS)
    return cur["wrmsse_mean"] - cand["wrmsse_mean"] > tau and better_folds >= 3


class Runner:
    def __init__(self, cfg: dict, out_path: Path):
        self.cfg, self.out_path = cfg, out_path
        self.folds = {f.name: f for f in build_folds(cfg)}
        ids_all, sales_all, prices_all = load_wide_matrices(cfg, cfg["validation"]["holdout_cutoff"])
        sub = ids_all["store_id"].astype(str).isin(STORES).to_numpy()
        self.ids = ids_all[sub].reset_index(drop=True)
        self.sales, self.prices = sales_all[sub], prices_all[sub]
        del sales_all, prices_all
        self.base_root = Path(cfg["paths"]["processed"]) / "base"
        self.res = (json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists()
                    else {"configs": {}, "stages": {}})

    def save(self) -> None:
        self.res["updated"] = datetime.now().isoformat(timespec="seconds")
        self.res["xgboost_version"] = xgb.__version__
        self.out_path.write_text(json.dumps(self.res, indent=2, default=str), encoding="utf-8")

    def evaluate(self, name: str, params: dict, opt) -> dict:
        if name in self.res["configs"]:
            return self.res["configs"][name]
        preds = {f: np.zeros((len(self.ids), 28), dtype=np.float32) for f in FOLDS}
        infos = []
        t0 = time.perf_counter()
        with PeakRSS() as mem:
            for store in STORES:
                base = pd.read_parquet(self.base_root / f"store_id={store}" / "part-0.parquet")
                feats = all_feature_names(base)
                for fname in FOLDS:
                    p, info = fit_predict(base, self.folds[fname], feats, params, options=opt)
                    to_wide(p, self.ids, store, self.folds[fname], preds[fname])
                    infos.append({"store_id": store, "fold": fname, "best_iteration": info["best_iteration"],
                                  "seconds": info["seconds"], "model_mb": info["model_mb"]})
                    print(f"  {name:40s} {store} {fname} best={info['best_iteration']:4d} "
                          f"{info['seconds']:5.1f}s", flush=True)
                del base
        per_fold = {}
        for fname in FOLDS:
            f = self.folds[fname]
            actual = self.sales[:, f.cutoff:f.cutoff + f.horizon].astype(np.float64)
            ev = WRMSSE(self.ids, self.sales, self.prices, cutoff=f.cutoff, horizon=f.horizon)
            per_fold[fname] = round(ev.score(actual, preds[fname])["wrmsse"], 5)
        w = np.array([per_fold[f] for f in FOLDS])
        r = {"params": {k: v for k, v in params.items()}, "per_fold": per_fold,
             "wrmsse_mean": round(float(w.mean()), 5), "wrmsse_std": round(float(w.std(ddof=1)), 5),
             "peak_rss_gb": round(mem.peak_gb, 2), "minutes": round((time.perf_counter() - t0) / 60, 1),
             "infos": infos}
        self.res["configs"][name] = r
        self.save()
        print(f"{name:40s} WRMSSE {r['wrmsse_mean']:.5f}  " + " ".join(f"{k}={v:.4f}" for k, v in per_fold.items())
              + f"  RAM {r['peak_rss_gb']} GB  {r['minutes']} mnt", flush=True)
        return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_path = Path(cfg["paths"]["reports"]) / "data" / "xgb_tuning.json"
    run = Runner(cfg, out_path)
    base, opt = model_params(cfg["model"]["xgb_point"])

    r42 = run.evaluate("base", base, opt)
    r43 = run.evaluate("base_seed43", {**base, "seed": 43}, opt)
    noise = abs(r42["wrmsse_mean"] - r43["wrmsse_mean"])
    tau = max(noise, MIN_TAU)
    run.res["noise"] = {"seed42": r42["wrmsse_mean"], "seed43": r43["wrmsse_mean"], "abs_diff": round(noise, 5),
                        "tau": round(tau, 5)}
    run.save()
    print(f"noise seed = {noise:.5f}, tau = {tau:.5f}", flush=True)

    cur_params, cur_name = dict(base), "base"
    for stage, cands in STAGES:
        cur = run.res["configs"][cur_name]
        results = {}
        for c in cands:
            p = {**cur_params, **c}
            name = config_name(p, base)
            results[name] = run.evaluate(name, p, opt)
        best = min(results, key=lambda k: results[k]["wrmsse_mean"])
        ok = accept(results[best], cur, tau)
        run.res["stages"][stage] = {"start": cur_name, "start_wrmsse": cur["wrmsse_mean"],
                                    "candidates": {k: v["wrmsse_mean"] for k, v in results.items()},
                                    "best_candidate": best, "accepted": ok}
        if ok:
            cur_params, cur_name = dict(results[best]["params"]), best
        run.res["current"] = cur_name
        run.save()
        print(f"[{stage}] terbaik {best} ({results[best]['wrmsse_mean']:.5f}) vs {cur['wrmsse_mean']:.5f}: "
              f"{'DITERIMA' if ok else 'ditolak'}", flush=True)
    run.res["final"] = {"name": cur_name, "params": cur_params,
                        "wrmsse_mean": run.res["configs"][cur_name]["wrmsse_mean"]}
    run.save()
    print("final:", cur_name, run.res["configs"][cur_name]["wrmsse_mean"])


if __name__ == "__main__":
    main()
