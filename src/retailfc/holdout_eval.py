"""Evaluasi holdout M5 (d_1914–d_1941), dijalankan SEKALI di akhir proyek.

Ditetapkan sebelum run (tidak ada yang diubah setelah hasil terlihat):
- Model: konfigurasi final apa adanya, `model.xgb_point` dan `model.xgb_quantile`
  (`config.yaml`), satu model per toko, cutoff d_1913, early stopping di d_1886–d_1913.
  Baseline: snaive, ma28, croston, tsb, classical (`baselines.run_fold`), cutoff yang sama.
- Metrik forecast: WRMSSE (bobot dari omzet d_1886–d_1913) beserta 12 level, MAE/RMSE/bias,
  akurasi 1 − WAPE per level, dan untuk model kuantil pinball, coverage, crossing.
- Simulasi stok: `inventory_sim.simulate_fold` dan `aggregate` dengan asumsi yang sama
  dengan CV (R = 7, L = 2, biaya dan sensitivitas di `config.yaml → inventory`).
- Pembanding: rata-rata dan rentang 4 fold CV dari laporan Fase 5 dan Fase 7.
- Tidak ada tuning, seleksi model, atau perubahan konfigurasi berdasarkan hasil holdout.

Run bisa dilanjutkan bila terhenti: prediksi tiap toko disimpan di
`data/processed/preds/_parts/holdout_<model>/`, jadi model yang sudah selesai tidak dilatih ulang.

Output: data/processed/preds/holdout/*.npy, reports/data/holdout_forecast.json,
reports/data/inventory_sim_holdout.json, reports/runs/<waktu>_holdout/.

Cara pakai:
    python -m retailfc.holdout_eval --config config.yaml
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
from retailfc.baselines import run_fold as run_baselines
from retailfc.features import all_feature_names
from retailfc.folds import build_folds
from retailfc.inventory_sim import aggregate, simulate_fold
from retailfc.io import load_calendar, load_config, load_wide_matrices
from retailfc.metrics import LEVELS, WRMSSE, basic_errors, build_aggregation
from retailfc.phase5_explain import wape_by_level
from retailfc.profiling import PeakRSS
from retailfc.quantile_final import evaluate_and_sort, quantile_params
from retailfc.quantile_final import score as quantile_score
from retailfc.train import fit_predict, to_wide
from retailfc.xgb_final import model_params


def train_store_parts(name: str, params: dict, opt, mcfg: dict, fold, stores, base_root: Path,
                      parts: Path, qcols: list[str] | None) -> list[dict]:
    """Latih satu model per toko untuk fold holdout; simpan prediksi tiap toko segera."""
    parts.mkdir(parents=True, exist_ok=True)
    infos = []
    for store in stores:
        info_path = parts / f"{store}.json"
        if info_path.exists():
            infos.append(json.loads(info_path.read_text(encoding="utf-8")))
            continue
        base = pd.read_parquet(base_root / f"store_id={store}" / "part-0.parquet")
        feats = all_feature_names(base, drop=tuple(mcfg.get("drop_features", [])))
        p, info = fit_predict(base, fold, feats, params, options=opt)
        del base
        cols = ["pred"]
        if qcols:
            info |= evaluate_and_sort(p, qcols)   # crossing dihitung sebelum diurutkan
            cols = qcols
        info |= {"store_id": store, "model": name}
        p[["item_id", "d"] + cols].to_parquet(parts / f"{store}.parquet")
        info_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        infos.append(info)
        print(f"{name} {store} best={info['best_iteration']:4d} {info['seconds']:6.1f}s", flush=True)
    return infos


def assemble(parts: Path, ids: pd.DataFrame, stores, fold, cols: list[str]) -> np.ndarray:
    out = np.zeros((len(ids), fold.horizon, len(cols)), dtype=np.float32)
    for store in stores:
        p = pd.read_parquet(parts / f"{store}.parquet")
        for j, c in enumerate(cols):
            to_wide(p, ids, store, fold, out[:, :, j], col=c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ho = next(f for f in build_folds(cfg) if f.role == "holdout")
    last_d = ho.cutoff + ho.horizon
    ids, sales, prices = load_wide_matrices(cfg, last_d)
    stores = sorted(ids["store_id"].astype(str).unique())
    base_root = Path(cfg["paths"]["processed"]) / "base"
    preds_root = Path(cfg["paths"]["processed"]) / "preds"
    out_dir = preds_root / ho.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Path(cfg["paths"]["reports"])

    with PeakRSS() as mem:
        # 1. XGBoost point
        mp = cfg["model"]["xgb_point"]
        params_p, opt_p = model_params(mp)
        infos_p = train_store_parts("xgb_point", params_p, opt_p, mp, ho, stores, base_root,
                                    preds_root / "_parts" / "holdout_xgb_point", None)
        point = assemble(preds_root / "_parts" / "holdout_xgb_point", ids, stores, ho, ["pred"])[:, :, 0]
        np.save(out_dir / "xgb_point.npy", point)

        # 2. XGBoost kuantil (satu model 4 kuantil per toko)
        mq = cfg["model"]["xgb_quantile"]
        params_q, opt_q = quantile_params(mq)
        alphas = [float(a) for a in mq["params"]["quantile_alpha"]]
        qcols = [f"pred_q{j}" for j in range(len(alphas))]
        infos_q = train_store_parts("xgb_quantile", params_q, opt_q, mq, ho, stores, base_root,
                                    preds_root / "_parts" / "holdout_xgb_quantile", qcols)
        qpred = assemble(preds_root / "_parts" / "holdout_xgb_quantile", ids, stores, ho, qcols)
        np.save(out_dir / "xgb_quantile.npy", qpred)
    peak_gb = round(mem.peak_gb, 2)

    # 3. Baseline (menyimpan npy + segments.parquet ke preds/holdout/)
    cal = load_calendar(Path(cfg["paths"]["raw_m5"]))
    base_res = run_baselines(ho, sales, ids, prices, cal, cfg, out_dir)

    # 4. Metrik forecast
    actual = sales[:, ho.cutoff:last_d].astype(np.float64)
    ev = WRMSSE(ids, sales, prices, cutoff=ho.cutoff, horizon=ho.horizon)
    S, labels = build_aggregation(ids)
    preds = {"xgb_point": point.astype(np.float64)}
    preds |= {m: np.load(out_dir / f"{m}.npy").astype(np.float64) for m in BASELINE_METHODS}
    forecast = {}
    for m, p in preds.items():
        sc = ev.score(actual, p)
        wape = wape_by_level(S, labels, actual, p)
        forecast[m] = {"wrmsse": round(sc["wrmsse"], 5), "per_level": sc["per_level"],
                       **{k: round(v, 5) for k, v in basic_errors(actual, p).items()},
                       "accuracy_pct": {lv: round(100 * (1 - wape[lv]), 2) for lv, _ in LEVELS}}
    quant = quantile_score(actual, qpred.astype(np.float64), alphas)
    quant["crossing_rate_mean"] = round(float(np.mean([i["crossing_rate"] for i in infos_q])), 5)

    result = {
        "fold": {"name": ho.name, "cutoff": ho.cutoff, "horizon": [ho.horizon_start, ho.horizon_end]},
        "forecast": forecast, "quantile": quant,
        "segment_counts": base_res["segment_counts"],
        "train": {"xgb_point": infos_p, "xgb_quantile": infos_q, "peak_rss_gb": peak_gb},
        "xgboost_version": xgb.__version__, "python": platform.python_version(),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (rep / "data" / "holdout_forecast.json").write_text(json.dumps(result, indent=2, default=str),
                                                        encoding="utf-8")
    for m, r in sorted(forecast.items(), key=lambda kv: kv[1]["wrmsse"]):
        print(f"{m:12s} holdout WRMSSE {r['wrmsse']:.4f}  acc total {r['accuracy_pct']['Level1']:.1f}%  "
              f"item-toko {r['accuracy_pct']['Level12']:.1f}%")
    print("kuantil coverage", quant["coverage"], "crossing", quant["crossing_rate_mean"])

    # 5. Simulasi stok, kode yang sama dengan CV
    inv = cfg["inventory"]
    sim = {"assumptions": inv, "folds": {ho.name: simulate_fold(ho, ids, sales, prices, cal, cfg, preds_root)}}
    sim |= aggregate(sim["folds"], [ho], inv)
    (rep / "data" / "inventory_sim_holdout.json").write_text(json.dumps(sim, indent=2, default=str),
                                                             encoding="utf-8")
    base_cost = {s["best"]: s for s in sim["sensitivity"]
                 if s["holding_annual"] == inv["holding_cost_annual_rate"] and s["stockout_rate"] == inv["stockout_cost_rate"]}
    print("stok: termurah pada asumsi dasar =", list(base_cost))

    run_dir = rep / "runs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_holdout"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"xgb_point": mp, "xgb_quantile": mq, "inventory": inv, "fold": result["fold"],
         "xgboost": xgb.__version__, "python": platform.python_version()}, indent=2, default=str), encoding="utf-8")
    (run_dir / "infos.json").write_text(json.dumps({"xgb_point": infos_p, "xgb_quantile": infos_q}, indent=2,
                                                   default=str), encoding="utf-8")
    print(f"run saved to {run_dir}")


if __name__ == "__main__":
    main()
