"""Fase 5: eksperimen model kuantil (untuk P3 / simulasi stok di Fase 7).

Rancangan (revisi 24 Okt 2026):
- Model: satu model multi-kuantil `reg:quantileerror`, alpha = [0,1; 0,5; 0,8; 0,9],
  `item_id` label encoding (Fase 1/4: kategorikal membuat RAM melonjak).
- Training tanpa deretan nol panjang item laris (`exclude_censored`), karena Fase 7
  mensimulasikan permintaan, bukan penjualan.
- Varian: eta 0,2 dan 0,3 di 2 toko (CA_3, WI_2) × 4 fold CV. Early stopping 100 iterasi,
  maks 1000. Run yang berhenti karena batas, bukan early stopping, dicatat (`hit_cap`):
  kalau sering terjadi, model belum selesai belajar dan batasnya harus dinaikkan.
- Acuan eta 0,1: 3 run yang selesai sebelum rancangan awal terhenti (lihat di bawah).
- Metrik: pinball loss per kuantil, coverage (porsi aktual <= prediksi kuantil),
  frekuensi kuantil bersilangan (sebelum diurutkan ulang), waktu.
- Aturan memilih (tidak berubah dari rancangan awal): pinball rata-rata (4 kuantil, semua
  fold) terendah; jika selisih < 1%, pilih yang paling cepat.

Riwayat rancangan: rancangan awal (eta 0,1 dan 0,2 di 2 toko × 4 fold, eta 0,05 di cv4,
maks 3000 iterasi) dihentikan sistem karena RAM laptop menipis setelah 3 run. Satu run
eta 0,1 butuh 293–524 detik. Rancangan direvisi karena waktu dan RAM, bukan karena nilai
pinball; aturan memilih tidak diubah. Acuan eta 0,05 tidak dijalankan.

Keputusan setelah eksperimen: eta 0,2. Ini menyimpang dari aturan (yang memilih eta 0,3,
selisih pinball 0,20% dan 11% lebih cepat). Alasannya dicatat di reports/phase5_models.md §8.2.

Kuantil yang bersilangan diurutkan ulang per baris (np.sort) sebelum dievaluasi, dan
frekuensinya dicatat. Hasil ditulis ke JSON setiap kali satu run selesai; run yang sudah
ada di file dilewati, jadi eksperimen bisa dilanjutkan kalau terhenti.

Cara pakai:
    python -m retailfc.quantile_experiments --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from retailfc.features import all_feature_names
from retailfc.folds import build_folds
from retailfc.io import load_config
from retailfc.metrics import pinball_loss
from retailfc.profiling import PeakRSS
from retailfc.train import TrainOptions, fit_predict

STORES = ["CA_3", "WI_2"]
FOLDS = ["cv1", "cv2", "cv3", "cv4"]
ALPHAS = [0.1, 0.5, 0.8, 0.9]
VARIANTS = {"eta_0.2": 0.2, "eta_0.3": 0.3}
MAX_ROUNDS, EARLY_STOPPING = 1000, 100

# Acuan eta 0,1 (maks 3000 iterasi) dari log run rancangan awal yang terhenti
# (24 Okt 2026). Log hanya mencatat pinball rata-rata 4 kuantil, tanpa coverage.
REFERENCE_ETA_01 = [
    {"store_id": "CA_3", "fold": "cv1", "best_iteration": 2989, "seconds": 523.5, "pinball_mean": 0.4611, "crossing_rate": 0.0},
    {"store_id": "CA_3", "fold": "cv2", "best_iteration": 1482, "seconds": 292.5, "pinball_mean": 0.4667, "crossing_rate": 0.0},
    {"store_id": "CA_3", "fold": "cv3", "best_iteration": 2142, "seconds": 413.9, "pinball_mean": 0.4746, "crossing_rate": 0.0},
]


def params(eta: float, seed: int = 42) -> dict:
    return {"device": "cuda", "tree_method": "hist", "max_bin": 256,
            "objective": "reg:quantileerror", "quantile_alpha": np.array(ALPHAS),
            "eta": eta, "max_depth": 8, "min_child_weight": 10,
            "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed}


def evaluate(p: pd.DataFrame) -> dict:
    q = p[[f"pred_q{j}" for j in range(len(ALPHAS))]].to_numpy()
    crossing = float((np.diff(q, axis=1) < 0).any(axis=1).mean())
    q = np.sort(q, axis=1)
    y = p["sales"].to_numpy().astype(np.float64)
    pin = {str(a): round(pinball_loss(y, q[:, j], a), 5) for j, a in enumerate(ALPHAS)}
    return {
        "crossing_rate": round(crossing, 5),
        "pinball": pin,
        "pinball_mean": round(float(np.mean(list(pin.values()))), 5),
        "coverage": {str(a): round(float((y <= q[:, j]).mean()), 4) for j, a in enumerate(ALPHAS)},
        "n_rows": int(len(y)),
    }


def ran_to_cap(best_iteration: int) -> bool:
    """True jika training berhenti karena batas iterasi, bukan early stopping
    (early stopping butuh `EARLY_STOPPING` iterasi tanpa perbaikan setelah iterasi terbaik)."""
    return best_iteration + EARLY_STOPPING >= MAX_ROUNDS


def summarize(out: dict) -> None:
    for r in out["runs"]:
        r["hit_cap"] = ran_to_cap(r["best_iteration"])
    runs = pd.DataFrame(out["runs"])
    out["summary"] = runs.groupby("variant").agg(
        pinball_mean=("pinball_mean", "mean"), seconds_mean=("seconds", "mean"),
        best_iter_mean=("best_iteration", "mean"), hit_cap=("hit_cap", "sum"),
        crossing_mean=("crossing_rate", "mean"), n=("fold", "size")).round(5).to_dict(orient="index")
    # Pembanding yang adil dengan acuan eta 0,1: hanya run di toko/fold yang sama.
    ref = pd.DataFrame(REFERENCE_ETA_01).assign(variant="eta_0.1_reference")
    keys = set(zip(ref["store_id"], ref["fold"]))
    same = runs[[(s, f) in keys for s, f in zip(runs["store_id"], runs["fold"])]]
    both = pd.concat([ref, same[ref.columns.intersection(same.columns)]])
    out["vs_eta_0.1_reference"] = both.groupby("variant").agg(
        pinball_mean=("pinball_mean", "mean"), seconds_mean=("seconds", "mean"),
        n=("fold", "size")).round(5).to_dict(orient="index")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    folds = {f.name: f for f in build_folds(cfg)}
    base_root = Path(cfg["paths"]["processed"]) / "base"
    out_path = Path(cfg["paths"]["reports"], "data", "quantile_experiments.json")
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"runs": []}
    out.update({"stores": STORES, "alphas": ALPHAS, "max_rounds": MAX_ROUNDS,
                "early_stopping": EARLY_STOPPING, "reference_eta_0.1": REFERENCE_ETA_01})
    done = {(r["variant"], r["store_id"], r["fold"]) for r in out["runs"]}
    opt = TrainOptions(exclude_censored=True, max_rounds=MAX_ROUNDS, early_stopping=EARLY_STOPPING)

    with PeakRSS() as mem:
        for store in STORES:
            todo = [(v, f) for f in FOLDS for v in VARIANTS if (v, store, f) not in done]
            if not todo:
                continue
            base = pd.read_parquet(base_root / f"store_id={store}" / "part-0.parquet")
            feats = all_feature_names(base)
            for name, fname in todo:
                p, info = fit_predict(base, folds[fname], feats, params(VARIANTS[name]), options=opt)
                r = {"variant": name, "store_id": store, "fold": fname, "eta": VARIANTS[name],
                     **info, "hit_cap": ran_to_cap(info["best_iteration"]), **evaluate(p)}
                out["runs"].append(r)
                out["peak_rss_gb"] = round(max(mem.peak_gb, out.get("peak_rss_gb", 0)), 2)
                out["updated"] = datetime.now().isoformat(timespec="seconds")
                out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
                print(f"{store} {fname} {name:8s} best={info['best_iteration']:4d} "
                      f"{info['seconds']:6.1f}s pinball={r['pinball_mean']:.4f} "
                      f"cross={r['crossing_rate']:.4f} cap={r['hit_cap']}", flush=True)
            del base

    summarize(out)
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"summary": out["summary"], "vs_eta_0.1_reference": out["vs_eta_0.1_reference"]},
                     indent=1))


if __name__ == "__main__":
    main()
