"""Fase 5: model kuantil final untuk 10 toko × 4 fold CV (bahan Fase 6–7).

Konfigurasi dibaca dari `config.yaml → model.xgb_quantile` (dipilih lewat
`quantile_experiments.py`, lihat reports/phase5_models.md §8). Parameter dipakai persis
seperti di eksperimen: tanpa `eval_metric` tambahan, jadi early stopping memakai metrik
bawaan objective kuantil.

Hanya fold CV (cv1–cv4). Fold khusus dan holdout (d_1914–d_1941) TIDAK dipakai di sini;
model holdout dibuat sekali di akhir proyek.

Tahan terhenti: setiap model (toko × fold) langsung disimpan ke
data/processed/preds/_parts/<nama run>/<fold>/<store>.parquet beserta info-nya, dan
bagian yang sudah ada dilewati saat skrip dijalankan lagi.

Output:
- data/processed/preds/<fold>/<nama run>.npy: (30.490 × 28 × n_kuantil), urutan kuantil
  sesuai `quantile_alpha`; sudah diurutkan per baris jika kuantil bersilangan;
- dengan `--checkpoints`: juga <nama run>_c<C>.npy untuk tiap batas C yang ditiru;
- reports/runs/<timestamp>_<nama run>/: config.json, infos.json, metrics.json
  (pinball, coverage, kuantil bersilangan per fold dan per toko, juga per batas C);
- cek konsistensi: pinball CA_3/WI_2 harus sama dengan reports/data/quantile_experiments.json
  (jika batasnya sama), dan model yang dilatih ulang harus sama dengan run acuan.

Uji batas iterasi (rencana di reports/quantile_cap1000.md §6): satu run berbatas 3.000
dengan titik baca 1.000/1.500/2.000/2.500/3.000. Training deterministik, jadi hasil di
titik C sama dengan run berbatas C (lihat `train.emulate_cap`). Titik 1.000 harus memberi
hasil yang sama persis dengan run batas 1.000 yang sudah ada; ini dicek otomatis.
Model yang di run acuan berhenti karena early stopping (bukan batas) akan berhenti di
iterasi yang sama, jadi hasilnya disalin dan ditandai `reused_from_run`. Model di
`--retrain` (format TOKO:fold) tetap dilatih ulang sebagai bukti.

Cara pakai:
    python -m retailfc.quantile_final --config config.yaml
    python -m retailfc.quantile_final --config config.yaml --max-rounds 3000 \\
        --checkpoints 1000 1500 2000 2500 3000 \\
        --reuse-uncapped-from reports/runs/20261024_075605_xgb_quantile --retrain CA_2:cv2 WI_2:cv2
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

from retailfc.features import all_feature_names
from retailfc.folds import build_folds
from retailfc.io import CAT_COLS, load_config, load_wide_matrices
from retailfc.metrics import pinball_loss
from retailfc.profiling import PeakRSS
from retailfc.quantile_experiments import ALPHAS as EXPERIMENT_ALPHAS
from retailfc.quantile_experiments import evaluate
from retailfc.train import TrainOptions, fit_predict, to_wide

NAME = "xgb_quantile"


def quantile_params(mcfg: dict) -> tuple[dict, TrainOptions]:
    params = {"device": "cuda", "tree_method": "hist", **mcfg["params"]}
    params["quantile_alpha"] = np.array(params["quantile_alpha"])
    return params, TrainOptions(**mcfg.get("options", {}))


def load_ids(cfg: dict) -> pd.DataFrame:
    """Kolom ID deret dengan urutan baris yang sama seperti `load_wide_matrices`."""
    path = Path(cfg["paths"]["raw_m5"]) / cfg["m5"]["sales_file"]
    return pd.read_csv(path, usecols=["id"] + CAT_COLS, dtype={c: "category" for c in CAT_COLS})[
        ["id"] + CAT_COLS]


def score(actual: np.ndarray, q: np.ndarray, alphas: list[float]) -> dict:
    """Pinball dan coverage per kuantil atas semua deret × hari. `q`: (n, h, k), sudah urut."""
    y = actual.ravel()
    pin = {str(a): round(pinball_loss(y, q[:, :, j].ravel(), a), 5) for j, a in enumerate(alphas)}
    return {"pinball": pin, "pinball_mean": round(float(np.mean(list(pin.values()))), 5),
            "coverage": {str(a): round(float((y <= q[:, :, j].ravel()).mean()), 4)
                         for j, a in enumerate(alphas)}}


def evaluate_and_sort(p: pd.DataFrame, cols: list[str]) -> dict:
    """Nilai satu set kolom kuantil (crossing dihitung sebelum diurutkan), lalu urutkan per baris."""
    ev = evaluate(p[cols + ["sales"]].set_axis([f"pred_q{j}" for j in range(len(cols))] + ["sales"], axis=1))
    p[cols] = np.sort(p[cols].to_numpy(), axis=1)
    return ev


def cv_summary(per_fold: dict, infos: list[dict], alphas: list[float], cap: str | None) -> dict:
    folds = list(per_fold)
    pm = [per_fold[f]["pinball_mean"] for f in folds]
    hit = [i["checkpoints"][cap]["hit_cap"] if cap else i["hit_cap"] for i in infos]
    cross = [i["checkpoints"][cap]["crossing_rate"] if cap else i["crossing_rate"] for i in infos]
    return {"pinball_mean": round(float(np.mean(pm)), 5), "pinball_std": round(float(np.std(pm, ddof=1)), 5),
            "coverage_mean": {str(a): round(float(np.mean([per_fold[f]["coverage"][str(a)] for f in folds])), 4)
                              for a in alphas},
            "crossing_rate_max": max(cross), "hit_cap": int(sum(hit)), "n_models": len(infos)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--checkpoints", type=int, nargs="*", default=[])
    ap.add_argument("--reuse-uncapped-from", default=None)
    ap.add_argument("--retrain", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"][NAME]
    params, opt = quantile_params(mcfg)
    name = NAME
    if args.max_rounds is not None and args.max_rounds != opt.max_rounds:
        opt.max_rounds = args.max_rounds
        name = f"{NAME}_r{args.max_rounds}"
    caps = sorted({c for c in args.checkpoints if c <= opt.max_rounds})
    opt.checkpoints = tuple(caps)
    alphas = [float(a) for a in mcfg["params"]["quantile_alpha"]]
    if alphas != EXPERIMENT_ALPHAS:
        raise ValueError("quantile_alpha differs from the experiment")
    k = len(alphas)
    qcols = [f"pred_q{j}" for j in range(k)]
    folds = [f for f in build_folds(cfg) if f.role == "cv"]

    reuse: dict = {}
    ref_infos: dict = {}
    ref_cap = None
    if args.reuse_uncapped_from:
        ref_dir = Path(args.reuse_uncapped_from)
        ref_cfg = json.loads((ref_dir / "config.json").read_text(encoding="utf-8"))
        ref_cap = int(ref_cfg["options"]["max_rounds"])
        ref_parts = Path(cfg["paths"]["processed"]) / "preds" / "_parts" / ref_cfg["name"]
        retrain = {tuple(s.split(":")) for s in args.retrain}
        for i in json.loads((ref_dir / "infos.json").read_text(encoding="utf-8")):
            ref_infos[(i["store_id"], i["fold"])] = i
            if not i["hit_cap"] and (i["store_id"], i["fold"]) not in retrain:
                reuse[(i["store_id"], i["fold"])] = (ref_parts, ref_dir.name, i)
        if caps and min(caps) < ref_cap:
            raise ValueError("checkpoints below the reference cap cannot reuse its uncapped models")

    ids = load_ids(cfg)
    stores = sorted(ids["store_id"].astype(str).unique())
    base_root = Path(cfg["paths"]["processed"]) / "base"
    preds_root = Path(cfg["paths"]["processed"]) / "preds"
    parts_root = preds_root / "_parts" / name

    # 0. Salin model yang di run acuan berhenti karena early stopping (hasil identik di
    #    semua batas >= batas acuan, karena iterasi terbaiknya + 100 < batas acuan).
    for (store, fname), (ref_parts, ref_run, info) in reuse.items():
        dst = parts_root / fname
        if (dst / f"{store}.json").exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        (dst / f"{store}.parquet").write_bytes((ref_parts / fname / f"{store}.parquet").read_bytes())
        ev = {key: info[key] for key in ("crossing_rate", "pinball", "pinball_mean", "coverage", "n_rows")}
        info = info | {"reused_from_run": ref_run,
                       "hit_cap": info["best_iteration"] + opt.early_stopping >= opt.max_rounds}
        if caps:
            info["checkpoints"] = {str(c): {"best_iteration": info["best_iteration"],
                                            "hit_cap": info["best_iteration"] + opt.early_stopping >= c, **ev}
                                   for c in caps}
        (dst / f"{store}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # 1. Training per toko × fold; setiap hasil langsung disimpan.
    with PeakRSS() as mem:
        for store in stores:
            todo = [f for f in folds if not (parts_root / f.name / f"{store}.json").exists()]
            if not todo:
                continue
            base = pd.read_parquet(base_root / f"store_id={store}" / "part-0.parquet")
            feats = all_feature_names(base, drop=tuple(mcfg.get("drop_features", [])))
            for f in todo:
                p, info = fit_predict(base, f, feats, params, options=opt)
                ev = evaluate_and_sort(p, qcols)
                info = info | {"store_id": store, "hit_cap": info["best_iteration"] + opt.early_stopping
                               >= opt.max_rounds, **ev}
                keep = list(qcols)
                for c in caps:
                    ccols = [f"{q}_c{c}" for q in qcols]
                    info["checkpoints"][str(c)].update(evaluate_and_sort(p, ccols))
                    keep += ccols
                (parts_root / f.name).mkdir(parents=True, exist_ok=True)
                p[["item_id", "d"] + keep].to_parquet(parts_root / f.name / f"{store}.parquet")
                (parts_root / f.name / f"{store}.json").write_text(json.dumps(info, indent=2),
                                                                  encoding="utf-8")
                extra = " ".join(f"c{c}={info['checkpoints'][str(c)]['pinball_mean']:.4f}" for c in caps)
                print(f"{store} {f.name} best={info['best_iteration']:4d} {info['seconds']:6.1f}s "
                      f"pinball={ev['pinball_mean']:.4f} cross={ev['crossing_rate']:.4f} "
                      f"cap={info['hit_cap']} {extra}", flush=True)
            del base

    # 2. Gabungkan ke matriks lebar dan nilai semua 30.490 deret, untuk run ini dan tiap batas C.
    infos = [json.loads((parts_root / f.name / f"{s}.json").read_text(encoding="utf-8"))
             for f in folds for s in stores]
    max_d = cfg["validation"]["holdout_cutoff"]
    ids_w, sales, _ = load_wide_matrices(cfg, max_d)
    if not ids_w["id"].astype(str).equals(ids["id"].astype(str)):
        raise ValueError("series order differs between load_ids and load_wide_matrices")
    store_arr = ids["store_id"].astype(str).to_numpy()
    variants = [("", None)] + [(f"_c{c}", str(c)) for c in caps]
    results = {cap: {"per_fold": {}, "per_store_fold": {}} for _, cap in variants}
    for f in folds:
        actual = sales[:, f.cutoff:f.cutoff + f.horizon].astype(np.float64)
        parts = {s: pd.read_parquet(parts_root / f.name / f"{s}.parquet") for s in stores}
        for suffix, cap in variants:
            q = np.zeros((len(ids), f.horizon, k), dtype=np.float32)
            for store, p in parts.items():
                for j, c in enumerate(qcols):
                    col = f"{c}{suffix}" if f"{c}{suffix}" in p.columns else c   # model salinan
                    to_wide(p, ids, store, f, q[:, :, j], col=col)
            np.save(preds_root / f.name / f"{name}{suffix}.npy", q)
            results[cap]["per_fold"][f.name] = score(actual, q, alphas)
            results[cap]["per_store_fold"][f.name] = {
                s: score(actual[store_arr == s], q[store_arr == s], alphas) for s in stores}
    metrics: dict = dict(results[None])
    metrics["cv_summary"] = cv_summary(results[None]["per_fold"], infos, alphas, None) | {
        "train_seconds_total": round(sum(i["seconds"] for i in infos), 1),
        "n_reused": sum("reused_from_run" in i for i in infos),
        "train_seconds_this_run": round(sum(i["seconds"] for i in infos if "reused_from_run" not in i), 1)}
    if caps:
        metrics["checkpoints"] = {
            cap: {"cv_summary": cv_summary(results[cap]["per_fold"], infos, alphas, cap), **results[cap]}
            for _, cap in variants[1:]}
    metrics["peak_rss_gb_training"] = round(mem.peak_gb, 2)

    # 3a. Cek terhadap run acuan: (i) model yang dilatih ulang di `--retrain` harus identik;
    #     (ii) titik baca = batas acuan harus identik dengan run acuan untuk semua model
    #     yang dilatih di run ini.
    if ref_infos:
        trained = [i for i in infos if "reused_from_run" not in i]
        checks = {}
        for i in trained:
            ref = ref_infos[(i["store_id"], i["fold"])]
            here = i["checkpoints"][str(ref_cap)] if str(ref_cap) in i.get("checkpoints", {}) else i
            checks[f"{i['store_id']}_{i['fold']}"] = {
                "best_iteration": [ref["best_iteration"], here["best_iteration"]],
                "pinball_abs_diff": round(abs(here["pinball_mean"] - ref["pinball_mean"]), 6)}
        metrics["check_vs_reference"] = {
            "reference_cap": ref_cap,
            "max_pinball_abs_diff": max(c["pinball_abs_diff"] for c in checks.values()) if checks else None,
            "best_iteration_mismatch": [key for key, c in checks.items()
                                        if c["best_iteration"][0] != c["best_iteration"][1]],
            "per_run": checks}

    # 3b. Cek konsistensi dengan eksperimen (training deterministik: harus sama persis).
    #     Hanya berlaku jika batas iterasi sama dengan eksperimen.
    exp_path = Path(cfg["paths"]["reports"]) / "data" / "quantile_experiments.json"
    exp_json = json.loads(exp_path.read_text(encoding="utf-8")) if exp_path.exists() else None
    if exp_json and exp_json.get("max_rounds") == opt.max_rounds:
        exp = {(r["store_id"], r["fold"]): r["pinball_mean"] for r in exp_json["runs"]
               if r["eta"] == mcfg["params"]["eta"]}
        diffs = {f"{i['store_id']}_{i['fold']}": round(abs(i["pinball_mean"] - exp[(i["store_id"], i["fold"])]), 6)
                 for i in infos if (i["store_id"], i["fold"]) in exp}
        metrics["check_vs_experiment"] = {"max_abs_diff": max(diffs.values()) if diffs else None,
                                          "per_run": diffs}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["paths"]["reports"]) / "runs" / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"name": name, "model": mcfg, "params_used": {k_: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                      for k_, v in params.items()},
         "options": {**opt.__dict__, "checkpoints": list(opt.checkpoints)},
         "folds": [f.name for f in folds], "xgboost": xgb.__version__,
         "python": platform.python_version(), "seed": params.get("seed"),
         "reference_run": args.reuse_uncapped_from, "retrain": args.retrain}, indent=2), encoding="utf-8")
    (run_dir / "infos.json").write_text(json.dumps(infos, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"cv_summary": metrics["cv_summary"],
                      "checkpoints": {c: v["cv_summary"] for c, v in metrics.get("checkpoints", {}).items()},
                      "check_vs_experiment": metrics.get("check_vs_experiment"),
                      "check_vs_reference": {key: v for key, v in metrics.get("check_vs_reference", {}).items()
                                             if key != "per_run"}}, indent=1))
    print(f"run saved to {run_dir}")


if __name__ == "__main__":
    main()
