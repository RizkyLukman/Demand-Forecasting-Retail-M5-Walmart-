"""Fase 1: ukur RAM / VRAM / waktu training XGBoost untuk satu toko.

Set fitur di sini SEMENTARA dan hanya dipakai agar profil punya bentuk yang
realistis (jumlah baris, jumlah fitur, kolom kategorikal). Fitur final yang
diuji terhadap cutoff dibuat di Fase 4; jalankan ulang profil ini saat itu.

Hanya baris dengan d <= cutoff yang dipakai. Early stopping memakai 28 hari
terakhir di dalam jendela itu, jadi horizon fold tidak tersentuh.

Cara pakai:
    python -m retailfc.hardware_profile --config config.yaml --store CA_1
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb

from retailfc.io import load_calendar, load_config
from retailfc.profiling import PeakRSS, PeakVRAM, system_ram

LAGS = [28, 35, 42, 49, 56, 364]
WINDOWS = [7, 14, 28, 56]


def provisional_features(long: pd.DataFrame, cal: pd.DataFrame, state: str) -> pd.DataFrame:
    """`long` harus satu toko, terurut per item_id lalu d, dengan hari berurutan per item."""
    df = long
    g = df.groupby("item_id", observed=True, sort=False)["sales"]
    for k in LAGS:
        df[f"lag_{k}"] = g.shift(k).astype(np.float32)
    g28 = df.groupby("item_id", observed=True, sort=False)["lag_28"]
    for w in WINDOWS:
        df[f"rmean_{w}"] = g28.transform(lambda x: x.rolling(w, min_periods=1).mean()).astype(np.float32)
        df[f"rstd_{w}"] = g28.transform(lambda x: x.rolling(w, min_periods=2).std()).astype(np.float32)

    c = cal.set_index("d_num")
    d = df["d"].to_numpy()
    df["wday"] = c["wday"].reindex(d).to_numpy().astype(np.int8)
    df["month"] = c["month"].reindex(d).to_numpy().astype(np.int8)
    df["mday"] = c["date"].reindex(d).dt.day.to_numpy().astype(np.int8)
    df["year"] = c["year"].reindex(d).to_numpy().astype(np.int16)
    df["snap"] = c[f"snap_{state}"].reindex(d).to_numpy().astype(np.int8)
    df["event_name_1"] = pd.Categorical(c["event_name_1"].reindex(d).to_numpy(),
                                        categories=sorted(cal["event_name_1"].dropna().unique()))
    df["event_type_1"] = pd.Categorical(c["event_type_1"].reindex(d).to_numpy(),
                                        categories=sorted(cal["event_type_1"].dropna().unique()))

    gp = df.groupby("item_id", observed=True, sort=False)["sell_price"]
    df["price_rel_item_mean"] = (df["sell_price"] / gp.transform("mean")).astype(np.float32)
    df["price_change_7d"] = (df["sell_price"] / gp.shift(7) - 1).astype(np.float32)
    return df


def train_once(params: dict, dtrain, dvalid, rounds: int, early: int) -> dict:
    with PeakVRAM() as vram, PeakRSS() as rss:
        t = time.perf_counter()
        booster = xgb.train(params, dtrain, num_boost_round=rounds,
                            evals=[(dvalid, "valid")], early_stopping_rounds=early,
                            verbose_eval=False)
        secs = time.perf_counter() - t
    model_mb = round(len(booster.save_raw("ubj")) / 2**20, 1)
    return {
        "model_mb": model_mb,
        "params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in params.items()},
        "seconds": round(secs, 1),
        "best_iteration": int(booster.best_iteration),
        "rounds_run": int(booster.num_boosted_rounds()),
        "sec_per_round": round(secs / booster.num_boosted_rounds(), 3),
        "vram_baseline_mib": vram.baseline_mib,
        "vram_peak_mib": vram.peak_mib,
        "vram_delta_mib": vram.peak_mib - vram.baseline_mib,
        "peak_rss_gb_during_train": round(rss.peak_gb, 2),
    }


def profile_final(args, cfg: dict) -> None:
    """Fase 4: profil ulang dengan fitur final (tabel dasar + fitur per fold).

    Setup sama dengan Fase 1 (toko, cutoff, parameter), supaya angkanya bisa dibandingkan.
    """
    from retailfc.features import add_fold_features, feature_names, to_model_frame

    out: dict = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "machine": {"cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
                    "ram_at_start": system_ram(), "xgboost": xgb.__version__,
                    "pandas": pd.__version__},
        "store": args.store, "cutoff": args.cutoff, "features_set": "final", "runs": [],
    }
    base_params = {"device": "cuda", "tree_method": "hist", "max_bin": 256, "eta": 0.05,
                   "max_depth": 8, "min_child_weight": 10, "subsample": 0.8,
                   "colsample_bytree": 0.8, "seed": cfg["seed"]}
    configs = [
        ("point_tweedie", "category", {"objective": "reg:tweedie", "tweedie_variance_power": 1.2,
                                       "eval_metric": "rmse"}),
        ("point_tweedie", "label", {"objective": "reg:tweedie", "tweedie_variance_power": 1.2,
                                    "eval_metric": "rmse"}),
        ("multi_quantile_4", "label", {"objective": "reg:quantileerror",
                                       "quantile_alpha": np.array([0.1, 0.5, 0.8, 0.9])}),
    ]
    with PeakRSS() as rss_all:
        t = time.perf_counter()
        base = pd.read_parquet(Path(cfg["paths"]["processed"]) / "base" / f"store_id={args.store}"
                               / "part-0.parquet")
        df = add_fold_features(base, args.cutoff, 0)            # hanya d <= cutoff
        del base
        feats = feature_names(df)
        closed = df["is_closed"].to_numpy().astype(bool)
        valid_start = args.cutoff - 27
        tr = (df["d"].to_numpy() < valid_start) & ~closed
        va = (df["d"].to_numpy() >= valid_start) & ~closed
        out["features"] = {"n": len(feats), "names": feats,
                           "seconds": round(time.perf_counter() - t, 1),
                           "table_mb": round(df.memory_usage(deep=True).sum() / 2**20, 1),
                           "peak_rss_gb": round(rss_all.peak_gb, 2)}
        y = df["sales"].to_numpy().astype(np.float32)
        for name, enc, obj in configs:
            X = to_model_frame(df, feats, enc)
            dtrain = xgb.QuantileDMatrix(X[tr], y[tr], enable_categorical=True, max_bin=256)
            dvalid = xgb.QuantileDMatrix(X[va], y[va], enable_categorical=True, ref=dtrain)
            run = {"name": name, "item_encoding": enc,
                   **train_once({**base_params, **obj}, dtrain, dvalid, args.rounds, 100)}
            out["runs"].append(run)
            print(json.dumps({k: v for k, v in run.items() if k != "params"}))
            del dtrain, dvalid, X
    out["peak_rss_gb_total"] = round(rss_all.peak_gb, 2)
    path = Path(cfg["paths"]["reports"]) / "hardware_profile_phase4.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Written to {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--store", default="CA_1")
    ap.add_argument("--cutoff", type=int, default=1885)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--features", choices=["provisional", "final"], default="provisional")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.features == "final":
        profile_final(args, cfg)
        return
    seed = cfg["seed"]

    out: dict = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "machine": {"cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
                    "ram_at_start": system_ram(), "xgboost": xgb.__version__,
                    "pandas": pd.__version__},
        "store": args.store, "cutoff": args.cutoff,
    }

    with PeakRSS() as rss_all:
        t = time.perf_counter()
        long = pd.read_parquet(
            Path(cfg["paths"]["interim"]) / "sales_long" / f"store_id={args.store}" / "part-0.parquet")
        long = long[long["d"] <= args.cutoff].reset_index(drop=True)
        cal = load_calendar(Path(cfg["paths"]["raw_m5"]))
        df = provisional_features(long, cal, state=args.store.split("_")[0])
        feats = [c for c in df.columns if c not in ("sales", "d", "wm_yr_wk", "state_id")]
        out["features"] = {"n": len(feats), "names": feats,
                           "seconds": round(time.perf_counter() - t, 1),
                           "table_mb": round(df.memory_usage(deep=True).sum() / 2**20, 1),
                           "peak_rss_gb": round(rss_all.peak_gb, 2)}

        valid_start = args.cutoff - 27
        tr = df["d"] < valid_start
        t = time.perf_counter()
        dtrain = xgb.QuantileDMatrix(df.loc[tr, feats], df.loc[tr, "sales"], enable_categorical=True,
                                     max_bin=256)
        dvalid = xgb.QuantileDMatrix(df.loc[~tr, feats], df.loc[~tr, "sales"], enable_categorical=True,
                                     ref=dtrain)
        out["dmatrix"] = {"train_rows": int(tr.sum()), "valid_rows": int((~tr).sum()),
                          "valid_days": f"d_{valid_start}..d_{args.cutoff}",
                          "seconds": round(time.perf_counter() - t, 1),
                          "peak_rss_gb_so_far": round(rss_all.peak_gb, 2)}
        del df, long

        base = {"device": "cuda", "tree_method": "hist", "max_bin": 256, "eta": 0.05,
                "max_depth": 8, "min_child_weight": 10, "subsample": 0.8,
                "colsample_bytree": 0.8, "seed": seed}
        out["runs"] = []
        tweedie = {**base, "objective": "reg:tweedie", "tweedie_variance_power": 1.2,
                   "eval_metric": "rmse"}
        out["runs"].append({"name": "point_tweedie", **train_once(tweedie, dtrain, dvalid, args.rounds, 100)})
        print(json.dumps(out["runs"][-1], indent=2))

        quant = {**base, "objective": "reg:quantileerror",
                 "quantile_alpha": np.array([0.1, 0.5, 0.8, 0.9])}
        out["runs"].append({"name": "multi_quantile_4", **train_once(quant, dtrain, dvalid, args.rounds, 100)})
        print(json.dumps(out["runs"][-1], indent=2))

    out["peak_rss_gb_total"] = round(rss_all.peak_gb, 2)
    rep = Path(cfg["paths"]["reports"])
    (rep / "hardware_profile.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ["features", "dmatrix", "peak_rss_gb_total"]}, indent=2,
                     default=str))
    print(f"Written to {rep / 'hardware_profile.json'}")


if __name__ == "__main__":
    main()
