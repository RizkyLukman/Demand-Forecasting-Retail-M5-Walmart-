"""Training XGBoost per toko per fold.

Fase 4 memakai `fit_predict` untuk ablation fitur. Fase 5 akan memperluas modul ini
(baseline pembanding, kuantil, laporan run).

Aturan yang dijaga di sini:
- fitting hanya pada baris d < awal jendela early stopping, bukan hari tutup;
- early stopping pada 28 hari terakhir sebelum cutoff, bukan horizon;
- prediksi hari tutup diisi 0.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from retailfc.features import add_fold_features, to_model_frame
from retailfc.folds import Fold


def fit_predict(
    base: pd.DataFrame,
    fold: Fold,
    feats: list[str],
    params: dict,
    item_encoding: str = "label",
    max_rounds: int = 3000,
    early_stopping: int = 100,
) -> tuple[pd.DataFrame, dict]:
    """Latih satu model untuk satu toko dan satu fold, lalu prediksi horizon-nya.

    Mengembalikan (prediksi horizon, info run).
    """
    df = add_fold_features(base, fold.cutoff, fold.horizon)
    d = df["d"].to_numpy()
    closed = df["is_closed"].to_numpy().astype(bool)
    fit, es, hz = fold.fit_mask(d, closed), fold.es_mask(d, closed), fold.horizon_mask(d)
    X = to_model_frame(df, feats, item_encoding)
    y = df["sales"].to_numpy().astype(np.float32)

    t = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(X[fit], y[fit], enable_categorical=True,
                                 max_bin=params.get("max_bin", 256))
    dvalid = xgb.QuantileDMatrix(X[es], y[es], enable_categorical=True, ref=dtrain)
    booster = xgb.train(params, dtrain, num_boost_round=max_rounds,
                        evals=[(dvalid, "es")], early_stopping_rounds=early_stopping,
                        verbose_eval=False)
    best = int(booster.best_iteration)
    # Data prediksi ada di memori CPU (pandas); samakan device booster agar XGBoost
    # tidak memindahkan data diam-diam (dan memunculkan peringatan device mismatch).
    booster.set_param({"device": "cpu"})
    pred = booster.inplace_predict(X[hz], iteration_range=(0, best + 1))
    pred = np.where(closed[hz], 0.0, pred).astype(np.float32)

    out = df.loc[hz, ["item_id", "dept_id", "cat_id", "d", "sales"]].copy()
    out["pred"] = pred
    info = {"fold": fold.name, "best_iteration": best, "rounds_run": int(booster.num_boosted_rounds()),
            "seconds": round(time.perf_counter() - t, 1), "n_fit": int(fit.sum()),
            "n_es": int(es.sum()), "n_horizon": int(hz.sum()), "n_features": len(feats)}
    return out, info
