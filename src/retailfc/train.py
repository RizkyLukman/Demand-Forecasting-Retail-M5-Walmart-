"""Training XGBoost per toko per fold.

Aturan yang dijaga di sini:
- fitting hanya pada baris d < awal jendela early stopping, bukan hari tutup;
- early stopping pada 28 hari terakhir sebelum cutoff, bukan horizon;
- horizon hanya diprediksi untuk item yang sudah pernah laku sampai cutoff; item lain
  diberi forecast 0 (tanggal mulai jual adalah informasi masa depan);
- prediksi hari tutup diisi 0.

Opsi (dipakai untuk eksperimen Fase 5):
- `exclude_censored`: keluarkan baris di dalam deretan nol >= `censor_run` hari dari
  item laris (laku >= `busy_share` hari) dari fitting dan early stopping. Deteksi hanya
  memakai data <= cutoff. Evaluasi tetap pada penjualan aktual.
- `refit`: setelah early stopping menentukan jumlah iterasi, latih ulang pada semua
  baris <= cutoff dengan jumlah iterasi itu.
- `checkpoints`: daftar batas iterasi yang ingin "ditiru" dari satu model berbatas
  `max_rounds`. Training deterministik, jadi pohon 1..C model ini identik dengan model
  berbatas C. Untuk tiap C, iterasi terbaik = skor validasi terbaik di antara iterasi
  1..C (sama dengan yang dipilih early stopping kalau batasnya C), lalu horizon diprediksi
  dengan pohon sampai iterasi itu. Kolom prediksi tambahan: `<kolom>_c<C>`.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from retailfc.features import add_fold_features, to_model_frame, zero_run_length
from retailfc.folds import Fold


@dataclass
class TrainOptions:
    item_encoding: str = "label"
    max_rounds: int = 3000
    early_stopping: int = 100
    exclude_censored: bool = False
    censor_run: int = 28
    busy_share: float = 0.5
    refit: bool = False
    checkpoints: tuple[int, ...] = ()
    save_model: str = ""        # jika diisi: path file model (.ubj) untuk disimpan setelah training


def emulate_cap(es_curve, cap: int, patience: int) -> tuple[int, bool]:
    """Iterasi terbaik dan status "berhenti di batas" untuk run berbatas `cap`, dibaca dari
    kurva validasi run berbatas lebih besar.

    Run berbatas `cap` menjalankan min(cap, len(kurva)) iterasi pertama yang sama persis.
    Early stopping XGBoost menganggap skor membaik hanya jika lebih kecil (bukan sama),
    jadi iterasi terbaiknya = kemunculan pertama nilai minimum (`np.argmin`).
    """
    n = min(cap, len(es_curve))
    best = int(np.argmin(np.asarray(es_curve[:n], dtype=np.float64)))
    return best, best + patience >= cap


def censored_mask(df: pd.DataFrame, cutoff: int, closed: np.ndarray, run_len: int,
                  busy_share: float) -> np.ndarray:
    """True untuk baris <= cutoff yang berada di dalam deretan nol >= `run_len` hari pada
    item yang laku >= `busy_share` hari bukanya (dihitung dari data <= cutoff saja)."""
    upto = df["d"].to_numpy() <= cutoff
    sub = df.loc[upto, ["item_id", "sales"]]
    cl = closed[upto]
    run = zero_run_length(sub, cl)
    item = sub["item_id"].cat.codes.to_numpy()
    sales = sub["sales"].to_numpy()
    new_item = np.r_[True, item[1:] != item[:-1]]
    grp = np.cumsum(((sales > 0) & ~cl) | new_item)
    run_total = pd.Series(run).groupby(grp).transform("max").to_numpy()
    is_zero = (sales == 0) & ~cl
    open_days = pd.Series(~cl).groupby(item).transform("sum").to_numpy()
    zeros = pd.Series(is_zero).groupby(item).transform("sum").to_numpy()
    busy = (1 - zeros / open_days) >= busy_share
    out = np.zeros(len(df), dtype=bool)
    out[upto] = is_zero & busy & (run_total >= run_len)
    return out


def fit_predict(
    base: pd.DataFrame,
    fold: Fold,
    feats: list[str],
    params: dict,
    item_encoding: str = "label",
    max_rounds: int = 3000,
    early_stopping: int = 100,
    options: TrainOptions | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Latih satu model untuk satu toko dan satu fold, lalu prediksi horizon-nya.

    Mengembalikan (prediksi horizon dalam format panjang, info run).
    """
    opt = options or TrainOptions(item_encoding=item_encoding, max_rounds=max_rounds,
                                  early_stopping=early_stopping)
    df = add_fold_features(base, fold.cutoff, fold.horizon)
    d = df["d"].to_numpy()
    closed = df["is_closed"].to_numpy().astype(bool)
    fit, es, hz = fold.fit_mask(d, closed), fold.es_mask(d, closed), fold.horizon_mask(d)

    # Item yang belum pernah laku sampai cutoff tidak diprediksi (forecast 0 nanti).
    first_d = df.groupby("item_id", observed=True)["d"].transform("min").to_numpy()
    hz &= first_d <= fold.cutoff

    n_censored = 0
    if opt.exclude_censored:
        cens = censored_mask(df, fold.cutoff, closed, opt.censor_run, opt.busy_share)
        n_censored = int((cens & (fit | es)).sum())
        fit &= ~cens
        es &= ~cens

    X = to_model_frame(df, feats, opt.item_encoding)
    y = df["sales"].to_numpy().astype(np.float32)

    # Hemat RAM: puncak terjadi saat QuantileDMatrix dibangun (terukur 4,87 GB di CA_3).
    # Ambil yang masih dibutuhkan untuk output, lalu lepas tabel fold sebelum itu, dan
    # lepas matriks fitur lengkap sebelum training. Data dan urutan baris tidak berubah.
    out = df.loc[hz, ["item_id", "dept_id", "cat_id", "d", "sales"]].copy()
    closed_hz = closed[hz]
    del df
    gc.collect()

    t = time.perf_counter()
    max_bin = params.get("max_bin", 256)
    dtrain = xgb.QuantileDMatrix(X[fit], y[fit], enable_categorical=True, max_bin=max_bin)
    # max_bin harus sama dengan dtrain; tanpa ini xgboost 3.4.1 menolak max_bin != 256.
    dvalid = xgb.QuantileDMatrix(X[es], y[es], enable_categorical=True, ref=dtrain, max_bin=max_bin)
    X_hz = X[hz]
    if not opt.refit:                             # refit masih butuh X untuk baris fit + es
        del X
        gc.collect()
    history: dict = {}
    booster = xgb.train(params, dtrain, num_boost_round=opt.max_rounds,
                        evals=[(dvalid, "es")], early_stopping_rounds=opt.early_stopping,
                        evals_result=history, verbose_eval=False)
    best = int(booster.best_iteration)
    n_rounds = best + 1
    # Skor validasi early stopping: nilai terbaik dan kurva tiap 100 iterasi (untuk melihat
    # apakah model masih membaik saat berhenti di batas iterasi).
    # Early stopping XGBoost memakai metrik terakhir di daftar; kurva yang sama dipakai di sini.
    es_metric, es_curve = list(history["es"].items())[-1]
    es_info = {"es_metric": es_metric, "es_best_score": round(float(es_curve[best]), 6),
               "es_curve_every_100": {i + 1: round(float(es_curve[i]), 6)
                                      for i in range(99, len(es_curve), 100)}}
    caps = sorted({int(c) for c in opt.checkpoints if int(c) <= opt.max_rounds})
    if caps:
        if opt.refit:
            raise ValueError("checkpoints cannot be combined with refit")
        # Pengaman: cara meniru batas harus memberi iterasi terbaik yang sama dengan
        # early stopping XGBoost pada batas run ini.
        if emulate_cap(es_curve, opt.max_rounds, opt.early_stopping)[0] != best:
            raise RuntimeError("emulated best iteration differs from booster.best_iteration")
        es_info["es_curve"] = [round(float(v), 6) for v in es_curve]
    if opt.refit:
        both = fit | es
        del dtrain, dvalid
        dall = xgb.QuantileDMatrix(X[both], y[both], enable_categorical=True, max_bin=max_bin)
        booster = xgb.train(params, dall, num_boost_round=n_rounds, verbose_eval=False)
        del dall
    # Data prediksi ada di memori CPU (pandas); samakan device booster agar XGBoost
    # tidak memindahkan data diam-diam (dan memunculkan peringatan device mismatch).
    booster.set_param({"device": "cpu"})

    def predict(n_trees: int) -> np.ndarray:
        p = booster.inplace_predict(X_hz, iteration_range=(0, n_trees))
        if p.ndim == 1:
            p = np.where(closed_hz, 0.0, p)
        else:                                     # model multi-kuantil: satu kolom per kuantil
            p = np.where(closed_hz[:, None], 0.0, p)
        return np.clip(p, 0, None).astype(np.float32)

    def put(p: np.ndarray, suffix: str = "") -> None:
        if p.ndim == 1:
            out[f"pred{suffix}"] = p
        else:
            for j in range(p.shape[1]):
                out[f"pred_q{j}{suffix}"] = p[:, j]

    pred = predict(n_rounds)
    put(pred)
    if caps:
        t_pred = time.perf_counter()
        cache = {best: pred}
        es_info["checkpoints"] = {}
        for c in caps:
            b, capped = emulate_cap(es_curve, c, opt.early_stopping)
            if b not in cache:
                cache[b] = predict(b + 1)
            put(cache[b], f"_c{c}")
            es_info["checkpoints"][str(c)] = {"best_iteration": b, "hit_cap": capped,
                                              "es_score": round(float(es_curve[b]), 6)}
        es_info["checkpoint_predict_seconds"] = round(time.perf_counter() - t_pred, 1)
    info = {"fold": fold.name, "best_iteration": best, "rounds_used": n_rounds,
            "seconds": round(time.perf_counter() - t, 1), "n_fit": int(fit.sum()),
            "n_es": int(es.sum()), "n_horizon": int(hz.sum()), "n_features": len(feats),
            "n_censored_excluded": n_censored, "refit": opt.refit,
            "item_encoding": opt.item_encoding, "model_mb": round(len(booster.save_raw("ubj")) / 2**20, 2),
            **es_info}
    if opt.save_model:
        # Model penuh disimpan; prediksi memakai pohon 0..best (`best_iteration` di info).
        booster.save_model(opt.save_model)
    return out, info


def to_wide(pred_long: pd.DataFrame, ids: pd.DataFrame, store: str, fold: Fold,
            out: np.ndarray, col: str = "pred") -> None:
    """Tulis prediksi format panjang satu toko ke matriks lebar `out` (n_deret × horizon).

    Baris/hari yang tidak ada di `pred_long` dibiarkan (harus diinisialisasi 0).
    """
    key = pd.MultiIndex.from_frame(ids[["item_id", "store_id"]].astype(str))
    rows = key.get_indexer(pd.MultiIndex.from_arrays(
        [pred_long["item_id"].astype(str), np.full(len(pred_long), store)]))
    if (rows < 0).any():
        raise ValueError("prediction for an unknown series")
    cols = pred_long["d"].to_numpy() - fold.horizon_start
    out[rows, cols] = pred_long[col].to_numpy()
