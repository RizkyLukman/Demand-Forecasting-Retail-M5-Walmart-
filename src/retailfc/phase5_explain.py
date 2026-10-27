"""Fase 5: angka dan grafik penjelas untuk laporan (akurasi 1 − WAPE, contoh hitungan WRMSSE).

Memakai prediksi yang sudah tersimpan di data/processed/preds/<fold>/ untuk fold cv1–cv4.
Data dimuat hanya sampai d_1913, jadi holdout tidak dibaca.

Cara pakai:
    python -m retailfc.phase5_explain --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retailfc.eda import INK, INK_2, MUTED, SERIES, SURFACE, style_axes
from retailfc.folds import build_folds
from retailfc.io import load_calendar, load_config, load_wide_matrices
from retailfc.metrics import LEVELS, WRMSSE, build_aggregation
from retailfc.phase5_report import LEVEL_NAMES, MODELS, dot_rows

CV = ["cv1", "cv2", "cv3", "cv4"]


def wape_by_level(S, labels: pd.DataFrame, actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """WAPE per level: total |aktual − forecast| dibagi total aktual, atas semua deret di level itu."""
    a, p = S @ actual, S @ pred
    lv = labels["Level_id"].to_numpy()
    abs_err = pd.Series(np.abs(a - p).sum(axis=1)).groupby(lv).sum()
    total = pd.Series(a.sum(axis=1)).groupby(lv).sum()
    return (abs_err / total).reindex([l for l, _ in LEVELS]).to_dict()


def worked_example(ev: WRMSSE, actual: np.ndarray, preds: dict[str, np.ndarray], row: int) -> dict:
    """Komponen RMSSE satu deret agregat (baris `row` di matriks S) untuk tiap model."""
    a = ev.S[row] @ actual
    out = {"label": ev.labels.iloc[row].to_dict(), "scale": float(ev.scale[row]),
           "naive_rmse": float(np.sqrt(ev.scale[row])), "weight": float(ev.weight[row]),
           "actual_mean": float(a.mean()), "models": {}}
    for k, p in preds.items():
        e = a - ev.S[row] @ p
        mse = float((e ** 2).mean())
        out["models"][k] = {"mse": mse, "rmse": float(np.sqrt(mse)),
                            "rmsse": float(np.sqrt(mse / ev.scale[row])),
                            "total_forecast": float((ev.S[row] @ p).sum()), "total_actual": float(a.sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep = Path(cfg["paths"]["reports"])
    pred_root = Path(cfg["paths"]["processed"]) / "preds"
    folds = {f.name: f for f in build_folds(cfg)}
    max_d = cfg["validation"]["holdout_cutoff"]
    ids, sales, prices = load_wide_matrices(cfg, max_d)
    S, labels = build_aggregation(ids)
    dates = load_calendar(Path(cfg["paths"]["raw_m5"])).set_index("d_num")["date"]
    keys = [k for k, _ in MODELS]

    result: dict = {"wape": {}, "total_28d": {}}
    for fname in CV:
        f = folds[fname]
        actual = sales[:, f.cutoff:f.cutoff + f.horizon].astype(np.float64)
        preds = {k: np.load(pred_root / fname / f"{k}.npy").astype(np.float64) for k in keys}
        result["wape"][fname] = {k: wape_by_level(S, labels, actual, preds[k]) for k in keys}
        result["total_28d"][fname] = {"actual": float(actual.sum()),
                                      **{k: float(preds[k].sum()) for k in keys}}

    # Contoh hitungan di cv4: deret total (Level1) dan deret item-toko berbobot terbesar.
    f = folds["cv4"]
    actual = sales[:, f.cutoff:f.cutoff + f.horizon].astype(np.float64)
    preds = {k: np.load(pred_root / "cv4" / f"{k}.npy").astype(np.float64) for k in keys}
    ev = WRMSSE(ids, sales, prices, cutoff=f.cutoff, horizon=f.horizon)
    lv = ev.labels["Level_id"].to_numpy()
    row_total = int(np.flatnonzero(lv == "Level1")[0])
    lv12 = np.flatnonzero(lv == "Level12")
    row_item = int(lv12[np.argmax(ev.weight[lv12])])
    result["example_cv4"] = {"total": worked_example(ev, actual, preds, row_total),
                             "top_item": worked_example(ev, actual, preds, row_item)}
    result["n_series_per_level"] = ev.labels["Level_id"].value_counts().reindex([l for l, _ in LEVELS]).to_dict()

    # 1. Total harian 10 toko di cv4: aktual vs forecast.
    days = np.arange(f.cutoff + 1, f.cutoff + f.horizon + 1)
    x = dates.loc[days].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    ax.plot(x, actual.sum(axis=0), color=INK, linewidth=2, label="aktual")
    for i, k in enumerate(["xgb_point", "classical"]):
        ax.plot(x, preds[k].sum(axis=0), color=SERIES[i], linewidth=2, label=dict(MODELS)[k])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("unit terjual per hari", color=INK_2, fontsize=9)
    span = f"{pd.Timestamp(x[0]):%d %b}–{pd.Timestamp(x[-1]):%d %b %Y}"
    fig.suptitle(f"Total penjualan 10 toko, cv4 ({span}): XGBoost mengikuti pola mingguan",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.93),
               labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(rep / "figures" / "phase5_total_cv4.png", facecolor=SURFACE)
    plt.close(fig)

    # 2. Contoh satu item-toko: riwayat 56 hari, aktual, dan forecast XGBoost.
    item_lab = ev.labels.iloc[row_item]
    bottom = int(np.flatnonzero(((ids["item_id"].astype(str) == item_lab["Agg_Level_1"])
                                 & (ids["store_id"].astype(str) == item_lab["Agg_Level_2"])).to_numpy())[0])
    hist_days = np.arange(f.cutoff - 55, f.cutoff + 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    ax.plot(dates.loc[hist_days].to_numpy(), sales[bottom, hist_days - 1], color=MUTED, linewidth=1.5,
            label="riwayat (sebelum cutoff)")
    ax.plot(x, actual[bottom], color=INK, linewidth=2, label="aktual")
    ax.plot(x, preds["xgb_point"][bottom], color=SERIES[0], linewidth=2, label="XGBoost")
    ax.axvline(dates.loc[f.cutoff + 1], color=MUTED, linewidth=0.8, linestyle=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("unit terjual per hari", color=INK_2, fontsize=9)
    fig.suptitle(f"{item_lab['Agg_Level_1']} di {item_lab['Agg_Level_2']}, cv4: forecast mengikuti "
                 "rata-rata, bukan lonjakan harian", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.93),
               labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(rep / "figures" / "phase5_item_example.png", facecolor=SURFACE)
    plt.close(fig)

    # 3. Akurasi (1 − WAPE) per level, rata-rata cv1–cv4.
    levels = [l for l, _ in LEVELS]
    acc = {k: [100 * (1 - np.mean([result["wape"][fn][k][l] for fn in CV])) for l in levels] for k in keys}
    result["accuracy_pct_cv_mean"] = {k: dict(zip(levels, np.round(v, 2))) for k, v in acc.items()}
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    dot_rows(ax, [LEVEL_NAMES[l] for l in levels], acc, offset=0.2)
    ax.set_xlabel("akurasi = 100% − WAPE, rata-rata cv1–cv4 (lebih besar lebih baik)", color=INK_2, fontsize=9)
    fig.suptitle("Akurasi tinggi untuk total, turun tajam di level item harian",
                 x=0.01, ha="left", fontsize=11, color=INK)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, ncol=3, loc="upper left",
               bbox_to_anchor=(0.01, 0.93), labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(rep / "figures" / "phase5_accuracy_by_level.png", facecolor=SURFACE)
    plt.close(fig)

    (rep / "data" / "phase5_explain.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"accuracy_pct_cv_mean": result["accuracy_pct_cv_mean"],
                      "total_28d": result["total_28d"], "example_cv4": result["example_cv4"]},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
