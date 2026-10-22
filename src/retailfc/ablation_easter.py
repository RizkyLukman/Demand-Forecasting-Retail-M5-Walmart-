"""Fase 4 (P4): ablation fitur hari libur bergerak untuk Easter.

Membandingkan model dengan vs tanpa `days_to_easter` / `days_since_easter`. Penanda
`event_name_1 == "Easter"` di hari-H tetap ada di kedua varian, jadi yang diuji adalah
nilai tambah informasi "berapa hari menuju/sejak Easter".

Kriteria (ditetapkan sebelum melihat hasil): fitur berguna jika error di jendela
±7 hari sekitar Easter turun konsisten di semua fold dan toko, dan penurunannya lebih
besar daripada selisih antar seed.

Cara pakai:
    python -m retailfc.ablation_easter --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retailfc.calendar_features import event_days
from retailfc.features import feature_names
from retailfc.folds import build_folds
from retailfc.io import load_calendar, load_config
from retailfc.profiling import PeakRSS
from retailfc.train import fit_predict

STORES = ["CA_1", "TX_1"]
FOLDS = ["easter_2014", "easter_2015", "cv3", "cv4"]
SEEDS = [42, 43]
WINDOW = 7
EASTER_FEATS = ("days_to_easter", "days_since_easter")


def params(seed: int) -> dict:
    return {"device": "cuda", "tree_method": "hist", "max_bin": 256,
            "objective": "reg:tweedie", "tweedie_variance_power": 1.2, "eval_metric": "rmse",
            "eta": 0.05, "max_depth": 8, "min_child_weight": 10,
            "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed}


def easter_offset(d: np.ndarray, easter: np.ndarray) -> np.ndarray:
    """Selisih hari ke Easter terdekat (negatif = sebelum)."""
    idx = np.abs(d[:, None] - easter[None, :]).argmin(axis=1)
    return d - easter[idx]


def summarise(pred: pd.DataFrame, easter: np.ndarray) -> pd.DataFrame:
    """Error per (store, fold, variant, seed): seluruh horizon dan jendela Easter."""
    pred = pred.assign(offset=easter_offset(pred["d"].to_numpy(), easter))
    pred["in_window"] = pred["offset"].abs() <= WINDOW
    pred["err"] = pred["pred"] - pred["sales"]
    keys = ["store_id", "fold", "variant", "seed"]
    rows = []
    for k, g in pred.groupby(keys, observed=True):
        w = g[g["in_window"]]
        # Total harian per kategori: apakah pola agregat di sekitar Easter tertangkap?
        tot = w.groupby(["cat_id", "d"], observed=True)[["sales", "pred"]].sum()
        rows.append(dict(zip(keys, k)) | {
            "rmse_horizon": float(np.sqrt((g["err"] ** 2).mean())),
            "rmse_window": float(np.sqrt((w["err"] ** 2).mean())),
            "mae_window": float(w["err"].abs().mean()),
            "bias_window": float(w["err"].mean()),
            "daily_total_wape_window": float((tot["pred"] - tot["sales"]).abs().sum() / tot["sales"].sum()),
            "window_days": int(w["d"].nunique()),
        })
    return pd.DataFrame(rows)


def plot_offsets(result_json: Path, path: Path) -> None:
    """Rasio total prediksi / total aktual per hari sekitar Easter, per kategori."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from retailfc.eda import BASE, INK, INK_2, MUTED, SERIES, SURFACE, style_axes

    r = json.loads(result_json.read_text(encoding="utf-8"))["pred_over_actual_by_offset"]
    cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    labels = {"without_easter": "tanpa fitur Easter", "with_easter": "dengan fitur Easter"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=150, sharey=True, facecolor=SURFACE)
    for ax, cat in zip(axes, cats):
        style_axes(ax)
        for i, v in enumerate(["without_easter", "with_easter"]):
            s = pd.Series(r[f"{v}|{cat}"]).rename(index=int).sort_index()
            ax.plot(s.index, s.to_numpy(), color=SERIES[i], linewidth=2, marker="o", markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=labels[v])
        ax.axhline(1.0, color=BASE, linewidth=1)
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(cat, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Hari dari Minggu Easter", color=INK_2, fontsize=9)
        ax.set_xticks([-7, -5, -3, -1, 0, 1, 3, 5, 7])
    axes[0].set_ylabel("Total prediksi / total aktual", color=INK_2, fontsize=9)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, loc="upper right", ncol=2,
               labelcolor=INK_2, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("Di FOODS, fitur Easter mengurangi under-forecast sebelum Easter",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--plot-only", action="store_true", help="buat grafik dari JSON yang sudah ada")
    args = ap.parse_args()
    cfg = load_config(args.config)
    reports = Path(cfg["paths"]["reports"])
    if args.plot_only:
        plot_offsets(reports / "easter_ablation.json", reports / "figures" / "easter_ablation.png")
        return
    cal = load_calendar(Path(cfg["paths"]["raw_m5"]))
    easter = event_days(cal, "Easter")
    folds = {f.name: f for f in build_folds(cfg)}
    base_root = Path(cfg["paths"]["processed"]) / "base"

    preds, infos = [], []
    with PeakRSS() as mem:
        for store in STORES:
            base = pd.read_parquet(base_root / f"store_id={store}" / "part-0.parquet")
            # fold_features menambah 4 fitur per fold; nama lengkap diambil dari satu contoh.
            all_feats = feature_names(base) + ["te_item_mean", "te_item_zero_share",
                                               "te_dept_mean", "price_rel_item"]
            variants = {"with_easter": all_feats,
                        "without_easter": [f for f in all_feats if f not in EASTER_FEATS]}
            for fold_name in FOLDS:
                for variant, feats in variants.items():
                    for seed in SEEDS:
                        p, info = fit_predict(base, folds[fold_name], feats, params(seed))
                        p = p.assign(store_id=store, fold=fold_name, variant=variant, seed=seed)
                        preds.append(p)
                        infos.append(info | {"store_id": store, "variant": variant, "seed": seed})
                        print(f"{store} {fold_name:12s} {variant:15s} seed={seed} "
                              f"best={info['best_iteration']:4d} {info['seconds']:5.1f}s")
            del base

    pred = pd.concat(preds, ignore_index=True)
    res = summarise(pred, easter)

    # Selisih with - without per (store, fold, seed); negatif = fitur Easter membantu.
    piv = res.pivot_table(index=["store_id", "fold", "seed"], columns="variant",
                          values=["rmse_window", "mae_window", "daily_total_wape_window", "rmse_horizon"])
    delta = pd.DataFrame({m: piv[(m, "with_easter")] - piv[(m, "without_easter")]
                          for m in ["rmse_window", "mae_window", "daily_total_wape_window", "rmse_horizon"]})
    # Gangguan antar seed: selisih seed 42 vs 43 dalam varian yang sama.
    seed_noise = res.pivot_table(index=["store_id", "fold", "variant"], columns="seed",
                                 values="rmse_window")
    seed_noise = (seed_noise[SEEDS[0]] - seed_noise[SEEDS[1]]).abs()

    # Profil error per offset hari (rata-rata semua toko, fold, seed).
    pred["offset"] = easter_offset(pred["d"].to_numpy(), easter)
    w = pred[pred["offset"].abs() <= WINDOW]
    by_offset = (w.groupby(["variant", "cat_id", "offset"], observed=True)[["sales", "pred"]].sum()
                 .assign(ratio=lambda x: x["pred"] / x["sales"])["ratio"].unstack("offset"))

    out = {
        "stores": STORES, "folds": FOLDS, "seeds": SEEDS, "window_days": WINDOW,
        "results": res.round(5).to_dict(orient="records"),
        "delta_with_minus_without": delta.round(5).reset_index().to_dict(orient="records"),
        "delta_mean": delta.mean().round(5).to_dict(),
        "delta_share_negative": (delta < 0).mean().round(3).to_dict(),
        "seed_noise_rmse_window_mean": round(float(seed_noise.mean()), 5),
        "pred_over_actual_by_offset": {f"{v}|{c}": row.round(3).to_dict()
                                       for (v, c), row in by_offset.iterrows()},
        "runs": infos,
        "peak_rss_gb": round(mem.peak_gb, 2),
    }
    (reports / "easter_ablation.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    plot_offsets(reports / "easter_ablation.json", reports / "figures" / "easter_ablation.png")
    print(delta.round(4).to_string())
    print("mean delta:", out["delta_mean"])
    print("share negative:", out["delta_share_negative"])
    print("seed noise (rmse_window):", out["seed_noise_rmse_window_mean"])
    print("peak RSS GB:", out["peak_rss_gb"])


if __name__ == "__main__":
    main()
