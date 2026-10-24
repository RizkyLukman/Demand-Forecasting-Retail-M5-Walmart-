"""Fase 6: grafik dari reports/error_analysis.json (dan prediksi tersimpan untuk studi kasus).

Cara pakai:
    python -m retailfc.error_report --config config.yaml
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

from retailfc.eda import GRID, INK, INK_2, MUTED, SERIES, SURFACE, style_axes
from retailfc.folds import build_folds
from retailfc.io import load_calendar, load_config, load_wide_matrices

CV = ["cv1", "cv2", "cv3", "cv4"]
SHOW = [("xgb_point", "XGBoost"), ("classical", "classical (ETS+TSB)"), ("croston", "Croston")]
SEG_ID = {"smooth": "smooth", "erratic": "erratic", "intermittent": "intermittent", "lumpy": "lumpy"}
DAY_ID = {"Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu", "Thursday": "Kamis",
          "Friday": "Jumat", "Saturday": "Sabtu", "Sunday": "Minggu"}


def cv_mean(r: dict, path: list[str], model: str, metric: str) -> float:
    vals = []
    for f in CV:
        g = r["folds"][f]
        for p in path:
            g = g[p]
        vals.append(g[model][metric])
    return float(np.mean(vals))


def dot_panel(ax, labels, values: dict, xlabel: str) -> None:
    y = np.arange(len(labels))[::-1]
    for i, (m, name) in enumerate(SHOW):
        ax.scatter(values[m], y + 0.18 * (1 - i), s=40, color=SERIES[i], edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=name)
    for yy in y:
        ax.axhline(yy, color=GRID, linewidth=0.6, zorder=1)
    ax.set_yticks(y, labels)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=8.5)


def legend_top(fig, ax) -> None:
    h, n = ax.get_legend_handles_labels()
    fig.legend(h, n, frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.93),
               labelcolor=INK_2)


def plot_segments(r: dict, out: Path) -> None:
    segs = list(SEG_ID)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), dpi=150, sharey=True, facecolor=SURFACE)
    for ax, metric, xl in [(axes[0], "wrmsse", "per item × toko (RMSSE tertimbang)"),
                           (axes[1], "agg_rmsse", "total harian segmen (RMSSE)")]:
        style_axes(ax)
        vals = {m: [cv_mean(r, ["groups", "segment", s], m, metric) for s in segs] for m, _ in SHOW}
        dot_panel(ax, [SEG_ID[s] for s in segs], vals, xl)
    fig.suptitle("Per item, XGBoost hanya tipis lebih baik; saat dijumlahkan, unggul jauh kecuali erratic",
                 x=0.01, ha="left", fontsize=10, color=INK)
    legend_top(fig, axes[0])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_erratic_bias(r: dict, out: Path) -> None:
    segs = list(SEG_ID)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(CV))
    for i, (m, name) in enumerate(SHOW[:2]):
        v = [r["folds"][f]["groups"]["segment"]["erratic"][m]["bias"] for f in CV]
        ax.plot(x, v, color=SERIES[i], linewidth=2, marker="o", markersize=6, label=name)
    others = [[r["folds"][f]["groups"]["segment"][s]["xgb_point"]["bias"] for f in CV] for s in segs if s != "erratic"]
    lo, hi = np.min(others), np.max(others)
    ax.axhspan(lo, hi, color=MUTED, alpha=0.18, linewidth=0, zorder=0)
    ax.text(x[-1] + 0.08, (lo + hi) / 2, "XGBoost,\nsegmen lain", fontsize=7.5, color=INK_2, va="center")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(x, CV)
    ax.set_xlim(-0.2, len(CV) - 0.4)
    ax.set_ylabel("bias (unit/hari per deret), + = terlalu tinggi", color=INK_2, fontsize=8.5)
    fig.suptitle("Segmen erratic: XGBoost makin terlalu tinggi dari cv1 ke cv4", x=0.01, ha="left",
                 fontsize=10.5, color=INK)
    legend_top(fig, ax)
    fig.tight_layout(rect=(0, 0, 0.97, 0.88))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_weekday(r: dict, out: Path) -> None:
    days = list(DAY_ID)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), dpi=150, sharey=True, facecolor=SURFACE)
    for ax, metric, xl in [(axes[0], "wrmsse", "per item × toko (RMSSE tertimbang)"),
                           (axes[1], "store_wape", "per toko (WAPE)")]:
        style_axes(ax)
        vals = {m: [cv_mean(r, ["by_day", "weekday", d], m, metric) for d in days] for m, _ in SHOW}
        dot_panel(ax, [DAY_ID[d] for d in days], vals, xl)
    fig.suptitle("Akhir pekan: per item XGBoost setara classical, per toko tetap unggul jauh",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    legend_top(fig, axes[0])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_cases(r: dict, cfg: dict, out: Path) -> None:
    folds = {f.name: f for f in build_folds(cfg)}
    ids, sales, _ = load_wide_matrices(cfg, cfg["validation"]["holdout_cutoff"])
    dates = load_calendar(Path(cfg["paths"]["raw_m5"])).set_index("d_num")["date"]
    root = Path(cfg["paths"]["processed"]) / "preds"
    cases = r["cases"]
    fig, axes = plt.subplots(len(cases), 1, figsize=(7.4, 1.75 * len(cases) + 0.9), dpi=150, facecolor=SURFACE)
    for ax, c in zip(axes, cases):
        style_axes(ax)
        f = folds[c["fold"]]
        i = c["row"]
        hd = np.arange(f.cutoff - 55, f.cutoff + 1)
        fd = np.arange(f.cutoff + 1, f.cutoff + f.horizon + 1)
        ax.plot(dates.loc[hd].to_numpy(), sales[i, hd - 1], color=MUTED, linewidth=1.2)
        ax.plot(dates.loc[fd].to_numpy(), sales[i, fd - 1], color=INK, linewidth=1.6,
                label="aktual (riwayat abu-abu)")
        for j, (m, name) in enumerate(SHOW[:2]):
            ax.plot(dates.loc[fd].to_numpy(), np.load(root / f.name / f"{m}.npy")[i], color=SERIES[j],
                    linewidth=1.6, label=name)
        ax.axvline(dates.loc[f.cutoff + 1], color=MUTED, linewidth=0.8, linestyle=":")
        name = c["id"].replace("_evaluation", "")
        rm = f"RMSSE XGB {c['rmsse_xgb']:.2f} / classical {c['rmsse_classical']:.2f}".replace(".", ",")
        ax.set_title(f"{name} · {c['fold']} · {c['segment']} · {rm}", fontsize=8.5, color=INK_2, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(labelsize=7.5)
    fig.suptitle("5 deret dengan kontribusi error terbesar (level item × toko)", x=0.01, ha="left",
                 fontsize=10.5, color=INK)
    h, n = axes[0].get_legend_handles_labels()
    fig.legend(h, n, frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.975),
               labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep = Path(cfg["paths"]["reports"])
    r = json.loads((rep / "error_analysis.json").read_text(encoding="utf-8"))
    fig_dir = rep / "figures"
    plot_segments(r, fig_dir / "phase6_segments.png")
    plot_erratic_bias(r, fig_dir / "phase6_erratic_bias.png")
    plot_weekday(r, fig_dir / "phase6_weekday.png")
    plot_cases(r, cfg, fig_dir / "phase6_cases.png")
    print("figures saved")


if __name__ == "__main__":
    main()
