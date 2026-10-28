"""Grafik evaluasi holdout: WRMSSE holdout vs rentang CV, dan kurva trade-off stok holdout vs CV.

Hanya membaca JSON yang sudah ada (reports/data/holdout_forecast.json, model_comparison.json,
inventory_sim_holdout.json, inventory_sim.json); tidak melatih apa pun.

Cara pakai:
    python -m retailfc.holdout_report --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from retailfc.eda import GRID, INK, INK_2, MUTED, SERIES, SURFACE, style_axes
from retailfc.inventory_report import STYLE, curve
from retailfc.io import load_config

MODELS = [("xgb_point", "XGBoost"), ("classical", "classical (ETS/TSB)"), ("snaive", "seasonal naive"),
          ("croston", "Croston"), ("tsb", "TSB"), ("ma28", "MA 28 hari")]


def fig_wrmsse(rep: Path) -> None:
    ho = json.loads((rep / "data" / "holdout_forecast.json").read_text(encoding="utf-8"))["forecast"]
    cv = json.loads((rep / "data" / "model_comparison.json").read_text(encoding="utf-8"))["cv_summary"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(MODELS))[::-1]
    for yy, (m, _) in zip(y, MODELS):
        pf = list(cv[m]["per_fold"].values())
        ax.plot([min(pf), max(pf)], [yy, yy], color=GRID, linewidth=6, solid_capstyle="round", zorder=1)
        ax.scatter(pf, [yy] * len(pf), s=14, color=MUTED, zorder=2)
        col = SERIES[0] if m == "xgb_point" else INK_2
        ax.scatter([ho[m]["wrmsse"]], [yy], s=60, color=col, marker="D", edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.annotate(f"{ho[m]['wrmsse']:.3f}".replace(".", ","), (ho[m]["wrmsse"], yy), xytext=(0, 8),
                    textcoords="offset points",
                    ha="center", fontsize=7.5, color=col)
    ax.set_yticks(y, [n for _, n in MODELS], fontsize=8)
    ax.set_xlabel("WRMSSE (lebih kecil lebih baik)", color=INK_2, fontsize=8.5)
    ax.scatter([], [], s=14, color=MUTED, label="4 fold CV")
    ax.scatter([], [], s=60, color=INK_2, marker="D", label="holdout (25 Apr–22 Mei 2016)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK_2)
    fig.suptitle("Holdout: XGBoost 0,563, di dalam rentang CV dan tetap jauh di depan baseline",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(rep / "figures" / "phase9_holdout_wrmsse.png", facecolor=SURFACE)
    plt.close(fig)


def fig_frontier(rep: Path) -> None:
    ho = json.loads((rep / "data" / "inventory_sim_holdout.json").read_text(encoding="utf-8"))
    cv = json.loads((rep / "data" / "inventory_sim.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=150, sharey=True, facecolor=SURFACE)
    for ax, (r, title) in zip(axes, [(cv, "4 fold CV"), (ho, "holdout")]):
        style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for m in ("xgb_normal", "empirical", "xgb_quantile", "classical_normal"):
            name, col, mk, ls = STYLE[m]
            x, yv, _ = curve(r["summary"], m, "all")
            ax.plot(x, yv, color=col, linewidth=2 if col != MUTED else 1.4, linestyle=ls, marker=mk, markersize=4,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=name)
        ax.set_xlim(3, 16)
        ax.set_ylim(70, 100)
        ax.set_title(title, fontsize=9, color=INK_2, loc="left")
        ax.set_xlabel("stok rata-rata (hari permintaan)", color=INK_2, fontsize=8.5)
    axes[0].set_ylabel("fill rate (%)", color=INK_2, fontsize=8.5)
    h, n = axes[0].get_legend_handles_labels()
    fig.legend(h, n, frameon=False, fontsize=7.5, ncol=4, loc="upper left", bbox_to_anchor=(0.01, 0.93),
               labelcolor=INK_2)
    fig.suptitle("Stok di holdout: urutan metode sama dengan CV, keunggulan XGBoost menyempit",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(rep / "figures" / "phase9_holdout_frontier.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    rep = Path(load_config(args.config)["paths"]["reports"])
    fig_wrmsse(rep)
    fig_frontier(rep)
    print("figures saved")


if __name__ == "__main__":
    main()
