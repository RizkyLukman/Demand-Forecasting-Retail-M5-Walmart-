"""Fase 6: grafik SHAP dari reports/data/shap_summary.json.

Cara pakai:
    python -m retailfc.shap_report --config config.yaml
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
from retailfc.io import load_config

SEGS = ["smooth", "erratic", "intermittent", "lumpy"]


def hbar(ax, names, values, color, xlabel) -> None:
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, names, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=8)


def plot_global(s: dict, out: Path, top: int = 15) -> None:
    items = list(s["global_mean_abs"].items())[:top]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    hbar(ax, [k for k, _ in items], [v for _, v in items], SERIES[0], "rata-rata |SHAP| (kontribusi ke log forecast)")
    top3 = sum(v for _, v in items[:3]) / sum(s["global_mean_abs"].values())
    fig.suptitle(f"Fitur terpenting (cv4, 10 toko): 3 teratas = {100 * top3:.0f}% total |SHAP|",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_segments(s: dict, out: Path, top: int = 8) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 3.6), dpi=150, sharex=True, facecolor=SURFACE)
    for ax, seg in zip(axes, SEGS):
        style_axes(ax)
        d = s["segment_mean_abs"][seg]
        tot = sum(d.values())
        items = list(d.items())[:top]
        hbar(ax, [k for k, _ in items], [100 * v / tot for _, v in items], SERIES[0], "% total |SHAP|")
        ax.set_title(seg, fontsize=9, color=INK_2)
    fig.suptitle("Fitur terpenting per segmen: porsi |SHAP| (cv4)", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_cases(s: dict, err: dict, out: Path) -> None:
    """Per studi kasus: rata-rata jangka panjang (te_item_mean) vs terbaru vs forecast vs aktual."""
    cls = {c["id"].replace("_evaluation", "") + "_" + c["fold"]: c["classical_sum"] / 28 for c in err["cases"]}
    cases = s["cases"]
    fig, axes = plt.subplots(len(cases), 1, figsize=(7.2, 1.55 * len(cases) + 1.0), dpi=150, facecolor=SURFACE)
    for ax, (name, c) in zip(np.atleast_1d(axes), cases.items()):
        style_axes(ax)
        fm = c["feature_mean"]
        rows = [("rata-rata seluruh riwayat (te_item_mean)", fm["te_item_mean"], MUTED),
                ("rata-rata 56 hari, berakhir 28 hari sebelumnya (rmean_56)", fm["rmean_56"], MUTED),
                ("forecast XGBoost", c["pred_mean"], SERIES[0]),
                ("forecast classical", cls.get(name, np.nan), SERIES[1]),
                ("aktual", c["actual_mean"], INK)]
        y = np.arange(len(rows))[::-1]
        for yy, (lab, v, col) in zip(y, rows):
            ax.plot([0, v], [yy, yy], color=GRID, linewidth=1, zorder=1)
            ax.scatter([v], [yy], s=46, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(f"{v:.1f}".replace(".", ","), (v, yy), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK_2)
        ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_xlim(0, max(r[1] for r in rows if np.isfinite(r[1])) * 1.18)
        parts = name.split("_")                      # contoh: HOBBIES_1_354_TX_2_cv1
        item, store, fold = "_".join(parts[:-3]), "_".join(parts[-3:-1]), parts[-1]
        ax.set_title(f"{item} · {store} · {fold} (unit per hari)", fontsize=8.5, color=INK_2, loc="left")
    fig.suptitle("Item yang tumbuh: forecast tertahan oleh rata-rata jangka panjang", x=0.01, ha="left",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    rep = Path(load_config(args.config)["paths"]["reports"])
    s = json.loads((rep / "data" / "shap_summary.json").read_text(encoding="utf-8"))
    plot_global(s, rep / "figures" / "phase6_shap_global.png")
    plot_segments(s, rep / "figures" / "phase6_shap_segments.png")
    err = json.loads((rep / "data" / "error_analysis.json").read_text(encoding="utf-8"))
    plot_cases(s, err, rep / "figures" / "phase6_shap_cases.png")
    print("figures saved")


if __name__ == "__main__":
    main()
