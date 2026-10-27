"""Fase 7: grafik dan tabel ringkasan dari reports/data/inventory_sim.json.

Cara pakai:
    python -m retailfc.inventory_report --config config.yaml
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

# Tiga metode utama berwarna; dua lainnya abu-abu dengan penanda berbeda.
STYLE = {
    "xgb_normal": ("XGBoost point + safety stock", SERIES[0], "o", "-"),
    "empirical": ("empiris (riwayat 1 tahun)", SERIES[1], "o", "-"),
    "xgb_quantile": ("XGBoost kuantil (jumlah harian)", SERIES[2], "o", "-"),
    "classical_normal": ("classical + safety stock", MUTED, "s", "--"),
    "xgb_empirical": ("gabungan XGBoost + empiris", MUTED, "^", ":"),
}


def curve(summary: dict, method: str, group: str) -> tuple[list[float], list[float], list[str]]:
    pts = sorted((v["groups"][group]["days_of_cover"], 100 * v["groups"][group]["fill_rate"], k.split("@")[1])
                 for k, v in summary.items() if k.startswith(method + "@") and v["groups"][group]["days_of_cover"])
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def plot_frontier(r: dict, ax, group: str, methods: list[str], title: str, label_alpha: bool = False) -> None:
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for m in methods:
        name, col, mk, ls = STYLE[m]
        x, y, a = curve(r["summary"], m, group)
        ax.plot(x, y, color=col, linewidth=2 if col != MUTED else 1.4, linestyle=ls, marker=mk, markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=name)
        if label_alpha and m in ("xgb_quantile",):
            for xx, yy, aa in zip(x, y, a):
                ax.annotate(f"P{int(float(aa) * 100)}", (xx, yy), xytext=(4, -10), textcoords="offset points",
                            fontsize=7, color=INK_2)
    ax.set_xlabel("stok rata-rata (hari permintaan)", color=INK_2, fontsize=8.5)
    ax.set_ylabel("fill rate (%)", color=INK_2, fontsize=8.5)
    ax.set_title(title, fontsize=9, color=INK_2, loc="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    rep = Path(load_config(args.config)["paths"]["reports"])
    r = json.loads((rep / "data" / "inventory_sim.json").read_text(encoding="utf-8"))
    fig_dir = rep / "figures"

    # 1. Kurva trade-off, semua deret.
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=150, facecolor=SURFACE)
    plot_frontier(r, ax, "all", list(STYLE), "semua deret, 4 fold CV", label_alpha=True)
    ax.set_ylim(60, 100)
    fig.suptitle("XGBoost paling efisien di layanan menengah; di atas ±97%, empiris paling efisien",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    h, n = ax.get_legend_handles_labels()
    fig.legend(h, n, frameon=False, fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.94),
               labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(fig_dir / "phase7_frontier.png", facecolor=SURFACE)
    plt.close(fig)

    # 2. Item tumbuh cepat vs lainnya.
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), dpi=150, sharey=True, facecolor=SURFACE)
    ms = ["xgb_normal", "empirical", "classical_normal"]
    plot_frontier(r, axes[0], "fast_growing", ms, "item tumbuh cepat (rasio > 2)")
    plot_frontier(r, axes[1], "not_fast_growing", ms, "item lainnya")
    for ax in axes:
        ax.set_ylim(70, 100)
    fig.suptitle("Item tumbuh cepat: classical setara atau lebih baik dari XGBoost", x=0.01, ha="left",
                 fontsize=10.5, color=INK)
    h, n = axes[0].get_legend_handles_labels()
    fig.legend(h, n, frameon=False, fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.93),
               labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(fig_dir / "phase7_fast_growing.png", facecolor=SURFACE)
    plt.close(fig)

    # 3. Sensitivitas biaya: biaya tiap metode (α terbaiknya) relatif ke metode termurah.
    sens = r["sensitivity"]
    labels = [f"simpan {int(s['holding_annual'] * 100)}%/thn · hilang {int(s['stockout_rate'] * 100)}%"
              for s in sens]
    y = np.arange(len(sens))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for m, (name, col, mk, _) in STYLE.items():
        rel = []
        for s in sens:
            best = min(v["cost"] for v in s["best_by_method"].values())
            rel.append(100 * (s["best_by_method"][m]["cost"] / best - 1))
        ax.scatter(rel, y, s=38, color=col, marker=mk, edgecolor=SURFACE, linewidth=1, zorder=3, label=name)
    for yy in y:
        ax.axhline(yy, color=GRID, linewidth=0.6, zorder=1)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel("biaya total di atas metode termurah (%), 0 = termurah", color=INK_2, fontsize=8.5)
    n_emp = sum(s["best"].startswith("empirical@") for s in sens)
    fig.suptitle(f"Sensitivitas biaya: empiris termurah di {n_emp} dari {len(sens)} asumsi", x=0.01,
                 ha="left", fontsize=10.5, color=INK)
    h, n = ax.get_legend_handles_labels()
    fig.legend(h, n, frameon=False, fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.94),
               labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(fig_dir / "phase7_cost_sensitivity.png", facecolor=SURFACE)
    plt.close(fig)
    print("figures saved")


if __name__ == "__main__":
    main()
