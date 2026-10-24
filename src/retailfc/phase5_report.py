"""Fase 5: grafik dan tabel ringkasan dari reports/model_comparison.json.

Cara pakai:
    python -m retailfc.phase5_report --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retailfc.eda import BASE, INK, INK_2, SERIES, SURFACE, style_axes
from retailfc.io import load_config
from retailfc.metrics import LEVELS

MODELS = [("xgb_point", "XGBoost"), ("classical", "classical (ETS+TSB)"), ("snaive", "seasonal naive")]
LEVEL_NAMES = {
    "Level1": "1 total", "Level2": "2 state", "Level3": "3 store", "Level4": "4 cat",
    "Level5": "5 dept", "Level6": "6 state×cat", "Level7": "7 state×dept",
    "Level8": "8 store×cat", "Level9": "9 store×dept", "Level10": "10 item",
    "Level11": "11 item×state", "Level12": "12 item×store",
}


def dot_rows(ax, labels: list[str], values: dict[str, list[float]], offset: float = 0.0) -> None:
    """Dot plot horizontal: satu baris per label, satu titik per model.

    `offset` menggeser tiap model sedikit ke atas/bawah supaya titik yang berimpit tetap terlihat.
    """
    y = np.arange(len(labels))[::-1]
    for i, (key, name) in enumerate(MODELS):
        ax.scatter(values[key], y + offset * (1 - i), s=46, color=SERIES[i], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=name)
    for yy in y:
        ax.axhline(yy, color=BASE, linewidth=0.6, zorder=1)
    ax.set_yticks(y, labels)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.grid(axis="y", visible=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep = Path(cfg["paths"]["reports"])
    comp = json.loads((rep / "model_comparison.json").read_text(encoding="utf-8"))["per_fold"]
    folds = list(comp.keys())
    cv = [f for f in folds if f.startswith("cv")]

    # 1. WRMSSE per fold.
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    dot_rows(ax, folds, {k: [comp[f][k]["wrmsse"] for f in folds] for k, _ in MODELS})
    ax.set_xlabel("WRMSSE (lebih kecil lebih baik)", color=INK_2, fontsize=9)
    fig.suptitle("XGBoost unggul di semua fold; selisih paling tipis di akhir tahun",
                 x=0.01, ha="left", fontsize=11, color=INK)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, ncol=3, loc="upper left",
               bbox_to_anchor=(0.01, 0.93), labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(rep / "figures" / "phase5_wrmsse_by_fold.png", facecolor=SURFACE)
    plt.close(fig)

    # 2. WRMSSE per level (rata-rata 4 fold CV). Bobot tiap level berjumlah 1, jadi nilai
    #    per level sudah berskala WRMSSE; WRMSSE total = rata-rata 12 nilai ini.
    levels = [lv for lv, _ in LEVELS]
    per_level = {k: [np.mean([comp[f][k]["per_level"][lv] for f in cv]) for lv in levels] for k, _ in MODELS}
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    dot_rows(ax, [LEVEL_NAMES[lv] for lv in levels], per_level)
    ax.set_xlabel("WRMSSE per level, rata-rata cv1–cv4", color=INK_2, fontsize=9)
    fig.suptitle("XGBoost jauh unggul di level agregat, tetapi tipis di level item",
                 x=0.01, ha="left", fontsize=11, color=INK)
    handles, names = ax.get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, ncol=3, loc="upper left",
               bbox_to_anchor=(0.01, 0.93), labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(rep / "figures" / "phase5_wrmsse_by_level.png", facecolor=SURFACE)
    plt.close(fig)

    table = pd.DataFrame(per_level, index=levels).rename(columns=dict(MODELS))
    table["xgb_vs_classical_%"] = 100 * (table["XGBoost"] / table["classical (ETS+TSB)"] - 1)
    print(table.round(4).to_string())
    (rep / "phase5_level_table.json").write_text(table.round(5).to_json(orient="index", indent=2),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
