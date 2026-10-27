"""Fase 8: grafik laporan dari reports/data/id_event_study_daily.csv dan reports/data/id_forecast.json.

Cara pakai:
    python -m retailfc.id_report --config config.yaml
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

from retailfc.eda import GRID, INK, INK_2, MUTED, SERIES, SURFACE, style_axes
from retailfc.id_event_study import load_daily
from retailfc.io import load_config

METHOD_LABELS = {"snaive7": "naive 7 hari", "weekday_mean4": "rata-rata hari sama 4 mg", "ma28": "MA 28",
                 "ets": "ETS", "croston": "Croston", "tsb": "TSB", "xgb": "XGBoost",
                 "xgb_noholiday": "XGBoost tanpa fitur libur"}


def fig_event_daily(cfg: dict, rep: Path) -> None:
    d = pd.read_csv(rep / "data" / "id_event_study_daily.csv", parse_dates=["date"]).set_index("date")
    full = load_daily(cfg)
    p = cfg["id_pharmacy"]["event_study"]
    ev0, ev1 = (pd.Timestamp(x) for x in p["event_period"])
    view = full.loc[ev0 - pd.offsets.Day(28): ev1 + pd.offsets.Day(28)]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.2), dpi=150, sharex=True, facecolor=SURFACE)
    for ax, s, name in zip(axes, ["notes_RJ", "notes_RI"], ["nota RJ", "nota RI"]):
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        ax.axvspan(ev0.to_pydatetime(), ev1.to_pydatetime(), color=GRID, alpha=0.5, linewidth=0)
        ax.plot(view.index, view[s], color=MUTED, linewidth=0.9, label="aktual")
        ax.plot(d.index, d[f"{s}_expected"], color=SERIES[0], linewidth=1.4, label="nilai harapan")
        for key in ("lebaran_cuti",):
            a, b = (pd.Timestamp(x) for x in p["windows"][key])
            ax.axvspan(a.to_pydatetime(), b.to_pydatetime(), color=SERIES[1], alpha=0.15, linewidth=0)
        ax.set_ylabel(name, color=INK_2, fontsize=8.5)
    axes[0].annotate("Lebaran + cuti bersama", (pd.Timestamp(p["windows"]["lebaran_cuti"][0]).to_pydatetime(),
                     axes[0].get_ylim()[1]), xytext=(2, -10), textcoords="offset points", fontsize=7.5,
                     color=SERIES[1])
    h, n = axes[0].get_legend_handles_labels()
    fig.legend(h, n, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.01, 0.95),
               labelcolor=INK_2, ncol=2)
    fig.suptitle("Pekan Lebaran normal; penurunan terjadi 1–2 minggu sesudahnya (area abu-abu = periode event)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(rep / "figures" / "phase8_event_daily.png", facecolor=SURFACE)
    plt.close(fig)


def fig_forecast(rep: Path) -> None:
    r = json.loads((rep / "data" / "id_forecast.json").read_text(encoding="utf-8"))
    agg, prod = r["cv_summary"]["agg"], r["cv_summary"]["products"]
    panels = [("nota RJ", {m: agg[m]["notes_RJ"]["rmsse_mean"] for m in agg}),
              ("nota RI", {m: agg[m]["notes_RI"]["rmsse_mean"] for m in agg}),
              ("unit total", {m: agg[m]["units_all"]["rmsse_mean"] for m in agg}),
              ("produk ±200 (berbobot)", {m: prod[m]["wrmsse_mean"] for m in prod})]
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.6), dpi=150, facecolor=SURFACE)
    for ax, (title, vals) in zip(axes, panels):
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        items = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
        y = np.arange(len(items))
        cols = [SERIES[0] if m.startswith("xgb") else MUTED for m, _ in items]
        ax.barh(y, [v for _, v in items], color=cols, height=0.6)
        ax.set_yticks(y, [METHOD_LABELS[m] for m, _ in items], fontsize=7)
        for yy, (_, v) in zip(y, items):
            ax.annotate(f"{v:.2f}", (v, yy), xytext=(3, -3), textcoords="offset points", fontsize=6.5,
                        color=INK_2)
        ax.set_title(title, fontsize=8.5, color=INK_2, loc="left")
        ax.set_xlim(0, max(vals.values()) * 1.25)
    fig.supxlabel("RMSSE rata-rata 8 fold CV (lebih kecil = lebih baik)", fontsize=8, color=INK_2)
    fig.suptitle("Data apotek kecil: XGBoost tidak mengalahkan baseline sederhana (horizon 14 hari)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(rep / "figures" / "phase8_forecast_cv.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep = Path(cfg["paths"]["reports"])
    fig_event_daily(cfg, rep)
    fig_forecast(rep)
    print("figures saved")


if __name__ == "__main__":
    main()
