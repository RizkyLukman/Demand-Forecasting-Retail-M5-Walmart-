"""Fase 5: rekap model kuantil final per batas iterasi, dan perbandingan dua batas.

Satu run (`--runs A`): grafik iterasi terbaik per toko × fold dan tabel ringkasan.
Dua run (`--runs A B`, A = batas lebih kecil): perbandingan pinball/coverage per fold,
per model yang mentok di A, serta skor validasi early stopping vs skor horizon. Jika B
punya titik baca (`--checkpoints` di quantile_final), juga tabel per batas, aturan
keputusan (`choose_cap`), dan grafik pinball per batas.

Cara pakai:
    python -m retailfc.quantile_cap_report --runs reports/runs/<stamp>_xgb_quantile
    python -m retailfc.quantile_cap_report --runs reports/runs/<A> reports/runs/<B>
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

from retailfc.eda import BASE, GRID, INK, INK_2, MUTED, SERIES, SURFACE, style_axes


def load_run(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return {"dir": run_dir, "name": cfg["name"], "max_rounds": cfg["options"]["max_rounds"],
            "early_stopping": cfg["options"]["early_stopping"],
            "infos": pd.DataFrame(json.loads((run_dir / "infos.json").read_text(encoding="utf-8"))),
            "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))}


def plot_best_iteration(run: dict, out: Path) -> None:
    """Small multiples per fold: iterasi terbaik tiap toko; titik oranye = berhenti di batas."""
    inf = run["infos"]
    folds = sorted(inf["fold"].unique())
    stores = sorted(inf["store_id"].unique())[::-1]
    limit = run["max_rounds"] - run["early_stopping"]
    fig, axes = plt.subplots(1, len(folds), figsize=(7.6, 3.9), dpi=150, sharey=True, facecolor=SURFACE)
    for ax, fname in zip(axes, folds):
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        g = inf[inf["fold"] == fname].set_index("store_id").loc[stores]
        y = np.arange(len(stores))
        cap = g["hit_cap"].to_numpy()
        ax.axvspan(limit, run["max_rounds"], color=BASE, alpha=0.35, linewidth=0, zorder=0)
        ax.scatter(g["best_iteration"][~cap], y[~cap], s=40, color=SERIES[0], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label="berhenti sendiri (early stopping)")
        ax.scatter(g["best_iteration"][cap], y[cap], s=40, color=SERIES[1], edgecolor=SURFACE,
                   linewidth=1.5, zorder=3, label=f"berhenti di batas {run['max_rounds']:,}".replace(",", "."))
        ax.set_xlim(0, run["max_rounds"] * 1.04)
        ax.set_title(f"{fname} ({int(cap.sum())}/{len(cap)} di batas)", fontsize=9, color=INK_2)
        ax.set_yticks(y, stores)
        ax.tick_params(axis="x", labelsize=7.5)
    axes[0].set_xlabel("iterasi terbaik", color=INK_2, fontsize=8.5)
    n_cap = int(inf["hit_cap"].sum())
    by_fold = inf.groupby("fold")["hit_cap"].sum()
    worst = by_fold.idxmax()
    fig.suptitle(f"{n_cap} dari {len(inf)} model berhenti di batas {run['max_rounds']:,} iterasi; "
                 f"terbanyak di {worst} ({int(by_fold.max())} dari {int((inf['fold'] == worst).sum())})"
                 .replace(",", "."), x=0.01, ha="left", fontsize=10.5, color=INK)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, ncol=2, loc="upper left",
               bbox_to_anchor=(0.01, 0.93), labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def compare(a: dict, b: dict) -> dict:
    """Bandingkan run batas kecil (a) dengan batas besar (b)."""
    ma, mb = a["metrics"], b["metrics"]
    per_fold = {}
    for f in ma["per_fold"]:
        pa, pb = ma["per_fold"][f], mb["per_fold"][f]
        per_fold[f] = {"pinball_a": pa["pinball_mean"], "pinball_b": pb["pinball_mean"],
                       "change_pct": round(100 * (pb["pinball_mean"] / pa["pinball_mean"] - 1), 3),
                       "coverage_a": pa["coverage"], "coverage_b": pb["coverage"]}
    sa, sb = ma["cv_summary"], mb["cv_summary"]
    key = ["store_id", "fold"]
    # Akhiran _a/_b dipasang eksplisit: kolom yang hanya ada di satu run (mis. kurva
    # validasi, titik baca) tidak akan diberi akhiran oleh merge biasa.
    ia = a["infos"].set_index(key).add_suffix("_a")
    ib = b["infos"].set_index(key).add_suffix("_b")
    m = ia.join(ib, how="inner").reset_index()
    capped = m[m["hit_cap_a"]].copy()
    capped["pinball_change_pct"] = 100 * (capped["pinball_mean_b"] / capped["pinball_mean_a"] - 1)
    rows = []
    for _, r in capped.iterrows():
        cps = r.get("checkpoints_b")
        if isinstance(cps, dict) and str(a["max_rounds"]) in cps:
            # Skor validasi terbaik di antara iterasi 1..batas A (= yang dipilih run A).
            es_at_a = cps[str(a["max_rounds"])]["es_score"]
        else:
            es_at_a = {int(k): v for k, v in r["es_curve_every_100_b"].items()}.get(a["max_rounds"])
        rows.append({"store_id": r["store_id"], "fold": r["fold"],
                     "best_iteration_b": int(r["best_iteration_b"]), "still_capped_b": bool(r["hit_cap_b"]),
                     "pinball_a": r["pinball_mean_a"], "pinball_b": r["pinball_mean_b"],
                     "pinball_change_pct": round(r["pinball_change_pct"], 3),
                     "es_score_at_a_limit": es_at_a, "es_best_score_b": r["es_best_score_b"],
                     "es_change_pct": (round(100 * (r["es_best_score_b"] / es_at_a - 1), 3)
                                       if es_at_a else None),
                     "seconds_a": r["seconds_a"], "seconds_b": r["seconds_b"]})
    return {"run_a": a["dir"].name, "run_b": b["dir"].name, "per_fold": per_fold,
            "cv_pinball_a": sa["pinball_mean"], "cv_pinball_b": sb["pinball_mean"],
            "cv_change_pct": round(100 * (sb["pinball_mean"] / sa["pinball_mean"] - 1), 3),
            "coverage_a": sa["coverage_mean"], "coverage_b": sb["coverage_mean"],
            "hit_cap_a": sa["hit_cap"], "hit_cap_b": sb["hit_cap"],
            "capped_models": rows}


COVERAGE_TARGETS = ("0.8", "0.9")


def cap_table(a: dict, b: dict) -> list[dict]:
    """Ringkasan per batas iterasi: batas run A, lalu tiap titik baca run B."""
    rows = [{"cap": a["max_rounds"], "source": a["dir"].name, **a["metrics"]["cv_summary"],
             "per_fold_pinball": {f: x["pinball_mean"] for f, x in a["metrics"]["per_fold"].items()}}]
    for cap, v in sorted(b["metrics"].get("checkpoints", {}).items(), key=lambda kv: int(kv[0])):
        if int(cap) != a["max_rounds"]:
            rows.append({"cap": int(cap), "source": f"{b['dir'].name} (titik {cap})", **v["cv_summary"],
                         "per_fold_pinball": {f: x["pinball_mean"] for f, x in v["per_fold"].items()}})
        else:   # titik baca = batas A: harus identik dengan run A (bukti cara meniru batas benar)
            rows[0]["checkpoint_matches_run_a"] = all(
                v["per_fold"][f]["pinball_mean"] == a["metrics"]["per_fold"][f]["pinball_mean"]
                for f in a["metrics"]["per_fold"])
    base = rows[0]["pinball_mean"]
    for r in rows:
        r["change_vs_a_pct"] = round(100 * (r["pinball_mean"] / base - 1), 3)
    return rows


def choose_cap(rows: list[dict], tol_pct: float = 1.0, cov_tol: float = 0.01) -> dict:
    """Aturan keputusan (reports/quantile_cap1000.md §6): batas terkecil yang pinball rata-rata
    4 fold-nya dalam `tol_pct`% dari pinball terbaik di semua batas, dan coverage P80/P90-nya
    tidak menjauh dari target lebih dari `cov_tol` dibanding batas acuan (baris pertama)."""
    best = min(r["pinball_mean"] for r in rows)
    ref_dev = {q: abs(rows[0]["coverage_mean"][q] - float(q)) for q in COVERAGE_TARGETS}
    for r in sorted(rows, key=lambda r: r["cap"]):
        within = r["pinball_mean"] <= best * (1 + tol_pct / 100)
        cov_ok = all(abs(r["coverage_mean"][q] - float(q)) <= ref_dev[q] + cov_tol for q in COVERAGE_TARGETS)
        if within and cov_ok:
            return {"chosen_cap": r["cap"], "best_pinball": best, "chosen_pinball": r["pinball_mean"],
                    "rule": f"smallest cap within {tol_pct}% of best pinball, coverage P80/P90 not "
                            f"worse than reference by more than {cov_tol}"}
    return {"chosen_cap": None, "best_pinball": best}


def plot_cap_curve(rows: list[dict], out: Path) -> None:
    """Pinball per batas iterasi: rata-rata 4 fold (tebal) dan tiap fold (tipis), relatif ke batas acuan."""
    caps = [r["cap"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    folds = sorted(rows[0]["per_fold_pinball"])
    ends = {}
    allv = [0.0]
    for f in folds:
        v = [100 * (r["per_fold_pinball"][f] / rows[0]["per_fold_pinball"][f] - 1) for r in rows]
        ax.plot(caps, v, color=MUTED, linewidth=1, marker="o", markersize=3)
        ends[f] = v[-1]
        allv += v
    mean = [r["change_vs_a_pct"] for r in rows]
    allv += mean
    lo, hi = min(allv), max(allv)
    pad = max(0.05, 0.15 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    # Label fold di ujung kanan. Garis yang ujungnya berdekatan diberi satu label gabungan,
    # supaya label tidak bergeser ke garis lain.
    gap = 0.08 * (hi - lo + 2 * pad)
    groups: list[list[tuple[str, float]]] = []
    for f, v in sorted(ends.items(), key=lambda kv: kv[1]):
        if groups and v - groups[-1][-1][1] < gap:
            groups[-1].append((f, v))
        else:
            groups.append([(f, v)])
    x_lab = caps[-1] + 0.02 * (caps[-1] - caps[0])
    for g in groups:
        ax.text(x_lab, float(np.mean([v for _, v in g])), ", ".join(f for f, _ in sorted(g)),
                fontsize=7.5, color=INK_2, va="center", ha="left")
    ax.plot(caps, mean, color=SERIES[0], linewidth=2, marker="o", markersize=6, label="rata-rata 4 fold")
    ax.axhline(0, color=BASE, linewidth=1)
    ax.set_xlim(caps[0] - 0.04 * (caps[-1] - caps[0]), caps[-1] + 0.16 * (caps[-1] - caps[0]))
    ax.set_xticks(caps, [f"{c:,}".replace(",", ".") for c in caps])
    ax.set_xlabel("batas iterasi", color=INK_2, fontsize=9)
    ax.set_ylabel(f"pinball vs batas {caps[0]:,}".replace(",", ".") + " (%); positif = lebih buruk",
                  color=INK_2, fontsize=8.5)
    worst = f"{max(mean):+.2f}".replace(".", ",")
    fig.suptitle(f"Batas lebih tinggi tidak memperbaiki pinball (maks {worst}%; ambang 1%)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.01, 0.93), labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_capped_change(cmp: dict, a: dict, b: dict, out: Path) -> None:
    """Perubahan skor validasi (early stopping) vs skor horizon untuk model yang mentok di A."""
    df = pd.DataFrame(cmp["capped_models"]).sort_values("pinball_change_pct")
    labels = [f"{s} {f}" for s, f in zip(df["store_id"], df["fold"])]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(df) + 1.6), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.scatter(df["es_change_pct"], y + 0.15, s=34, color=SERIES[2], edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label="skor validasi (28 hari sebelum cutoff)")
    ax.scatter(df["pinball_change_pct"], y - 0.15, s=34, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label="pinball horizon (28 hari yang diramal)")
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel(f"perubahan dari batas {a['max_rounds']:,} ke {b['max_rounds']:,} iterasi (%), negatif = lebih baik"
                  .replace(",", "."), color=INK_2, fontsize=8.5)
    fig.suptitle("Model yang mentok: perubahan skor setelah batas dinaikkan", x=0.01, ha="left",
                 fontsize=10.5, color=INK)
    fig.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0.01, 0.955),
               labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    rep = Path(args.reports)
    runs = [load_run(Path(r)) for r in args.runs]
    a = runs[0]
    plot_best_iteration(a, rep / "figures" / f"quantile_best_iter_r{a['max_rounds']}.png")
    summary = {"run": a["dir"].name, "cv_summary": a["metrics"]["cv_summary"],
               "hit_cap_by_fold": a["infos"].groupby("fold")["hit_cap"].sum().astype(int).to_dict(),
               "hit_cap_by_store": a["infos"].groupby("store_id")["hit_cap"].sum().astype(int).to_dict()}
    out = {"summary_a": summary}
    if len(runs) == 2:
        b = runs[1]
        plot_best_iteration(b, rep / "figures" / f"quantile_best_iter_r{b['max_rounds']}.png")
        out["comparison"] = compare(a, b)
        plot_capped_change(out["comparison"], a, b, rep / "figures" /
                           f"quantile_cap_r{a['max_rounds']}_vs_r{b['max_rounds']}.png")
        if b["metrics"].get("checkpoints"):
            rows = cap_table(a, b)
            out["cap_table"] = rows
            out["decision"] = choose_cap(rows)
            out["capped_models_still_capped_by_cap"] = {
                cap: int(sum(i["checkpoints"][cap]["hit_cap"] for i in b["infos"].to_dict("records")))
                for cap in b["metrics"]["checkpoints"]}
            plot_cap_curve(rows, rep / "figures" / f"quantile_cap_curve_r{b['max_rounds']}.png")
    path = rep / ("quantile_cap_comparison.json" if len(runs) == 2 else f"quantile_cap_r{a['max_rounds']}.json")
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
