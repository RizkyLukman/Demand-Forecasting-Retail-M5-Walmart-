"""Fase 8: studi efek Ramadan/Lebaran pada data apotek, dengan uji plasebo.

Dua tahap, dijalankan terpisah:
1. `--calibrate`: memilih cara menghitung nilai harapan (baseline) dan panjang jendela
   HANYA dari hari di luar periode event. Periode event asli (Ramadan s.d. pasca Lebaran)
   tidak dilihat di tahap ini.
2. tanpa flag: menghitung efek di jendela asli dengan pilihan dari tahap 1, beserta
   distribusi plasebonya.

Uji plasebo: periode event palsu dengan panjang yang sama dengan periode event asli
digeser di sepanjang data non-event (Agu-Des). Model baseline dilatih tanpa periode palsu
itu, lalu dipakai menebak isinya. Error tebakan di jendela palsu (posisi relatif sama
dengan jendela asli) menjadi ukuran fluktuasi biasa.

Cara pakai:
    python -m retailfc.id_event_study --config config.yaml --calibrate
    python -m retailfc.id_event_study --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retailfc.id_calendar import DAY_OFF
from retailfc.io import load_config

METHODS = {
    "weekday_median": {},
    "weekday_loglinear": {},
    "local_level_14": {"margin": 14},
    "local_level_28": {"margin": 28},
    "local_level_42": {"margin": 42},
}


def load_daily(cfg: dict) -> pd.DataFrame:
    """Nota per hari (total, RJ, RI) dan unit per hari (total) pada periode lengkap."""
    src = Path(cfg["paths"]["interim"]) / "id_pharmacy"
    p = cfg["id_pharmacy"]
    start, end = pd.Timestamp(p["complete_start"]), pd.Timestamp(p["end"])
    ms = pd.read_parquet(src / "ms_sales.parquet", columns=["NO_RESEP", "TGL"])
    det = pd.read_parquet(src / "det_sales.parquet", columns=["NO_RESEP", "QTY"])
    ms = ms[(ms["TGL"] >= start) & (ms["TGL"] <= end)]
    pre = ms["NO_RESEP"].str.extract(r"^([A-Z]+)")[0]
    days = pd.date_range(start, end, freq="D")
    out = pd.DataFrame(index=days)
    out["notes_all"] = ms.groupby("TGL").size()
    for k in ("RJ", "RI"):
        out[f"notes_{k}"] = ms[pre == k].groupby("TGL").size()
    lines = det.merge(ms, on="NO_RESEP")
    out["units_all"] = lines.groupby("TGL")["QTY"].sum()
    return out.fillna(0.0)


def day_off_dates(cfg: dict) -> set[pd.Timestamp]:
    cal = pd.read_csv(Path(cfg["paths"]["external"]) / "id_calendar.csv", parse_dates=["date"])
    return set(cal.loc[cal["event"].isin(DAY_OFF), "date"])


def expected(y: pd.Series, fit: np.ndarray, gap: np.ndarray, method: str) -> pd.Series:
    """Nilai harapan untuk hari-hari di `gap`, dilatih hanya dari hari `fit` (bool, sejajar y)."""
    wd = y.index.weekday.to_numpy()
    yf, wf = y.to_numpy()[fit], wd[fit]
    if method == "weekday_median":
        med = pd.Series(yf).groupby(wf).median()
        return pd.Series(med.reindex(wd[gap]).to_numpy(), index=y.index[gap])
    if method == "weekday_loglinear":
        t = np.arange(len(y), dtype=float)
        X = np.column_stack([np.eye(7)[wd], t / len(y)])
        beta, *_ = np.linalg.lstsq(X[fit], np.log1p(yf), rcond=None)
        resid = np.log1p(yf) - X[fit] @ beta
        pred = np.expm1(X[gap] @ beta) * np.mean(np.exp(resid))  # koreksi bias retransformasi
        return pd.Series(pred, index=y.index[gap])
    margin = METHODS[method]["margin"]
    # Faktor hari dalam minggu dari semua hari fit; level = rata-rata deret tanpa pola mingguan
    # pada `margin` hari fit terdekat di kiri dan kanan gap, diinterpolasi linear.
    factor = pd.Series(yf).groupby(wf).mean()
    factor = factor / factor.mean()
    z = y.to_numpy() / factor.reindex(wd).to_numpy()
    idx = np.flatnonzero(fit)
    g = np.flatnonzero(gap)
    left, right = idx[idx < g.min()][-margin:], idx[idx > g.max()][:margin]
    if len(left) == 0 or len(right) == 0:
        raise ValueError("local_level butuh hari fit di kedua sisi gap")
    lv_l, lv_r = z[left].mean(), z[right].mean()
    x_l, x_r = left.mean(), right.mean()
    level = lv_l + (lv_r - lv_l) * (g - x_l) / (x_r - x_l)
    return pd.Series(level * factor.reindex(wd[g]).to_numpy(), index=y.index[g])


def window_offsets(windows: dict[str, list[str]], event_start: pd.Timestamp) -> dict[str, tuple[int, int]]:
    """Posisi jendela relatif terhadap awal periode event: (offset awal, offset akhir) inklusif."""
    return {k: ((pd.Timestamp(a) - event_start).days, (pd.Timestamp(b) - event_start).days)
            for k, (a, b) in windows.items()}


def split_windows(windows: dict[str, list[str]], length: int) -> dict[str, list[str]]:
    """Pecah jendela yang panjangnya kelipatan `length` menjadi potongan `length` hari."""
    out = {}
    for k, (a, b) in windows.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        n = (b - a).days + 1
        if n % length or n == length:
            out[k] = [str(a.date()), str(b.date())]
            continue
        for i in range(n // length):
            s = a + pd.offsets.Day(i * length)
            out[f"{k}_{i + 1}"] = [str(s.date()), str((s + pd.offsets.Day(length - 1)).date())]
    return out


def log_ratio(actual: pd.Series, exp: pd.Series, keep: np.ndarray) -> float:
    a, e = actual[keep].sum(), exp[keep].sum()
    return float(np.log(a / e)) if a > 0 and e > 0 else float("nan")


def placebo(y: pd.Series, off: set, event: tuple[pd.Timestamp, pd.Timestamp],
            offsets: dict[str, tuple[int, int]], method: str, min_side: int = 14) -> pd.DataFrame:
    """Log rasio aktual/harapan di jendela palsu, untuk semua posisi periode event palsu."""
    days = y.index
    n_event = (event[1] - event[0]).days + 1
    is_event = (days >= event[0]) & (days <= event[1])
    is_off = days.isin(list(off))
    rows = []
    for s in range(len(days) - n_event + 1):
        gap = np.zeros(len(days), dtype=bool)
        gap[s:s + n_event] = True
        if (gap & is_event).any():
            continue
        fit = ~is_event & ~is_off & ~gap
        if fit[:s].sum() < min_side or fit[s + n_event:].sum() < min_side:
            continue
        exp_gap = expected(y, fit, gap, method)
        act_gap = y[gap]
        keep_gap = ~is_off[gap]
        for name, (a, b) in offsets.items():
            keep = np.zeros(n_event, dtype=bool)
            keep[a:b + 1] = True
            keep &= keep_gap
            if keep.sum() == 0:
                continue
            rows.append({"start": str(days[s].date()), "window": name,
                         "log_ratio": log_ratio(act_gap, exp_gap, keep)})
    return pd.DataFrame(rows)


def calibrate(cfg: dict) -> dict:
    p = cfg["id_pharmacy"]["event_study"]
    y_all = load_daily(cfg)
    off = day_off_dates(cfg)
    event = tuple(pd.Timestamp(d) for d in p["event_period"])
    series = ["notes_all", "notes_RJ", "notes_RI"]
    res = {"methods": {}, "window_length": {}}
    off14 = window_offsets(p["windows"], event[0])
    for m in METHODS:
        per = {}
        for s in series:
            pl = placebo(y_all[s], off, event, off14, m)
            per[s] = {"rms_log_ratio": float(np.sqrt(np.nanmean(pl["log_ratio"] ** 2))),
                      "mean_log_ratio": float(pl["log_ratio"].mean()),
                      "n_positions": int(pl["start"].nunique())}
        per["mean_rms"] = float(np.mean([per[s]["rms_log_ratio"] for s in series]))
        res["methods"][m] = per
    best = min(res["methods"], key=lambda k: res["methods"][k]["mean_rms"])
    res["chosen_method"] = best
    for length, wins in ((14, p["windows"]), (7, split_windows(p["windows"], 7))):
        offs = window_offsets(wins, event[0])
        per = {}
        for s in series:
            pl = placebo(y_all[s], off, event, offs, best)
            q = pl["log_ratio"].quantile([0.05, 0.95])
            per[s] = {"rms_log_ratio": float(np.sqrt(np.nanmean(pl["log_ratio"] ** 2))),
                      "p05_effect": float(np.expm1(q.iloc[0])), "p95_effect": float(np.expm1(q.iloc[1]))}
        res["window_length"][str(length)] = per
    return res


LEBARAN_EVENTS = {"cuti_bersama_idulfitri", "idulfitri_holiday"}
SERIES = ["notes_all", "notes_RJ", "notes_RI", "units_all"]


def other_day_off(cfg: dict) -> set[pd.Timestamp]:
    """Hari libur selain Lebaran/cuti bersama Idulfitri (dikeluarkan dari jumlah jendela)."""
    cal = pd.read_csv(Path(cfg["paths"]["external"]) / "id_calendar.csv", parse_dates=["date"])
    lebaran = set(cal.loc[cal["event"].isin(LEBARAN_EVENTS), "date"])
    return day_off_dates(cfg) - lebaran


def effects(cfg: dict, method: str | None = None) -> tuple[dict, pd.DataFrame]:
    """Efek di jendela asli + pita plasebo, dengan pilihan dari tahap kalibrasi.

    `method` hanya diisi untuk uji sensitivitas; hasil utama memakai config.
    """
    p = cfg["id_pharmacy"]["event_study"]
    method = method or p["baseline_method"]
    y_all = load_daily(cfg)
    off = day_off_dates(cfg)
    other = other_day_off(cfg)
    event = tuple(pd.Timestamp(d) for d in p["event_period"])
    wins = split_windows(p["windows"], p["window_days"])
    offs = window_offsets(wins, event[0])
    days = y_all.index
    is_event = (days >= event[0]) & (days <= event[1])
    fit = ~is_event & ~days.isin(list(off))
    keep_gap = ~days[is_event].isin(list(other))
    out = {"method": method, "window_days": p["window_days"], "windows": wins, "series": {}}
    daily = pd.DataFrame(index=days[is_event])
    for s in SERIES:
        y = y_all[s]
        exp_gap = expected(y, fit, is_event, method)
        daily[f"{s}_actual"], daily[f"{s}_expected"] = y[is_event].to_numpy(), exp_gap.to_numpy()
        pl = placebo(y, off, event, offs, method)
        res = {}
        for name, (a, b) in offs.items():
            keep = np.zeros(is_event.sum(), dtype=bool)
            keep[a:b + 1] = True
            keep &= keep_gap
            lr = log_ratio(y[is_event], exp_gap, keep)
            ref = pl.loc[pl["window"] == name, "log_ratio"].dropna()
            q05, q95 = ref.quantile([0.05, 0.95])
            res[name] = {
                "effect": float(np.expm1(lr)),
                "actual": float(y[is_event][keep].sum()), "expected": float(exp_gap[keep].sum()),
                "days_used": int(keep.sum()),
                "placebo_p05": float(np.expm1(q05)), "placebo_p95": float(np.expm1(q95)),
                "placebo_rank": float((ref <= lr).mean()), "placebo_n": int(len(ref)),
                "outside_band": bool(lr < q05 or lr > q95),
            }
        out["series"][s] = res
    return out, daily


WINDOW_LABELS = {
    "ramadan_w1_2_1": "Ramadan mg 1", "ramadan_w1_2_2": "Ramadan mg 2", "ramadan_w3_4_1": "Ramadan mg 3",
    "ramadan_w3_4_2": "Ramadan mg 4", "lebaran_cuti": "Lebaran+cuti", "post_1_1": "pasca mg 1",
    "post_1_2": "pasca mg 2", "post_2_1": "pasca mg 3", "post_2_2": "pasca mg 4",
}
SERIES_LABELS = {"notes_all": "nota total", "notes_RJ": "nota RJ", "notes_RI": "nota RI", "units_all": "unit total"}


def fig_effects(r: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from retailfc.eda import GRID, INK, INK_2, MUTED, SERIES as COLORS, SURFACE, style_axes

    names = list(r["windows"])
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9), dpi=150, sharey=True, facecolor=SURFACE)
    col = COLORS[0]
    for ax, s in zip(axes, ["notes_all", "notes_RJ", "notes_RI"]):
        style_axes(ax)
        ax.grid(axis="x", visible=False)
        res = r["series"][s]
        lo = np.array([100 * res[n]["placebo_p05"] for n in names])
        hi = np.array([100 * res[n]["placebo_p95"] for n in names])
        eff = np.array([100 * res[n]["effect"] for n in names])
        ax.bar(x, hi - lo, bottom=lo, width=0.7, color=GRID, label="pita plasebo 5–95%")
        out = np.array([res[n]["outside_band"] for n in names])
        ax.scatter(x[~out], eff[~out], s=30, color=MUTED, zorder=3, edgecolor=SURFACE, label="dalam pita")
        ax.scatter(x[out], eff[out], s=34, color=col, zorder=3, edgecolor=SURFACE, label="di luar pita")
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_xticks(x, [WINDOW_LABELS.get(n, n) for n in names], rotation=60, ha="right", fontsize=7)
        ax.set_title(SERIES_LABELS[s], fontsize=9, color=INK_2, loc="left")
    axes[0].set_ylabel("efek vs nilai harapan (%)", color=INK_2, fontsize=8.5)
    h, n = axes[0].get_legend_handles_labels()
    fig.legend(h, n, frameon=False, fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.92),
               labelcolor=INK_2)
    fig.suptitle("Efek Ramadan dan Lebaran per minggu, dibanding fluktuasi minggu biasa", x=0.01, ha="left",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep = Path(cfg["paths"]["reports"])
    if args.calibrate:
        r = calibrate(cfg)
        (rep / "data" / "id_event_calibration.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(json.dumps(r, indent=2))
        return
    r, daily = effects(cfg)
    # Sensitivitas: efek dengan kandidat baseline lain (bukan hasil utama).
    r["sensitivity"] = {}
    for m in METHODS:
        if m == r["method"]:
            continue
        alt, _ = effects(cfg, m)
        r["sensitivity"][m] = {s: {k: {"effect": v["effect"], "outside_band": v["outside_band"]}
                                   for k, v in res.items()} for s, res in alt["series"].items()}
    (rep / "data" / "id_event_study.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    daily.round(2).to_csv(rep / "data" / "id_event_study_daily.csv", index_label="date")
    fig_effects(r, rep / "figures" / "phase8_event_effects.png")
    for s, res in r["series"].items():
        print(s)
        for k, v in res.items():
            print(f"  {k:16s} efek {100 * v['effect']:+6.1f}%  pita [{100 * v['placebo_p05']:+6.1f}, "
                  f"{100 * v['placebo_p95']:+6.1f}]  rank {v['placebo_rank']:.2f}  {'LUAR' if v['outside_band'] else ''}")


if __name__ == "__main__":
    main()
