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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep = Path(cfg["paths"]["reports"])
    if args.calibrate:
        r = calibrate(cfg)
        (rep / "id_event_calibration.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(json.dumps(r, indent=2))
        return
    raise SystemExit("tahap efek belum diaktifkan: jalankan --calibrate dulu dan tetapkan pilihannya di config")


if __name__ == "__main__":
    main()
