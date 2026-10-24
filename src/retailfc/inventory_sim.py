"""Fase 7: simulasi stok (P3), kebijakan periodic review order-up-to.

Kebijakan (sama untuk semua metode):
- stok dicek setiap `review_period_days` (R) hari, mulai hari pertama horizon;
- pesanan tiba `lead_time_days` (L) hari kemudian, di awal hari;
- target stok S = kuantil permintaan selama periode proteksi R + L hari
  (dipotong di akhir horizon), pesanan = max(0, S − (stok di tangan + pesanan di jalan));
- di hari pertama, stok di tangan = S (sistem dianggap sudah berjalan);
- permintaan yang tidak terpenuhi hilang (lost sales);
- "permintaan" = penjualan aktual M5. Penjualan yang tercatat saat stok habis lebih rendah
  dari permintaan asli (censored demand), jadi unit hilang di sini adalah batas bawah.

Metode target stok S (untuk tiap kuantil α):
- `xgb_quantile`: jumlah kuantil harian model kuantil XGBoost selama periode proteksi
  (bukan kuantil dari jumlah; seberapa jauh bedanya diukur lewat coverage S);
- `xgb_normal`: jumlah forecast point XGBoost + z_α · σ · √n;
- `classical_normal`: jumlah forecast classical (ETS+TSB) + z_α · σ · √n;
- `empirical`: kuantil α dari jumlah penjualan n hari berturut-turut di riwayat
  `empirical_window_days` hari sebelum cutoff;
- `xgb_empirical` (gabungan): target `empirical` dikalikan rasio jumlah forecast point
  XGBoost selama periode proteksi terhadap rata-rata historis n hari di jendela yang sama.
  Bentuk ekor dari riwayat, level dari XGBoost. Tanpa riwayat di jendela itu: jumlah forecast.
  Aturan keputusan (ditetapkan sebelum run): lebih baik dari `empirical` jika biayanya ≤
  empirical pada asumsi dasar DAN pada ≥ 7 dari 12 kombinasi sensitivitas.
σ = simpangan baku penjualan harian `sigma_window_days` hari terakhir sebelum cutoff.

Biaya (ASUMSI, M5 tidak memuat data biaya): biaya simpan = tarif tahunan / 365 × harga ×
unit-hari stok; biaya kehabisan = tarif × harga × unit hilang. Karena biaya linear, nilai
rupiah-nya (harga × unit) disimpan sekali, lalu biaya untuk setiap kombinasi tarif dihitung
ulang tanpa simulasi ulang (analisis sensitivitas).

Hanya fold CV (data ≤ d_1913). Holdout disimulasikan sekali di akhir proyek.

Cara pakai:
    python -m retailfc.inventory_sim --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from retailfc.baselines import fill_closed
from retailfc.error_analysis import growth_ratio
from retailfc.folds import build_folds, closed_day_mask
from retailfc.io import load_calendar, load_config, load_wide_matrices

METHODS = ["xgb_quantile", "xgb_normal", "classical_normal", "empirical", "xgb_empirical"]
QUANTILE_ALPHAS = [0.1, 0.5, 0.8, 0.9]          # urutan kolom di xgb_quantile.npy


def review_days(h: int, review: int) -> list[int]:
    return list(range(0, h, review))


def protection_windows(h: int, review: int, lead: int) -> list[tuple[int, int]]:
    """(awal, akhir-eksklusif) periode proteksi untuk tiap hari review, dipotong di akhir horizon."""
    return [(t, min(t + review + lead, h)) for t in review_days(h, review)]


def simulate(demand: np.ndarray, price: np.ndarray, targets: np.ndarray, review: int, lead: int) -> dict:
    """Simulasi lost-sales order-up-to untuk banyak deret sekaligus.

    `demand`, `price`: (n × h). `targets`: (n × jumlah review), S untuk tiap hari review.
    Mengembalikan total per deret: permintaan, terjual, hilang, unit-hari stok, nilai stok
    (harga × unit-hari), nilai hilang (harga × unit), stok akhir, dan jumlah unit dipesan.
    """
    n, h = demand.shape
    reviews = review_days(h, review)
    if targets.shape != (n, len(reviews)):
        raise ValueError("targets must be (n_series, n_reviews)")
    on_hand = np.maximum(targets[:, 0], 0.0).astype(np.float64)
    arrivals = np.zeros((n, h + lead + 1))
    out = {k: np.zeros(n) for k in ("demand", "sold", "lost", "hold_units", "hold_value", "lost_value", "ordered")}
    for d in range(h):
        on_hand += arrivals[:, d]
        if d in reviews and d > 0:
            r = reviews.index(d)
            position = on_hand + arrivals[:, d + 1:].sum(axis=1)
            order = np.maximum(targets[:, r] - position, 0.0)
            arrivals[:, d + lead] += order
            out["ordered"] += order
        dem = demand[:, d]
        sold = np.minimum(on_hand, dem)
        lost = dem - sold
        on_hand -= sold
        out["demand"] += dem
        out["sold"] += sold
        out["lost"] += lost
        out["hold_units"] += on_hand
        out["hold_value"] += on_hand * price[:, d]
        out["lost_value"] += lost * price[:, d]
    out["ending_stock"] = on_hand
    return out


def empirical_targets(hist: np.ndarray, first: np.ndarray, lengths: list[int], alpha: float,
                      window: int) -> dict[int, np.ndarray]:
    """Kuantil α dari jumlah `n` hari berturut-turut di `window` hari terakhir riwayat,
    hanya dari hari setelah penjualan pertama. NaN (riwayat terlalu pendek) → 0."""
    h = hist[:, -window:].astype(np.float64)
    start_col = hist.shape[1] - window
    pre = np.arange(window)[None, :] + start_col < first[:, None]
    h = np.where(pre, np.nan, h)
    out = {}
    for n_days in sorted(set(lengths)):
        sums = np.lib.stride_tricks.sliding_window_view(h, n_days, axis=1).sum(axis=2)
        with warnings.catch_warnings():               # baris tanpa jendela lengkap → NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            q = np.nanquantile(sums, alpha, axis=1)
        out[n_days] = np.nan_to_num(q, nan=0.0)
    return out


def history_mean(hist: np.ndarray, first: np.ndarray, window: int) -> np.ndarray:
    """Rata-rata penjualan harian di `window` hari terakhir, hanya setelah penjualan pertama."""
    h = hist[:, -window:].astype(np.float64)
    valid = np.arange(window)[None, :] + hist.shape[1] - window >= first[:, None]
    cnt = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (h * valid).sum(axis=1) / cnt, 0.0)


def build_targets(method: str, alpha: float, windows, point: dict, qpred: np.ndarray, sigma: np.ndarray,
                  emp: dict, hist_mean: np.ndarray | None = None) -> np.ndarray:
    cols = []
    for a, b in windows:
        n_days = b - a
        if method == "xgb_quantile":
            j = QUANTILE_ALPHAS.index(alpha)
            cols.append(qpred[:, a:b, j].sum(axis=1))
        elif method in ("xgb_normal", "classical_normal"):
            p = point["xgb_point" if method == "xgb_normal" else "classical"]
            cols.append(p[:, a:b].sum(axis=1) + norm.ppf(alpha) * sigma * np.sqrt(n_days))
        elif method == "empirical":
            cols.append(emp[(alpha, n_days)])
        elif method == "xgb_empirical":
            fc = point["xgb_point"][:, a:b].sum(axis=1)
            base = hist_mean * n_days
            with np.errstate(invalid="ignore", divide="ignore"):
                cols.append(np.where(base > 0, emp[(alpha, n_days)] * fc / base, fc))
        else:
            raise ValueError(method)
    return np.maximum(np.stack(cols, axis=1), 0.0)


def coverage(demand: np.ndarray, targets: np.ndarray, windows) -> float:
    """Porsi (deret × review) di mana permintaan aktual selama periode proteksi ≤ S."""
    hits = [(demand[:, a:b].sum(axis=1) <= targets[:, r]) for r, (a, b) in enumerate(windows)]
    return float(np.mean(hits))


def summarize(res: dict, sel: np.ndarray, h: int) -> dict:
    s = {k: float(v[sel].sum()) for k, v in res.items()}
    s["n_series"] = int(sel.sum())
    s["fill_rate"] = s["sold"] / s["demand"] if s["demand"] > 0 else None
    s["avg_on_hand"] = s["hold_units"] / h
    s["days_of_cover"] = s["avg_on_hand"] / (s["demand"] / h) if s["demand"] > 0 else None
    return s


def cost(s: dict, holding_annual: float, stockout_rate: float) -> float:
    return s["hold_value"] * holding_annual / 365 + s["lost_value"] * stockout_rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    inv = cfg["inventory"]
    R, L = inv["review_period_days"], inv["lead_time_days"]
    alphas = [a for a in inv["service_quantiles"] if a in QUANTILE_ALPHAS]   # xgb_quantile
    base_alphas = inv["baseline_quantiles"]
    folds = [f for f in build_folds(cfg) if f.role == "cv"]
    ids, sales, prices = load_wide_matrices(cfg, cfg["validation"]["holdout_cutoff"])
    cal = load_calendar(Path(cfg["paths"]["raw_m5"]))
    root = Path(cfg["paths"]["processed"]) / "preds"
    cats = ids["cat_id"].astype(str).to_numpy()

    out: dict = {"assumptions": inv, "methods": METHODS, "folds": {}}
    for f in folds:
        c, h = f.cutoff, f.horizon
        closed = closed_day_mask(np.arange(1, c + 1), cal, cfg)
        hist = fill_closed(sales[:, :c], closed)
        sold_hist = hist > 0
        has = sold_hist.any(axis=1)
        first = np.where(has, sold_hist.argmax(axis=1), c)
        win = inv["sigma_window_days"]
        recent = np.where(np.arange(c - win, c)[None, :] >= first[:, None], hist[:, -win:], np.nan)
        with warnings.catch_warnings():               # deret tanpa riwayat → NaN → 0
            warnings.simplefilter("ignore", RuntimeWarning)
            sigma = np.nan_to_num(np.nanstd(recent, axis=1), nan=0.0)
        demand = sales[:, c:c + h].astype(np.float64)
        with warnings.catch_warnings():               # item tanpa harga 28 hari terakhir → 0
            warnings.simplefilter("ignore", RuntimeWarning)
            p_ref = np.nan_to_num(np.nanmean(prices[:, c - 28:c], axis=1), nan=0.0)
        price = np.where(np.isnan(prices[:, c:c + h]), p_ref[:, None], prices[:, c:c + h]).astype(np.float64)
        point = {m: np.load(root / f.name / f"{m}.npy").astype(np.float64) for m in ("xgb_point", "classical")}
        qpred = np.load(root / f.name / "xgb_quantile.npy").astype(np.float64)
        windows = protection_windows(h, R, L)
        lengths = [b - a for a, b in windows]
        emp = {}
        for a_ in sorted(set(alphas) | set(base_alphas)):
            for n_days, v in empirical_targets(hist, first, lengths, a_, inv["empirical_window_days"]).items():
                emp[(a_, n_days)] = v
        hist_mean = history_mean(hist, first, inv["empirical_window_days"])
        seg = pd.read_parquet(root / f.name / "segments.parquet")["segment"].astype(str).to_numpy()
        growth = growth_ratio(sales[:, :c])
        groups = {"all": np.ones(len(ids), bool), "has_history": has,
                  "fast_growing": growth > inv["growth_risk_ratio"],
                  "not_fast_growing": has & ~(growth > inv["growth_risk_ratio"])}
        groups |= {f"segment={s}": seg == s for s in ["smooth", "erratic", "intermittent", "lumpy"]}
        groups |= {f"cat={k}": cats == k for k in sorted(set(cats))}

        fold_out: dict = {}
        for m in METHODS:
            for a_ in (alphas if m == "xgb_quantile" else base_alphas):
                S = build_targets(m, a_, windows, point, qpred, sigma, emp, hist_mean)
                res = simulate(demand, price, S, R, L)
                key = f"{m}@{a_}"
                fold_out[key] = {"coverage_S": round(coverage(demand, S, windows), 4),
                                 "groups": {g: summarize(res, sel, h) for g, sel in groups.items()}}
        out["folds"][f.name] = fold_out
        g = fold_out
        print(f"{f.name}: " + " ".join(f"{k} fill={v['groups']['all']['fill_rate']:.3f}"
                                         f" cov={v['coverage_S']:.3f}" for k, v in g.items() if k.endswith("0.9")),
              flush=True)

    # Ringkasan: jumlah 4 fold per metode × kuantil × kelompok, lalu biaya untuk tiap asumsi.
    keys = list(out["folds"][folds[0].name])
    group_names = list(out["folds"][folds[0].name][keys[0]]["groups"])
    summary = {}
    for k in keys:
        summary[k] = {"coverage_S_mean": round(float(np.mean([out["folds"][f.name][k]["coverage_S"] for f in folds])), 4),
                      "groups": {}}
        for gname in group_names:
            tot = {m: sum(out["folds"][f.name][k]["groups"][gname][m] for f in folds)
                   for m in ("demand", "sold", "lost", "hold_units", "hold_value", "lost_value", "ending_stock", "ordered")}
            tot["fill_rate"] = tot["sold"] / tot["demand"] if tot["demand"] > 0 else None
            tot["avg_on_hand_per_day"] = tot["hold_units"] / (len(folds) * folds[0].horizon)
            tot["days_of_cover"] = tot["hold_units"] / tot["demand"] if tot["demand"] > 0 else None
            tot["cost_base"] = cost(tot, inv["holding_cost_annual_rate"], inv["stockout_cost_rate"])
            tot["cost_holding_base"] = tot["hold_value"] * inv["holding_cost_annual_rate"] / 365
            tot["cost_stockout_base"] = tot["lost_value"] * inv["stockout_cost_rate"]
            summary[k]["groups"][gname] = {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in tot.items()}
    sens = []
    for hr in inv["sensitivity"]["holding_cost_annual_rate"]:
        for sr in inv["sensitivity"]["stockout_cost_rate"]:
            costs = {k: cost(summary[k]["groups"]["all"], hr, sr) for k in keys}
            best = min(costs, key=costs.get)
            best_by_method = {m: min((k for k in keys if k.startswith(m + "@")), key=costs.get) for m in METHODS}
            sens.append({"holding_annual": hr, "stockout_rate": sr, "best": best,
                         "best_by_method": {m: {"choice": b, "cost": round(costs[b], 1)} for m, b in best_by_method.items()},
                         "costs": {k: round(v, 1) for k, v in costs.items()}})
    # Kurva trade-off: fill rate pada stok rata-rata (hari) yang sama, interpolasi linear di
    # antara titik-titik α tiap metode. Di luar rentang titik metode itu: tidak dihitung (None).
    frontier = {}
    for gname in group_names:
        frontier[gname] = {}
        for m in METHODS:
            pts = sorted((summary[k]["groups"][gname]["days_of_cover"], summary[k]["groups"][gname]["fill_rate"])
                         for k in keys if k.startswith(m + "@") and summary[k]["groups"][gname]["days_of_cover"])
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            frontier[gname][m] = {str(x): (round(float(np.interp(x, xs, ys)), 4) if xs and xs[0] <= x <= xs[-1] else None)
                                  for x in inv["cover_grid_days"]}
    out["frontier_fill_rate_at_cover_days"] = frontier
    out["summary"] = summary
    out["sensitivity"] = sens
    rep = Path(cfg["paths"]["reports"])
    (rep / "inventory_sim.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for k in keys:
        a = summary[k]["groups"]["all"]
        print(f"{k:22s} fill={a['fill_rate']:.4f} lost={a['lost']:.0f} cover={a['days_of_cover']:.2f}d "
              f"cost={a['cost_base']:.0f} covS={summary[k]['coverage_S_mean']:.3f}")


if __name__ == "__main__":
    main()
