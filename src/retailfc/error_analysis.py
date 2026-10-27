"""Fase 6: analisis error per segmen (P2), dari prediksi Fase 5 yang sudah tersimpan.

Tidak melatih model. Membaca data/processed/preds/<fold>/{xgb_point, baseline}.npy dan
segments.parquet (segmen Syntetos–Boylan dihitung dari data train tiap fold). Data dimuat
hanya sampai d_1913, jadi holdout tidak dibaca.

Ukuran per kelompok deret (di level item × toko):
- `wrmsse`: rata-rata RMSSE tertimbang omzet, bobot dinormalisasi di dalam kelompok
  (= skor level 12 WRMSSE, dibatasi ke kelompok itu);
- `agg_rmsse`: RMSSE dari total harian kelompok (semua deret kelompok dijumlahkan dulu),
  untuk melihat apakah error saling menetralkan saat dijumlahkan;
- `mae`, `bias` (rata-rata forecast − aktual), `wape`;
- `win_share`: porsi deret (dan porsi bobot) di mana RMSSE XGBoost < RMSSE baseline.

Ukuran per kelompok hari (hari dalam minggu, event, SNAP): RMSSE tertimbang yang dihitung
hanya dari hari-hari di kelompok itu, ditambah WAPE di level toko. Hari tutup (25 Des)
dikeluarkan dari kelompok hari.

Cara pakai:
    python -m retailfc.error_analysis --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from retailfc.baselines import METHODS as BASELINES
from retailfc.folds import build_folds, closed_day_mask
from retailfc.io import load_calendar, load_config, load_wide_matrices
from retailfc.metrics import WRMSSE, naive_scale

MODEL = "xgb_point"
SEGMENTS = ["smooth", "erratic", "intermittent", "lumpy", "no_history"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def bottom_scale_weight(ev: WRMSSE, ids: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Skala dan bobot level 12 (item × toko) dalam urutan baris `ids`."""
    lv12 = (ev.labels["Level_id"] == "Level12").to_numpy()
    lab = ev.labels[lv12].reset_index(drop=True)
    key = pd.MultiIndex.from_frame(ids[["item_id", "store_id"]].astype(str))
    pos = pd.MultiIndex.from_frame(lab[["Agg_Level_1", "Agg_Level_2"]]).get_indexer(key)
    if (pos < 0).any():
        raise ValueError("series missing from Level12 labels")
    return ev.scale[lv12][pos], ev.weight[lv12][pos]


def series_rmsse(err: np.ndarray, scale: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """RMSSE per deret; dengan `mask` (n × h) hanya hari yang True yang dihitung."""
    e2 = err ** 2
    if mask is None:
        mse = e2.mean(axis=1)
    else:
        cnt = mask.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mse = np.where(cnt > 0, (e2 * mask).sum(axis=1) / cnt, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(mse / scale)


def weighted(values: np.ndarray, w: np.ndarray, sel: np.ndarray) -> float:
    ok = sel & np.isfinite(values) & (w > 0)
    return float((w[ok] * values[ok]).sum() / w[ok].sum()) if ok.any() else float("nan")


def group_block(sel, actual, preds, hist, scale, w, rmsse) -> dict:
    """Ukuran satu kelompok deret untuk semua model."""
    out = {"n_series": int(sel.sum()), "weight_share": round(float(w[sel].sum()), 5)}
    if not sel.any():
        return out
    hist_tot = hist[sel].sum(axis=0, dtype=np.float64)[None, :]
    agg_scale = float(naive_scale(hist_tot)[0])
    a = actual[sel]
    for m, p in preds.items():
        e = a - p[sel]
        agg_err = e.sum(axis=0)
        out[m] = {"wrmsse": round(weighted(rmsse[m], w, sel), 5),
                  "agg_rmsse": round(float(np.sqrt((agg_err ** 2).mean() / agg_scale)), 5),
                  "mae": round(float(np.abs(e).mean()), 5), "bias": round(float(-e.mean()), 5),
                  "wape": round(float(np.abs(e).sum() / a.sum()), 5) if a.sum() > 0 else None}
    ok = sel & np.isfinite(rmsse[MODEL]) & (w > 0)
    out["win_share"] = {}
    for b in BASELINES:
        win = rmsse[MODEL][ok] < rmsse[b][ok]
        out["win_share"][b] = {"series": round(float(win.mean()), 4) if ok.any() else None,
                               "weight": round(float(w[ok][win].sum() / w[ok].sum()), 4) if ok.any() else None}
    return out


def day_block(mask, actual, preds, scale, w, store_codes, n_store) -> dict:
    """Ukuran satu kelompok hari (mask n × h) untuk semua model."""
    out = {"n_days": int(mask.any(axis=0).sum())}
    if not mask.any():
        return out
    for m, p in preds.items():
        e = actual - p
        r = series_rmsse(e, scale, mask)
        # WAPE level toko: jumlahkan per toko per hari, lalu hanya hari di mask.
        a_st = np.zeros((n_store, actual.shape[1]))
        e_st = np.zeros_like(a_st)
        np.add.at(a_st, store_codes, actual * mask)
        np.add.at(e_st, store_codes, e * mask)
        out[m] = {"wrmsse": round(weighted(r, w, np.ones(len(w), bool)), 5),
                  "store_wape": round(float(np.abs(e_st).sum() / a_st.sum()), 5),
                  "bias": round(float(-(e * mask).sum() / mask.sum()), 5)}
    return out


def zero_run_max(x: np.ndarray) -> int:
    best = cur = 0
    for v in x:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return best


def analyze_fold(f, ids, sales, prices, cal, cfg, pred_root: Path) -> tuple[dict, dict]:
    c, h = f.cutoff, f.horizon
    actual = sales[:, c:c + h].astype(np.float64)
    hist = sales[:, :c]
    preds = {m: np.load(pred_root / f.name / f"{m}.npy").astype(np.float64) for m in [MODEL] + BASELINES}
    seg = pd.read_parquet(pred_root / f.name / "segments.parquet")
    if not seg["id"].astype(str).equals(ids["id"].astype(str)):
        raise ValueError("segments order differs from ids")
    ev = WRMSSE(ids, sales, prices, cutoff=c, horizon=h)
    scale, w = bottom_scale_weight(ev, ids)
    rmsse = {m: series_rmsse(actual - p, scale) for m, p in preds.items()}

    groups: dict = {"all": group_block(np.ones(len(ids), bool), actual, preds, hist, scale, w, rmsse)}
    for col, values in [("segment", SEGMENTS), ("cat_id", None), ("store_id", None), ("dept_id", None)]:
        src = seg["segment"].astype(str) if col == "segment" else ids[col].astype(str)
        vals = values or sorted(src.unique())
        groups[col] = {v: group_block((src == v).to_numpy(), actual, preds, hist, scale, w, rmsse) for v in vals}
    combo = seg["segment"].astype(str) + "|" + ids["cat_id"].astype(str)
    groups["segment_x_cat"] = {v: group_block((combo == v).to_numpy(), actual, preds, hist, scale, w, rmsse)
                               for v in sorted(combo.unique())}

    # Kelompok hari.
    days = np.arange(c + 1, c + h + 1)
    ch = cal.set_index("d_num").loc[days]
    closed = closed_day_mask(days, cal, cfg)
    open_ = np.broadcast_to(~closed, actual.shape)
    store_codes = ids["store_id"].cat.codes.to_numpy() if hasattr(ids["store_id"], "cat") else \
        pd.factorize(ids["store_id"])[0]
    n_store = int(store_codes.max()) + 1
    by_day: dict = {"weekday": {}}
    for wd in WEEKDAYS:
        m = np.broadcast_to((ch["weekday"].to_numpy() == wd), actual.shape) & open_
        by_day["weekday"][wd] = day_block(m, actual, preds, scale, w, store_codes, n_store)
    is_event = (ch["event_name_1"].notna() | ch["event_name_2"].notna()).to_numpy()
    by_day["event"] = {k: day_block(np.broadcast_to(v, actual.shape) & open_, actual, preds, scale, w,
                                    store_codes, n_store)
                       for k, v in {"event": is_event, "non_event": ~is_event}.items()}
    by_day["event_names"] = sorted({e for col in ("event_name_1", "event_name_2") for e in ch[col].dropna()})
    state = ids["state_id"].astype(str).to_numpy()
    snap = np.stack([ch[f"snap_{s}"].to_numpy().astype(bool) for s in state])
    by_day["snap"] = {k: day_block(v & open_, actual, preds, scale, w, store_codes, n_store)
                      for k, v in {"snap": snap, "non_snap": ~snap}.items()}
    by_day["n_closed_days_excluded"] = int(closed.sum())

    # Kontribusi per deret ke skor level 12 (untuk studi kasus).
    contrib = {"w": w, "rmsse_xgb": rmsse[MODEL], "rmsse_classical": rmsse["classical"],
               "segment": seg["segment"].astype(str).to_numpy()}
    return {"groups": groups, "by_day": by_day, "n_excluded_series": ev.n_excluded_series}, contrib


def case_diagnostics(i: int, f, ids, sales, prices, seg: str, pred_root: Path) -> dict:
    """Sinyal untuk menjelaskan error besar satu deret: stok habis, harga, produk baru, lonjakan."""
    c, h = f.cutoff, f.horizon
    a = sales[i, c:c + h].astype(float)
    x = np.load(pred_root / f.name / f"{MODEL}.npy")[i].astype(float)
    cl = np.load(pred_root / f.name / "classical.npy")[i].astype(float)
    hist = sales[i, :c].astype(float)
    first = int(np.argmax(hist > 0)) + 1                      # d_ penjualan pertama
    p_before = np.nanmean(prices[i, c - 28:c])
    p_h = prices[i, c:c + h]
    return {
        "id": str(ids["id"].iloc[i]), "fold": f.name, "segment": seg,
        "actual_sum": float(a.sum()), "xgb_sum": round(float(x.sum()), 1), "classical_sum": round(float(cl.sum()), 1),
        "actual_mean_last28_before": round(float(hist[-28:].mean()), 2),
        "actual_mean_prev_year_same_window": round(float(sales[i, c - 364:c - 364 + h].mean()), 2) if c >= 364 else None,
        "max_zero_run_horizon": zero_run_max(a), "max_zero_run_last56_before": zero_run_max(hist[-56:]),
        "horizon_max_over_mean": round(float(a.max() / a.mean()), 2) if a.mean() > 0 else None,
        "days_since_first_sale": c - first + 1,
        "price_last28_before": round(float(p_before), 2),
        "price_horizon_min": round(float(np.nanmin(p_h)), 2), "price_horizon_max": round(float(np.nanmax(p_h)), 2),
        "price_change_pct": round(100 * (float(np.nanmean(p_h)) / float(p_before) - 1), 1),
    }


GROWTH_EDGES = [0, 0.5, 0.8, 1.25, 2, 4, np.inf]
GROWTH_LABELS = ["<0,5", "0,5–0,8", "0,8–1,25", "1,25–2", "2–4", ">4"]
BLEND_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.5, 0.0]


def growth_ratio(hist: np.ndarray, recent: int = 56) -> np.ndarray:
    """Rata-rata `recent` hari terakhir sebelum cutoff ÷ rata-rata sejak penjualan pertama.
    Hanya memakai data <= cutoff. NaN untuk deret tanpa penjualan."""
    h = hist.astype(np.float64)
    sold = h > 0
    has = sold.any(axis=1)
    first = np.where(has, sold.argmax(axis=1), h.shape[1])
    valid = np.arange(h.shape[1])[None, :] >= first[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        longrun = (h * valid).sum(axis=1) / valid.sum(axis=1)
        return np.where(has, h[:, -recent:].mean(axis=1) / longrun, np.nan)


def growth_and_blend(folds, ids, sales, prices, pred_root: Path) -> dict:
    """Bias per kelompok rasio pertumbuhan (point dan kuantil), dan uji campuran XGBoost +
    classical sebagai diagnosis: apakah informasi level terbaru belum dimanfaatkan penuh."""
    acc = {lab: {"n": 0, "act": 0.0, "xgb": 0.0, "cls": 0.0, "cov": np.zeros(3), "k": 0} for lab in GROWTH_LABELS}
    blend = {a: {"wrmsse": [], "Level1": [], "Level3": [], "Level12": [],
                 "bins": {lab: [0.0, 0.0] for lab in GROWTH_LABELS}} for a in BLEND_WEIGHTS}
    total_units = 0.0
    for f in folds:
        c = f.cutoff
        a_ = sales[:, c:c + f.horizon].astype(np.float64)
        x = np.load(pred_root / f.name / f"{MODEL}.npy").astype(np.float64)
        cl = np.load(pred_root / f.name / "classical.npy").astype(np.float64)
        qpath = pred_root / f.name / "xgb_quantile.npy"
        q = np.load(qpath).astype(np.float64) if qpath.exists() else None
        bins = pd.cut(growth_ratio(sales[:, :c]), GROWTH_EDGES, labels=GROWTH_LABELS).astype(str)
        total_units += a_.sum()
        for lab in GROWTH_LABELS:
            m = (bins == lab)
            g = acc[lab]
            g["n"] += int(m.sum()); g["act"] += a_[m].sum(); g["xgb"] += x[m].sum(); g["cls"] += cl[m].sum()
            if q is not None:
                g["cov"] += [(a_[m] <= q[m, :, j]).mean() for j in (1, 2, 3)]   # P50, P80, P90
                g["k"] += 1
        ev = WRMSSE(ids, sales, prices, cutoff=c, horizon=f.horizon)
        for w in BLEND_WEIGHTS:
            p = w * x + (1 - w) * cl
            s = ev.score(a_, p)
            b = blend[w]
            b["wrmsse"].append(s["wrmsse"])
            for lv in ("Level1", "Level3", "Level12"):
                b[lv].append(s["per_level"][lv])
            for lab in GROWTH_LABELS:
                m = bins == lab
                b["bins"][lab][0] += p[m].sum(); b["bins"][lab][1] += a_[m].sum()
    n_f = len(folds)
    growth = {lab: {"n_series_mean": round(g["n"] / n_f, 1), "units_share_pct": round(100 * g["act"] / total_units, 2),
                    "xgb_bias_pct": round(100 * (g["xgb"] / g["act"] - 1), 2),
                    "classical_bias_pct": round(100 * (g["cls"] / g["act"] - 1), 2),
                    **({"quantile_coverage": dict(zip(["0.5", "0.8", "0.9"], np.round(g["cov"] / g["k"], 4).tolist()))}
                       if g["k"] else {})} for lab, g in acc.items()}
    base = float(np.mean(blend[1.0]["wrmsse"]))
    blends = {str(w): {"wrmsse": round(float(np.mean(b["wrmsse"])), 5),
                       "change_pct": round(100 * (float(np.mean(b["wrmsse"])) / base - 1), 2),
                       **{lv: round(float(np.mean(b[lv])), 5) for lv in ("Level1", "Level3", "Level12")},
                       "bias_pct_by_growth": {lab: round(100 * (v[0] / v[1] - 1), 1) for lab, v in b["bins"].items()}}
              for w, b in blend.items()}
    return {"growth_ratio": "mean of last 56 days before cutoff / mean since first sale (data <= cutoff)",
            "by_growth": growth, "blend_xgb_weight": blends}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n-cases", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep = Path(cfg["paths"]["reports"])
    pred_root = Path(cfg["paths"]["processed"]) / "preds"
    folds = [f for f in build_folds(cfg) if f.role != "holdout"]
    ids, sales, prices = load_wide_matrices(cfg, cfg["validation"]["holdout_cutoff"])
    cal = load_calendar(Path(cfg["paths"]["raw_m5"]))

    result: dict = {"folds": {}, "models": [MODEL] + BASELINES}
    contribs = {}
    for f in folds:
        result["folds"][f.name], contribs[f.name] = analyze_fold(f, ids, sales, prices, cal, cfg, pred_root)
        print(f"{f.name}: done", flush=True)

    # Studi kasus: deret dengan kontribusi rata-rata terbesar ke skor level 12 (fold CV).
    cv = [f for f in folds if f.role == "cv"]
    share = np.stack([contribs[f.name]["w"] * np.nan_to_num(contribs[f.name]["rmsse_xgb"]) for f in cv])
    lvl12 = share.sum(axis=1)                                   # skor level 12 per fold
    mean_share = (share / lvl12[:, None]).mean(axis=0)
    order = np.argsort(mean_share)[::-1]
    result["concentration"] = {f"top_{k}": round(float(mean_share[order[:k]].sum()), 4)
                               for k in (5, 50, 500, 3000)}
    cases = []
    for i in order[:args.n_cases]:
        worst = int(np.argmax(share[:, i]))
        f = cv[worst]
        d = case_diagnostics(int(i), f, ids, sales, prices, contribs[f.name]["segment"][i], pred_root)
        d["level12_share_mean_pct"] = round(100 * float(mean_share[i]), 3)
        d["level12_share_worst_fold_pct"] = round(100 * float(share[worst, i] / lvl12[worst]), 3)
        d["rmsse_xgb"] = round(float(contribs[f.name]["rmsse_xgb"][i]), 3)
        d["rmsse_classical"] = round(float(contribs[f.name]["rmsse_classical"][i]), 3)
        d["row"] = int(i)
        cases.append(d)
    result["cases"] = cases
    result["growth_and_blend"] = growth_and_blend(cv, ids, sales, prices, pred_root)
    (rep / "data" / "error_analysis.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"concentration": result["concentration"], "cases": cases}, indent=1, default=str))


if __name__ == "__main__":
    main()
