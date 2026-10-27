"""Fase 8: forecast 14 hari pada data apotek (agregat dan produk), XGBoost vs baseline.

Desain: reports/id_phase8_design.md §3, parameter: config.yaml → id_pharmacy.forecast.

- Deret agregat: nota RJ, nota RI, unit total (satu model gabungan).
- Deret produk: unit per produk, untuk produk yang terjual di >= 50% hari train fold itu.
- Fitur dihitung dari matriks yang nilainya setelah cutoff sudah diganti NaN, jadi data
  setelah cutoff tidak mungkin terpakai (lag dan rolling semuanya >= horizon).
- Varian `xgb_noholiday` = XGBoost tanpa fitur hari libur, untuk menguji fitur libur umum.

Cara pakai:
    python -m retailfc.id_forecast --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from retailfc.baselines import moving_average, seasonal_naive, statsforecast_predict
from retailfc.id_event_study import day_off_dates
from retailfc.io import load_config

HOLIDAY_FEATURES = ["day_off", "days_to_off", "days_since_off"]
AGG_SERIES = ["notes_RJ", "notes_RI", "units_all"]


def load_panels(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(agregat hari × deret, unit produk × hari, omzet produk × hari) pada periode lengkap."""
    src = Path(cfg["paths"]["interim"]) / "id_pharmacy"
    p = cfg["id_pharmacy"]
    start, end = pd.Timestamp(p["complete_start"]), pd.Timestamp(p["end"])
    days = pd.date_range(start, end, freq="D")
    ms = pd.read_parquet(src / "ms_sales.parquet", columns=["NO_RESEP", "TGL"])
    ms = ms[(ms["TGL"] >= start) & (ms["TGL"] <= end)]
    det = pd.read_parquet(src / "det_sales.parquet", columns=["NO_RESEP", "KD_OBAT", "QTY", "HJ"])
    lines = det.merge(ms, on="NO_RESEP")
    pre = ms["NO_RESEP"].str.extract(r"^([A-Z]+)")[0]
    agg = pd.DataFrame(index=days)
    for k in ("RJ", "RI"):
        agg[f"notes_{k}"] = ms[pre == k].groupby("TGL").size()
    agg["units_all"] = lines.groupby("TGL")["QTY"].sum()
    agg = agg.fillna(0.0)
    units = lines.pivot_table(index="KD_OBAT", columns="TGL", values="QTY", aggfunc="sum", fill_value=0.0)
    units = units.reindex(columns=days, fill_value=0.0).clip(lower=0.0)   # retur bersih -> 0
    rev = (lines.assign(rev=lines["QTY"] * lines["HJ"])
           .pivot_table(index="KD_OBAT", columns="TGL", values="rev", aggfunc="sum", fill_value=0.0)
           .reindex(index=units.index, columns=days, fill_value=0.0))
    return agg, units, rev


def calendar_frame(days: pd.DatetimeIndex, off: set, clip: int = 7) -> pd.DataFrame:
    is_off = days.isin(list(off))
    pos = np.arange(len(days))
    off_pos = pos[is_off]
    nxt = np.array([off_pos[off_pos >= i].min() - i if (off_pos >= i).any() else clip for i in pos])
    prv = np.array([i - off_pos[off_pos <= i].max() if (off_pos <= i).any() else clip for i in pos])
    return pd.DataFrame({"weekday": days.weekday, "day_of_month": days.day, "day_off": is_off.astype(int),
                         "days_to_off": np.minimum(nxt, clip), "days_since_off": np.minimum(prv, clip)},
                        index=days)


def build_features(Y: np.ndarray, cal: pd.DataFrame, cutoff: int, horizon: int,
                   lags: list[int], windows: list[int]) -> pd.DataFrame:
    """Baris (deret, hari) untuk hari 0..cutoff+horizon. Nilai Y setelah `cutoff` tidak dipakai."""
    assert min(lags) >= horizon, "lag harus >= horizon (tanpa prediksi rekursif)"
    n, _ = Y.shape
    last = cutoff + horizon
    H = Y[:, :cutoff + 1].astype(np.float64)
    Hpad = np.concatenate([H, np.full((n, horizon), np.nan)], axis=1)     # hari > cutoff = NaN
    cols = {}
    for lag in lags:
        sh = np.full((n, last + 1), np.nan)
        sh[:, lag:] = Hpad[:, :last + 1 - lag]
        cols[f"lag_{lag}"] = sh
    base = cols[f"lag_{min(lags)}"]
    for w in windows:
        cols[f"roll_mean_{w}"] = pd.DataFrame(base.T).rolling(w, min_periods=w).mean().to_numpy().T
    te = np.nanmean(H, axis=1)
    rows = []
    for i in range(n):
        df = pd.DataFrame({k: v[i] for k, v in cols.items()})
        df["series"] = i
        df["te_mean"] = te[i]
        df["day"] = np.arange(last + 1)
        rows.append(df)
    X = pd.concat(rows, ignore_index=True)
    calv = cal.iloc[:last + 1].reset_index(drop=True)
    return pd.concat([X, calv.loc[X["day"]].reset_index(drop=True)], axis=1)


def series_scale(Y: np.ndarray, cutoff: int) -> np.ndarray:
    """Rata-rata train tiap deret (data <= cutoff), dipakai untuk menyamakan skala antar deret."""
    s = np.nanmean(Y[:, :cutoff + 1], axis=1)
    return np.where(s > 0, s, 1.0)


def fit_predict_xgb(Y: np.ndarray, cal: pd.DataFrame, cutoff: int, fc: dict, use_holidays: bool,
                    scaled: bool = True) -> tuple[np.ndarray, int]:
    """Latih di hari < (cutoff - early_stopping_days + 1), early stopping di sisa train, prediksi horizon.

    `scaled=True`: target dan fitur dibagi rata-rata train tiap deret, prediksi dikalikan
    kembali. Tanpa ini, deret berskala besar mendominasi loss model gabungan
    (run pertama, reports/data/id_forecast_unscaled.json).
    """
    h = fc["horizon"]
    scale = series_scale(Y, cutoff) if scaled else np.ones(Y.shape[0])
    Y = Y / scale[:, None]
    X = build_features(Y, cal, cutoff, h, fc["lags"], fc["rolling_windows"])
    if scaled:
        X["te_mean"] = np.log(scale[X["series"].to_numpy()])   # level asli deret sebagai fitur
    feats = [c for c in X.columns if c != "day" and (use_holidays or c not in HOLIDAY_FEATURES)]
    valid_rows = X[feats].drop(columns=[c for c in HOLIDAY_FEATURES if c in feats]).notna().all(axis=1)
    target = np.full(len(X), np.nan)
    in_hist = X["day"].to_numpy() <= cutoff
    target[in_hist] = Y[X["series"].to_numpy()[in_hist], X["day"].to_numpy()[in_hist]]
    es_start = cutoff - fc["early_stopping_days"] + 1
    day = X["day"].to_numpy()
    tr = valid_rows.to_numpy() & (day < es_start)
    va = valid_rows.to_numpy() & (day >= es_start) & (day <= cutoff)
    te = (day > cutoff)
    params = dict(fc["params"], tree_method="hist", device="cpu", nthread=4)
    dtr = xgb.DMatrix(X.loc[tr, feats], label=target[tr])
    dva = xgb.DMatrix(X.loc[va, feats], label=target[va])
    booster = xgb.train(params, dtr, num_boost_round=fc["max_rounds"], evals=[(dva, "valid")],
                        early_stopping_rounds=fc["early_stopping"], verbose_eval=False)
    pred = booster.predict(xgb.DMatrix(X.loc[te, feats]), iteration_range=(0, booster.best_iteration + 1))
    out = np.zeros((Y.shape[0], h))
    out[X.loc[te, "series"].to_numpy(), day[te] - cutoff - 1] = pred
    return out * scale[:, None], int(booster.best_iteration + 1)


def weekday_mean(hist: np.ndarray, h: int, weeks: int = 4) -> np.ndarray:
    """Rata-rata hari dalam minggu yang sama dari `weeks` minggu terakhir."""
    last = hist[:, -7 * weeks:].reshape(hist.shape[0], weeks, 7).mean(axis=1)
    return np.tile(last, int(np.ceil(h / 7)))[:, :h]


def baselines(hist: np.ndarray, h: int, kind: str) -> dict[str, np.ndarray]:
    from statsforecast.models import TSB, AutoETS, CrostonOptimized

    out = {"snaive7": seasonal_naive(hist, h), "weekday_mean4": weekday_mean(hist, h),
           "ma28": moving_average(hist, h)}
    rows, first = np.arange(hist.shape[0]), np.zeros(hist.shape[0], dtype=int)
    if kind == "agg":
        out["ets"] = statsforecast_predict(hist, rows, first, [AutoETS(season_length=7)], h)["AutoETS"]
    else:
        sf = statsforecast_predict(hist, rows, first, [CrostonOptimized(), TSB(alpha_d=0.1, alpha_p=0.1)], h)
        out["croston"], out["tsb"] = sf["CrostonOptimized"], sf["TSB"]
    return out


def scores(actual: np.ndarray, pred: np.ndarray, hist: np.ndarray) -> dict:
    """RMSSE per deret (skala = MSE naive 7 hari di train), WAPE dan bias total."""
    scale = np.mean((hist[:, 7:] - hist[:, :-7]) ** 2, axis=1)
    mse = np.mean((pred - actual) ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmsse = np.sqrt(mse / scale)
    e = pred - actual
    return {"rmsse": rmsse, "wape": float(np.abs(e).sum() / actual.sum()), "bias": float(e.sum() / actual.sum())}


def run_fold(name: str, cutoff_date: str, agg: pd.DataFrame, units: pd.DataFrame, rev: pd.DataFrame,
             cal: pd.DataFrame, fc: dict) -> dict:
    days = agg.index
    c = int(days.get_loc(pd.Timestamp(cutoff_date)))
    h = fc["horizon"]
    res = {"cutoff": cutoff_date, "horizon": [str(days[c + 1].date()), str(days[c + h].date())]}
    # Agregat
    Ya = agg[AGG_SERIES].to_numpy().T
    preds = baselines(Ya[:, :c + 1], h, "agg")
    preds["xgb"], it = fit_predict_xgb(Ya, cal, c, fc, True)
    preds["xgb_noholiday"], it2 = fit_predict_xgb(Ya, cal, c, fc, False)
    act = Ya[:, c + 1:c + 1 + h]
    res["agg"] = {}
    for m, p in preds.items():
        res["agg"][m] = {}
        for i, s in enumerate(AGG_SERIES):
            sc = scores(act[i:i + 1], p[i:i + 1], Ya[i:i + 1, :c + 1])
            res["agg"][m][s] = {"rmsse": float(sc["rmsse"][0]), "wape": sc["wape"], "bias": sc["bias"]}
    res["agg_daily"] = {"actual": act.tolist(), **{m: p.tolist() for m, p in preds.items()}}
    # Produk: dipilih dari data train fold ini saja
    share = (units.iloc[:, :c + 1] > 0).mean(axis=1)
    sel = units.index[share >= fc["product_min_share_days"]]
    Yp = units.loc[sel].to_numpy()
    w = rev.loc[sel].iloc[:, c - 27:c + 1].sum(axis=1).clip(lower=0).to_numpy()
    w = w / w.sum()
    ppreds = baselines(Yp[:, :c + 1], h, "prod")
    ppreds["xgb"], it3 = fit_predict_xgb(Yp, cal, c, fc, True)
    ppreds["xgb_noholiday"], _ = fit_predict_xgb(Yp, cal, c, fc, False)
    pact = Yp[:, c + 1:c + 1 + h]
    res["products"] = {"n": int(len(sel))}
    for m, p in ppreds.items():
        sc = scores(pact, p, Yp[:, :c + 1])
        ok = np.isfinite(sc["rmsse"])
        tot_e = p.sum(axis=0) - pact.sum(axis=0)
        res["products"][m] = {
            "wrmsse": float(np.sum(w[ok] * sc["rmsse"][ok]) / w[ok].sum()),
            "wape_item": sc["wape"], "bias": sc["bias"],
            "wape_total": float(np.abs(tot_e).sum() / pact.sum()),
        }
    res["best_iteration"] = {"agg_xgb": it, "agg_xgb_noholiday": it2, "prod_xgb": it3}
    return res


def summarize(folds: dict, names: list[str]) -> dict:
    out = {"agg": {}, "products": {}}
    first = folds[names[0]]
    for m in first["agg"]:
        out["agg"][m] = {}
        for s in AGG_SERIES:
            v = np.array([folds[f]["agg"][m][s]["rmsse"] for f in names])
            wp = np.array([folds[f]["agg"][m][s]["wape"] for f in names])
            out["agg"][m][s] = {"rmsse_mean": float(v.mean()), "rmsse_std": float(v.std(ddof=1)),
                                "wape_mean": float(wp.mean())}
        wins = sum(all(folds[f]["agg"][m][s]["rmsse"] <= folds[f]["agg"][o][s]["rmsse"]
                       for o in first["agg"] if o != m) for f in names for s in AGG_SERIES)
        out["agg"][m]["n_best"] = int(wins)
    for m in (k for k in first["products"] if k != "n"):
        v = np.array([folds[f]["products"][m]["wrmsse"] for f in names])
        out["products"][m] = {"wrmsse_mean": float(v.mean()), "wrmsse_std": float(v.std(ddof=1)),
                              "wape_item_mean": float(np.mean([folds[f]["products"][m]["wape_item"] for f in names])),
                              "wape_total_mean": float(np.mean([folds[f]["products"][m]["wape_total"] for f in names]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fc = cfg["id_pharmacy"]["forecast"]
    agg, units, rev = load_panels(cfg)
    cal = calendar_frame(agg.index, day_off_dates(cfg))
    folds = {}
    for i, cdate in enumerate(fc["cv_cutoffs"], 1):
        folds[f"cv{i}"] = run_fold(f"cv{i}", cdate, agg, units, rev, cal, fc)
        print(f"cv{i} {cdate} selesai")
    for name, cdate in fc["extra_folds"].items():
        folds[name] = run_fold(name, cdate, agg, units, rev, cal, fc)
        print(f"{name} {cdate} selesai")
    cv = [f"cv{i}" for i in range(1, len(fc["cv_cutoffs"]) + 1)]
    out = {"xgboost_version": xgb.__version__, "cv_summary": summarize(folds, cv), "folds": folds}
    rep = Path(cfg["paths"]["reports"])
    (rep / "data" / "id_forecast.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["cv_summary"], indent=1))


if __name__ == "__main__":
    main()
