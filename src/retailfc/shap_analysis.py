"""Fase 6: SHAP untuk model point XGBoost (kepentingan fitur global, per segmen, studi kasus).

Langkah:
1. Latih ulang model point (konfigurasi final `config.yaml → model.xgb_point`) untuk 10 toko
   di fold cv4, ditambah model untuk studi kasus di fold lain (`CASE_MODELS`). Model disimpan
   ke data/processed/models/xgb_point/<fold>/<store>.ubj (tidak di-track git).
2. Bukti setara: prediksi hasil training ulang harus identik dengan prediksi Fase 5 di
   data/processed/preds/<fold>/xgb_point.npy.
3. SHAP dengan TreeSHAP bawaan XGBoost (`pred_contribs=True`) untuk semua baris horizon
   (±85 ribu per toko), memakai pohon sampai iterasi terbaik.

SHAP dihitung di **CPU**. Di xgboost 3.4.1, `pred_contribs` di GPU memberi hasil yang tidak
aditif untuk model dengan fitur kategorikal (diuji pada CA_4 cv4: jumlah SHAP vs margin
berselisih hingga 0,57, pada 98% baris; di CPU selisih maksimum 5e-6). Setiap toko dicek:
selisih jumlah SHAP vs margin harus < 1e-3, kalau tidak, skrip berhenti.

Objective tweedie memakai link log, jadi nilai SHAP adalah kontribusi ke **log** forecast.
Kepentingan fitur = rata-rata |SHAP|. Fitur harga dibaca sebagai asosiasi, bukan elastisitas.

Cara pakai:
    python -m retailfc.shap_analysis --config config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from retailfc.features import add_fold_features, all_feature_names, to_model_frame
from retailfc.folds import build_folds
from retailfc.io import load_config, load_wide_matrices
from retailfc.train import TrainOptions, fit_predict, to_wide
from retailfc.xgb_final import model_params

MAIN_FOLD = "cv4"
# Studi kasus Fase 6 (reports/error_analysis.md §7) yang fold terburuknya bukan cv4.
CASE_MODELS = [("WI_2", "cv1"), ("TX_2", "cv1")]
CASES = [("FOODS_1_096", "WI_2", "cv1"), ("HOBBIES_1_354", "TX_2", "cv1")]


def horizon_frame(base: pd.DataFrame, fold, feats: list[str], encoding: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matriks fitur dan metadata baris horizon, dengan aturan yang sama seperti `fit_predict`."""
    df = add_fold_features(base, fold.cutoff, fold.horizon)
    hz = fold.horizon_mask(df["d"].to_numpy())
    first_d = df.groupby("item_id", observed=True)["d"].transform("min").to_numpy()
    hz &= first_d <= fold.cutoff
    meta = df.loc[hz, ["item_id", "d", "sales"]].reset_index(drop=True)
    meta["item_id"] = meta["item_id"].astype(str)
    return to_model_frame(df, feats, encoding)[hz].reset_index(drop=True), meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stores", nargs="*", default=None, help="subset toko (untuk uji cepat)")
    ap.add_argument("--out", default="shap_summary.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mcfg = cfg["model"]["xgb_point"]
    params, base_opt = model_params(mcfg)
    folds = {f.name: f for f in build_folds(cfg)}
    ids, _, _ = load_wide_matrices(cfg, cfg["validation"]["holdout_cutoff"])
    stores = args.stores or sorted(ids["store_id"].astype(str).unique())
    proc = Path(cfg["paths"]["processed"])
    models_root = proc / "models" / "xgb_point"
    preds_root = proc / "preds"
    seg = {f: pd.read_parquet(preds_root / f / "segments.parquet").set_index("id")["segment"].astype(str)
           for f in {MAIN_FOLD} | {c[1] for c in CASE_MODELS}}

    jobs = [(s, MAIN_FOLD) for s in stores] + CASE_MODELS
    feat_names: list[str] = []
    rows_abs, rows_signed, per_series = [], [], []
    checks, cases = {}, {}
    for store in stores:
        mine = [fn for s, fn in jobs if s == store]
        if not mine:
            continue
        base = pd.read_parquet(proc / "base" / f"store_id={store}" / "part-0.parquet")
        feats = all_feature_names(base, drop=tuple(mcfg.get("drop_features", [])))
        feat_names = feats
        for fname in mine:
            f = folds[fname]
            path = models_root / fname / f"{store}.ubj"
            info_path = path.with_suffix(".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                opt = TrainOptions(**{**base_opt.__dict__, "save_model": str(path)})
                p, info = fit_predict(base, f, feats, params, options=opt)
                # Bukti setara dengan prediksi Fase 5.
                saved = np.load(preds_root / fname / "xgb_point.npy")
                mine_rows = (ids["store_id"].astype(str) == store).to_numpy()
                now = np.zeros_like(saved)
                to_wide(p, ids, store, f, now)
                info["max_abs_diff_vs_phase5"] = float(np.abs(now[mine_rows] - saved[mine_rows]).max())
                info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
            info = json.loads(info_path.read_text(encoding="utf-8"))
            checks[f"{store}_{fname}"] = {"best_iteration": info["best_iteration"],
                                          "max_abs_diff_vs_phase5": info["max_abs_diff_vs_phase5"]}
            print(f"{store} {fname} best={info['best_iteration']} diff={info['max_abs_diff_vs_phase5']:.2e}",
                  flush=True)

            X, meta = horizon_frame(base, f, feats, base_opt.item_encoding)
            booster = xgb.Booster(model_file=str(path))
            booster.set_param({"device": "cpu"})   # GPU tidak aditif untuk fitur kategorikal
            dm = xgb.DMatrix(X, enable_categorical=True)
            rng = (0, info["best_iteration"] + 1)
            contrib = booster.predict(dm, pred_contribs=True, iteration_range=rng)
            margin = booster.predict(dm, output_margin=True, iteration_range=rng)
            pred = booster.predict(dm, iteration_range=rng)
            additivity = float(np.abs(contrib.sum(axis=1) - margin).max())
            checks[f"{store}_{fname}"]["shap_sum_vs_margin_max_abs"] = additivity
            if additivity > 1e-3:
                raise RuntimeError(f"{store} {fname}: SHAP not additive (max diff {additivity:.3g})")
            sv = contrib[:, :-1]

            meta["id"] = meta["item_id"] + f"_{store}_evaluation"
            meta["segment"] = meta["id"].map(seg[fname]).fillna("no_history")
            meta["pred"] = pred
            if fname == MAIN_FOLD:
                # Kolom kelompok diberi awalan "_" karena `cat_id` juga nama fitur.
                keys = ["_segment", "_cat"]
                df_abs = pd.DataFrame(np.abs(sv), columns=feats).assign(
                    _segment=meta["segment"].to_numpy(), _cat=meta["item_id"].str.split("_").str[0].to_numpy())
                rows_abs.append(df_abs.groupby(keys)[feats].sum().assign(n=df_abs.groupby(keys).size()))
                df_s = pd.DataFrame(sv, columns=feats).assign(_segment=meta["segment"].to_numpy())
                rows_signed.append(df_s.groupby("_segment")[feats].sum().assign(n=df_s.groupby("_segment").size()))
                # Per deret: bias dan rata-rata SHAP (untuk mencari pendorong bias).
                g = pd.DataFrame(sv, columns=feats).assign(_id=meta["id"].to_numpy()).groupby("_id")[feats].mean()
                g["bias"] = (meta["pred"] - meta["sales"]).groupby(meta["id"]).mean()
                g["actual_mean"] = meta["sales"].groupby(meta["id"]).mean()
                g["_segment"] = meta.groupby("id")["segment"].first()
                per_series.append(g)
            for item, cs, cf in CASES:
                if cs == store and cf == fname:
                    m = (meta["item_id"] == item).to_numpy()
                    cases[f"{item}_{cs}_{cf}"] = {
                        "mean_shap": dict(zip(feats, np.round(sv[m].mean(axis=0), 4).tolist())),
                        "base_value": round(float(contrib[m, -1].mean()), 4),
                        "feature_mean": {c: round(float(pd.to_numeric(X.loc[m, c], errors="coerce").mean()), 3)
                                         for c in feats if pd.api.types.is_numeric_dtype(X[c])},
                        "pred_mean": round(float(pred[m].mean()), 2), "actual_mean": round(float(meta.loc[m, "sales"].mean()), 2)}
        del base

    # Agregasi (fold cv4, 10 toko).
    A = pd.concat(rows_abs).groupby(level=[0, 1]).sum()
    total = A[feat_names].sum()
    n_all = A["n"].sum()
    out: dict = {"fold": MAIN_FOLD, "n_rows": int(n_all), "checks": checks, "link": "log (reg:tweedie)"}
    out["global_mean_abs"] = (total / n_all).sort_values(ascending=False).round(5).to_dict()
    by_seg = A.groupby(level=0).sum()
    out["segment_mean_abs"] = {s: (by_seg.loc[s, feat_names] / by_seg.loc[s, "n"]).sort_values(ascending=False)
                               .round(5).to_dict() for s in by_seg.index}
    out["segment_n_rows"] = by_seg["n"].astype(int).to_dict()
    by_cat = A.groupby(level=1).sum()
    out["category_mean_abs"] = {c: (by_cat.loc[c, feat_names] / by_cat.loc[c, "n"]).sort_values(ascending=False)
                                .round(5).to_dict() for c in by_cat.index}
    S = pd.concat(rows_signed).groupby(level=0).sum()
    out["segment_mean_signed"] = {s: (S.loc[s, feat_names] / S.loc[s, "n"]).round(5).to_dict() for s in S.index}
    P = pd.concat(per_series)
    out["bias_drivers"] = {}
    for s in ["smooth", "erratic", "intermittent", "lumpy"]:
        q = P[P["_segment"] == s]
        corr = q[feat_names].corrwith(q["bias"], method="spearman").dropna()
        out["bias_drivers"][s] = {"n_series": int(len(q)), "bias_mean": round(float(q["bias"].mean()), 4),
                                  "spearman_top": corr.reindex(corr.abs().sort_values(ascending=False).index)
                                  .head(8).round(3).to_dict()}
    out["cases"] = cases
    rep = Path(cfg["paths"]["reports"])
    (rep / args.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"checks": checks, "global_top10": dict(list(out["global_mean_abs"].items())[:10]),
                      "bias_drivers": out["bias_drivers"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
