"""Fase 2: EDA terarah. Menjawab empat pertanyaan dari PROJECT_PLAN.md dengan angka.

1. Proporsi hari tanpa penjualan per kategori (setelah penjualan pertama tiap item).
2. Kekuatan pola mingguan, dan efek SNAP per negara bagian (dengan placebo non-makanan).
3. Hari toko tutup (mis. 25 Desember): diverifikasi dari penjualan, bukan diasumsikan.
4. Penjualan di sekitar Easter (hari libur bergerak), sebagai persiapan P4.

Hanya hari d <= holdout_cutoff yang dipakai, jadi holdout final tidak tersentuh.

Cara pakai:
    python -m retailfc.eda --config config.yaml
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

from retailfc.io import load_calendar, load_config

# Palet referensi (mode terang): tiga slot kategorikal pertama lolos uji buta warna
# untuk semua pasangan warna.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CATS = ["FOODS", "HOBBIES", "HOUSEHOLD"]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def daily_store_cat(long_root: Path, max_d: int) -> pd.DataFrame:
    """Per toko x kategori x hari: total unit, jumlah baris (item aktif), baris bernilai nol."""
    parts = []
    for p in sorted(long_root.glob("store_id=*/part-0.parquet")):
        store = p.parent.name.split("=", 1)[1]
        df = pd.read_parquet(p, columns=["cat_id", "d", "sales"])
        df = df[df["d"] <= max_d]
        g = df.assign(zero=(df["sales"] == 0)).groupby(["cat_id", "d"], observed=True)
        a = g.agg(units=("sales", "sum"), rows=("sales", "size"), zeros=("zero", "sum")).reset_index()
        a["store_id"] = store
        parts.append(a)
    out = pd.concat(parts, ignore_index=True)
    out["cat_id"] = out["cat_id"].astype(str)
    out["state_id"] = out["store_id"].str.slice(0, 2)
    return out


def detrended(series: pd.Series, window: int = 28) -> pd.Series:
    """Nilai harian dibagi rata-rata rolling berpusat (menghilangkan tren dan musim tahunan)."""
    return series / series.rolling(window, center=True, min_periods=window // 2).mean()


def q1_zero_share(daily: pd.DataFrame) -> dict:
    by_cat = daily.groupby("cat_id")[["zeros", "rows"]].sum()
    by_cat_store = daily.groupby(["cat_id", "store_id"])[["zeros", "rows"]].sum()
    zs = (by_cat_store["zeros"] / by_cat_store["rows"]).unstack()
    return {
        "zero_share_by_cat": (by_cat["zeros"] / by_cat["rows"]).round(4).to_dict(),
        "zero_share_all": round(float(daily["zeros"].sum() / daily["rows"].sum()), 4),
        "zero_share_by_cat_store_min": zs.min(axis=1).round(4).to_dict(),
        "zero_share_by_cat_store_max": zs.max(axis=1).round(4).to_dict(),
        "zero_share_by_cat_store": zs.round(4).to_dict(orient="index"),
    }


def q2_weekly_and_snap(daily: pd.DataFrame, cal: pd.DataFrame, closed: set[int]) -> tuple[dict, pd.DataFrame]:
    c = cal.set_index("d_num")
    out: dict = {"weekday_index": {}, "weekday_max_over_min": {}, "snap": {}}

    tot = daily[~daily["d"].isin(closed)].groupby(["cat_id", "d"])["units"].sum().unstack(0)
    ratio = tot.apply(detrended)
    wd = c.loc[ratio.index, "weekday"]
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    prof = ratio.groupby(wd.values).mean().reindex(order)
    prof = prof / prof.mean()
    for cat in CATS:
        out["weekday_index"][cat] = prof[cat].round(3).to_dict()
        out["weekday_max_over_min"][cat] = round(float(prof[cat].max() / prof[cat].min()), 3)

    st = daily[~daily["d"].isin(closed)].groupby(["state_id", "cat_id", "d"])["units"].sum()
    for (state, cat), s in st.groupby(level=[0, 1]):
        s = s.droplevel([0, 1])
        r = detrended(s).dropna()
        snap = c.loc[r.index, f"snap_{state}"].to_numpy() == 1
        wdays = c.loc[r.index, "wday"].to_numpy()
        # Disesuaikan per hari: bandingkan SNAP vs non-SNAP di hari yang sama, lalu rata-ratakan.
        lifts = []
        for w in range(1, 8):
            m = wdays == w
            if (m & snap).any() and (m & ~snap).any():
                lifts.append(r[m & snap].mean() / r[m & ~snap].mean() - 1)
        out["snap"].setdefault(state, {})[cat] = round(float(np.mean(lifts)), 4)
    out["snap_days_of_month"] = {
        s: sorted(cal.loc[cal[f"snap_{s}"] == 1, "date"].dt.day.unique().tolist()) for s in ["CA", "TX", "WI"]
    }
    return out, prof


def q3_closed_days(daily: pd.DataFrame, cal: pd.DataFrame) -> tuple[dict, set[int]]:
    c = cal.set_index("d_num")
    per_store = daily.groupby(["store_id", "d"])["units"].sum().unstack(0)
    total = per_store.sum(axis=1)
    zero_all = total.index[total == 0].tolist()
    store_zero = {s: per_store.index[per_store[s] == 0].tolist() for s in per_store.columns}
    # Hari ketika seluruh jaringan menjual < 10% rata-rata 28 hari berpusat: hampir tutup.
    rel = detrended(total)
    near = rel.index[(rel < 0.10) & (total > 0)].tolist()

    def label(ds):
        return [f"d_{d} {c.loc[d, 'date'].date()} {c.loc[d, 'weekday']}" for d in ds]

    christmas = cal.loc[(cal["date"].dt.month == 12) & (cal["date"].dt.day == 25), "d_num"].tolist()
    out = {
        "days_zero_all_stores": label(zero_all),
        "christmas_days_in_range": label([d for d in christmas if d in total.index]),
        "christmas_total_units": {str(c.loc[d, "date"].date()): int(total.loc[d]) for d in christmas if d in total.index},
        "days_below_10pct_of_rolling_mean_but_nonzero": label(near),
        "store_days_with_zero_total": {s: label(v) for s, v in store_zero.items() if v},
    }
    # Hari yang benar-benar nol maupun hampir nol dianggap tutup untuk analisis berikutnya.
    return out, set(zero_all) | set(near)


def q4_easter(daily: pd.DataFrame, cal: pd.DataFrame, max_d: int, before: int = 14, after: int = 7) -> tuple[dict, pd.DataFrame]:
    """Unit per kategori di setiap hari sekitar Easter, relatif terhadap hari yang sama
    2-5 minggu sebelum dan sesudahnya (jendela event sendiri tidak ikut dihitung)."""
    tot = daily.groupby(["cat_id", "d"])["units"].sum().unstack(0)
    easter = cal.loc[(cal["event_name_1"] == "Easter") | (cal["event_name_2"] == "Easter"), "d_num"]
    easter = [int(d) for d in easter if d + after + 35 <= max_d]
    rows = []
    for e in easter:
        for off in range(-before, after + 1):
            d = e + off
            base_days = [d + 7 * k for k in (-5, -4, -3, -2, 2, 3, 4, 5)]
            base = tot.loc[[b for b in base_days if b in tot.index]].mean()
            for cat in CATS:
                rows.append({"easter_d": e, "year": int(cal.loc[cal["d_num"] == e, "year"].iloc[0]),
                             "offset": off, "cat_id": cat, "ratio": tot.loc[d, cat] / base[cat]})
    r = pd.DataFrame(rows)
    mean = r.groupby(["cat_id", "offset"])["ratio"].mean().unstack(0)
    out = {
        "easter_years_used": sorted(r["year"].unique().tolist()),
        "excluded_easter_reason": f"needs d+{after}+35 <= d_{max_d} for the baseline",
        "mean_ratio_by_offset": mean.round(3).to_dict(),
        "peak_offset": {cat: int(mean[cat].idxmax()) for cat in CATS},
        "peak_ratio": {cat: round(float(mean[cat].max()), 3) for cat in CATS},
        "easter_day_ratio": {cat: round(float(mean.loc[0, cat]), 3) for cat in CATS},
    }
    return out, r


def q5_data_quality(long_root: Path, raw: Path, cal: pd.DataFrame, cfg: dict, max_d: int) -> dict:
    """Kualitas data: harga anomali dan deretan nol panjang (indikasi stok habis)."""
    from retailfc.features import zero_run_length
    from retailfc.folds import closed_day_mask
    from retailfc.io import load_prices

    # Harga: hanya minggu sampai minggu dari d_max (holdout tidak disentuh).
    max_wk = int(cal.loc[cal["d_num"] == max_d, "wm_yr_wk"].iloc[0])
    pr = load_prices(raw)
    pr = pr[pr["wm_yr_wk"] <= max_wk]
    med = pr.groupby(["store_id", "item_id"], observed=True)["sell_price"].transform("median")
    ratio = pr["sell_price"] / med
    low = pr["sell_price"] < 0.10
    ex = (pr[low].groupby("item_id", observed=True)["sell_price"].agg(["min", "size"])
          .join(pr.groupby("item_id", observed=True)["sell_price"].median().rename("median"))
          .sort_values("size", ascending=False).head(8))
    prices = {
        "rows_checked": int(len(pr)),
        "rows_below_0_10": int(low.sum()),
        "items_below_0_10": int(pr.loc[low, "item_id"].nunique()),
        "rows_below_20pct_of_item_store_median": int((ratio < 0.2).sum()),
        "rows_above_5x_item_store_median": int((ratio > 5).sum()),
        "examples_below_0_10": {i: {"min": round(float(r["min"]), 2), "weeks": int(r["size"]),
                                    "median": round(float(r["median"]), 2)} for i, r in ex.iterrows()},
    }

    # Deretan nol: dihitung per deret item-toko, hari tutup tidak ikut dihitung.
    per_series, zero_rows, zero_rows_long, zero_rows_busy, zero_rows_busy_long = [], 0, 0, 0, 0
    for p in sorted(long_root.glob("store_id=*/part-0.parquet")):
        df = pd.read_parquet(p, columns=["item_id", "d", "sales"])
        df = df[df["d"] <= max_d].reset_index(drop=True)
        closed = closed_day_mask(df["d"].to_numpy(), cal, cfg)
        run = zero_run_length(df, closed)
        item = df["item_id"].cat.codes.to_numpy()
        new_item = np.r_[True, item[1:] != item[:-1]]
        grp = np.cumsum(((df["sales"].to_numpy() > 0) & ~closed) | new_item)
        run_total = pd.Series(run).groupby(grp).transform("max").to_numpy()   # panjang deretan penuh
        is_zero = (df["sales"].to_numpy() == 0) & ~closed
        open_day = ~closed
        g = pd.DataFrame({"item": item, "run": run, "zero": is_zero, "open": open_day})
        s = g.groupby("item", sort=False).agg(longest=("run", "max"), trailing=("run", "last"),
                                              zeros=("zero", "sum"), open_days=("open", "sum"))
        s["sell_share"] = 1 - s["zeros"] / s["open_days"]
        per_series.append(s)
        busy_item = s["sell_share"].reindex(item).to_numpy() >= 0.5
        zero_rows += int(is_zero.sum())
        zero_rows_long += int((is_zero & (run_total >= 28)).sum())
        zero_rows_busy += int((is_zero & busy_item).sum())
        zero_rows_busy_long += int((is_zero & busy_item & (run_total >= 28)).sum())
    t = pd.concat(per_series, ignore_index=True)
    busy = t["sell_share"] >= 0.5
    runs = {
        "series": int(len(t)),
        "busy_series_sell_share_ge_0_5": int(busy.sum()),
        **{f"busy_series_with_zero_run_ge_{L}": int((busy & (t["longest"] >= L)).sum()) for L in (28, 56, 90)},
        "all_series_with_zero_run_ge_90": int((t["longest"] >= 90).sum()),
        "series_zero_for_last_28plus_days": int((t["trailing"] >= 28).sum()),
        "busy_series_zero_for_last_28plus_days": int((busy & (t["trailing"] >= 28)).sum()),
        "share_of_zero_days_inside_runs_ge_28": round(zero_rows_long / zero_rows, 4),
        "share_of_zero_days_inside_runs_ge_28_busy_series": round(zero_rows_busy_long / zero_rows_busy, 4),
    }
    return {"prices": prices, "zero_runs": runs}


def fig_weekday(prof: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(7)
    for i, cat in enumerate(CATS):
        ax.plot(x, prof[cat].to_numpy(), color=SERIES[i], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=cat)
    # Label langsung di titik terakhir, digeser agar tidak saling menumpuk.
    ends = sorted(((prof[c].iloc[6], c) for c in CATS))
    min_gap, placed = 0.03, []
    for y, cat in ends:
        y_lab = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_lab)
        ax.annotate(cat, (6, y), xytext=(6.25, y_lab), textcoords="data", va="center",
                    fontsize=9, color=INK_2)
    ax.axhline(1.0, color=BASE, linewidth=1)
    ax.set_xticks(x, ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"])
    ax.set_xlim(-0.3, 7.6)
    ax.set_ylabel("Unit vs. rata-rata mingguan", color=INK_2, fontsize=9)
    ax.set_title("Penjualan memuncak di akhir pekan; HOBBIES turun lagi di hari Minggu",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_easter(r: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=150, sharey=True, facecolor=SURFACE)
    for ax, (i, cat) in zip(axes, enumerate(CATS)):
        style_axes(ax)
        sub = r[r["cat_id"] == cat]
        for _, y in sub.groupby("year"):
            ax.plot(y["offset"], y["ratio"], color=SERIES[i], alpha=0.25, linewidth=1)
        m = sub.groupby("offset")["ratio"].mean()
        ax.plot(m.index, m.to_numpy(), color=SERIES[i], linewidth=2)
        ax.axhline(1.0, color=BASE, linewidth=1)
        ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(cat, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("Hari dari Minggu Easter", color=INK_2, fontsize=9)
    axes[0].set_ylabel("Unit vs. hari sama, ±2–5 minggu", color=INK_2, fontsize=9)
    fig.suptitle("Penjualan di sekitar Easter (tipis: per tahun, tebal: rata-rata)", x=0.01,
                 ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    max_d = cfg["validation"]["holdout_cutoff"]
    cal = load_calendar(Path(cfg["paths"]["raw_m5"]))
    reports = Path(cfg["paths"]["reports"])
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    daily = daily_store_cat(Path(cfg["paths"]["interim"]) / "sales_long", max_d)
    daily.to_parquet(Path(cfg["paths"]["interim"]) / "daily_store_cat.parquet", index=False)

    out: dict = {"days_used": f"d_1..d_{max_d}"}
    out["q1_zero_share"] = q1_zero_share(daily)
    out["q3_closed_days"], closed = q3_closed_days(daily, cal)
    out["q2_weekly_snap"], prof = q2_weekly_and_snap(daily, cal, closed)
    out["q4_easter"], easter = q4_easter(daily, cal, max_d)
    out["q5_data_quality"] = q5_data_quality(Path(cfg["paths"]["interim"]) / "sales_long",
                                             Path(cfg["paths"]["raw_m5"]), cal, cfg, max_d)

    fig_weekday(prof, figs / "eda_weekday.png")
    fig_easter(easter, figs / "eda_easter.png")
    (reports / "data" / "eda.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
