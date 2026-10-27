"""Fase 8 langkah 1: audit kelayakan data apotek Indonesia.

Membaca tabel hasil `retailfc.id_pharmacy` (data/interim/id_pharmacy/*.parquet) dan
menulis angka audit ke reports/data/id_data_audit.json serta satu grafik harian.

Tanggal nota diambil dari header (`ms_sales.TGL`). Baris detail tanpa header tidak
bisa diberi tanggal sehingga tidak dipakai untuk deret harian; jumlahnya dilaporkan.

Libur nasional di audit ini berasal dari paket `holidays` (ID) dan BELUM diverifikasi
terhadap SKB; verifikasi dilakukan saat membuat data/external/id_calendar.csv.

Cara pakai:
    python -m retailfc.id_audit --config config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import holidays
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retailfc.eda import GRID, INK, INK_2, MUTED, SERIES, SURFACE, style_axes
from retailfc.io import load_config
from retailfc.segments import classify

MONTH_TAG = re.compile(r"-(\d\d)\.(\d{4})-")
EST_ROWS = re.compile(r"^-- Dumping data for table \w+\.(\w+): ~([\d,]+) rows")
WEEKDAYS = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def header_estimates(sql: Path) -> dict[str, int]:
    """Perkiraan jumlah baris yang ditulis HeidiSQL di komentar dump (bukan hitungan pasti)."""
    out = {}
    with open(sql, encoding="utf-8") as f:
        for line in f:
            m = EST_ROWS.match(line)
            if m:
                out[m.group(1)] = int(m.group(2).replace(",", ""))
    return out


def month_tag(no_resep: pd.Series) -> pd.Series:
    """Bulan-tahun yang tertulis di nomor nota, mis. 'RJ-01.2015-08-0001' -> '2015-01'."""
    m = no_resep.str.extract(MONTH_TAG)
    return (m[1] + "-" + m[0]).where(m[0].notna())


def header_coverage(det: pd.DataFrame, ms: pd.DataFrame) -> pd.DataFrame:
    """Per bulan di nomor nota: jumlah nota detail yang punya / tidak punya header."""
    notes = det[["NO_RESEP"]].drop_duplicates()
    notes["tag"] = month_tag(notes["NO_RESEP"]).fillna("tanpa_tag")
    notes["has_header"] = notes["NO_RESEP"].isin(ms["NO_RESEP"])
    g = notes.groupby(["tag", "has_header"]).size().unstack(fill_value=0)
    g = g.reindex(columns=[True, False], fill_value=0)
    g.columns = ["with_header", "without_header"]
    g["missing_share"] = g["without_header"] / g.sum(axis=1)
    return g


def complete_period(cov: pd.DataFrame, ms: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Periode sejak bulan pertama yang semua notanya punya header, sampai tanggal terakhir."""
    tagged = cov.drop(index="tanpa_tag", errors="ignore").sort_index()
    bad = tagged.index[tagged["without_header"] > 0]
    later = tagged.index[tagged.index > bad.max()] if len(bad) else tagged.index
    start = pd.Timestamp(later.min() + "-01")
    return start, ms["TGL"].max()


def transaction_vs_join(det: pd.DataFrame, ms: pd.DataFrame, tr: pd.DataFrame) -> dict:
    """Bandingkan tabel `transaction` (versi penulis) dengan det_sales ⋈ ms_sales."""
    key = ["NO_RESEP", "KD_OBAT", "QTY", "HNA", "HJ"]
    j = det.merge(ms[["NO_RESEP"]], on="NO_RESEP")
    j = j.assign(k=j.groupby(key).cumcount())
    t = tr.assign(k=tr.groupby(key).cumcount())
    m = j.merge(t[key + ["k"]].assign(in_tr=True), on=key + ["k"], how="outer")
    only_tr = int(m["PPN_JUAL"].isna().sum())
    dropped = m[m["in_tr"].isna()]
    x = tr[["NO_RESEP", "TGL"]].drop_duplicates().merge(ms[["NO_RESEP", "TGL"]], on="NO_RESEP",
                                                       suffixes=("_tr", "_ms"))
    diff = (x["TGL_tr"] - x["TGL_ms"]).dt.days
    return {
        "join_rows": len(j), "transaction_rows": len(tr),
        "rows_only_in_join": len(dropped), "rows_only_in_transaction": only_tr,
        "dropped_with_qty_le_0": int((dropped["QTY"] <= 0).sum()),
        "dropped_ppn_share_zero": float((dropped["PPN_JUAL"] == 0).mean()),
        "kept_ppn_share_zero": float((m.loc[m["in_tr"].notna(), "PPN_JUAL"] == 0).mean()),
        "notes_with_other_date": int((diff != 0).sum()),
        "date_diff_days_abs_median": float(diff[diff != 0].abs().median()) if (diff != 0).any() else 0.0,
    }


def quality(det: pd.DataFrame, ms: pd.DataFrame, pr: pd.DataFrame) -> dict:
    return {
        "det_exact_duplicate_rows": int(det.duplicated().sum()),
        "det_qty_negative": int((det["QTY"] < 0).sum()),
        "det_qty_zero": int((det["QTY"] == 0).sum()),
        "det_qty_non_integer": int((det["QTY"] % 1 != 0).sum()),
        "det_price_zero": int((det["HJ"] == 0).sum()),
        "det_product_not_in_master": int((~det["KD_OBAT"].isin(pr["KD_OBAT"])).sum()),
        "ms_duplicate_notes": int(ms["NO_RESEP"].duplicated().sum()),
        "ms_notes_without_month_tag": int(month_tag(ms["NO_RESEP"]).isna().sum()),
        "ms_month_tag_mismatch_date": int(
            (month_tag(ms["NO_RESEP"]).dropna() != ms["TGL"].dt.strftime("%Y-%m")[month_tag(ms["NO_RESEP"]).notna()]).sum()),
        "master_products": len(pr),
        "master_products_never_sold": int((~pr["KD_OBAT"].isin(det["KD_OBAT"])).sum()),
        "master_price_zero": int((pr["HJ_RP"] == 0).sum()),
        "note_prefix_counts": ms["NO_RESEP"].str.extract(r"^([A-Z]+)")[0].value_counts().to_dict(),
        "reg_as_counts": ms["REG_AS"].value_counts(dropna=False).to_dict(),
        "racik_counts": ms["RACIK"].value_counts(dropna=False).to_dict(),
        "customer_umum_share": float((ms["KD_CUST"] == "UMUM").mean()),
    }


def daily_tables(lines: pd.DataFrame, start, end) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Total harian dan matriks unit produk × hari pada [start, end]."""
    days = pd.date_range(start, end, freq="D")
    sub = lines[(lines["TGL"] >= start) & (lines["TGL"] <= end)]
    daily = pd.DataFrame({
        "notes": sub.groupby("TGL")["NO_RESEP"].nunique(),
        "lines": sub.groupby("TGL").size(),
        "units": sub.groupby("TGL")["QTY"].sum(),
        "revenue": (sub["QTY"] * sub["HJ"]).groupby(sub["TGL"]).sum(),
    }).reindex(days, fill_value=0)
    units = sub.pivot_table(index="KD_OBAT", columns="TGL", values="QTY", aggfunc="sum", fill_value=0)
    return daily, units.reindex(columns=days, fill_value=0)


def product_stats(units: pd.DataFrame, lines: pd.DataFrame, start, end) -> dict:
    n_days = units.shape[1]
    sold_days = (units > 0).sum(axis=1)
    share = sold_days / n_days
    sub = lines[(lines["TGL"] >= start) & (lines["TGL"] <= end)]
    rev = (sub["QTY"] * sub["HJ"]).groupby(sub["KD_OBAT"]).sum().sort_values(ascending=False)
    cum = rev.cumsum() / rev.sum()
    seg = classify(units.to_numpy(np.float64))
    return {
        "days": n_days,
        "products_sold": int((sold_days > 0).sum()),
        "products_by_share_of_days_sold": {
            f">={int(t * 100)}%": int((share >= t).sum()) for t in (0.1, 0.25, 0.5, 0.8)},
        "zero_day_share_all_products": float(1 - (units > 0).to_numpy().mean()),
        "zero_day_share_median_product": float(1 - share[sold_days > 0].median()),
        "products_for_80pct_revenue": int((cum < 0.8).sum() + 1),
        "segments": seg["segment"].value_counts().reindex(
            ["smooth", "erratic", "intermittent", "lumpy"]).astype(int).to_dict(),
    }


def notes_by_prefix(ms: pd.DataFrame, start, end) -> pd.DataFrame:
    """Jumlah nota per hari × jenis nota (huruf awal nomor nota)."""
    sub = ms[(ms["TGL"] >= start) & (ms["TGL"] <= end)]
    pre = sub["NO_RESEP"].str.extract(r"^([A-Z]+)")[0]
    out = sub.groupby([sub["TGL"], pre]).size().unstack(fill_value=0)
    return out.reindex(pd.date_range(start, end, freq="D"), fill_value=0)


def sequence_vs_date(ms: pd.DataFrame) -> dict:
    """Apakah nomor urut nota naik searah tanggal? Korelasi Spearman per (jenis, bulan, unit)."""
    m = ms["NO_RESEP"].str.extract(r"^([A-Z]+)-(\d\d)\.(\d{4})-(\d+)-(\d+)$")
    ok = ms.assign(pre=m[0], tag=m[1], unit=m[3], seq=pd.to_numeric(m[4])).dropna(subset=["seq"])
    rho = []
    for _, g in ok.groupby(["pre", "tag", "unit"]):
        if len(g) > 20:
            rho.append(g["seq"].corr(g["TGL"].rank(), method="spearman"))
    rho = pd.Series(rho).dropna()
    return {"notes_parsed": len(ok), "groups": len(rho), "spearman_median": float(rho.median()),
            "share_groups_below_0_95": float((rho < 0.95).mean())}


def holiday_table(start, end, daily: pd.DataFrame) -> list[dict]:
    """Libur nasional + cuti bersama (paket `holidays`) dan jumlah nota pada hari itu."""
    years = range(start.year, end.year + 1)
    h = dict(holidays.country_holidays("ID", years=years))
    h.update(holidays.country_holidays("ID", years=years, categories=("government",)))
    wd_median = daily["notes"].groupby(daily.index.weekday).median()
    out = []
    for d, name in sorted(h.items()):
        ts = pd.Timestamp(d)
        if start <= ts <= end:
            out.append({"date": str(d), "name": name, "notes": int(daily.at[ts, "notes"]),
                        "weekday_median_notes": float(wd_median[ts.weekday()])})
    return out


def fig_daily(daily_full: pd.DataFrame, start, hol: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=150, facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    s = daily_full["notes"]
    ax.axvspan(s.index.min().to_pydatetime(), (start - pd.offsets.Day(1)).to_pydatetime(),
               color=GRID, alpha=0.6, linewidth=0)
    ax.plot(s.index, s.values, color=MUTED, linewidth=0.7)
    ax.plot(s.index, s.rolling(7, center=True).mean().values, color=SERIES[0], linewidth=1.8,
            label="rata-rata 7 hari")
    for d in hol:
        if "Idul Fitri" in d["name"] and "Cuti" not in d["name"] and "kedua" not in d["name"]:
            ts = pd.Timestamp(d["date"])
            ax.axvline(ts, color=SERIES[1], linewidth=1.2, linestyle="--")
            ax.annotate("Idulfitri", (ts, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                        fontsize=7.5, color=SERIES[1])
    ax.annotate("header nota\nbanyak hilang", (s.index.min(), ax.get_ylim()[1]), xytext=(4, -22),
                textcoords="offset points", fontsize=7.5, color=INK_2)
    ax.set_ylabel("nota per hari", color=INK_2, fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK_2)
    fig.suptitle("Data apotek: satu tahun (2015), deret lengkap hanya April–Desember",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def run(cfg: dict) -> dict:
    raw = Path(cfg["paths"]["raw_id_pharmacy"])
    src = Path(cfg["paths"]["interim"]) / "id_pharmacy"
    rep = Path(cfg["paths"]["reports"])
    t = {n: pd.read_parquet(src / f"{n}.parquet") for n in ("det_sales", "ms_sales", "ms_product", "transaction")}
    det, ms, pr, tr = t["det_sales"], t["ms_sales"], t["ms_product"], t["transaction"]

    cov = header_coverage(det, ms)
    start, end = complete_period(cov, ms)
    lines = det.merge(ms[["NO_RESEP", "TGL", "JAM_JUAL"]], on="NO_RESEP")
    daily_full, _ = daily_tables(lines, ms["TGL"].min(), end)
    daily, units = daily_tables(lines, start, end)
    hol = holiday_table(ms["TGL"].min(), end, daily_full)
    hours = pd.to_timedelta(ms["JAM_JUAL"]).dt.components["hours"]

    idul = sorted({h["date"][:4] for h in hol if h["name"] == "Hari Raya Idul Fitri"})
    by_pre = notes_by_prefix(ms, start, end)
    idul_dates = [pd.Timestamp(h["date"]) for h in hol if h["name"] == "Hari Raya Idul Fitri"]
    idul_window = {}
    for d0 in idul_dates:
        w = by_pre.loc[d0 - pd.offsets.Day(7): d0 + pd.offsets.Day(21)]
        idul_window[str(d0.date())] = {
            str(d.date()): {"weekday": WEEKDAYS[d.weekday()], **{k: int(v) for k, v in row.items()}}
            for d, row in w.iterrows()}
    years = (end - ms["TGL"].min()).days / 365.25
    result = {
        "source": {
            "doi": "10.17632/2ym7v78wtd.1", "license": "CC BY 4.0",
            "file": "sales.zip", "bytes": (raw / "sales.zip").stat().st_size,
            "sha256_zip": sha256(raw / "sales.zip"), "sha256_sql": sha256(raw / "sales.sql"),
        },
        "tables": {n: {"rows": len(d), "header_estimate": header_estimates(raw / "sales.sql").get(n)}
                   for n, d in t.items()},
        "period": {
            "first_date": str(ms["TGL"].min().date()), "last_date": str(end.date()),
            "distinct_dates": int(ms["TGL"].nunique()), "years_span": round(years, 2),
            "complete_start": str(start.date()), "complete_days": int((end - start).days + 1),
        },
        "header_coverage_by_month": cov.reset_index().to_dict(orient="records"),
        "det_rows_without_header": int((~det["NO_RESEP"].isin(ms["NO_RESEP"])).sum()),
        "transaction_vs_join": transaction_vs_join(det, ms, tr),
        "quality": quality(det, ms, pr),
        "daily_complete_period": {
            c: {"min": float(daily[c].min()), "median": float(daily[c].median()), "max": float(daily[c].max())}
            for c in daily.columns},
        "days_without_notes_complete_period": int((daily["notes"] == 0).sum()),
        "notes_by_weekday_mean": {WEEKDAYS[k]: float(v) for k, v in
                                  daily["notes"].groupby(daily.index.weekday).mean().items()},
        "notes_by_weekday_and_prefix_mean": {
            WEEKDAYS[k]: {c: round(float(v), 1) for c, v in row.items()}
            for k, row in by_pre.groupby(by_pre.index.weekday).mean().iterrows()},
        "notes_share_outside_07_21": float(((hours < 7) | (hours >= 21)).mean()),
        "note_sequence_vs_date": sequence_vs_date(ms),
        "idulfitri_window_notes_by_prefix": idul_window,
        "products_complete_period": product_stats(units, lines, start, end),
        "holidays_in_data": hol,
        "idulfitri_years_in_data": idul,
        "feasibility": {
            "years_span_ge_2": bool(years >= 2),
            "ramadan_count_ge_2": len(idul) >= 2,
        },
    }
    fig_daily(daily_full, start, hol, rep / "figures" / "phase8_daily_notes.png")
    (rep / "data" / "id_data_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                                            encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    r = run(load_config(args.config))
    print(json.dumps({k: r[k] for k in ("period", "feasibility", "idulfitri_years_in_data")}, indent=2))


if __name__ == "__main__":
    main()
