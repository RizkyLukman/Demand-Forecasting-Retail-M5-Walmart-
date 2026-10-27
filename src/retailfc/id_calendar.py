"""Fase 8: kalender Indonesia 2015 dengan sumber per tanggal -> data/external/id_calendar.csv.

Setiap tanggal di bawah diambil dari sumber yang disebut di kolom `source`, bukan dari
ingatan dan bukan langsung dari paket `holidays`. Paket `holidays` hanya dipakai sebagai
pembanding; selisihnya dilaporkan saat skrip dijalankan.

Catatan sumber (dicek 28 Okt 2026):
- Libur nasional dan cuti bersama: Setkab, "Ingat, Tidak Seperti 26 Desember, 2 Januari
  Masuk Hari Kerja Biasa" (29 Des 2014), mengacu SKB 3 Menteri 7 Mei 2014
  (No. 5/2014, 3/SKB/MEN/V/2014, 02/SKB/MENPAN/V/2014). Versi Mei 2014 (JPNN, 8 Mei 2014)
  menulis cuti bersama Idulfitri 15, 16, 20 Juli; versi Setkab Desember 2014 menulis
  16, 20, 21 Juli. Yang dipakai versi Setkab.
- Maulid 24 Des 2015: SKB perubahan No. 310 Tahun 2014 (JDIH Kemnaker, peraturan 549).
  Setkab Des 2014 mencatat 24 Des sebagai cuti bersama Natal. Kedua status dicatat;
  tanggal itu hari libur di kedua versi.
- Pilkada 9 Des 2015: Keppres No. 25 Tahun 2015 (Setkab).
- 1 Ramadan 1436 H = 18 Jun 2015: pengumuman Menteri Agama hasil sidang isbat 16 Jun 2015
  (Suara.com, 16 Jun 2015).
- 1 Syawal 1436 H = 17 Jul 2015: pengumuman Menteri Agama hasil sidang isbat 16 Jul 2015
  (The Jakarta Post, 16 Jul 2015). Akhir Ramadan = sehari sebelum 1 Syawal.

Cara pakai:
    python -m retailfc.id_calendar --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import holidays
import pandas as pd

from retailfc.io import load_config

SRC_SKB = ("Setkab 29-12-2014 'Ingat, Tidak Seperti 26 Desember, 2 Januari Masuk Hari Kerja Biasa' "
           "(SKB 3 Menteri 7-5-2014) https://setkab.go.id/ingat-tidak-seperti-26-desember-2-januari-masuk-hari-kerja-biasa/")
SRC_SKB_310 = ("SKB perubahan No. 310 Tahun 2014 (JDIH Kemnaker) "
               "https://jdih.kemnaker.go.id/peraturan/detail/549/")
SRC_PILKADA = ("Keppres No. 25 Tahun 2015 (Setkab) "
               "https://setkab.go.id/sukseskan-pilkada-serentak-pemerintah-tetapkan-9-desember-2015-sebagai-hari-libur-nasional/")
SRC_RAMADAN = ("Isbat 16-6-2015, pengumuman Menag (Suara.com 16-6-2015) "
               "https://www.suara.com/foto/2015/06/16/201752/puasa-jatuh-hari-kamis")
SRC_SYAWAL = ("Isbat 16-7-2015, pengumuman Menag (The Jakarta Post 16-7-2015) "
              "https://www.thejakartapost.com/news/2015/07/16/idul-fitri-falls-friday-says-govt")
SRC_DERIVED = "turunan: sehari sebelum 1 Syawal 1436 H (lihat baris idulfitri)"

# (tanggal, event). Nama event dipakai sebagai kunci fitur, jadi berbahasa Inggris.
EVENTS_2015: list[tuple[str, str, str]] = [
    ("2015-01-01", "new_year", SRC_SKB),
    ("2015-01-03", "maulid", SRC_SKB),
    ("2015-02-19", "chinese_new_year", SRC_SKB),
    ("2015-03-21", "nyepi", SRC_SKB),
    ("2015-04-03", "good_friday", SRC_SKB),
    ("2015-05-01", "labour_day", SRC_SKB),
    ("2015-05-14", "ascension", SRC_SKB),
    ("2015-05-16", "isra_miraj", SRC_SKB),
    ("2015-06-02", "vesak", SRC_SKB),
    ("2015-06-18", "ramadan_start", SRC_RAMADAN),
    ("2015-07-16", "ramadan_end", SRC_DERIVED),
    ("2015-07-16", "cuti_bersama_idulfitri", SRC_SKB),
    ("2015-07-17", "idulfitri", SRC_SYAWAL),
    ("2015-07-17", "idulfitri_holiday", SRC_SKB),
    ("2015-07-18", "idulfitri_holiday", SRC_SKB),
    ("2015-07-20", "cuti_bersama_idulfitri", SRC_SKB),
    ("2015-07-21", "cuti_bersama_idulfitri", SRC_SKB),
    ("2015-08-17", "independence_day", SRC_SKB),
    ("2015-09-24", "idul_adha", SRC_SKB),
    ("2015-10-14", "islamic_new_year", SRC_SKB),
    ("2015-12-09", "regional_election", SRC_PILKADA),
    ("2015-12-24", "maulid", SRC_SKB_310),
    ("2015-12-24", "cuti_bersama_christmas", SRC_SKB),
    ("2015-12-25", "christmas", SRC_SKB),
]

# Event yang berarti hari libur (dipakai untuk membandingkan dengan paket `holidays`).
DAY_OFF = {"new_year", "maulid", "chinese_new_year", "nyepi", "good_friday", "labour_day", "ascension",
           "isra_miraj", "vesak", "idulfitri_holiday", "independence_day", "idul_adha", "islamic_new_year",
           "regional_election", "christmas", "cuti_bersama_idulfitri", "cuti_bersama_christmas"}


def build_calendar() -> pd.DataFrame:
    cal = pd.DataFrame(EVENTS_2015, columns=["date", "event", "source"])
    cal["date"] = pd.to_datetime(cal["date"], format="%Y-%m-%d")
    return cal.sort_values(["date", "event"]).reset_index(drop=True)


def compare_with_package(cal: pd.DataFrame, year: int = 2015) -> dict[str, list[str]]:
    """Tanggal libur yang hanya ada di kalender ini atau hanya di paket `holidays`."""
    pkg = set(holidays.country_holidays("ID", years=year))
    pkg |= set(holidays.country_holidays("ID", years=year, categories=("government",)))
    ours = set(cal.loc[cal["event"].isin(DAY_OFF), "date"].dt.date)
    return {"only_here": sorted(str(d) for d in ours - pkg),
            "only_package": sorted(str(d) for d in pkg - ours)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    out = Path(load_config(args.config)["paths"]["external"]) / "id_calendar.csv"
    cal = build_calendar()
    cal.assign(date=cal["date"].dt.strftime("%Y-%m-%d")).to_csv(out, index=False, encoding="utf-8")
    print(f"{len(cal)} baris ditulis")
    print("selisih dengan paket holidays:", compare_with_package(cal))


if __name__ == "__main__":
    main()
