"""Cek konsistensi kalender Indonesia 2015 (tanpa akses internet)."""

import pandas as pd

from retailfc.id_calendar import build_calendar, compare_with_package


def test_calendar_shape_and_sources():
    cal = build_calendar()
    assert list(cal.columns) == ["date", "event", "source"]
    assert cal["source"].str.len().min() > 0
    assert not cal.duplicated(["date", "event"]).any()
    assert (cal["date"].dt.year == 2015).all()


def test_ramadan_and_idulfitri_consistent():
    cal = build_calendar().set_index("event")["date"]
    start, end, idul = cal["ramadan_start"], cal["ramadan_end"], cal["idulfitri"]
    assert end == idul - pd.offsets.Day(1)
    assert (end - start).days + 1 in (29, 30)  # bulan hijriah 29 atau 30 hari
    assert start.weekday() == 3 and idul.weekday() == 4  # Kamis 18 Jun, Jumat 17 Jul


def test_only_documented_differences_with_package():
    # Satu-satunya selisih yang diketahui: cuti bersama Natal 24 Des tidak ada di paket,
    # tetapi 24 Des tetap libur (Maulid) di paket, jadi tanggalnya cocok.
    assert compare_with_package(build_calendar()) == {"only_here": [], "only_package": []}
