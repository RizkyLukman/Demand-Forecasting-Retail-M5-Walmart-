"""Tes simulasi stok (Fase 7) dengan contoh kecil yang dihitung manual.

FIXTURE: angka permintaan, harga, dan target stok di bawah ditulis tangan untuk menguji
logika simulasi (bukan data M5).
"""

import numpy as np
import pytest

from retailfc.inventory_sim import coverage, empirical_targets, protection_windows, simulate


def test_protection_windows_truncate_at_horizon():
    assert protection_windows(10, review=7, lead=2) == [(0, 9), (7, 10)]
    assert protection_windows(28, review=7, lead=2) == [(0, 9), (7, 16), (14, 23), (21, 28)]


def test_simulate_hand_calculated():
    # FIXTURE: 1 deret, 10 hari, permintaan 3/hari, S = 20 lalu 15, R = 7, L = 2.
    demand = np.full((1, 10), 3.0)
    price = np.ones((1, 10))
    r = simulate(demand, price, np.array([[20.0, 15.0]]), review=7, lead=2)
    # Hari 0–5 terjual 3 (stok 20 → 2), hari 6 terjual 2 (hilang 1), hari 7–8 stok 0
    # (hilang 3 + 3), pesanan 15 dari hari 7 tiba hari 9, terjual 3 → sisa 12.
    assert r["demand"][0] == 30
    assert r["sold"][0] == 23
    assert r["lost"][0] == 7
    assert r["ordered"][0] == 15
    assert r["ending_stock"][0] == 12
    assert r["hold_units"][0] == 17 + 14 + 11 + 8 + 5 + 2 + 0 + 0 + 0 + 12
    assert r["lost_value"][0] == 7


def test_simulate_no_order_when_position_covers_target():
    # FIXTURE: pesanan di jalan ikut dihitung di posisi stok, jadi tidak dipesan dua kali.
    demand = np.zeros((1, 15))
    r = simulate(demand, np.ones((1, 15)), np.array([[10.0, 10.0, 10.0]]), review=7, lead=2)
    assert r["ordered"][0] == 0 and r["ending_stock"][0] == 10


def test_coverage():
    demand = np.full((1, 10), 3.0)
    # Periode (0, 9): 27 > 20 → tidak tercakup; periode (7, 10): 9 ≤ 15 → tercakup.
    assert coverage(demand, np.array([[20.0, 15.0]]), [(0, 9), (7, 10)]) == pytest.approx(0.5)


def test_empirical_targets_ignore_days_before_first_sale():
    # FIXTURE: 2 deret, riwayat 6 hari. Deret 1 baru laku sejak hari ke-4 (indeks 3).
    hist = np.array([[1, 2, 3, 4, 5, 6], [0, 0, 0, 2, 2, 2]], dtype=float)
    first = np.array([0, 3])
    q = empirical_targets(hist, first, lengths=[2], alpha=1.0, window=6)[2]
    assert q[0] == 11          # jumlah 2 hari terbesar: 5 + 6
    assert q[1] == 4           # hanya jendela setelah penjualan pertama: 2 + 2


def test_xgb_empirical_scales_empirical_target_by_forecast_level():
    from retailfc.inventory_sim import build_targets, history_mean

    # FIXTURE: 2 deret. Deret 0: rata-rata historis 2/hari, forecast 4/hari (level ×2).
    # Deret 1: tidak pernah laku di jendela → target = jumlah forecast.
    hist = np.array([[2.0] * 6, [0.0] * 6])
    first = np.array([0, 6])
    hm = history_mean(hist, first, window=6)
    assert list(hm) == [2.0, 0.0]
    point = {"xgb_point": np.array([[4.0, 4.0], [1.0, 1.0]]), "classical": np.zeros((2, 2))}
    emp = {(0.9, 2): np.array([5.0, 0.0])}
    s = build_targets("xgb_empirical", 0.9, [(0, 2)], point, None, None, emp, hm)
    assert s[0, 0] == pytest.approx(5.0 * 8.0 / 4.0)   # target empiris × (forecast 8 / historis 4)
    assert s[1, 0] == pytest.approx(2.0)               # tanpa riwayat: jumlah forecast
