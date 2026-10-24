"""Tes fungsi baseline dan klasifikasi segmen (Fase 5). Fixture kecil buatan tangan,
ditandai jelas: hanya untuk menguji rumus, bukan data proyek."""

import numpy as np
import pytest

from retailfc.baselines import fill_closed, moving_average, seasonal_naive
from retailfc.segments import classify

# FIXTURE BUATAN TANGAN (bukan data M5).
HIST = np.array([
    [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],          # laku setiap hari sejak hari ke-3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],          # tidak pernah laku
], dtype=float)


def test_seasonal_naive_repeats_last_week():
    p = seasonal_naive(HIST, 10)
    assert p[0].tolist() == [2, 3, 4, 5, 6, 7, 8, 2, 3, 4]


def test_moving_average_uses_last_window():
    assert moving_average(HIST, 3, window=4)[0].tolist() == [6.5, 6.5, 6.5]


def test_fill_closed_uses_value_seven_days_earlier():
    closed = np.zeros(10, dtype=bool)
    closed[9] = True
    assert fill_closed(HIST, closed)[0, 9] == HIST[0, 2]


def test_classify_thresholds():
    # smooth: laku setiap hari dengan ukuran stabil; lumpy: jarang dan bervariasi.
    smooth = np.r_[np.zeros(3), np.full(30, 5.0)]
    lumpy = np.r_[[1.0], np.tile([0, 0, 0, 20, 0, 0, 1, 0], 4)]
    intermittent = np.r_[[3.0], np.tile([0, 0, 3, 0], 8)]
    erratic = np.r_[np.tile([1, 30], 16), [1.0]]
    hist = np.vstack([smooth, lumpy, intermittent, erratic, np.zeros(33)])
    seg = classify(hist)
    assert seg["segment"].astype(str).tolist() == ["smooth", "lumpy", "intermittent", "erratic", "no_history"]
    assert seg.loc[0, "adi"] == pytest.approx(1.0) and seg.loc[0, "cv2"] == pytest.approx(0.0)


def test_classify_ignores_closed_days():
    x = np.r_[[2.0], np.zeros(1), np.full(8, 2.0)][None, :]
    closed = np.zeros(10, dtype=bool)
    closed[1] = True
    assert classify(x, closed).loc[0, "adi"] == pytest.approx(1.0)
