"""Tes anti-leakage dan baseline forecast apotek (fixture deterministik, bukan data apotek)."""

import numpy as np
import pandas as pd

from retailfc.id_forecast import build_features, calendar_frame, weekday_mean

DAYS = pd.date_range("2015-04-01", periods=120, freq="D")
CAL = calendar_frame(DAYS, {pd.Timestamp("2015-05-01"), pd.Timestamp("2015-06-02")})


def test_features_ignore_data_after_cutoff():
    rng = np.random.default_rng(0)
    Y = rng.poisson(20, size=(3, len(DAYS))).astype(float)
    cutoff, h = 90, 14
    a = build_features(Y, CAL, cutoff, h, [14, 21, 28], [7, 14, 28])
    Y2 = Y.copy()
    Y2[:, cutoff + 1:] = 1e6          # ubah masa depan: fitur tidak boleh berubah
    b = build_features(Y2, CAL, cutoff, h, [14, 21, 28], [7, 14, 28])
    pd.testing.assert_frame_equal(a, b)
    assert a["day"].max() == cutoff + h


def test_lag_values_and_target_mean():
    Y = np.arange(len(DAYS), dtype=float)[None, :]
    X = build_features(Y, CAL, 90, 14, [14, 21, 28], [7])
    row = X[X["day"] == 100].iloc[0]
    assert row["lag_14"] == 86 and row["lag_28"] == 72
    assert row["roll_mean_7"] == np.mean(np.arange(80, 87))
    assert row["te_mean"] == np.mean(np.arange(91))


def test_calendar_distance_features():
    r = CAL.loc["2015-04-29"]
    assert (r["day_off"], r["days_to_off"]) == (0, 2)
    assert CAL.loc["2015-05-01", "day_off"] == 1
    assert CAL.loc["2015-05-03", "days_since_off"] == 2
    assert CAL.loc["2015-04-01", "days_since_off"] == 7      # belum ada libur sebelumnya: dipotong


def test_scaled_pooling_handles_different_scales():
    """Dua deret dengan pola mingguan sama tetapi skala 1 : 100 harus sama-sama tertebak."""
    from retailfc.id_forecast import fit_predict_xgb

    pattern = np.array([30.0, 300, 260, 270, 250, 230, 40])[DAYS.weekday]
    Y = np.vstack([pattern, 100 * pattern])
    fc = {"horizon": 14, "lags": [14, 21, 28], "rolling_windows": [7, 14, 28], "early_stopping_days": 14,
          "params": {"objective": "reg:tweedie", "tweedie_variance_power": 1.1, "eta": 0.1, "max_depth": 4,
                     "min_child_weight": 1, "seed": 0},
          "max_rounds": 500, "early_stopping": 50}
    pred, _ = fit_predict_xgb(Y, CAL, 90, fc, use_holidays=False)
    act = Y[:, 91:105]
    np.testing.assert_allclose(pred.sum(axis=1) / act.sum(axis=1), 1.0, atol=0.05)


def test_weekday_mean():
    hist = np.tile(np.arange(7, dtype=float), 4)[None, :]
    np.testing.assert_array_equal(weekday_mean(hist, 14)[0], np.tile(np.arange(7), 2))
