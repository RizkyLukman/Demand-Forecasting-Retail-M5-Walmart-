"""Tes baseline studi efek pada deret buatan yang jelas ditandai (bukan data pengganti).

Deret di sini adalah fixture deterministik untuk menguji rumus, bukan data apotek.
"""

import numpy as np
import pandas as pd
import pytest

from retailfc.id_event_study import METHODS, expected, placebo, split_windows, window_offsets

DAYS = pd.date_range("2015-04-01", "2015-12-31", freq="D")
WEEKLY = np.array([30.0, 300, 260, 270, 250, 230, 40])  # Sen..Min


def weekly_series(trend: float = 0.0) -> pd.Series:
    t = np.arange(len(DAYS))
    return pd.Series(WEEKLY[DAYS.weekday] * (1 + trend * t / len(DAYS)), index=DAYS)


@pytest.mark.parametrize("method", list(METHODS))
def test_expected_recovers_pure_weekly_pattern(method):
    y = weekly_series()
    gap = (DAYS >= "2015-06-18") & (DAYS <= "2015-08-18")
    e = expected(y, ~gap, gap, method)
    np.testing.assert_allclose(e.to_numpy(), y[gap].to_numpy(), rtol=1e-6)


@pytest.mark.parametrize("method", ["weekday_loglinear", "local_level_28"])
def test_trend_methods_follow_linear_trend(method):
    y = weekly_series(trend=0.3)
    gap = (DAYS >= "2015-06-18") & (DAYS <= "2015-08-18")
    e = expected(y, ~gap, gap, method)
    assert np.abs(e.sum() / y[gap].sum() - 1) < 0.02


def test_placebo_is_zero_without_event_effect():
    y = weekly_series()
    event = (pd.Timestamp("2015-06-18"), pd.Timestamp("2015-08-18"))
    offs = window_offsets({"w": ["2015-06-18", "2015-07-01"]}, event[0])
    pl = placebo(y, set(), event, offs, "weekday_median")
    assert len(pl) > 0
    np.testing.assert_allclose(pl["log_ratio"].to_numpy(), 0.0, atol=1e-9)


def test_split_windows_keeps_short_window():
    w = split_windows({"a": ["2015-06-18", "2015-07-01"], "b": ["2015-07-16", "2015-07-21"]}, 7)
    assert w == {"a_1": ["2015-06-18", "2015-06-24"], "a_2": ["2015-06-25", "2015-07-01"],
                 "b": ["2015-07-16", "2015-07-21"]}
