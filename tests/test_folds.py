"""Aturan desain fold dari PROJECT_PLAN.md, Fase 3."""

from pathlib import Path

import copy

import numpy as np
import pytest

from retailfc.folds import Fold, build_folds, closed_day_mask, validate_folds
from retailfc.io import load_calendar, load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
RAW = ROOT / CFG["paths"]["raw_m5"]


@pytest.fixture(scope="module")
def folds():
    return {f.name: f for f in build_folds(CFG)}


def test_holdout_is_last_28_days(folds):
    ho = folds["holdout"]
    assert (ho.horizon_start, ho.horizon_end) == (1914, 1941)


def test_no_fold_touches_holdout(folds):
    for f in folds.values():
        if f.role != "holdout":
            assert f.horizon_end < folds["holdout"].horizon_start, f.name


def test_cv_folds_tile_d1802_to_d1913(folds):
    cv = sorted((f for f in folds.values() if f.role == "cv"), key=lambda f: f.cutoff)
    assert len(cv) >= 4
    days = np.concatenate([np.arange(f.horizon_start, f.horizon_end + 1) for f in cv])
    assert np.array_equal(days, np.arange(1802, 1914))   # berurutan, tanpa celah, tanpa tumpang tindih


def test_masks_are_disjoint_and_ordered(folds):
    d = np.arange(1, 1942)
    for f in folds.values():
        fit, es, hz = f.fit_mask(d), f.es_mask(d), f.horizon_mask(d)
        assert not (fit & es).any() and not (fit & hz).any() and not (es & hz).any()
        assert d[es].max() == f.cutoff and es.sum() == f.es_days
        assert d[fit].max() < d[es].min() and d[es].max() < d[hz].min()
        assert hz.sum() == f.horizon


@pytest.mark.parametrize("fold,event_d", [("easter_2014", 1178), ("easter_2015", 1528),
                                          ("year_end_2015", 1792)])
def test_special_folds_contain_their_event(folds, fold, event_d):
    f = folds[fold]
    assert f.role == "special"
    assert f.horizon_start <= event_d <= f.horizon_end


def test_overlap_with_holdout_is_rejected():
    bad = [Fold("cv1", "cv", 1801, 28, 28), Fold("cv2", "cv", 1829, 28, 28),
           Fold("cv3", "cv", 1857, 28, 28), Fold("cv4", "cv", 1900, 28, 28),
           Fold("holdout", "holdout", 1913, 28, 28)]
    with pytest.raises(ValueError, match="overlaps the holdout"):
        validate_folds(bad, min_history=420, last_day=1941)


def test_overlapping_cv_horizons_are_rejected():
    cfg = copy.deepcopy(CFG)
    cfg["validation"]["cv_cutoffs"] = [1801, 1820, 1857, 1885]
    with pytest.raises(ValueError, match="cv horizons overlap"):
        build_folds(cfg)


def test_too_little_history_is_rejected():
    cfg = copy.deepcopy(CFG)
    cfg["validation"]["extra_folds"] = {"too_early": 300}
    with pytest.raises(ValueError, match="days before early stopping"):
        build_folds(cfg)


@pytest.mark.skipif(not (RAW / "calendar.csv").exists(), reason="raw M5 data not present")
def test_closed_days_are_christmas_only_and_outside_cv_and_holdout(folds):
    cal = load_calendar(RAW)
    d = cal["d_num"].to_numpy()
    closed = closed_day_mask(d, cal, CFG)
    dates = cal.loc[closed, "date"]
    assert ((dates.dt.month == 12) & (dates.dt.day == 25)).all()
    assert closed.sum() == 5   # 2011-2015 di dalam d_1..d_1969
    for f in folds.values():
        if f.role in ("cv", "holdout"):
            assert not closed[f.horizon_mask(d)].any(), f.name
