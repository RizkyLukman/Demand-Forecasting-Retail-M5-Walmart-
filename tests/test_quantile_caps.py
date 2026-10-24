"""Tes logika uji batas iterasi (Fase 5): meniru batas dari kurva validasi, dan aturan
keputusan pemilihan batas. Tidak melatih model.

Fixture: kurva validasi dan baris ringkasan di bawah ditulis tangan untuk menguji logika
(bukan data hasil model), dan ditandai FIXTURE.
"""

from retailfc.quantile_cap_report import choose_cap
from retailfc.train import emulate_cap

# FIXTURE: kurva skor validasi (lebih kecil lebih baik), indeks = iterasi (0-based).
CURVE = [5.0, 4.0, 3.0, 2.5, 2.5, 2.4, 2.45, 2.46, 2.47, 2.48, 2.49, 2.5]


def test_emulate_cap_picks_best_within_cap():
    # Batas 4 iterasi: hanya iterasi 0..3 yang dijalankan; terbaik = 3; 3 + 2 >= 4 -> di batas.
    assert emulate_cap(CURVE, cap=4, patience=2) == (3, True)
    # Batas 12 (seluruh kurva): terbaik = 5; 5 + 2 < 12 -> berhenti karena early stopping.
    assert emulate_cap(CURVE, cap=12, patience=2) == (5, False)


def test_emulate_cap_ties_keep_first_occurrence():
    # Iterasi 3 dan 4 sama-sama 2,5. Early stopping XGBoost hanya menganggap skor yang
    # lebih kecil sebagai perbaikan, jadi yang dipilih iterasi 3.
    assert emulate_cap(CURVE, cap=5, patience=10)[0] == 3


def test_emulate_cap_beyond_curve_length_equals_full_run():
    assert emulate_cap(CURVE, cap=100, patience=2) == emulate_cap(CURVE, cap=len(CURVE), patience=2)


def _row(cap, pinball, p80=0.81, p90=0.894):
    # FIXTURE: baris ringkasan per batas.
    return {"cap": cap, "pinball_mean": pinball, "coverage_mean": {"0.8": p80, "0.9": p90}}


def test_choose_cap_prefers_smallest_within_one_percent():
    rows = [_row(1000, 0.3413), _row(1500, 0.3400), _row(2000, 0.3395), _row(3000, 0.3394)]
    # 0.3413 <= 0.3394 * 1.01 = 0.3428 -> batas 1.000 sudah cukup.
    assert choose_cap(rows)["chosen_cap"] == 1000


def test_choose_cap_moves_up_when_gain_is_large():
    rows = [_row(1000, 0.3500), _row(1500, 0.3420), _row(2000, 0.3400)]
    # 0.35 > 0.34 * 1.01 = 0.3434; 0.342 <= 0.3434 -> batas 1.500.
    assert choose_cap(rows)["chosen_cap"] == 1500


def test_choose_cap_rejects_cap_with_worse_coverage():
    rows = [_row(1000, 0.3500), _row(1500, 0.3420, p90=0.87), _row(2000, 0.3410)]
    # 1.500 cukup dekat pinball terbaik, tetapi P90 menjauh dari target 0,9 sebanyak 0,03
    # (acuan 0,006 + toleransi 0,01 = 0,016) -> ditolak; pilih 2.000.
    assert choose_cap(rows)["chosen_cap"] == 2000


def _write_run(path, name, max_rounds, infos, per_fold, checkpoints=None):
    # FIXTURE: folder run minimal dengan skema yang sama seperti keluaran quantile_final.
    import json
    path.mkdir()
    (path / "config.json").write_text(json.dumps(
        {"name": name, "options": {"max_rounds": max_rounds, "early_stopping": 100}}))
    (path / "infos.json").write_text(json.dumps(infos))
    summary = {"pinball_mean": sum(v["pinball_mean"] for v in per_fold.values()) / len(per_fold),
               "coverage_mean": {"0.8": 0.81, "0.9": 0.894}, "hit_cap": sum(i["hit_cap"] for i in infos)}
    metrics = {"per_fold": per_fold, "cv_summary": summary}
    if checkpoints:
        metrics["checkpoints"] = checkpoints
    (path / "metrics.json").write_text(json.dumps(metrics))


def test_compare_and_cap_table_with_checkpoint_run(tmp_path):
    from retailfc.quantile_cap_report import cap_table, compare, load_run

    pf = {"cv1": {"pinball_mean": 0.40, "coverage": {}}, "cv2": {"pinball_mean": 0.30, "coverage": {}}}
    a_infos = [{"store_id": "S1", "fold": "cv1", "best_iteration": 999, "hit_cap": True,
                "pinball_mean": 0.40, "seconds": 10.0},
               {"store_id": "S1", "fold": "cv2", "best_iteration": 300, "hit_cap": False,
                "pinball_mean": 0.30, "seconds": 5.0}]
    b_infos = [{"store_id": "S1", "fold": "cv1", "best_iteration": 1800, "hit_cap": False,
                "pinball_mean": 0.39, "seconds": 20.0, "es_best_score": 0.50,
                "es_curve_every_100": {"1000": 0.52},
                "checkpoints": {"1000": {"best_iteration": 999, "hit_cap": True, "es_score": 0.52},
                                "2000": {"best_iteration": 1800, "hit_cap": False, "es_score": 0.50}}},
               {"store_id": "S1", "fold": "cv2", "best_iteration": 300, "hit_cap": False,
                "pinball_mean": 0.30, "seconds": 5.0, "reused_from_run": "A",
                "checkpoints": {"1000": {"best_iteration": 300, "hit_cap": False},
                                "2000": {"best_iteration": 300, "hit_cap": False}}}]
    pf_b = {"cv1": {"pinball_mean": 0.39, "coverage": {}}, "cv2": {"pinball_mean": 0.30, "coverage": {}}}
    cps = {"1000": {"per_fold": pf, "cv_summary": {"pinball_mean": 0.35, "coverage_mean": {"0.8": 0.81, "0.9": 0.894},
                                                   "hit_cap": 1}},
           "2000": {"per_fold": pf_b, "cv_summary": {"pinball_mean": 0.345, "coverage_mean": {"0.8": 0.81, "0.9": 0.894},
                                                     "hit_cap": 0}}}
    _write_run(tmp_path / "A", "xgb_quantile", 1000, a_infos, pf)
    _write_run(tmp_path / "B", "xgb_quantile_r2000", 2000, b_infos, pf_b, cps)
    a, b = load_run(tmp_path / "A"), load_run(tmp_path / "B")

    cmp = compare(a, b)
    assert len(cmp["capped_models"]) == 1                      # hanya model yang mentok di A
    row = cmp["capped_models"][0]
    assert row["es_score_at_a_limit"] == 0.52                  # skor terbaik dalam 1..1.000
    assert row["pinball_change_pct"] == round(100 * (0.39 / 0.40 - 1), 3)

    rows = cap_table(a, b)
    assert [r["cap"] for r in rows] == [1000, 2000]
    assert rows[0]["checkpoint_matches_run_a"] is True
    assert rows[1]["change_vs_a_pct"] == round(100 * (0.345 / 0.35 - 1), 3)
