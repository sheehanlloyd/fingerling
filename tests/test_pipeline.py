"""Pipeline, storage and CLI tests.

These are wiring tests. The maths is covered in test_calibrate/test_measure/
test_trust; what's left to get wrong here is plumbing — a stage that doesn't get
timed, a number that reaches the database as the wrong column, a correction that
overwrites the machine's own verdict.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from cli import batch as batch_cli
from pipeline import capture
from pipeline.config import load_config
from pipeline.detect import StubDetector, StubSettings, build_detector
from pipeline.pipeline import Pipeline
from store import db as store_db
from tests import synthetic as syn


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def sample_dir(tmp_path):
    """A handful of frames with a real card in them, so calibration has work to
    do rather than being skipped."""
    d = tmp_path / "frames"
    d.mkdir()
    import cv2

    from cli.make_sample import frames as sample_frames

    for i, frame in enumerate(sample_frames(6, seed=1)):
        cv2.imwrite(str(d / f"f{i:03d}.png"), frame)
    return d


def make_pipeline(cfg, tmp_path, mode="normal", conn=None):
    return Pipeline(
        cfg, conn=conn, detector=StubDetector(StubSettings(mode=mode, seed=5))
    )


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def test_image_directory_yields_every_frame_in_filename_order(sample_dir, cfg):
    frames = list(capture.open_source(sample_dir, cfg))
    assert len(frames) == 6
    assert [f.index for f in frames] == list(range(6))
    assert all(f.path is not None for f in frames)


def test_video_and_image_sources_produce_the_same_frame_count(sample_dir, cfg, tmp_path):
    import cv2

    video = tmp_path / "clip.mp4"
    imgs = sorted(sample_dir.iterdir())
    first = cv2.imread(str(imgs[0]))
    h, w = first.shape[:2]
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for p in imgs:
        wr.write(cv2.imread(str(p)))
    wr.release()

    assert len(list(capture.open_source(video, cfg))) == len(imgs)


def test_stride_takes_every_nth_frame(sample_dir, cfg, tmp_path):
    import cv2

    video = tmp_path / "clip.mp4"
    imgs = sorted(sample_dir.iterdir())
    h, w = cv2.imread(str(imgs[0])).shape[:2]
    wr = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for p in imgs:
        wr.write(cv2.imread(str(p)))
    wr.release()

    assert len(list(capture.open_source(video, cfg, stride=3))) == 2


def test_oversized_frames_are_downscaled_to_the_configured_long_edge(sample_dir, cfg):
    cfg = dict(cfg)
    cfg["capture"] = dict(cfg["capture"], max_long_edge_px=320)
    frames = list(capture.open_source(sample_dir, cfg))
    assert max(frames[0].image.shape[:2]) == 320
    assert frames[0].scale > 1.0


def test_a_missing_source_raises_capture_error_not_something_cryptic(cfg):
    with pytest.raises(capture.CaptureError):
        list(capture.open_source("no/such/file.mp4", cfg))


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------


def test_every_stage_is_timed_and_the_total_is_at_least_their_sum(cfg, sample_dir, tmp_path):
    pipe = make_pipeline(cfg, tmp_path)
    frame = next(iter(capture.open_source(sample_dir, cfg)))
    rec = pipe.process(frame)

    for stage in ("detect", "calibrate", "measure", "trust", "decide", "total"):
        assert stage in rec.latency_ms, f"{stage} was not timed"
        assert rec.latency_ms[stage] >= 0.0

    parts = sum(v for k, v in rec.latency_ms.items() if k != "total")
    assert rec.latency_ms["total"] >= parts * 0.99


def test_a_frame_with_a_card_in_it_actually_calibrates(cfg, sample_dir, tmp_path):
    """If this fails the pipeline is running but calibration isn't reaching it,
    and every millimetre in the database is missing rather than wrong."""
    pipe = make_pipeline(cfg, tmp_path)
    rec = pipe.process(next(iter(capture.open_source(sample_dir, cfg))))
    assert rec.calibration.calibrated
    assert rec.calibration.target == "card"
    assert rec.measures.fork_length_mm is not None


def test_a_frame_with_no_target_still_grades_in_pixels(cfg, tmp_path):
    pipe = make_pipeline(cfg, tmp_path)
    blank = capture.Frame(syn.blank_scene(), 0, "test")
    rec = pipe.process(blank)
    assert not rec.calibration.calibrated
    assert rec.measures.fork_length_mm is None
    assert rec.measures.fork_length_px is not None
    assert rec.measures.depth_ratio is not None  # dimensionless traits survive
    assert rec.decision.decision in ("PASS", "CULL", "REVIEW")


def test_no_detection_produces_a_record_and_does_not_crash(cfg, tmp_path):
    pipe = make_pipeline(cfg, tmp_path, mode="no_detection")
    rec = pipe.process(capture.Frame(syn.blank_scene(), 0, "test"))
    assert rec.landmarks.detected is False
    assert rec.decision.decision == "REVIEW"


def test_the_record_says_which_detector_produced_it(cfg, tmp_path):
    """A row that doesn't say whether the stub or a real model made it is a row
    you can't interpret six months from now."""
    pipe = make_pipeline(cfg, tmp_path)
    rec = pipe.process(capture.Frame(syn.blank_scene(), 0, "test"))
    assert rec.to_row()["detector_source"] == "stub:normal"


def test_an_aruco_residual_is_not_written_to_the_residual_column(cfg, tmp_path):
    """It's ~0 by construction on a 4-point solve. Storing it under a column
    called 'residual' invites someone to read it as a quality figure later."""
    from pipeline.calibrate import Calibration

    pipe = make_pipeline(cfg, tmp_path)
    rec = pipe.process(capture.Frame(syn.blank_scene(), 0, "test"))
    rec.calibration = Calibration(
        calibrated=True, target="aruco", H=np.eye(3), residual_px=0.0004,
        residual_meaningful=False, obliquity=1.0, mm_per_px=0.1,
    )
    assert rec.to_row()["calibration_residual"] is None


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------


def test_a_record_survives_a_round_trip_through_sqlite(cfg, sample_dir, tmp_path):
    conn = store_db.connect(tmp_path / "t.db")
    pipe = make_pipeline(cfg, tmp_path, conn=conn)
    rec = pipe.process(next(iter(capture.open_source(sample_dir, cfg))))
    fid = store_db.insert(conn, rec.to_row())

    got = store_db.get(conn, fid)
    assert got["decision"] == rec.decision.decision
    assert got["fork_length_mm"] == pytest.approx(rec.measures.fork_length_mm.value)
    assert got["trust_score"] == pytest.approx(rec.trust.trust_score)
    # The raw landmarks have to come back, not just the derived numbers.
    lm = json.loads(got["landmarks_json"])
    assert "snout_tip" in lm and "x" in lm["snout_tip"]
    conn.close()


def test_uncertainties_are_stored_and_not_quietly_dropped(cfg, sample_dir, tmp_path):
    conn = store_db.connect(tmp_path / "t.db")
    pipe = make_pipeline(cfg, tmp_path, conn=conn)
    rec = pipe.process(next(iter(capture.open_source(sample_dir, cfg))))
    got = store_db.get(conn, store_db.insert(conn, rec.to_row()))
    assert got["fork_length_mm_sigma"] > 0
    assert got["curvature_index_sigma"] > 0
    conn.close()


def test_failed_constraints_are_stored_by_name(cfg, tmp_path):
    conn = store_db.connect(tmp_path / "t.db")
    pipe = make_pipeline(cfg, tmp_path, mode="implausible", conn=conn)
    rec = pipe.process(capture.Frame(syn.blank_scene(), 0, "test"))
    got = store_db.get(conn, store_db.insert(conn, rec.to_row()))
    names = json.loads(got["failed_constraints_json"])
    assert "peduncle_points_not_swapped" in names
    assert got["decision"] == "REVIEW"
    conn.close()


def test_a_correction_records_the_human_without_erasing_the_machine(cfg, tmp_path):
    """The pair (what the model said, what the person said) is the training data
    for the next model. Overwrite `decision` and that signal is gone."""
    conn = store_db.connect(tmp_path / "t.db")
    pipe = make_pipeline(cfg, tmp_path, mode="implausible", conn=conn)
    rec = pipe.process(capture.Frame(syn.blank_scene(), 0, "test"))
    fid = store_db.insert(conn, rec.to_row())

    updated = store_db.correct(conn, fid, "PASS", note="looked fine to me")
    assert updated["human_corrected"] == 1
    assert updated["human_decision"] == "PASS"
    assert updated["decision"] == "REVIEW"  # the machine's verdict is untouched
    assert updated["corrected_at"]
    conn.close()


def test_the_review_queue_only_shows_untouched_reviews(cfg, tmp_path):
    conn = store_db.connect(tmp_path / "t.db")
    pipe = make_pipeline(cfg, tmp_path, mode="implausible", conn=conn)
    ids = [
        store_db.insert(conn, pipe.process(capture.Frame(syn.blank_scene(), i, "t")).to_row())
        for i in range(3)
    ]
    assert len(store_db.list_fish(conn, needs_review=True)) == 3
    store_db.correct(conn, ids[0], "CULL")
    assert len(store_db.list_fish(conn, needs_review=True)) == 2
    conn.close()


def test_latency_percentiles_come_back_per_stage(cfg, sample_dir, tmp_path):
    conn = store_db.connect(tmp_path / "t.db")
    pipe = make_pipeline(cfg, tmp_path, conn=conn)
    for frame in capture.open_source(sample_dir, cfg):
        store_db.insert(conn, pipe.process(frame).to_row())

    stats = store_db.latency_percentiles(conn)
    assert {"detect", "calibrate", "measure", "trust", "decide", "total"} <= set(stats)
    assert stats["total"]["n"] == 6
    assert stats["total"]["p95"] >= stats["total"]["p50"]
    conn.close()


# ---------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------


def test_batch_writes_one_csv_row_per_graded_fish_with_timing(sample_dir, tmp_path):
    out = tmp_path / "results.csv"
    rc = batch_cli.main(
        [str(sample_dir), "--out", str(out), "--db", str(tmp_path / "b.db"), "--quiet"]
    )
    assert rc == 0

    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 6
    for r in rows:
        assert r["decision"] in ("PASS", "CULL", "REVIEW")
        assert float(r["latency_total_ms"]) > 0
        assert float(r["latency_detect_ms"]) >= 0
        assert float(r["fork_length_mm"]) > 0


def test_batch_no_store_leaves_no_database(sample_dir, tmp_path):
    out = tmp_path / "r.csv"
    db = tmp_path / "should_not_exist.db"
    batch_cli.main([str(sample_dir), "--out", str(out), "--db", str(db), "--no-store", "--quiet"])
    assert out.exists()
    assert not db.exists()


def test_batch_on_a_bad_source_exits_nonzero(tmp_path):
    rc = batch_cli.main(["nope/nothing.mp4", "--out", str(tmp_path / "x.csv"), "--no-store"])
    assert rc == 2


# ---------------------------------------------------------------------------
# detector selection
# ---------------------------------------------------------------------------


def test_config_selects_the_stub_by_default(cfg):
    assert build_detector(cfg).name == "stub"


def test_a_yolo_backend_with_no_weights_falls_back_to_the_stub(cfg, capsys):
    """It must not crash a grading station because a weights file moved — but it
    must say so, loudly, because silently grading fish with a made-up model is
    the worst failure this project has."""
    cfg = dict(cfg)
    cfg["detector"] = dict(cfg["detector"], backend="yolo", weights="nope/missing.pt")
    det = build_detector(cfg)
    assert det.name == "stub"
    assert "falling back to the stub" in capsys.readouterr().out


def test_an_unknown_backend_is_an_error_not_a_silent_stub(cfg):
    cfg = dict(cfg)
    cfg["detector"] = dict(cfg["detector"], backend="magic")
    with pytest.raises(ValueError):
        build_detector(cfg)
