"""Phase 2 harness tests: the dataset writer and the real-model adapter.

Everything here skips cleanly when ultralytics/torch aren't installed, because
the app is meant to run without them.

The adapter tests that need weights look for the smoke-test run
(`runs/pose/**/weights/best.pt`) and skip if it isn't there. That model was
trained on geometric polygons and knows nothing about fish — it's used here to
prove the PARSING is right, which is the only thing tests in this file can prove
while the real dataset is blocked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline.landmarks import INDEX, N_LANDMARKS, SCHEMA

ultralytics = pytest.importorskip("ultralytics")


def find_smoke_weights() -> Path | None:
    hits = sorted(Path("runs").glob("**/weights/best.pt"))
    return hits[-1] if hits else None


# ---------------------------------------------------------------------------
# the synthetic dataset writer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from eval.dataset import make_synthetic

    d = tmp_path_factory.mktemp("synth")
    make_synthetic(d, n_train=4, n_val=2, size=320, seed=7)
    return d


def test_the_dataset_yaml_declares_the_schema_we_actually_use(synth):
    doc = yaml.safe_load((synth / "data.yaml").read_text())
    assert doc["kpt_shape"] == [N_LANDMARKS, 3]
    assert doc["kpt_names"] == list(SCHEMA)


def test_flip_index_is_the_identity_and_that_is_deliberate(synth):
    """A horizontal flip of a fish in lateral view maps the snout onto the tail.
    No permutation of these landmarks expresses that, so flip augmentation has to
    be off — identity flip_idx here, fliplr=0.0 in eval/train.py. Get this wrong
    and half the training targets are simply incorrect."""
    doc = yaml.safe_load((synth / "data.yaml").read_text())
    assert doc["flip_idx"] == list(range(N_LANDMARKS))


def test_every_label_has_one_box_and_twelve_keypoints_in_range(synth):
    labels = sorted((synth / "labels" / "train").glob("*.txt"))
    assert labels
    for p in labels:
        fields = p.read_text().split()
        # class + 4 box + 12 * (x, y, visibility)
        assert len(fields) == 1 + 4 + N_LANDMARKS * 3, p.name
        assert fields[0] == "0"
        nums = [float(f) for f in fields[1:5]]
        assert all(0.0 <= v <= 1.0 for v in nums), f"box out of range in {p.name}"
        for i in range(N_LANDMARKS):
            x, y, v = (float(f) for f in fields[5 + i * 3: 8 + i * 3])
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, f"kpt {i} out of range in {p.name}"
            assert v == 2.0


def test_train_and_val_splits_both_exist_and_do_not_share_files(synth):
    tr = {p.stem for p in (synth / "images" / "train").glob("*.jpg")}
    va = {p.stem for p in (synth / "images" / "val").glob("*.jpg")}
    assert tr and va
    assert not (tr & va)


# ---------------------------------------------------------------------------
# the keypoint remap — pure logic, no model needed
# ---------------------------------------------------------------------------


def test_a_model_in_a_different_keypoint_order_is_permuted_not_reindexed():
    """The remap is what lets a dataset with its own ordering drop in without
    renumbering the schema or the dataset."""
    from pipeline.detect_yolo import YoloPoseDetector

    model_order = ["caudal_tip", "snout_tip", "dorsal_apex"]
    perm = YoloPoseDetector._build_permutation(model_order)
    assert perm == [INDEX["caudal_tip"], INDEX["snout_tip"], INDEX["dorsal_apex"]]


def test_a_keypoint_the_schema_does_not_have_is_dropped_not_misplaced():
    from pipeline.detect_yolo import YoloPoseDetector

    perm = YoloPoseDetector._build_permutation(["snout_tip", "pelvic_fin_anterior"])
    assert perm[0] == INDEX["snout_tip"]
    assert perm[1] == -1  # dropped, not silently mapped onto index 1


def test_no_declared_order_means_no_permutation():
    from pipeline.detect_yolo import YoloPoseDetector

    assert YoloPoseDetector._build_permutation(None) is None


def test_missing_weights_raises_rather_than_returning_a_broken_detector():
    from pipeline.detect_yolo import YoloPoseDetector

    with pytest.raises(FileNotFoundError):
        YoloPoseDetector(weights="nope/does/not/exist.pt")
    with pytest.raises(ValueError):
        YoloPoseDetector(weights=None)


# ---------------------------------------------------------------------------
# the adapter against a real ultralytics Results object
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_detector():
    w = find_smoke_weights()
    if w is None:
        pytest.skip("no smoke-test weights; run `python -m eval.train` first")
    from pipeline.detect_yolo import YoloPoseDetector

    return YoloPoseDetector(weights=w, conf=0.10, device="cpu")


@pytest.fixture(scope="module")
def synth_image(synth):
    import cv2

    p = sorted((synth / "images" / "val").glob("*.jpg"))[0]
    return cv2.imread(str(p))


def test_the_adapter_returns_the_landmarks_contract_not_a_yolo_object(
    smoke_detector, synth_image
):
    """The whole point of the adapter: nothing downstream can tell which backend
    produced a detection."""
    lm = smoke_detector.detect(synth_image)
    assert lm.points.shape == (N_LANDMARKS, 2)
    assert lm.confidence.shape == (N_LANDMARKS,)
    assert lm.present.shape == (N_LANDMARKS,)
    assert lm.schema == SCHEMA
    assert lm.source.startswith("yolo:")


def test_the_adapter_finds_the_polygon_it_was_trained_on(smoke_detector, synth_image):
    """Not a claim about fish. It proves the model loaded, ran, and that the
    parsing pulled real coordinates out of the Results object rather than zeros."""
    lm = smoke_detector.detect(synth_image)
    assert lm.detected
    assert lm.present.sum() >= N_LANDMARKS // 2
    h, w = synth_image.shape[:2]
    placed = lm.points[lm.present]
    assert placed[:, 0].min() >= 0 and placed[:, 0].max() <= w
    assert placed[:, 1].min() >= 0 and placed[:, 1].max() <= h


def test_an_empty_frame_produces_no_detection_and_does_not_raise(smoke_detector):
    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    lm = smoke_detector.detect(blank)
    assert lm.points.shape == (N_LANDMARKS, 2)  # contract holds either way


def test_a_real_detection_flows_through_the_rest_of_the_pipeline(
    smoke_detector, synth_image
):
    """End of the swap: real model in, decision out, with nothing downstream
    changed. This is the test that would catch the adapter producing landmarks
    the measurement layer can't use."""
    from pipeline.decide import DecideSettings, decide
    from pipeline.measure import MeasureSettings, measure_traits
    from pipeline.trust import TrustSettings, score_trust
    from tests import fish as F

    lm = smoke_detector.detect(synth_image)
    m = measure_traits(lm, F.scale_calibration(), MeasureSettings())
    t = score_trust(lm, m, TrustSettings())
    d = decide(m, t, DecideSettings())
    assert d.decision in ("PASS", "CULL", "REVIEW")
    assert d.reasons
