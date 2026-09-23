"""Trust and routing tests.

The one that matters most is
`test_the_implausible_stub_goes_to_review_with_named_constraints` — that's the
checkpoint 5 done-condition, and it's the case a confidence threshold on its own
cannot catch, because the stub deliberately keeps its confidence high while
returning a fish that cannot exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.decide import CULL, PASS, REVIEW, DecideSettings, decide
from pipeline.detect import StubDetector, StubSettings
from pipeline.landmarks import Landmarks
from pipeline.measure import MeasureSettings, measure_traits
from pipeline.trust import TrustSettings, body_frame, score_trust
from tests import fish as F

TRUST = TrustSettings()
MEASURE = MeasureSettings()
DECIDE = DecideSettings()

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def run(lm: Landmarks, calib=None):
    """Landmarks in, (measures, trust, decision) out — the back half of the
    pipeline, without the pipeline."""
    calib = calib if calib is not None else F.scale_calibration()
    m = measure_traits(lm, calib, MEASURE)
    t = score_trust(lm, m, TRUST)
    d = decide(m, t, DECIDE)
    return m, t, d


def stub(mode: str, seed: int = 3) -> Landmarks:
    det = StubDetector(StubSettings(mode=mode, jitter_px=0.5, seed=seed))
    return det.detect(FRAME)


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------


def test_a_clean_fish_passes_every_constraint():
    _, t, d = run(F.place(confidence=0.92))
    assert t.failed_names == []
    assert t.plausibility_score == 1.0
    assert d.decision == PASS


def test_the_normal_stub_passes():
    _, t, d = run(stub("normal"))
    assert t.failed_names == []
    assert d.decision == PASS


def test_constraints_do_not_care_which_way_the_fish_faces():
    """Every check is expressed in the body frame, so a fish facing the other way
    has to score identically. If this fails, some constraint is reading raw image
    coordinates and only works for fish pointing right."""
    for angle in (0.0, 90.0, 180.0, 217.0):
        _, t, d = run(F.place(angle_deg=angle, confidence=0.92))
        assert t.failed_names == [], f"failed at {angle} degrees: {t.failed_names}"
        assert d.decision == PASS


# ---------------------------------------------------------------------------
# checkpoint 5's done-condition
# ---------------------------------------------------------------------------


def test_the_implausible_stub_goes_to_review_with_named_constraints():
    _, t, d = run(stub("implausible"))
    assert d.decision == REVIEW
    # The three violations the stub actually injects, each caught by name.
    assert "peduncle_points_not_swapped" in t.failed_names
    assert "eye_between_snout_and_operculum" in t.failed_names
    assert "landmarks_not_degenerate" in t.failed_names


def test_the_implausible_stub_is_still_confident_which_is_the_whole_point():
    """If this ever drops below the review threshold on confidence alone, the
    test above stops proving anything — it would be passing for the wrong
    reason."""
    _, t, _ = run(stub("implausible"))
    assert t.landmark_confidence > TRUST.review_below


def test_the_failed_constraints_carry_a_readable_detail():
    _, t, _ = run(stub("implausible"))
    for c in t.failed:
        assert c.detail, f"{c.name} failed without saying why"


# ---------------------------------------------------------------------------
# one hard failure at a time
# ---------------------------------------------------------------------------


def _mutate(**moves) -> Landmarks:
    """A clean fish with individual landmarks moved to specified mm positions."""
    body = dict(F.FISH_200MM)
    body.update(moves)
    return F.place(body, confidence=0.92)


def test_swapped_peduncle_points_fire_exactly_that_constraint():
    lm = _mutate(peduncle_dorsal=(178.0, 9.6), peduncle_ventral=(178.0, -9.6))
    _, t, d = run(lm)
    assert t.failed_names == ["peduncle_points_not_swapped"]
    assert d.decision == REVIEW


def test_a_dorsal_fin_inserted_backwards_fires_exactly_that_constraint():
    lm = _mutate(dorsal_origin=(125.0, -25.0), dorsal_insertion=(90.0, -21.0))
    _, t, _ = run(lm)
    assert "dorsal_insertion_posterior_to_origin" in t.failed_names


def test_a_tail_in_front_of_the_body_fires_the_fork_constraint():
    lm = _mutate(caudal_fork=(60.0, 0.0), caudal_tip=(70.0, 0.0))
    _, t, _ = run(lm)
    assert "caudal_fork_is_most_posterior" in t.failed_names


def test_a_tail_tip_in_front_of_the_fork_fires_its_own_constraint():
    lm = _mutate(caudal_tip=(190.0, 0.0))
    _, t, _ = run(lm)
    assert "caudal_tip_posterior_to_fork" in t.failed_names


def test_two_landmarks_on_top_of_each_other_are_caught():
    lm = _mutate(eye_posterior=(16.0, -6.0))  # collapsed onto eye_anterior
    _, t, _ = run(lm)
    assert "landmarks_not_degenerate" in t.failed_names


def test_a_flat_fish_is_caught_as_collinear():
    body = {k: (x, 0.0) for k, (x, y) in F.FISH_200MM.items()}
    _, t, _ = run(F.place(body, confidence=0.92))
    assert "body_not_collinear" in t.failed_names


def test_a_dorsal_apex_inside_the_body_is_caught():
    lm = _mutate(dorsal_apex=(105.0, -10.0))  # inboard of the fin base at -25
    _, t, _ = run(lm)
    assert "dorsal_apex_outside_body" in t.failed_names


def test_an_absurd_depth_ratio_fires_the_soft_constraint():
    """A fish three times deeper than it should be. Soft, because a genuinely
    deep-bodied species would land here too and that's a judgement call, not an
    impossibility."""
    lm = _mutate(dorsal_origin=(90.0, -75.0), ventral_margin=(90.0, 75.0),
                 dorsal_apex=(105.0, -94.0), peduncle_dorsal=(178.0, -33.0),
                 peduncle_ventral=(178.0, 33.0))
    _, t, _ = run(lm)
    assert "depth_ratio_in_range" in t.failed_names
    assert all(c.kind == "soft" for c in t.failed)


# ---------------------------------------------------------------------------
# how the two signals combine
# ---------------------------------------------------------------------------


def test_one_hard_failure_zeroes_plausibility_outright():
    lm = _mutate(peduncle_dorsal=(178.0, 9.6), peduncle_ventral=(178.0, -9.6))
    _, t, _ = run(lm)
    assert t.plausibility_score == 0.0


def test_a_soft_failure_only_dents_plausibility():
    lm = _mutate(dorsal_origin=(90.0, -75.0), ventral_margin=(90.0, 75.0),
                 dorsal_apex=(105.0, -94.0), peduncle_dorsal=(178.0, -33.0),
                 peduncle_ventral=(178.0, 33.0))
    _, t, _ = run(lm)
    assert 0.0 < t.plausibility_score < 1.0


def test_a_confident_but_impossible_fish_cannot_pass():
    """The reason plausibility is a veto and not an average. Confidence 0.99 and
    one impossible geometry: a straight weighted mean would give 0.5*0.99 + 0.5*0.9
    = 0.945 and wave it through."""
    lm = _mutate(peduncle_dorsal=(178.0, 9.6), peduncle_ventral=(178.0, -9.6))
    lm = Landmarks.from_mapping(
        {n: tuple(lm.points[i]) for i, n in enumerate(lm.schema) if lm.present[i]},
        confidence=0.99,
    )
    _, t, d = run(lm)
    assert t.landmark_confidence == pytest.approx(0.99)
    assert d.decision == REVIEW


def test_low_confidence_alone_routes_to_review_with_no_constraint_failures():
    """The other half of the design: geometry fine, model unsure. This has to
    reach review by a different route than the impossible fish does."""
    _, t, d = run(stub("low_confidence"))
    assert t.failed_names == []
    assert t.plausibility_score == 1.0
    assert d.decision == REVIEW


def test_no_detection_scores_zero_trust_and_reviews():
    _, t, d = run(Landmarks.empty())
    assert t.trust_score == 0.0
    assert d.decision == REVIEW


# ---------------------------------------------------------------------------
# routing order
# ---------------------------------------------------------------------------


def test_the_deformed_stub_is_culled():
    m, t, d = run(stub("deformed"))
    assert t.failed_names == []
    assert m.curvature_index.value > DECIDE.curvature_cull_above
    assert d.decision == CULL


def test_a_bent_but_untrusted_fish_reviews_rather_than_culls():
    """The ordering that matters. A curvature reading off landmarks you don't
    believe is evidence of a bad detection, not of a deformed fish — culling on
    it would destroy healthy stock because the model had a bad frame."""
    lm = F.place(bend_mm=25.0, confidence=0.25)
    m, t, d = run(lm)
    assert m.curvature_index.value > DECIDE.curvature_cull_above
    assert d.decision == REVIEW


def test_a_trusted_fish_with_no_curvature_reviews_rather_than_passes():
    """Default behaviour: this detector is supposed to measure curvature, so a
    fish that didn't produce one is a fish a human should look at."""
    lm = F.place(
        confidence=0.95,
        drop=("dorsal_origin", "ventral_margin", "peduncle_dorsal", "peduncle_ventral"),
    )
    m, t, d = run(lm)
    assert m.curvature_index is None
    assert d.decision == REVIEW


def test_a_detector_that_cannot_measure_curvature_does_not_review_every_fish():
    """The four-point trained model has no midline landmarks, so curvature is
    None on every fish. Under the default rule that put 25 of 25 real fish in the
    review queue at a median trust of 0.96 — a queue holding all the stock.

    With assess_deformity false the same fish passes on trust alone.
    """
    lm = F.place(
        confidence=0.95,
        drop=("dorsal_origin", "ventral_margin", "peduncle_dorsal", "peduncle_ventral"),
    )
    settings = DecideSettings(assess_deformity=False)
    m, t, _ = run(lm)
    d = decide(m, t, settings)
    assert m.curvature_index is None
    assert d.decision == PASS


def test_not_assessing_deformity_is_stated_on_every_record_it_affects():
    """The switch must not be a silent way to turn culling off. A PASS produced
    without a curvature measurement has to say so, or someone reads it as
    'checked for deformity and fine' — which is exactly the confidently-wrong
    output this whole project exists to avoid."""
    lm = F.place(
        confidence=0.95,
        drop=("dorsal_origin", "ventral_margin", "peduncle_dorsal", "peduncle_ventral"),
    )
    m, t, _ = run(lm)
    d = decide(m, t, DecideSettings(assess_deformity=False))
    assert d.decision == PASS
    assert any("DEFORMITY NOT ASSESSED" in r for r in d.reasons)


def test_the_switch_does_not_rescue_an_untrusted_fish():
    """assess_deformity relaxes the curvature rule and nothing else. A fish whose
    landmarks aren't believable still goes to a human."""
    lm = F.place(
        confidence=0.05,
        drop=("dorsal_origin", "ventral_margin", "peduncle_dorsal", "peduncle_ventral"),
    )
    m, t, _ = run(lm)
    assert decide(m, t, DecideSettings(assess_deformity=False)).decision == REVIEW


# ---------------------------------------------------------------------------
# the body frame itself
# ---------------------------------------------------------------------------


def test_the_body_frame_points_lateral_at_the_dorsal_side():
    """Every dorsal/ventral check is a coin flip if this is wrong, so it gets its
    own test rather than being inferred from the constraints passing."""
    for angle in (0.0, 90.0, 180.0, 270.0):
        lm = F.place(angle_deg=angle)
        f = body_frame(lm)
        assert f.lateral_of(lm.point("dorsal_origin")) > 0
        assert f.lateral_of(lm.point("ventral_margin")) < 0


def test_no_body_frame_without_a_snout():
    lm = F.place(drop=("snout_tip",))
    assert body_frame(lm) is None
    _, t, d = run(lm)
    assert d.decision == REVIEW
