"""How much to believe one detection.

Two signals, and the spec is right that they have to be independent:

  Landmark confidence — what the model says about itself. Cheap, and useless
  exactly when it matters, because the dangerous failure is a model that is
  confidently wrong.

  Anatomical plausibility — geometry that doesn't involve the model at all. An
  eye cannot be behind the gill cover. The tail cannot be in front of the dorsal
  fin. These are facts about fish, and a prediction that violates one is wrong
  no matter how high its confidence was.

WHY PLAUSIBILITY IS NOT JUST AVERAGED IN

I built this as a straight weighted mean first and it doesn't work. Take the
stub's `implausible` mode: confidence 0.9, three of ten constraints failing.
Fraction-passed is 0.7, the blend is 0.5*0.9 + 0.5*0.7 = 0.80, and a fish with
its peduncle upside down sails through as a PASS. Averaging a probability with a
contradiction is a category error — one of those quantities can be 0.7 and the
other cannot.

So constraints are tagged. A `hard` constraint is a geometric impossibility and
failing one drives plausibility to exactly 0.0, which drags the blended trust
under any sane threshold on its own. A `soft` constraint is a range check on a
proportion, where an unusual but real fish could legitimately fall outside, and
those only cost a fraction of the score.

That's a deviation from a literal reading of the spec and it's logged in
docs/DECISIONS.md for review. The alternative I rejected was a separate veto in
decide.py; keeping it inside the score means there's one number to threshold and
one place routing happens.

AND THE BLEND IS GEOMETRIC, NOT ARITHMETIC

Second thing a test caught. With arithmetic weights of 0.5, clean geometry
contributes 0.5 to the total by itself, so trust has a floor of 0.5 no matter how
unsure the model is — the stub's low_confidence mode scored 0.625 and passed. A
weighted geometric mean says the thing I actually mean: both signals are
necessary conditions, and either one near zero takes the product with it.

Every constraint that ran is stored with its name and its verdict, not just the
total. When someone asks why fish #4412 went to review, the answer has to be
"peduncle_points_not_swapped and eye_between_snout_and_operculum failed", not
"the score was 0.45".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from pipeline.landmarks import Landmarks
from pipeline.measure import MeasurementSet


@dataclass(frozen=True)
class TrustSettings:
    weight_landmark_confidence: float = 0.5
    weight_plausibility: float = 0.5
    review_below: float = 0.6
    depth_ratio_min: float = 0.12
    depth_ratio_max: float = 0.45
    peduncle_to_depth_ratio_min: float = 0.20
    peduncle_to_depth_ratio_max: float = 0.85
    min_landmark_separation_px: float = 2.0
    # Measured off the 245 ground-truth annotations in Fish Measurement, then
    # widened. These are NOT placeholders — see docs/DATASETS.md for the
    # distributions they came from.
    eye_axial_min: float = 0.02
    eye_axial_max: float = 0.25
    dorsal_axial_min: float = 0.30
    dorsal_axial_max: float = 0.65

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "TrustSettings":
        t = cfg.get("trust", {})
        p = cfg.get("plausibility", {})
        return cls(
            weight_landmark_confidence=float(t.get("weight_landmark_confidence", 0.5)),
            weight_plausibility=float(t.get("weight_plausibility", 0.5)),
            review_below=float(t.get("review_below", 0.6)),
            depth_ratio_min=float(p.get("depth_ratio_min", 0.12)),
            depth_ratio_max=float(p.get("depth_ratio_max", 0.45)),
            peduncle_to_depth_ratio_min=float(p.get("peduncle_to_depth_ratio_min", 0.20)),
            peduncle_to_depth_ratio_max=float(p.get("peduncle_to_depth_ratio_max", 0.85)),
            min_landmark_separation_px=float(p.get("min_landmark_separation_px", 2.0)),
            eye_axial_min=float(p.get("eye_axial_min", 0.02)),
            eye_axial_max=float(p.get("eye_axial_max", 0.25)),
            dorsal_axial_min=float(p.get("dorsal_axial_min", 0.30)),
            dorsal_axial_max=float(p.get("dorsal_axial_max", 0.65)),
        )


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    kind: str  # "hard" | "soft"
    passed: bool
    applicable: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "passed": self.passed,
            "applicable": self.applicable,
            "detail": self.detail,
        }


@dataclass
class TrustResult:
    landmark_confidence: float
    plausibility_score: float
    trust_score: float
    constraints: list[ConstraintResult] = field(default_factory=list)

    @property
    def failed(self) -> list[ConstraintResult]:
        return [c for c in self.constraints if c.applicable and not c.passed]

    @property
    def failed_names(self) -> list[str]:
        return [c.name for c in self.failed]

    @property
    def hard_failure(self) -> bool:
        return any(c.kind == "hard" for c in self.failed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "landmark_confidence": self.landmark_confidence,
            "plausibility_score": self.plausibility_score,
            "trust_score": self.trust_score,
            "constraints": [c.as_dict() for c in self.constraints],
            "failed": self.failed_names,
        }


# ---------------------------------------------------------------------------
# the body frame every constraint is expressed in
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyFrame:
    """Snout at the origin, +axial toward the tail, +lateral toward the dorsal fin.

    Expressing constraints in this frame is what makes them independent of how
    the fish happens to be lying under the camera. A fish facing left, upside
    down, at 40 degrees, gives exactly the same numbers as one lying flat facing
    right, so none of the checks below need a special case for orientation.
    """

    origin: np.ndarray
    axial: np.ndarray  # unit vector snout -> tail
    lateral: np.ndarray  # unit vector, dorsal positive
    length_px: float

    def axial_of(self, p: np.ndarray) -> float:
        return float((p - self.origin) @ self.axial)

    def lateral_of(self, p: np.ndarray) -> float:
        return float((p - self.origin) @ self.lateral)


def body_frame(lm: Landmarks) -> BodyFrame | None:
    """Build the body frame, or None if the landmarks can't support one."""
    tail = "caudal_fork" if lm.has("caudal_fork") else "caudal_tip"
    if not lm.has("snout_tip", tail):
        return None
    a = lm.point("snout_tip")
    b = lm.point(tail)
    v = b - a
    L = float(np.hypot(*v))
    if L < 1e-6:
        return None
    axial = v / L
    lateral = np.array([-axial[1], axial[0]])
    # Point `lateral` at whichever side the dorsal landmarks are on. Without this
    # the sign is arbitrary and every dorsal/ventral check becomes a coin flip.
    ref = None
    for name in ("dorsal_origin", "dorsal_apex", "dorsal_insertion"):
        if lm.has(name):
            ref = lm.point(name)
            break
    if ref is not None and (ref - a) @ lateral < 0:
        lateral = -lateral
    return BodyFrame(a, axial, lateral, L)


# ---------------------------------------------------------------------------
# the constraints
#
# Each takes (lm, frame, measures, settings) and returns a ConstraintResult.
# They're a plain list so the whole set is readable in one screen and adding one
# is adding one function. Order here is the order they're reported in.
# ---------------------------------------------------------------------------


def _na(name: str, kind: str, why: str) -> ConstraintResult:
    return ConstraintResult(name, kind, passed=True, applicable=False, detail=why)


def _verdict(name: str, kind: str, ok: bool, detail: str) -> ConstraintResult:
    return ConstraintResult(name, kind, passed=bool(ok), applicable=True, detail=detail)


def c_eye_between_snout_and_operculum(lm, f, m, s) -> ConstraintResult:
    name, kind = "eye_between_snout_and_operculum", "hard"
    if not lm.has("eye_anterior", "eye_posterior", "operculum_posterior"):
        return _na(name, kind, "eye or operculum landmark missing")
    eye = (lm.point("eye_anterior") + lm.point("eye_posterior")) / 2
    a_eye = f.axial_of(eye)
    a_op = f.axial_of(lm.point("operculum_posterior"))
    ok = 0.0 < a_eye < a_op
    return _verdict(
        name, kind, ok,
        f"eye at {a_eye/f.length_px:.3f} of body length, operculum at {a_op/f.length_px:.3f}",
    )


def c_dorsal_insertion_posterior_to_origin(lm, f, m, s) -> ConstraintResult:
    name, kind = "dorsal_insertion_posterior_to_origin", "hard"
    if not lm.has("dorsal_origin", "dorsal_insertion"):
        return _na(name, kind, "dorsal fin landmarks missing")
    a0 = f.axial_of(lm.point("dorsal_origin"))
    a1 = f.axial_of(lm.point("dorsal_insertion"))
    return _verdict(
        name, kind, a1 > a0,
        f"origin at {a0/f.length_px:.3f}, insertion at {a1/f.length_px:.3f}",
    )


def c_caudal_fork_is_most_posterior(lm, f, m, s) -> ConstraintResult:
    """Every body landmark has to sit in front of the fork. The tail tip is the
    one exception and it gets its own constraint below."""
    name, kind = "caudal_fork_is_most_posterior", "hard"
    if not lm.has("caudal_fork"):
        return _na(name, kind, "caudal_fork missing")
    a_fork = f.axial_of(lm.point("caudal_fork"))
    behind = [
        n for n in lm.schema
        if n not in ("caudal_fork", "caudal_tip")
        and lm.has(n)
        and f.axial_of(lm.point(n)) > a_fork
    ]
    return _verdict(
        name, kind, not behind,
        "clear" if not behind else "behind the fork: " + ", ".join(behind),
    )


def c_caudal_tip_posterior_to_fork(lm, f, m, s) -> ConstraintResult:
    name, kind = "caudal_tip_posterior_to_fork", "hard"
    if not lm.has("caudal_fork", "caudal_tip"):
        return _na(name, kind, "fork or tip missing")
    a_fork = f.axial_of(lm.point("caudal_fork"))
    a_tip = f.axial_of(lm.point("caudal_tip"))
    return _verdict(
        name, kind, a_tip > a_fork,
        f"fork at {a_fork/f.length_px:.3f}, tip at {a_tip/f.length_px:.3f}",
    )


def c_peduncle_points_not_swapped(lm, f, m, s) -> ConstraintResult:
    name, kind = "peduncle_points_not_swapped", "hard"
    if not lm.has("peduncle_dorsal", "peduncle_ventral"):
        return _na(name, kind, "peduncle landmarks missing")
    ld = f.lateral_of(lm.point("peduncle_dorsal"))
    lv = f.lateral_of(lm.point("peduncle_ventral"))
    return _verdict(
        name, kind, ld > lv,
        f"dorsal side offset {ld:.1f} px, ventral {lv:.1f} px (dorsal must be greater)",
    )


def c_dorsal_apex_outside_body(lm, f, m, s) -> ConstraintResult:
    """The fin margin has to stick out further than where the fin meets the body."""
    name, kind = "dorsal_apex_outside_body", "hard"
    if not lm.has("dorsal_apex", "dorsal_origin"):
        return _na(name, kind, "dorsal apex or origin missing")
    apex = f.lateral_of(lm.point("dorsal_apex"))
    origin = f.lateral_of(lm.point("dorsal_origin"))
    return _verdict(
        name, kind, apex > origin,
        f"apex at {apex:.1f} px, fin base at {origin:.1f} px",
    )


def c_landmarks_not_degenerate(lm, f, m, s) -> ConstraintResult:
    """No two distinct landmarks on top of each other. A collapsed pair is a
    classic heatmap failure — two joints share one peak — and it silently
    produces a zero-length trait."""
    name, kind = "landmarks_not_degenerate", "hard"
    names = [n for n in lm.schema if lm.has(n)]
    if len(names) < 2:
        return _na(name, kind, "fewer than two landmarks present")
    bad = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(np.hypot(*(lm.point(a) - lm.point(b))))
            if d < s.min_landmark_separation_px:
                bad.append(f"{a}/{b} {d:.2f}px")
    return _verdict(
        name, kind, not bad,
        "clear" if not bad else "collapsed pairs: " + "; ".join(bad),
    )


def c_body_not_collinear(lm, f, m, s) -> ConstraintResult:
    """A fish has area. If every landmark falls on one line the pose collapsed."""
    name, kind = "body_not_collinear", "hard"
    names = [n for n in lm.schema if lm.has(n)]
    if len(names) < 4:
        return _na(name, kind, "fewer than four landmarks present")
    lat = np.array([f.lateral_of(lm.point(n)) for n in names])
    spread = float(lat.max() - lat.min())
    frac = spread / f.length_px if f.length_px else 0.0
    return _verdict(
        name, kind, frac > 0.02,
        f"lateral spread is {frac:.4f} of body length",
    )


def c_depth_ratio_in_range(lm, f, m, s) -> ConstraintResult:
    name, kind = "depth_ratio_in_range", "soft"
    if m is None or m.depth_ratio is None:
        return _na(name, kind, "depth ratio not computable")
    v = m.depth_ratio.value
    ok = s.depth_ratio_min <= v <= s.depth_ratio_max
    return _verdict(
        name, kind, ok,
        f"{v:.3f}, allowed {s.depth_ratio_min}..{s.depth_ratio_max}",
    )


def c_peduncle_to_depth_in_range(lm, f, m, s) -> ConstraintResult:
    name, kind = "peduncle_to_depth_ratio_in_range", "soft"
    if m is None or m.peduncle_depth_px is None or m.body_depth_px is None:
        return _na(name, kind, "peduncle or body depth not computable")
    if m.body_depth_px.value <= 0:
        return _na(name, kind, "body depth is zero")
    v = m.peduncle_depth_px.value / m.body_depth_px.value
    ok = s.peduncle_to_depth_ratio_min <= v <= s.peduncle_to_depth_ratio_max
    return _verdict(
        name, kind, ok,
        f"{v:.3f}, allowed {s.peduncle_to_depth_ratio_min}..{s.peduncle_to_depth_ratio_max}",
    )


# --- constraints that work on the four points the real dataset actually gives ---
#
# The originals above were written against a twelve-point schema that no public
# dataset turned out to annotate. On the real model only three of them apply, so
# plausibility had almost nothing to say. These three are checkable with
# snout_tip, caudal_fork, dorsal_origin and eye_centre, which is all the trained
# model emits. Their ranges were measured off the dataset's own ground truth
# rather than guessed — that work is in docs/DATASETS.md.


def c_eye_anterior_to_dorsal_origin(lm, f, m, s) -> ConstraintResult:
    """The eye sits in the head; the dorsal fin does not. If the eye computes as
    posterior to the dorsal origin, the body axis is reversed — which is exactly
    the failure a snout/tail swap produces.

    This fires on 1 of the dataset's own 245 annotations, and that annotation is
    genuinely mislabelled. See docs/DATASETS.md.
    """
    name, kind = "eye_anterior_to_dorsal_origin", "hard"
    if not lm.has("eye_centre", "dorsal_origin"):
        return _na(name, kind, "eye_centre or dorsal_origin missing")
    a_eye = f.axial_of(lm.point("eye_centre")) / f.length_px
    a_dor = f.axial_of(lm.point("dorsal_origin")) / f.length_px
    return _verdict(
        name, kind, a_eye < a_dor,
        f"eye at {a_eye:.3f} of fork length, dorsal origin at {a_dor:.3f}",
    )


def c_eye_in_head_region(lm, f, m, s) -> ConstraintResult:
    """Ground truth puts the eye between 0.042 and 0.162 of fork length back from
    the snout. Outside a widened version of that, the point is somewhere a real
    eye isn't. Soft: an unusual head shape shouldn't veto a detection outright."""
    name, kind = "eye_in_head_region", "soft"
    if not lm.has("eye_centre"):
        return _na(name, kind, "eye_centre missing")
    a = f.axial_of(lm.point("eye_centre")) / f.length_px
    return _verdict(
        name, kind, s.eye_axial_min <= a <= s.eye_axial_max,
        f"eye at {a:.3f} of fork length (allowed {s.eye_axial_min}-{s.eye_axial_max})",
    )


def c_dorsal_origin_in_mid_body(lm, f, m, s) -> ConstraintResult:
    """The tightest thing in the ground truth: every one of the 245 fish has its
    dorsal origin between 0.404 and 0.546 of fork length. Widened to 0.30-0.65
    here so real biological variation isn't punished."""
    name, kind = "dorsal_origin_in_mid_body", "soft"
    if not lm.has("dorsal_origin"):
        return _na(name, kind, "dorsal_origin missing")
    a = f.axial_of(lm.point("dorsal_origin")) / f.length_px
    return _verdict(
        name, kind, s.dorsal_axial_min <= a <= s.dorsal_axial_max,
        f"dorsal origin at {a:.3f} of fork length (allowed {s.dorsal_axial_min}-{s.dorsal_axial_max})",
    )


CONSTRAINTS: tuple[Callable[..., ConstraintResult], ...] = (
    c_eye_between_snout_and_operculum,
    c_dorsal_insertion_posterior_to_origin,
    c_caudal_fork_is_most_posterior,
    c_caudal_tip_posterior_to_fork,
    c_peduncle_points_not_swapped,
    c_dorsal_apex_outside_body,
    c_landmarks_not_degenerate,
    c_body_not_collinear,
    c_depth_ratio_in_range,
    c_peduncle_to_depth_in_range,
    c_eye_anterior_to_dorsal_origin,
    c_eye_in_head_region,
    c_dorsal_origin_in_mid_body,
)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def check_constraints(
    lm: Landmarks, measures: MeasurementSet | None, settings: TrustSettings
) -> list[ConstraintResult]:
    """Run every constraint. Nothing here raises; a constraint that can't be
    evaluated comes back applicable=False with the reason."""
    if not lm.detected:
        return [
            ConstraintResult(fn.__name__.removeprefix("c_"), "hard", True, False, "no detection")
            for fn in CONSTRAINTS
        ]
    frame = body_frame(lm)
    if frame is None:
        return [
            ConstraintResult(
                fn.__name__.removeprefix("c_"), "hard", False, True,
                "no body frame: snout or tail landmark missing",
            )
            for fn in CONSTRAINTS
        ]
    return [fn(lm, frame, measures, settings) for fn in CONSTRAINTS]


def plausibility_score(constraints: list[ConstraintResult]) -> float:
    """0.0 the moment any hard constraint fails; otherwise the fraction of soft
    constraints that passed. See the module docstring for why it's shaped like
    a veto and not like an average."""
    applicable = [c for c in constraints if c.applicable]
    if not applicable:
        return 0.0
    if any(c.kind == "hard" and not c.passed for c in applicable):
        return 0.0
    soft = [c for c in applicable if c.kind == "soft"]
    if not soft:
        return 1.0
    return sum(1 for c in soft if c.passed) / len(soft)


def score_trust(
    lm: Landmarks, measures: MeasurementSet | None, settings: TrustSettings
) -> TrustResult:
    constraints = check_constraints(lm, measures, settings)
    plaus = plausibility_score(constraints)
    conf = lm.mean_confidence() if lm.detected else 0.0

    # WEIGHTED GEOMETRIC mean, not arithmetic. I wrote it arithmetic first and a
    # test caught the problem: with weights of 0.5, perfect geometry contributes
    # 0.5 to the total on its own, so trust can never fall below 0.5 however
    # unsure the model is. The stub's low_confidence mode — mean confidence about
    # 0.25, geometry flawless — scored 0.625 and passed. A detection the model
    # barely believes has to be able to reach a human.
    #
    # A geometric mean says what I actually mean: these are two necessary
    # conditions, not two opinions to average. Either one near zero drags the
    # product to zero. It also makes the hard-constraint veto fall out of the
    # arithmetic instead of needing a special case downstream.
    w1, w2 = settings.weight_landmark_confidence, settings.weight_plausibility
    total = w1 + w2
    if total <= 0 or not lm.detected:
        trust = 0.0
    elif conf <= 0.0 or plaus <= 0.0:
        trust = 0.0
    else:
        trust = float(conf ** (w1 / total) * plaus ** (w2 / total))
    return TrustResult(
        landmark_confidence=float(conf),
        plausibility_score=float(plaus),
        trust_score=float(trust),
        constraints=constraints,
    )
