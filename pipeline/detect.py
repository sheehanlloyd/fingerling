"""Finding the fish and its landmarks.

Two backends behind one interface:

  StubDetector — a parametric fish, no model, no weights. It exists so every
  other stage in this pipeline could be built and tested before a model existed,
  and so the failure modes the trust layer is supposed to catch can be produced
  on demand instead of waited for.

  YoloPoseDetector — the real thing, in pipeline/detect_yolo.py. Imported lazily
  so the app runs with none of the training dependencies installed.

Which one you get is `detector.backend` in config.yaml. It defaults to stub, and
it falls back to stub with a loud note if the real weights aren't on disk, rather
than crashing a grading station because a file moved.

The stub's canonical fish below is *made up*. It's a plausible salmonid in
proportion and nothing more — no dataset was measured to produce it. Every number
that comes out of the stub is therefore about the plumbing, never about fish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from pipeline.landmarks import INDEX, N_LANDMARKS, Landmarks

# A fish in its own frame: x runs 0 at the snout to 1.0 at the tail tip, y is
# perpendicular to the midline, positive downward, in the same units. Eyeballed
# proportions, not measured ones.
CANONICAL_FISH: dict[str, tuple[float, float]] = {
    "snout_tip": (0.000, 0.000),
    "eye_anterior": (0.075, -0.030),
    "eye_posterior": (0.115, -0.030),
    "operculum_posterior": (0.215, 0.000),
    "dorsal_origin": (0.400, -0.125),
    "dorsal_insertion": (0.560, -0.105),
    "dorsal_apex": (0.470, -0.215),
    "ventral_margin": (0.420, 0.125),
    "peduncle_dorsal": (0.800, -0.048),
    "peduncle_ventral": (0.800, 0.048),
    "caudal_fork": (0.900, 0.000),
    "caudal_tip": (1.000, 0.000),
}

STUB_MODES = ("normal", "low_confidence", "implausible", "deformed", "no_detection")


class Detector(Protocol):
    """What the pipeline needs from a detector. Both backends satisfy it."""

    name: str

    def detect(self, frame: np.ndarray) -> Landmarks: ...


@dataclass
class StubSettings:
    mode: str = "normal"
    jitter_px: float = 1.5
    length_frac: float = 0.55  # fish total length as a fraction of frame width
    angle_deg: float = 0.0
    bend: float = 0.0  # extra midline bow, as a fraction of total length
    seed: int = 0


def _fish_points(
    length_px: float, angle_deg: float, centre: np.ndarray, bend: float
) -> dict[str, np.ndarray]:
    """Place the canonical fish in image pixels.

    `bend` bows the whole body: every landmark gets pushed perpendicular to the
    body axis by `bend * length * sin(pi * x)`, which is zero at snout and tail
    and maximal in the middle. That's a scoliotic fish, and it's what the
    curvature index is supposed to notice.
    """
    t = np.deg2rad(angle_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    out: dict[str, np.ndarray] = {}
    for name, (fx, fy) in CANONICAL_FISH.items():
        fy = fy + bend * np.sin(np.pi * fx)
        local = np.array([fx - 0.5, fy]) * length_px
        out[name] = centre + R @ local
    return out


class StubDetector:
    """Fish-shaped landmarks with frame-to-frame drift, and four ways to be wrong.

    The modes aren't decoration. `implausible` in particular is the whole
    justification for the plausibility layer: it returns landmarks with *high*
    confidence that are anatomically impossible, which is the failure mode a
    confidence threshold alone cannot catch.
    """

    name = "stub"

    def __init__(self, settings: StubSettings | None = None):
        self.settings = settings or StubSettings()
        if self.settings.mode not in STUB_MODES:
            raise ValueError(
                f"unknown stub mode {self.settings.mode!r}; one of {STUB_MODES}"
            )
        self._rng = np.random.default_rng(self.settings.seed)
        self._frame_index = 0

    def detect(self, frame: np.ndarray) -> Landmarks:
        s = self.settings
        self._frame_index += 1
        if s.mode == "no_detection":
            return Landmarks.empty(source="stub:no_detection")

        h, w = frame.shape[:2]
        length_px = s.length_frac * w

        # Slow lissajous drift so the overlay moves like something real is under
        # the camera, plus per-landmark jitter so confidence has something to be
        # about. Deterministic given the seed, so tests are repeatable.
        k = self._frame_index
        centre = np.array(
            [w / 2 + 0.03 * w * np.sin(k / 37.0), h / 2 + 0.02 * h * np.sin(k / 23.0)]
        )
        angle = s.angle_deg + 2.0 * np.sin(k / 51.0)

        # `deformed` is just `normal` with a bowed midline. 0.10 of body length
        # is comfortably over the placeholder cull threshold of 0.06, which is
        # the only reason that number is what it is — it exists to make the CULL
        # branch reachable, not because any real fish bends by 10%.
        bend = s.bend if s.mode != "deformed" else max(s.bend, 0.10)
        pts = _fish_points(length_px, angle, centre, bend)
        for name in pts:
            pts[name] = pts[name] + self._rng.normal(0.0, s.jitter_px, size=2)

        if s.mode == "low_confidence":
            conf = {n: float(self._rng.uniform(0.15, 0.35)) for n in pts}
        else:
            conf = {n: float(self._rng.uniform(0.82, 0.97)) for n in pts}

        if s.mode == "implausible":
            pts, conf = self._break_anatomy(pts, conf)

        return Landmarks.from_mapping(
            {n: (float(p[0]), float(p[1])) for n, p in pts.items()},
            confidence=conf,
            source=f"stub:{s.mode}",
            box_px=_bbox(pts),
            detection_confidence=float(np.mean(list(conf.values()))),
        )

    @staticmethod
    def _break_anatomy(
        pts: dict[str, np.ndarray], conf: dict[str, float]
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Produce a fish that cannot exist, while staying confident about it.

        Three separate violations, chosen to fire three different named
        constraints so the review queue shows a list and not a single flag:
          - the two caudal peduncle points are swapped top for bottom
          - the eye is dragged behind the operculum
          - the two eye corners are collapsed onto each other (degenerate)
        """
        pts = {k: v.copy() for k, v in pts.items()}
        pts["peduncle_dorsal"], pts["peduncle_ventral"] = (
            pts["peduncle_ventral"],
            pts["peduncle_dorsal"],
        )
        shift = pts["operculum_posterior"] - pts["eye_posterior"]
        pts["eye_anterior"] = pts["eye_anterior"] + shift * 1.6
        pts["eye_posterior"] = pts["eye_anterior"].copy()
        # Confidence stays high on purpose. That's the point of the mode.
        return pts, conf


def _bbox(pts: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    arr = np.asarray(list(pts.values()), dtype=np.float64)
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].max()),
    )


def stub_settings_from_config(cfg: dict[str, Any]) -> StubSettings:
    d = cfg.get("detector", {})
    return StubSettings(
        mode=d.get("stub_mode", "normal"),
        jitter_px=float(d.get("stub_jitter_px", 1.5)),
        bend=float(d.get("stub_bend", 0.0)),
    )


def build_detector(cfg: dict[str, Any]) -> Detector:
    """Pick a backend from config. Falls back to the stub, never crashes.

    The fallback prints. A grading station silently running on a made-up fish
    model would be the worst possible failure of this whole project, so it says
    so on stdout and it says so in every record it writes (`source` on the
    Landmarks carries the backend name downstream).
    """
    d = cfg.get("detector", {})
    backend = d.get("backend", "stub")

    if backend == "stub":
        return StubDetector(stub_settings_from_config(cfg))

    if backend == "yolo":
        weights = d.get("weights")
        try:
            from pipeline.detect_yolo import YoloPoseDetector  # lazy: heavy import

            return YoloPoseDetector(
                weights=weights,
                conf=float(d.get("conf_threshold", 0.25)),
                device=d.get("device", "mps"),
                keypoint_order=d.get("keypoint_order"),
            )
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            print(
                f"[detect] yolo backend unavailable ({exc}); "
                f"falling back to the stub. Numbers from this run are NOT about fish."
            )
            return StubDetector(stub_settings_from_config(cfg))

    raise ValueError(f"unknown detector backend {backend!r}")
