"""The landmark schema, and the contract every detector returns.

The spec says the public dataset decides the schema. It finally got to. The
dataset is Fish Measurement (Roboflow, CC BY 4.0, 245 images of salmonid parr in
a tray) and it annotates FOUR points per fish, not twelve:

    model index 0 -> snout_tip
    model index 1 -> caudal_fork
    model index 2 -> dorsal_origin
    model index 3 -> eye_centre

I read that off the images, not off a schema file — the export ships no keypoint
names at all, only `kpt_shape: [4, 3]`. docs/DATASETS.md has the renders.

The names below are still the full vocabulary, deliberately. The four real points
are a subset of it, everything else is simply absent, and the machinery that was
built for a missing landmark does the rest: `TRAIT_REQUIREMENTS` nulls the traits
that need a point nobody annotated, and each trust constraint reports itself "not
applicable" rather than passing by default. That's the whole reason nothing
downstream refers to a landmark by index.

What it costs is written down plainly rather than hidden: with the real model,
fork length is computable and body depth, depth ratio, peduncle depth, total
length and the spinal curvature index are not. Curvature is the one that hurts,
because it's the deformity proxy and therefore the entire CULL criterion. No
public fish keypoint dataset I could find annotates a midline.

`eye_centre` is new and exists because the dataset marks one eye point, not the
anterior/posterior pair the rest of this schema assumed. Naming it for what it
actually is beats forcing it into `eye_anterior` and quietly moving the eye
forward by half an eye-width.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# Order is the array order. Index positions are an implementation detail; nothing
# outside this module should use them.
SCHEMA: tuple[str, ...] = (
    "snout_tip",
    "eye_anterior",
    "eye_posterior",
    "eye_centre",
    "operculum_posterior",
    "dorsal_origin",
    "dorsal_insertion",
    "dorsal_apex",
    "ventral_margin",
    "peduncle_dorsal",
    "peduncle_ventral",
    "caudal_fork",
    "caudal_tip",
)

INDEX: dict[str, int] = {name: i for i, name in enumerate(SCHEMA)}
N_LANDMARKS = len(SCHEMA)

# What each point means, in the anatomical sense, so a future me reading an
# annotation guide can match them up without guessing.
DESCRIPTIONS: dict[str, str] = {
    "snout_tip": "most anterior point of the closed mouth",
    "eye_anterior": "anterior margin of the eye",
    "eye_posterior": "posterior margin of the eye",
    "eye_centre": "centre of the eye — the single eye point the real dataset annotates",
    "operculum_posterior": "posterior edge of the gill cover",
    "dorsal_origin": "anterior insertion of the dorsal fin at the body",
    "dorsal_insertion": "posterior insertion of the dorsal fin at the body",
    "dorsal_apex": "outermost point of the dorsal fin margin",
    "ventral_margin": "lowest point of the ventral body outline below dorsal origin",
    "peduncle_dorsal": "top of the caudal peduncle at its narrowest",
    "peduncle_ventral": "bottom of the caudal peduncle at its narrowest",
    "caudal_fork": "the notch between the two tail lobes",
    "caudal_tip": "most posterior point of the tail fin",
}

# Which landmarks each trait needs. measure.py reads this rather than hardcoding,
# so a schema that's missing a point degrades to a null trait instead of a wrong
# one. Keys match the field names on MeasurementSet.
TRAIT_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "fork_length_mm": ("snout_tip", "caudal_fork"),
    "total_length_mm": ("snout_tip", "caudal_tip"),
    "body_depth_mm": ("dorsal_origin", "ventral_margin"),
    "peduncle_depth_mm": ("peduncle_dorsal", "peduncle_ventral"),
    "depth_ratio": ("dorsal_origin", "ventral_margin", "snout_tip", "caudal_fork"),
    "curvature_index": (
        "snout_tip",
        "dorsal_origin",
        "ventral_margin",
        "peduncle_dorsal",
        "peduncle_ventral",
        "caudal_fork",
    ),
}

# The midline is derived, not annotated. Each entry is the set of landmarks whose
# centroid gives one midline point, in head-to-tail order. Two of the four are
# single points that already sit on the midline; two are averages of a dorsal and
# a ventral point.
#
# Four points is a coarse spine. A fish bent *between* two of them doesn't show
# up at all. That limitation is real and it is not fixable without more
# landmarks — see docs/DATASETS.md.
MIDLINE_CHAIN: tuple[tuple[str, ...], ...] = (
    ("snout_tip",),
    ("dorsal_origin", "ventral_margin"),
    ("peduncle_dorsal", "peduncle_ventral"),
    ("caudal_fork",),
)


@dataclass(frozen=True)
class Landmarks:
    """One detection. This is the contract — the stub returns it today and the
    trained model returns it later, and nothing downstream can tell which.

    `points` is (N_LANDMARKS, 2) in image pixels, `confidence` is (N_LANDMARKS,)
    in 0..1. A landmark the detector couldn't place is marked False in `present`
    and its coordinates are meaningless — check `present` before reading a point.

    `detected` False means no fish at all, in which case the arrays are zeros.
    Kept as a separate flag from "all landmarks absent" because they're different
    situations: no fish in frame is normal, a fish with no usable landmarks is a
    detector failure worth seeing in the record.
    """

    detected: bool
    points: np.ndarray  # (N, 2) float64, image pixels
    confidence: np.ndarray  # (N,) float64, 0..1
    present: np.ndarray  # (N,) bool
    schema: tuple[str, ...] = SCHEMA
    source: str = "stub"  # which detector produced this
    box_px: tuple[float, float, float, float] | None = None  # x1,y1,x2,y2 if any
    detection_confidence: float | None = None  # box-level score, if the model has one

    @classmethod
    def empty(cls, source: str = "stub") -> "Landmarks":
        return cls(
            detected=False,
            points=np.zeros((N_LANDMARKS, 2), dtype=np.float64),
            confidence=np.zeros(N_LANDMARKS, dtype=np.float64),
            present=np.zeros(N_LANDMARKS, dtype=bool),
            source=source,
        )

    @classmethod
    def from_mapping(
        cls,
        points: dict[str, tuple[float, float]],
        confidence: dict[str, float] | float = 1.0,
        source: str = "stub",
        **kw,
    ) -> "Landmarks":
        """Build from a name -> (x, y) dict. Anything not in the dict is absent.

        This is how tests construct fixtures, and it's the reason a test can be
        read: `{"snout_tip": (100, 200), ...}` says what it means, `pts[0]`
        doesn't.
        """
        pts = np.zeros((N_LANDMARKS, 2), dtype=np.float64)
        conf = np.zeros(N_LANDMARKS, dtype=np.float64)
        present = np.zeros(N_LANDMARKS, dtype=bool)
        for name, xy in points.items():
            if name not in INDEX:
                raise KeyError(f"{name!r} is not in the schema")
            i = INDEX[name]
            pts[i] = xy
            present[i] = True
            conf[i] = (
                confidence if isinstance(confidence, (int, float))
                else confidence.get(name, 0.0)
            )
        return cls(
            detected=True, points=pts, confidence=conf, present=present, source=source, **kw
        )

    def has(self, *names: str) -> bool:
        """True only if every named landmark was actually placed."""
        return self.detected and all(bool(self.present[INDEX[n]]) for n in names)

    def point(self, name: str) -> np.ndarray:
        """(2,) pixel coordinates. Raises if the landmark isn't present — reading
        a missing landmark should be a crash in a test, not a zero in a record."""
        i = INDEX[name]
        if not self.detected or not self.present[i]:
            raise KeyError(f"landmark {name!r} is not present in this detection")
        return self.points[i].copy()

    def conf(self, name: str) -> float:
        i = INDEX[name]
        return float(self.confidence[i]) if self.present[i] else 0.0

    def mean_confidence(self, names: Iterable[str] | None = None) -> float:
        """Mean confidence over present landmarks. 0.0 if none are present."""
        if names is None:
            mask = self.present
            vals = self.confidence[mask]
        else:
            vals = np.array([self.conf(n) for n in names if self.has(n)])
        return float(vals.mean()) if len(vals) else 0.0

    def as_dict(self) -> dict[str, dict[str, float]]:
        """Name-keyed dump for the DB and the wire. Absent landmarks are omitted
        entirely rather than written as nulls — a missing key is harder to
        misread than a null coordinate."""
        return {
            name: {
                "x": float(self.points[i, 0]),
                "y": float(self.points[i, 1]),
                "c": float(self.confidence[i]),
            }
            for i, name in enumerate(self.schema)
            if self.present[i]
        }


def midline_points(lm: Landmarks) -> np.ndarray | None:
    """Derived head-to-tail midline, (M, 2) pixels, or None if it can't be built.

    Needs at least three of the four chain points to be meaningful — two points
    fit a line exactly and would score zero curvature by construction, which is
    an answer that looks confident and means nothing.
    """
    out = []
    for group in MIDLINE_CHAIN:
        if all(lm.has(n) for n in group):
            out.append(np.mean([lm.point(n) for n in group], axis=0))
    if len(out) < 3:
        return None
    return np.asarray(out, dtype=np.float64)
