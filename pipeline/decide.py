"""PASS, CULL, or REVIEW.

The ordering matters and it's the whole design: trust is checked BEFORE the
deformity threshold. A curvature reading from a detection you don't believe is
not evidence of a deformed fish, it's evidence of a bad detection, and culling on
it would destroy healthy stock on the strength of a model error.

So an untrusted fish goes to a human no matter what its curvature says.

The second thing here is `assess_deformity`, and it exists because of a real
result rather than a hypothetical. The trained model emits four landmarks —
snout, caudal fork, dorsal origin, eye — because that is what the only suitable
public dataset annotates. None of those build a midline, so `curvature_index` is
None on every single fish. With the original rule ("trusted but no curvature ->
REVIEW") the first real run put all 25 of 25 fish in the review queue at a median
trust of 0.96. A queue holding 100% of the stock is not a grading station.

There are two different situations wearing the same None, and they deserve
opposite treatment:

  - This detector CAN measure curvature and didn't, for this fish. That's
    suspicious. Review it.
  - This detector can NEVER measure curvature, because its schema has no midline
    points. That's a known, documented capability limit, and re-litigating it
    once per fish tells nobody anything.

`assess_deformity: false` says the second. It does NOT quietly turn culling off:
every decision made under it carries an explicit reason saying deformity was not
assessed, so a PASS can never be misread as "checked for deformity and fine".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipeline.measure import MeasurementSet
from pipeline.trust import TrustResult

PASS = "PASS"
CULL = "CULL"
REVIEW = "REVIEW"


@dataclass(frozen=True)
class DecideSettings:
    review_below: float = 0.6
    curvature_cull_above: float = 0.06
    # Whether the running detector is capable of measuring curvature at all.
    # True with the stub (twelve landmarks, midline derivable). False with the
    # four-point trained model. See the module docstring.
    assess_deformity: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DecideSettings":
        d = cfg.get("decide", {})
        return cls(
            review_below=float(cfg.get("trust", {}).get("review_below", 0.6)),
            curvature_cull_above=float(d.get("curvature_cull_above", 0.06)),
            assess_deformity=bool(d.get("assess_deformity", True)),
        )


@dataclass
class Decision:
    decision: str
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"decision": self.decision, "reasons": list(self.reasons)}


def decide(
    measures: MeasurementSet, trust: TrustResult, settings: DecideSettings
) -> Decision:
    reasons: list[str] = []

    if trust.trust_score < settings.review_below:
        reasons.append(
            f"trust {trust.trust_score:.2f} is below the review threshold "
            f"{settings.review_below:.2f}"
        )
        if trust.failed_names:
            reasons.append("failed constraints: " + ", ".join(trust.failed_names))
        return Decision(REVIEW, reasons)

    k = measures.curvature_index
    if k is None:
        if settings.assess_deformity:
            # This detector is supposed to be able to measure curvature and
            # didn't, for this fish. Passing it on a trait that was never
            # measured is the same class of mistake as reporting millimetres
            # with no calibration. If EVERY fish lands here, the detector can't
            # do it at all and `decide.assess_deformity` should be false.
            reasons.append(
                "curvature index could not be computed for a trusted detection "
                "(if this detector has no midline landmarks, set "
                "decide.assess_deformity: false)"
            )
            return Decision(REVIEW, reasons)
        # Known capability limit. Route on trust, and say so on every record so
        # a PASS is never mistaken for a clean deformity check.
        reasons.append(
            "DEFORMITY NOT ASSESSED: this detector has no midline landmarks, so "
            "spinal curvature was never measured for this fish"
        )
        reasons.append(f"trust {trust.trust_score:.2f} is above the review threshold")
        return Decision(PASS, reasons)

    if k.value > settings.curvature_cull_above:
        reasons.append(
            f"curvature index {k.value:.4f} ± {k.sigma:.4f} is above the cull "
            f"threshold {settings.curvature_cull_above:.4f}"
        )
        return Decision(CULL, reasons)

    reasons.append(
        f"trust {trust.trust_score:.2f}, curvature {k.value:.4f} within limits"
    )
    return Decision(PASS, reasons)
