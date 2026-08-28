"""PASS, CULL, or REVIEW.

The ordering matters and it's the whole design: trust is checked BEFORE the
deformity threshold. A curvature reading from a detection you don't believe is
not evidence of a deformed fish, it's evidence of a bad detection, and culling on
it would destroy healthy stock on the strength of a model error.

So an untrusted fish goes to a human no matter what its curvature says.
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

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DecideSettings":
        return cls(
            review_below=float(cfg.get("trust", {}).get("review_below", 0.6)),
            curvature_cull_above=float(
                cfg.get("decide", {}).get("curvature_cull_above", 0.06)
            ),
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
        # Trusted landmarks but no curvature means the midline couldn't be built,
        # which is a schema problem, not a fish problem. A human should see it
        # rather than the fish being passed on a trait that was never measured.
        reasons.append("curvature index could not be computed for a trusted detection")
        return Decision(REVIEW, reasons)

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
