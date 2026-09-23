"""REST: list records, correct one, export CSV, read timing stats.

No auth, no accounts, no settings endpoints. The spec is explicit that this is a
single-station operator tool and if a login form ever appears here something has
gone wrong.

The correction endpoint is the one that earns its keep. It writes
`human_corrected` and the human's verdict WITHOUT touching the machine's own
decision, because the pair of the two is the training data for the next model.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from pipeline.decide import CULL, PASS, REVIEW
from store import db as store_db

router = APIRouter(prefix="/api")

VALID_DECISIONS = {PASS, CULL, REVIEW}

# Columns the CSV export writes. Same idea as the batch CLI: the derived numbers
# and the timing, not the landmark blob.
EXPORT_COLUMNS = (
    "id", "timestamp", "source", "decision", "human_corrected", "human_decision",
    "human_note", "corrected_at", "trust_score", "landmark_confidence",
    "plausibility_score", "failed_constraints_json", "calibrated",
    "calibration_target", "calibration_residual", "calibration_obliquity",
    "calibration_reliable", "fork_length_mm", "fork_length_mm_sigma",
    "total_length_mm", "body_depth_mm", "body_depth_mm_sigma", "depth_ratio",
    "depth_ratio_sigma", "peduncle_depth_mm", "peduncle_depth_mm_sigma",
    "curvature_index", "curvature_index_sigma", "weight_g", "condition_factor",
    "detector_source", "frame_path", "latency_ms_json",
)


def _conn(request: Request):
    conn = getattr(request.app.state, "conn", None)
    if conn is None:
        raise HTTPException(503, "no database on this server")
    return conn


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    """Turn the JSON-in-a-column fields back into real structures for the wire.
    The browser shouldn't be parsing JSON out of strings."""
    out = dict(row)
    for col, key in (
        ("failed_constraints_json", "failed_constraints"),
        ("latency_ms_json", "latency_ms"),
        ("landmarks_json", "landmarks"),
        ("decision_reasons_json", "decision_reasons"),
    ):
        try:
            out[key] = json.loads(out.get(col) or "null")
        except (TypeError, ValueError):
            out[key] = None
    return out


@router.get("/records")
def list_records(
    request: Request,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    decision: str | None = None,
    needs_review: bool = False,
):
    rows = store_db.list_fish(
        _conn(request), limit=limit, offset=offset,
        decision=decision, needs_review=needs_review,
    )
    return {"records": [_decode(r) for r in rows], "limit": limit, "offset": offset}


@router.get("/records/{fish_id}")
def get_record(request: Request, fish_id: int):
    row = store_db.get(_conn(request), fish_id)
    if row is None:
        raise HTTPException(404, f"no fish {fish_id}")
    return _decode(row)


@router.post("/records/{fish_id}/correct")
def correct_record(
    request: Request,
    fish_id: int,
    payload: dict = Body(...),
):
    """Record a human verdict. `decision` accepts PASS, CULL or REVIEW.

    Accepting REVIEW as a human verdict is deliberate. "I looked and I still
    can't tell" is a real answer and it's different from never having looked.
    """
    decision = str(payload.get("decision", "")).upper()
    if decision not in VALID_DECISIONS:
        raise HTTPException(400, f"decision must be one of {sorted(VALID_DECISIONS)}")
    note = payload.get("note")
    row = store_db.correct(_conn(request), fish_id, decision, note)
    if row is None:
        raise HTTPException(404, f"no fish {fish_id}")
    return _decode(row)


@router.get("/stats")
def stats(request: Request):
    conn = _conn(request)
    return {
        "counts": {
            "total": store_db.count(conn),
            "PASS": store_db.count(conn, PASS),
            "CULL": store_db.count(conn, CULL),
            "REVIEW": store_db.count(conn, REVIEW),
            "awaiting_review": len(
                store_db.list_fish(conn, limit=1000, needs_review=True)
            ),
        },
        "latency_ms": store_db.latency_percentiles(conn),
    }


@router.get("/export.csv")
def export_csv(request: Request, limit: int = Query(100000, ge=1)):
    """Everything, newest first, as a CSV download."""
    rows = store_db.list_fish(_conn(request), limit=limit)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(EXPORT_COLUMNS), extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="fingerling.csv"'},
    )


@router.get("/live")
def live_status(request: Request):
    return request.app.state.live.status()


@router.post("/live/start")
async def live_start(request: Request, payload: dict = Body(default={})):
    from api.ws import LiveSettings

    current = request.app.state.live.settings
    settings = LiveSettings(
        source=str(payload.get("source", current.source)),
        fps=float(payload.get("fps", current.fps)),
        loop=bool(payload.get("loop", current.loop)),
    )
    return await request.app.state.live.start(settings)


@router.post("/live/stop")
async def live_stop(request: Request):
    return await request.app.state.live.stop()
