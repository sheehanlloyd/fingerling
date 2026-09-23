"""SQLite, stdlib only, one table.

No ORM on purpose. There's one table and about six queries, and an ORM would be
more code than the queries it replaced.

This module is the only place that knows SQL. Everything else hands it a
FishRecord and gets dicts back.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# The order rows come back in and the order the CSV export writes. Kept explicit
# rather than SELECT *, so adding a column doesn't silently reshape an export
# somebody's spreadsheet depends on.
COLUMNS = (
    "id", "timestamp", "source", "frame_path", "landmarks_json", "calibrated",
    "calibration_residual", "fork_length_mm", "body_depth_mm", "depth_ratio",
    "peduncle_depth_mm", "curvature_index", "weight_g", "condition_factor",
    "landmark_confidence", "plausibility_score", "failed_constraints_json",
    "trust_score", "decision", "human_corrected", "human_decision",
    "latency_ms_json", "fork_length_mm_sigma", "body_depth_mm_sigma",
    "depth_ratio_sigma", "peduncle_depth_mm_sigma", "curvature_index_sigma",
    "detector_source", "calibration_target", "calibration_obliquity",
    "calibration_reliable", "total_length_mm", "decision_reasons_json",
    "corrected_at", "human_note",
)


# One lock per connection. FastAPI runs `def` endpoints in a threadpool, so the
# same connection legitimately gets used from several threads, and sqlite3's
# default `check_same_thread=True` turns that into a hard error. A test caught it
# the first time the API served a request. Turning the check off without adding a
# lock would just trade a loud failure for interleaved cursors, so: check off,
# lock on, every access serialised. Fine for one station; this is not a database
# that needs concurrency.
_LOCKS: "dict[int, threading.RLock]" = {}


@contextmanager
def _locked(conn: sqlite3.Connection):
    lock = _LOCKS.get(id(conn))
    if lock is None:
        lock = _LOCKS.setdefault(id(conn), threading.RLock())
    with lock:
        yield conn


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) and make sure the schema is there."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL so the API can read while the batch CLI is writing. Without it a long
    # batch run locks the reader out and the UI just stops.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()
    _LOCKS[id(conn)] = threading.RLock()
    return conn


def insert(conn: sqlite3.Connection, row: dict[str, Any]) -> int:
    cols = [c for c in COLUMNS if c != "id"]
    sql = (
        f"INSERT INTO fish ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})"
    )
    with _locked(conn):
        cur = conn.execute(sql, [row.get(c) for c in cols])
        conn.commit()
        return int(cur.lastrowid)


def insert_many(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    """One transaction for the whole batch. On a few thousand frames this is the
    difference between a batch run taking seconds and taking minutes."""
    cols = [c for c in COLUMNS if c != "id"]
    sql = (
        f"INSERT INTO fish ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})"
    )
    payload = [[r.get(c) for c in cols] for r in rows]
    with _locked(conn):
        conn.executemany(sql, payload)
        conn.commit()
    return len(payload)


def get(conn: sqlite3.Connection, fish_id: int) -> dict[str, Any] | None:
    with _locked(conn):
        cur = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM fish WHERE id = ?", (fish_id,))
        r = cur.fetchone()
    return dict(r) if r else None


def list_fish(
    conn: sqlite3.Connection,
    limit: int = 100,
    offset: int = 0,
    decision: str | None = None,
    needs_review: bool = False,
) -> list[dict[str, Any]]:
    """Newest first. `needs_review` is the review queue: routed to REVIEW and no
    human has touched it yet."""
    where: list[str] = []
    args: list[Any] = []
    if decision:
        where.append("decision = ?")
        args.append(decision)
    if needs_review:
        where.append("decision = 'REVIEW' AND human_corrected = 0")
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    with _locked(conn):
        cur = conn.execute(
            f"SELECT {', '.join(COLUMNS)} FROM fish {clause} "
            f"ORDER BY id DESC LIMIT ? OFFSET ?",
            (*args, limit, offset),
        )
        return [dict(r) for r in cur.fetchall()]


def count(conn: sqlite3.Connection, decision: str | None = None) -> int:
    with _locked(conn):
        if decision:
            cur = conn.execute("SELECT COUNT(*) FROM fish WHERE decision = ?", (decision,))
        else:
            cur = conn.execute("SELECT COUNT(*) FROM fish")
        return int(cur.fetchone()[0])


def correct(
    conn: sqlite3.Connection,
    fish_id: int,
    human_decision: str,
    note: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any] | None:
    """Record a human's verdict on a fish.

    The machine's own `decision` is NOT overwritten. Keeping both is the entire
    point of the review queue: the pairs of (what the model said, what a person
    said) are the training data for the next model. Overwrite the original and
    that signal is gone forever.
    """
    from datetime import datetime, timezone

    ts = timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _locked(conn):
        conn.execute(
            "UPDATE fish SET human_corrected = 1, human_decision = ?, human_note = ?, "
            "corrected_at = ? WHERE id = ?",
            (human_decision, note, ts, fish_id),
        )
        conn.commit()
    return get(conn, fish_id)


def constraint_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """How many times each named constraint failed, across every row.

    This exists because I got the number wrong by hand. `failed_constraints_json`
    is a list per row, and counting the lists as whole values buckets a row that
    failed three constraints under "three-constraints-at-once" and a row that
    failed none under the empty list, so the per-constraint totals come out
    low and there's a mystery empty bucket. Flattening is the only correct way
    to read it, so it lives here rather than in whatever one-liner is to hand.

    Rows with no failures contribute nothing; use `count()` for the denominator.
    """
    with _locked(conn):
        rows = conn.execute("SELECT failed_constraints_json FROM fish").fetchall()
    out: dict[str, int] = {}
    for (blob,) in rows:
        try:
            names = json.loads(blob) or []
        except (TypeError, ValueError):
            continue
        for name in names:
            out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def latency_percentiles(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    """p50 and p95 per pipeline stage, over every row in the database.

    Computed in Python from the stored per-row JSON rather than kept as a running
    aggregate, because the raw breakdown is what's stored and any summary should
    be re-derivable from it. Fine at this scale; it reads every row.
    """
    import statistics

    with _locked(conn):
        rows = conn.execute("SELECT latency_ms_json FROM fish").fetchall()
    buckets: dict[str, list[float]] = {}
    for (blob,) in rows:
        try:
            d = json.loads(blob)
        except (TypeError, ValueError):
            continue
        for stage, ms in d.items():
            if isinstance(ms, (int, float)):
                buckets.setdefault(stage, []).append(float(ms))

    out: dict[str, dict[str, float]] = {}
    for stage, vals in buckets.items():
        vals.sort()
        out[stage] = {
            "n": len(vals),
            "p50": statistics.median(vals),
            # Nearest-rank p95. With fewer than 20 samples this is just the max,
            # which is the honest answer, because you can't have a p95 from 5 numbers.
            "p95": vals[min(len(vals) - 1, int(round(0.95 * len(vals))) - 1 if len(vals) > 1 else 0)],
            "mean": sum(vals) / len(vals),
        }
    return out
