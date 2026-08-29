-- One row per graded fish. One table, stdlib sqlite3, no ORM.
--
-- The columns down to latency_ms_json are the schema CLAUDE.md specifies, in
-- that order. Everything under the EXTENSIONS heading I added, and each one has
-- a reason next to it — they're flagged rather than folded in so the difference
-- between "the spec asked for this" and "I decided this" stays visible.
--
-- Raw landmarks and the frame path are stored, not just the derived numbers. If
-- someone questions a grade I have to be able to show exactly why the system
-- said what it said, and a fork length with no landmarks behind it can't do that.

CREATE TABLE IF NOT EXISTS fish (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT    NOT NULL,   -- ISO 8601, UTC
    source                  TEXT    NOT NULL,   -- "video:tray.mp4", "webcam:0", ...
    frame_path              TEXT,               -- saved frame on disk, if any
    landmarks_json          TEXT    NOT NULL,   -- {name: {x, y, c}}, absent points omitted
    calibrated              INTEGER NOT NULL,   -- 0/1
    calibration_residual    REAL,               -- px; NULL when not meaningful
    fork_length_mm          REAL,
    body_depth_mm           REAL,
    depth_ratio             REAL,
    peduncle_depth_mm       REAL,
    curvature_index         REAL,
    weight_g                REAL,               -- hand-entered, usually NULL
    condition_factor        REAL,
    landmark_confidence     REAL    NOT NULL,
    plausibility_score      REAL    NOT NULL,
    failed_constraints_json TEXT    NOT NULL,   -- ["peduncle_points_not_swapped", ...]
    trust_score             REAL    NOT NULL,
    decision                TEXT    NOT NULL,   -- PASS | CULL | REVIEW
    human_corrected         INTEGER NOT NULL DEFAULT 0,
    human_decision          TEXT,
    latency_ms_json         TEXT    NOT NULL,   -- {stage: ms}, per-stage breakdown

    -- ---- EXTENSIONS (mine, not in CLAUDE.md's list) ----

    -- Every trait in this project is supposed to carry an error bar. Dropping
    -- the sigmas at the storage layer would throw away the one thing that makes
    -- a measurement here different from a number off a ruler.
    fork_length_mm_sigma    REAL,
    body_depth_mm_sigma     REAL,
    depth_ratio_sigma       REAL,
    peduncle_depth_mm_sigma REAL,
    curvature_index_sigma   REAL,

    -- Which detector actually produced this row. With a stub and a real model
    -- behind the same interface, a record that doesn't say which one ran is a
    -- record you cannot interpret later.
    detector_source         TEXT    NOT NULL DEFAULT 'unknown',

    -- Calibration geometry. `reliable` is the gate that actually catches a bad
    -- view (see docs/CALIBRATION.md); storing obliquity means a bad batch can be
    -- diagnosed after the fact instead of re-shot.
    calibration_target      TEXT,
    calibration_obliquity   REAL,
    calibration_reliable    INTEGER,

    -- Reported for completeness; fork length is the primary length trait.
    total_length_mm         REAL,

    -- Why the pipeline decided what it decided, in words, for the review queue.
    decision_reasons_json   TEXT,

    -- Set when a human touches the row in the review queue.
    corrected_at            TEXT,
    human_note              TEXT
);

CREATE INDEX IF NOT EXISTS idx_fish_decision  ON fish (decision);
CREATE INDEX IF NOT EXISTS idx_fish_timestamp ON fish (timestamp);
-- The review queue is the hot query: undecided reviews, newest last.
CREATE INDEX IF NOT EXISTS idx_fish_review    ON fish (decision, human_corrected);
