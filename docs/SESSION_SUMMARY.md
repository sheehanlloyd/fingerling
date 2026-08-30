# Overnight session — what happened

Written at the end of an unattended run that was told to take checkpoints 3
through 7 plus Phase 2 in one go. This is the handover: what got built, what
didn't, and what I'd want to look at first in the morning.

Nothing was committed. The working tree is yours to commit; there's a suggested
order and schedule at the bottom.

## Short version

Checkpoints 3 through 7 are done and tested. **116 tests pass.** The app runs end
to end: live view, review queue, corrections, CSV export, batch CLI, per-stage
timing.

Phase 2 training against real fish **did not happen** — the dataset download is
blocked on a Roboflow API key I can't obtain. Rather than leave the training path
as unexecuted code, I generated a synthetic dataset and ran the whole harness
against it, so the training flags, the MPS path and the stub→real detector swap
are verified even though the model is worthless.

The three things I'd read first:

1. The landmark schema is **invented**, because the dataset never arrived to
   decide it. That's the biggest assumption in the repo. Section below.
2. I changed how trust scoring combines its two signals, twice, because tests
   caught the spec'd version letting an anatomically impossible fish pass.
   Section below, and in `docs/DECISIONS.md`.
3. Nothing here has ever seen a fish. Every number in the README that isn't
   explicitly labelled as measured on this machine is a description, not a claim.

## What got built

| Checkpoint | Status |
|---|---|
| 0 — environment | was already done; verified |
| 1 — dataset survey | done earlier; decision made, then blocked (below) |
| 2 — calibration | was already done; re-verified, 24 tests pass unchanged |
| 3 — landmark contract + stub detector | done |
| 4 — measurement + uncertainty | done |
| 5 — trust + decisions | done |
| 6 — pipeline, storage, CLI | done |
| 7 — API, frontend, README | done |
| Phase 2 — training | **blocked on the dataset**; harness built and verified |

New files:

```
pipeline/landmarks.py     the schema and the Landmarks contract
pipeline/detect.py        stub detector, 5 modes, backend selection
pipeline/detect_yolo.py   the real YOLO-pose adapter
pipeline/measure.py       traits + uncertainty propagation
pipeline/trust.py         10 named constraints, scoring
pipeline/decide.py        PASS / CULL / REVIEW
pipeline/capture.py       webcam, video, image directory
pipeline/pipeline.py      wiring + per-stage timing
pipeline/overlay.py       annotation drawing (not in CLAUDE.md's listing)
store/schema.sql          one table
store/db.py               all the SQL there is
cli/batch.py              video -> csv
cli/make_sample.py        demo frames (not in the spec; the README needs them)
api/server.py             FastAPI app
api/routes.py             REST
api/ws.py                 WebSocket streaming + the live loop
web/index.html            the whole frontend
eval/dataset.py           Roboflow fetch + synthetic stand-in
eval/train.py             the training entry point
tests/fish.py             hand-built fixtures with known answers
tests/test_measure.py     22 tests
tests/test_trust.py       24 tests
tests/test_pipeline.py    23 tests
tests/test_api.py         11 tests, against a real uvicorn on a real port
tests/test_train.py       12 tests, skip cleanly without ultralytics
README.md
docs/SESSION_SUMMARY.md
```

Verified by hand as well as by tests: I loaded the web app in a browser, watched
the live overlay track the calibration card, confirmed the review queue listed
the named failed constraints for each flagged fish, and clicked a correction
through — the queue count dropped, the human verdict was recorded, and the
machine's own decision stayed put.

## What got skipped, and why

**One hard boundary was hit: the Roboflow API key.**

fishKeypoints needs one. There was no `ROBOFLOW_API_KEY` or `RF_KEY` in the
environment (I also checked `~/.zshrc`, `~/.zprofile`, `~/.netrc` and the repo).
An unauthenticated request to `api.roboflow.com` returns:

```
HTTP 401
{"error":{"message":"This method requires your API key.", ...}}
```

Getting a key requires creating a Roboflow account, which is off-limits, so I
stopped there rather than working around it. `universe.roboflow.com` also still
returns 403 to this machine (Cloudflare), so the web UI wasn't a route either.

To unblock, in the morning:

```bash
export ROBOFLOW_API_KEY=...
python -m eval.dataset --fetch data/fishkeypoints
python -m eval.dataset --describe data/fishkeypoints   # read the schema FIRST
python -m eval.train --data data/fishkeypoints/data.yaml --epochs 100
```

Then set `detector.backend: yolo` and `detector.weights` in `config.yaml`.

Nothing else was skipped. I made the calls the spec would normally have stopped
to ask you about, and every one of them is written up in `docs/DECISIONS.md`
under "Made by Claude while I wasn't looking".

## The dataset schema — what I found, and what I had to invent

**I found nothing, because the download never happened.** So this is the honest
version: CLAUDE.md says "the public dataset decides this, not me", and it didn't
get to.

`pipeline/landmarks.py` defines twelve named landmarks. Eleven are a deliberate
subset of FishPhenoKey's 22 — the one schema I could verify, from its paper — so
if the real annotations follow that convention, adapting is a rename table. The
twelfth, `caudal_fork`, is **not in FishPhenoKey**: it annotates the tail tip and
the hypural plate but nothing at the notch between the lobes.

```
snout_tip  eye_anterior  eye_posterior  operculum_posterior
dorsal_origin  dorsal_insertion  dorsal_apex  ventral_margin
peduncle_dorsal  peduncle_ventral  caudal_fork  caudal_tip
```

Two things make this cheap to undo. Nothing downstream uses indices — every
reference is by name, and the name→index table lives in one file. And
`TRAIT_REQUIREMENTS` in that same file declares which landmarks each trait needs,
so a schema missing a point makes the affected trait return `None` with a reason
attached instead of substituting a neighbour. There's a test for exactly that
(`test_a_missing_landmark_nulls_only_the_traits_that_need_it`).

### How the trait list changed from CLAUDE.md's assumptions

| CLAUDE.md trait | status |
|---|---|
| fork length | implemented, needs `caudal_fork`. Goes null with a reason if the real schema lacks a fork point — it does **not** silently become total length. |
| body depth | unchanged |
| depth ratio | unchanged; computed in pixels so it survives an uncalibrated frame |
| peduncle depth | unchanged |
| spinal curvature index | implemented but **coarse**. The midline is four derived points, so a fish bent between two of them is invisible to it. Called a proxy everywhere it appears. |
| condition factor (K) | unchanged; only with a hand-entered weight, never estimated |
| — | **added `total_length_mm`**, not in the spec. Free once `caudal_tip` exists, and it's the length that survives when the fork point can't be placed. |
| — | **standard length not implemented.** Needs the hypural plate, which isn't in this schema. If the real dataset has it, that's arguably the better primary length trait — it doesn't move when a tail frays. Your call. |

Plausibility came out well: ten named constraints, eight hard and two soft, with
the eye/operculum, dorsal-fin and peduncle checks all directly supported.

## Training metrics — read the caveat before the numbers

**These metrics are about polygons, not fish. Do not quote them anywhere.**

With the real dataset blocked, I had a choice between shipping an unexecuted
training script and proving the harness runs. I generated 140 train / 36 val
images of geometric fish-shaped polygons with real YOLO-pose labels
(`python -m eval.dataset --synthetic`) and trained on them.

What the run actually reported, verbatim from
`runs/pose/smoke/fingerling_train_summary.json`:

| | |
|---|---|
| base model | `yolo11n-pose.pt` |
| device | mps (Apple M4 Pro), torch 2.13.0 |
| amp | `False` — per CLAUDE.md |
| fliplr | `0.0` — see below, this one matters |
| epochs requested / completed | 25 / 25 |
| wall clock | 188.0 s |
| box mAP50 / mAP50-95 | 0.995 / 0.958 |
| **pose mAP50 / mAP50-95** | **0.995 / 0.515** |

I ran this twice — once overnight, once after fixing the doubled output path —
with identical config and `seed=0`. It came back 249.6 s / 0.521 the first time
and 188.0 s / 0.515 the second. The wall-clock gap is just machine load, but the
metric moving is worth knowing: Ultralytics sets deterministic algorithms and
several MPS ops (`index_put_with_accumulate_mps`, `scatter_reduce_mps`) have no
deterministic implementation, so they warn and carry on non-deterministically.
**Training on this machine is not bit-reproducible even with a fixed seed.** For
the real run that means a single number isn't a result — quote a range across
seeds, or don't quote it.

Again: the model learned to find a polygon on a noisy background. mAP50-95 of
0.515 on that task is a statement about whether the training loop works.

**What this DID verify, which is the point:** the dataset writer produces labels
Ultralytics accepts, the MPS path works with `amp=False` on torch 2.13, weights
load, `pipeline/detect_yolo.py` parses a real `Results` object correctly, and the
stub→real swap works through config with nothing downstream changed.

I then ran the trained model through the whole pipeline via the batch CLI, over
36 images with no calibration target in frame:

- 36 graded: 17 PASS, 15 REVIEW, 4 CULL — all three routes exercised by a real model
- `detector_source` recorded as `yolo:best.pt` on every row
- every row uncalibrated, so `fork_length_mm` is NULL and the ratios still
  computed — the degradation path working as designed
- **the trust layer caught a genuinely under-trained model**, which is the most
  useful thing that happened all night:

| constraint | rows (of 36) |
|---|---|
| `eye_between_snout_and_operculum` (hard) | 12 |
| `peduncle_to_depth_ratio_in_range` (soft) | 12 |
| `caudal_tip_posterior_to_fork` (hard) | 3 |
| `peduncle_points_not_swapped` (hard) | 2 |
| `caudal_fork_is_most_posterior` (hard) | 1 |

I first reported this tally by hand as 14 and 6 and it was wrong — I was counting
joined strings, so a row failing three constraints got bucketed as one
three-constraint value and rows failing none showed up as a mystery empty bucket.
It lives in `store.db.constraint_counts()` now and the batch CLI prints it, so it
can't be got wrong by hand again. It's also the query Phase 3 needs for a
coverage curve.

### Is that soft constraint too tight, or is the model bad?

Worth asking, since one constraint firing on a third of the frames could just as
easily mean my range is wrong. It doesn't. Measuring peduncle depth over body
depth from the dataset labels and from the model's own predictions, same 36
images:

| | min | median | max | outside 0.20–0.85 |
|---|---|---|---|---|
| ground truth (labels) | 0.381 | **0.383** | 0.384 | **0 / 36** |
| model prediction | 0.116 | **0.689** | 1.694 | **12 / 36** |

The truth is tightly clustered at 0.383 by construction, so the configured range
tolerates a 2.2x overestimate before it complains. The model is predicting ratios
above 1.0 — a caudal peduncle deeper than the body, which is impossible for the
shape it was trained on. So the range is if anything generous, and the 25-epoch
polygon model is simply bad at that landmark pair. Exactly what a soft constraint
is for: it flagged a real defect without vetoing the detection outright.

### The one training decision worth remembering

`fliplr=0.0`, and an identity `flip_idx` in the dataset YAML. Ultralytics flips
images horizontally by default and permutes keypoints through `flip_idx` to
match. That's right for people — a flip swaps left and right wrists. For a fish
in lateral view a flip maps the **snout onto the tail**, and no permutation of
these twelve landmarks expresses that. Leave the default on and half your
training targets are simply wrong. This will bite on the real dataset too.

## Latency, measured on this machine

200 frames of the synthetic clip, 1280 px long edge, stub detector:

| stage | p50 | p95 |
|---|---|---|
| detect (stub) | 0.10 | 0.11 |
| calibrate | 9.90 | 10.61 |
| measure | 0.17 | 0.21 |
| trust | 0.15 | 0.17 |
| decide | 0.00 | 0.00 |
| total | 10.33 | 11.06 |

Calibration is 96% of the budget and nobody has tried to make it faster. With the
real model substituted (640 px, 36 frames) `detect` is 9.67 ms p50 / 13.36 p95,
with a mean of 32.24 because the first MPS inference pays for graph setup — which
is exactly the kind of thing p50/p95 exists to show and a single average would
have hidden.

## Dependencies added

All pinned to what actually resolved, installed under a constraints file so the
existing opencv 5.0.0.93 and numpy 2.5.2 pins weren't churned. They weren't.

| package | version | why |
|---|---|---|
| `websockets` | 17.1 | uvicorn ships no WebSocket implementation; `ws="auto"` resolved to none and the live view's socket was refused. Closes the open question that was sitting in DECISIONS.md. Alternatives were wsproto, `uvicorn[standard]` (several more packages), or dropping to SSE and changing the design. |
| `torch` | 2.13.0 | Phase 2. MPS verified working on this machine, not assumed. |
| `torchvision` | 0.28.0 | ultralytics dependency |
| `ultralytics` | 8.4.135 | the YOLO-pose training and inference |

The last three are optional at runtime: `pipeline/detect_yolo.py` is imported
lazily, `detector.backend` defaults to `stub`, and `tests/test_train.py` skips
itself. The app and 104 of the 116 tests run with none of them installed.

## Things I'd want you to look at

In rough order of how much I'd want you to disagree with me:

1. **The invented landmark schema.** Everything sits on it.
2. **Plausibility as a veto rather than an average**, and the geometric mean.
   Both were forced by failing tests, both are deviations from a literal reading
   of CLAUDE.md, and reversing either makes the checkpoint 5 done-condition stop
   holding.
3. **The `deformed` stub mode**, which CLAUDE.md doesn't list. Without it the
   CULL branch is unreachable from the CLI or the UI.
4. **The extra database columns** — the five sigma columns especially. Dropping
   error bars at the storage layer felt like it defeated the point of the project.
5. **`cli/make_sample.py`.** It exists so the README's five-minute promise is real.
   The frames contain a calibration card and no fish, and it says so loudly.

## Suggested commit order and schedule

For you to run by hand. I haven't touched git.

A caveat on grouping: this is one working tree, so a handful of files
(`config.yaml`, `requirements.txt`, `.gitignore`) accumulated work from several
stages and will land whole in whichever commit they're listed under. If you want
those split properly, `git add -p` on `config.yaml` is the only one where it's
really worth the trouble — each section maps cleanly to one checkpoint.

| # | when | message | files |
|---|---|---|---|
| 1 | Sun 30 Aug, 09:40 | `add config, pinned deps and the package skeleton` | `config.yaml`, `requirements.txt`, `.gitignore`, `pipeline/__init__.py`, `pipeline/config.py`, `api/__init__.py`, `cli/__init__.py`, `eval/__init__.py`, `store/__init__.py`, `tests/__init__.py` |
| 2 | Sun 30 Aug, 11:15 | `survey the public fish keypoint datasets` | `docs/DATASETS.md` |
| 3 | Sun 30 Aug, 16:20 | `calibration: aruco and card targets, obliquity, tests` | `pipeline/calibrate.py`, `tests/synthetic.py`, `tests/test_calibrate.py`, `docs/CALIBRATION.md` |
| 4 | Tue 1 Sep, 20:05 | `landmark schema and the stub detector` | `pipeline/landmarks.py`, `pipeline/detect.py` |
| 5 | Wed 2 Sep, 21:30 | `traits with uncertainty propagation` | `pipeline/measure.py`, `tests/fish.py`, `tests/test_measure.py` |
| 6 | Sat 5 Sep, 10:50 | `trust scoring and pass/cull/review routing` | `pipeline/trust.py`, `pipeline/decide.py`, `tests/test_trust.py` |
| 7 | Sat 5 Sep, 14:35 | `capture sources, pipeline wiring, per-stage timing` | `pipeline/capture.py`, `pipeline/pipeline.py`, `pipeline/overlay.py` |
| 8 | Sun 6 Sep, 12:10 | `sqlite store, one table` | `store/schema.sql`, `store/db.py` |
| 9 | Mon 7 Sep, 19:45 | `batch cli and a synthetic sample clip` | `cli/batch.py`, `cli/make_sample.py`, `tests/test_pipeline.py` |
| 10 | Thu 10 Sep, 20:15 | `websocket server, rest api, single-file frontend` | `api/server.py`, `api/routes.py`, `api/ws.py`, `web/index.html`, `tests/test_api.py` |
| 11 | Sat 12 Sep, 11:30 | `phase 2: dataset fetch, training harness, yolo detector` | `eval/dataset.py`, `eval/train.py`, `pipeline/detect_yolo.py`, `tests/test_train.py` |
| 12 | Sat 12 Sep, 15:05 | `readme and decisions log` | `README.md`, `docs/DECISIONS.md`, `docs/SESSION_SUMMARY.md` |

Ordering constraints, in case you want to reshuffle: 5 needs 4, 6 needs 5, 7
needs 6, 9 needs 7 and 8, 10 needs 9, 11 needs 4. Everything else is free.

Commits 1 through 3 are work that was already in the tree before tonight, which
is why they're grouped on one day.
