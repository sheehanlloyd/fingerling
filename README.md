# fingerling

A single-station fish grading app. Camera or video in, real-world measurements
and a sort decision out.

A fingerling is a juvenile fish, roughly finger-sized — the life stage where fish
farms do quality inspection before deciding which fish move forward in a breeding
program.

## Why

Fish farms still grade and phenotype by hand. Someone nets a fish, measures it
with calipers, eyeballs it for deformities, writes it down. Five minutes a fish.
So farms only ever measure a tiny sample of their stock, and you can't run a
serious breeding program off a sample that small.

The interesting engineering here isn't "can a neural net find a fish" — that's
basically solved. It's everything between a model output and a decision someone
will act on:

- predictions come out in pixels, breeding decisions need millimetres with a
  known error
- a wrong measurement delivered confidently is worse than no measurement, so the
  system has to know when to shut up and ask a human
- every pipeline stage costs time, and if this ever feeds a sorting machine the
  time budget is fixed by belt speed
- the data the model was trained on is not the data it will see

So the weight of this project is on the system around the model, not on squeezing
out accuracy points.

## The honest status, up front

**There is no trained model.** The plan was to fine-tune a YOLO-pose model on the
fishKeypoints dataset from Roboflow Universe. That needs an API key, there wasn't
one, and getting one means making an account. So the detector you get is a stub:
a parametric fish drawn from made-up proportions that doesn't look at the image
at all.

Everything else is real and tested — calibration, measurement, uncertainty, the
trust layer, routing, storage, timing, the API, the UI, and the training harness
itself. But **nothing in this repo has ever seen a fish**, and every record it
writes says which detector produced it so you can't mistake one for the other.

The training harness isn't theoretical, though. Rather than ship an unexecuted
training script, I generated a dataset of geometric fish-shaped polygons and ran
the whole thing: 25 epochs of yolo11n-pose on MPS with `amp=False`, 188 seconds,
and then pushed the resulting model through the real pipeline end to end. So the
dataset writer, the training flags, the MPS path, the weights loading, the
keypoint parsing and the stub/real swap are all verified. **The model that came
out has learned to find a polygon and its metrics say nothing whatsoever about
fish** — see [docs/SESSION_SUMMARY.md](docs/SESSION_SUMMARY.md).

To unblock the real thing, see [docs/DATASETS.md](docs/DATASETS.md).

## Demo, from a fresh clone

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

If you only want to run the app and not train anything, the last three lines of
`requirements.txt` (torch, torchvision, ultralytics) are optional — everything
except `tests/test_train.py` works without them, and that file skips itself.

Make something to point at, since there's no dataset in the repo:

```bash
python -m cli.make_sample
```

That renders 60 frames with a credit-card-shaped calibration target moving
around. **There is no fish in those frames.** The stub doesn't look at the image,
so this exercises real calibration geometry against an imaginary animal.

Grade it to a CSV:

```bash
python -m cli.batch data/sample --out results.csv
```

Or run the web app and watch it:

```bash
python -m api.server
```

Then open http://127.0.0.1:8000. Live view with the landmark overlay, the current
fish's numbers, the records table, the review queue, and an export button.

To see the parts that matter, edit `detector.stub_mode` in `config.yaml`:

- `normal` — everything passes
- `deformed` — bent midline, routes to CULL
- `implausible` — high confidence, impossible anatomy, routes to REVIEW with the
  specific failed constraints listed. This is the case a confidence threshold
  cannot catch and it's the reason the trust layer exists.
- `low_confidence` — clean geometry, unsure model, routes to REVIEW by a
  different path
- `no_detection` — nothing in frame

## How it works

```
capture -> detect -> calibrate -> measure -> trust -> decide -> record
```

Every stage is timed separately and the breakdown is stored per fish.

### Calibration is the part I care most about

An object whose physical size I know exactly sits in the frame. Find its corners,
solve a homography to its plane, and any point on that plane converts to
millimetres. Two targets: an ArUco marker on a phone screen (no printer scaling
error), or a credit card — ISO/IEC 7810 ID-1 is 85.60 × 53.98 mm worldwide, so I
know its size to a hundredth of a millimetre without owning a caliper.

The thing worth reading is [docs/CALIBRATION.md](docs/CALIBRATION.md), because
the obvious quality metric doesn't work. A homography has 8 degrees of freedom
and four corners give exactly 8 equations, so the reprojection residual is ~0 no
matter how badly the target is angled. My near-edge-on test scene reports 0.000
px and it's a terrible view. What actually catches a bad view is *obliquity*,
read off the homography's local Jacobian: 1.0 flat-on, unchanged by in-plane
rotation, climbing with tilt.

If there's no target in frame, the record is marked uncalibrated and no
millimetre figure is reported at all. Ratios still are.

### Measurements carry error bars

Fork length, total length, body depth, depth ratio, peduncle depth, spinal
curvature index, and condition factor (only with a hand-entered weight — never
estimated). Each with an uncertainty.

The uncertainty model is spelled out in full at the top of
`pipeline/measure.py`. Short version: first-order propagation of independent
Gaussian landmark errors, plus a *relative* calibration scale term. Two
consequences worth knowing:

- Landmark errors are assumed independent, which is the assumption most likely to
  be wrong — a model that misses the whole fish by 5 px moves every landmark
  together and that cancels in a distance. So for correlated error this
  **over-estimates**. I'd rather be wrong in that direction.
- Scale error cancels in a ratio, so `depth_ratio` and `curvature_index` carry no
  calibration term at all. That's the whole argument for preferring ratios, and
  it's why the deformity trait is the dimensionless one.

### Trust is two independent signals

Model confidence, and geometric plausibility that doesn't involve the model at
all — an eye can't be behind the gill cover, a tail can't be in front of the
dorsal fin, two landmarks can't sit on the same pixel. Ten named constraints,
eight of them hard (impossibilities) and two soft (range checks on proportions).
Which ones fired is stored, not just the score.

I built the scoring the way the spec described it first and it doesn't work: a
50/50 arithmetic blend lets a fish with its caudal peduncle upside down score
0.80 and pass. So a hard failure zeroes plausibility outright, and the two
signals combine as a weighted **geometric** mean, because they're two necessary
conditions rather than two opinions to average. Both changes are in
[docs/DECISIONS.md](docs/DECISIONS.md) for review.

### Routing

Trust below threshold → REVIEW. Trusted and curvature above threshold → CULL.
Otherwise PASS.

Trust is checked *first*, deliberately. A curvature reading off landmarks you
don't believe is evidence of a bad detection, not of a deformed fish, and culling
on it destroys healthy stock because the model had a bad frame.

Both thresholds are placeholders in `config.yaml`. Phase 3 picks them off a real
coverage curve.

### The review queue is the point

Corrections write `human_corrected` and the human's verdict *without* overwriting
the machine's. The pair of the two is the training data for the next model —
overwrite the original and that signal is gone forever.

## What's measured, and what isn't

Numbers below came out of this machine. Everything else in this README is
description, not measurement.

**Per-stage latency**, 200 frames of the synthetic sample clip at 1280 px long
edge, M4 Pro:

| stage | n | p50 | p95 | mean |
|---|---|---|---|---|
| detect (stub) | 200 | 0.10 | 0.11 | 0.10 |
| calibrate | 200 | 9.90 | 10.61 | 9.96 |
| measure | 200 | 0.17 | 0.21 | 0.17 |
| trust | 200 | 0.15 | 0.17 | 0.15 |
| decide | 200 | 0.00 | 0.00 | 0.00 |
| **total** | 200 | **10.33** | **11.06** | **10.39** |

Calibration is 96% of it, and it's the card path — Canny plus Otsu plus contour
finding over a 1280 px frame, twice. Nobody has tried to make that faster because
nothing has needed it to be yet.

For comparison, the same table with the real (polygon-trained) YOLO-pose model
substituted for the stub, over the 36 synthetic validation images at 640 px and
no calibration target in frame:

| stage | n | p50 | p95 | mean |
|---|---|---|---|---|
| detect (yolo11n-pose, MPS) | 36 | 9.67 | 13.36 | 32.24 |
| calibrate (no target found) | 36 | 0.59 | 0.82 | 0.73 |
| measure | 36 | 0.09 | 0.14 | 0.10 |
| trust | 36 | 0.15 | 0.19 | 0.15 |
| **total** | 36 | **10.50** | **14.23** | **33.24** |

The mean being three times the p50 on `detect` is the first-frame warmup — the
first inference on MPS pays for graph setup. That's exactly the kind of thing a
p50/p95 breakdown is for and a single average would have hidden.

Read that with two caveats. `detect` is the stub, which does no work — a real
model will dominate this table completely and probably reorder it. And
`calibrate` is finding a card on a clean synthetic background, which is the easy
case.

**Calibration accuracy on synthetic scenes** is in
[docs/CALIBRATION.md](docs/CALIBRATION.md) — a 200 mm span measured back to
within about 0.3 mm through a rendered homography. That's a check that the maths
is right, not an accuracy claim about a camera.

**Measurement accuracy in millimetres against real objects: not measured.** That's
Phase 3 — calibrate on one object of known size, measure the others, report the
error distribution.

**Model accuracy against fish: not measured, because there is no fish model.**
The smoke-test run reported pose mAP50 0.995 and mAP50-95 0.515 on its own
validation split — of synthetic polygons it was trained to find. (Two runs at the
same seed gave 0.521 and 0.515: several MPS ops have no deterministic
implementation, so training here is not bit-reproducible.) That is a
statement about whether the training loop works, not about fish, and it must
never be quoted as anything else.
<!-- TODO: my call — replace this whole paragraph once the Roboflow key exists
and eval/train.py has run against real fish. -->

## What breaks

Stated plainly, because most of these aren't fixable by trying harder.

- **The fish is not on the calibration plane.** The whole method assumes the
  target and the fish are coplanar. They aren't — a fish has thickness and its
  midline sits above the board by roughly half its body depth, so it images
  larger and every length reads long. Roughly `d / (d - h)` for camera distance
  `d` and midline height `h`: about 4% at a 500 mm working distance with a 20 mm
  half-depth, which on a 200 mm fish is 8 mm. **It's systematic, averaging frames
  won't help, and it is larger than every other error in this project put
  together.** I haven't measured my working distance, so there's no correction
  applied and no number claimed. <!-- TODO: my call -->
- **Two overlapping fish.** The detector takes the highest-confidence detection
  and ignores the rest. Nothing here tracks a fish across frames either — each
  frame is an independent grading event, so a video of one fish produces one row
  per frame, not one row per fish. Use `--stride` to thin that out.
- **The curvature index is coarse.** The midline is four derived points, so a
  fish bent *between* two of them doesn't register at all. It's a deformity
  proxy, not a spine measurement.
- **The card aspect filter is the only thing identifying a card.** Anything
  bright, convex, four-sided and roughly 1.586:1 gets measured against. A
  paperback at the right angle would do it.
- **No lens distortion model.** No intrinsics, no undistortion. A wide-angle lens
  bends straight lines and a homography can't express that. On the card path the
  outline residual will at least notice; on the ArUco path nothing will.
- **The ArUco path is unusable right now** — `marker_length_mm` is null in config
  and the code refuses to invent a scale. The card path works.
- **The landmark schema is invented.** No public dataset got to decide it, so the
  twelve landmark names may not correspond to anything a real annotator drew.
  Everything downstream refers to landmarks by name and declares which ones each
  trait needs, so a schema swap is a rename table and a missing point nulls the
  affected traits rather than substituting a neighbour — but it's still an
  assumption sitting under everything.
- **Every threshold in `config.yaml` marked PLACEHOLDER is a guess**, including
  the confidence-to-pixels mapping that sets the width of every error bar
  printed. Don't quote them at anyone.

## Layout

```
pipeline/    capture, detect, calibrate, measure, trust, decide, wiring, overlay
             landmarks.py holds the schema — the one place names map to indices
api/         FastAPI server, WebSocket streaming, REST
store/       sqlite, one table, no ORM
cli/         batch.py (video -> csv), make_sample.py (demo frames)
eval/        dataset.py (Roboflow pull + synthetic stand-in), train.py
web/         index.html — the whole frontend, one file, no build step
tests/       116 tests, 104 of which run without torch installed
docs/        CALIBRATION.md, DATASETS.md, DECISIONS.md, SESSION_SUMMARY.md
```

`pipeline/` imports nothing from `api/`, so the pipeline is usable as a library
and testable without a server. The batch CLI and the web server run the same
`Pipeline.process`, so anything the UI shows, the CSV shows too.

## Tests

```bash
pytest -v
```

The ones worth reading are `tests/test_calibrate.py`, because calibration is the
part most likely to be silently wrong, and
`test_the_implausible_stub_goes_to_review_with_named_constraints` in
`tests/test_trust.py`, because that's the failure mode the whole trust layer
exists for.

Every test is built from an answer I chose in advance — a 200 mm fish has to
measure 200 mm, a bent midline has to score above a straight one. A test that
only asserted "returns a MeasurementSet" would pass on code that computed
nonsense.

## Not in scope

No auth, no accounts, no cloud, no multi-camera, no settings page, no dark mode.
It's a single-station operator tool. If a login form appears in this repo,
something has gone wrong.
