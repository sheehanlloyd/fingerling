# Decisions

One line per decision, with why. Anything still open sits at the top so I can't
lose track of it.

## Blocked — needs something only I can do

- **The dataset never downloaded.** fishKeypoints on Roboflow Universe needs an
  API key. There was no `ROBOFLOW_API_KEY` (or `RF_KEY`) in the environment, an
  unauthenticated call to `api.roboflow.com` returns HTTP 401 with
  `"This method requires your API key."`, and getting a key means creating a
  Roboflow account — which is off-limits. So Phase 2 training against real fish
  did not happen, and nothing in this repo has ever seen a fish.

  To unblock, one command and one re-run:

  ```
  export ROBOFLOW_API_KEY=...
  python -m eval.dataset --fetch data/fishkeypoints
  python -m eval.dataset --describe data/fishkeypoints   # READ THE SCHEMA FIRST
  python -m eval.train --data data/fishkeypoints/data.yaml --epochs 100
  ```

  Everything downstream was built against the stub detector instead, which is
  what checkpoint 3 exists for. See "the schema I had to invent" below — that's
  the one consequence of this that isn't just "no model yet".

- **ArUco marker physical size.** `calibration.aruco.marker_length_mm` is still
  null in `config.yaml`. It's `displayed_side_px / phone_PPI * 25.4` and I
  haven't looked up my phone's PPI. The code refuses to calibrate against a
  marker rather than guess a scale, so the card path works today and the marker
  path doesn't. Unchanged from before.

## Open — my call, not yet made

- **Every threshold tagged PLACEHOLDER in `config.yaml`.** By design — Phase 3
  picks them off a real coverage curve. Listed here so nobody mistakes them for
  measured values. There are more of them now: `measure.sigma_at_conf1_px`,
  `measure.sigma_at_conf0_px` and `measure.assumed_corner_error_px` joined the
  list, and they're the ones I'd least want quoted, because they set the width of
  every error bar the app prints.

- **Whether the invented landmark schema survives contact with real data.** See
  below. It's twelve points chosen to be a subset of FishPhenoKey's 22 plus one
  that FishPhenoKey doesn't have. When the real annotations land, this either
  needs a rename table or it needs the trait list adjusted, and that's my call.

- **The coplanarity bias.** Still unmeasured and still the largest error in the
  system — bigger than everything in docs/CALIBRATION.md put together. Needs me
  to measure a working distance with a tape measure. Nothing in software fixes
  it.

## Resolved since last time

- **WebSocket transport — settled, `websockets` added.** uvicorn ships no
  WebSocket implementation, `ws="auto"` resolved to none, and the connection was
  simply refused. Options were `websockets`, `wsproto`, `uvicorn[standard]`
  (several more packages for things this doesn't use), or dropping to SSE and
  changing CLAUDE.md's design. Took `websockets==17.1` — one package, for the
  thing the design already asked for. `tests/test_api.py` runs a real uvicorn on
  a real port and opens a real socket, so this is verified rather than assumed.
  One wrinkle: `ws="websockets"` is deprecated in this uvicorn, so the code
  passes `ws="auto"`, which picks the installed implementation.

- **Which dataset — decided, then blocked.** fishKeypoints over FishPhenoKey,
  because FishPhenoKey needs a signed agreement emailed to a maintainer with an
  unknown turnaround and that cannot complete unattended. The tradeoff is a much
  smaller set (593 images vs 10,327 annotated) and a schema nobody has read. Then
  the download itself was blocked on the API key. Written up in docs/DATASETS.md.

## Made by Claude while I wasn't looking, flagged for me to review

These weren't in the spec. I want to either ratify or reverse each one.

### From the calibration work (earlier)

- **ArUco subpixel corner refinement is on** (`CORNER_REFINE_SUBPIX`). Off by
  default, the detector reports the index of the outermost black pixel, which is
  a systematic inward bias of about one pixel — 0.28% on a 360 px marker in my
  synthetic scenes. It's a bias, not noise, so averaging frames won't remove it.
  With refinement the same scene measures within 0.03%. Costs a little time per
  frame; checkpoint 6 will show how much.
- **Card corners get `cv2.cornerSubPix` too**, same reason — `approxPolyDP` picks
  its vertices off a binarised contour so they land on whole pixels. Guarded: if
  refinement moves a corner more than 4 px it's found something else, so the
  original is kept.
- **Added an "obliquity" metric, and `reliable` gates on it rather than on the
  reprojection residual.** The spec says a near-edge-on target should produce a
  high residual. It doesn't, and it can't: a homography has 8 degrees of freedom
  and four corner correspondences give exactly 8 equations, so the solve is exact
  and the residual is ~0 no matter how badly the target is angled. My synthetic
  edge-on scene has a residual of 0.000 px and an obliquity of 15.5. Obliquity
  reads the tilt straight off the homography's local Jacobian: 1.0 is flat-on,
  in-plane rotation stays 1.0, tilt climbs. Details in `docs/CALIBRATION.md`.
- **The card gets a genuine over-determined residual from its outline**, not from
  its corners: every point of the detected outline is mapped into the board plane
  and measured against the ideal rectangle's edges. That catches a bent card or a
  distorting lens, which the corner-only solve cannot see. Flat card scores
  0.92 px; the same card bowed by 1.5 mm scores 7.11 px.
- **`residual_meaningful` flag on the result.** ArUco results carry it as False so
  a structurally-zero residual never gets read downstream as a quality signal.
- **Card detector rejects quads that touch the frame border, and anything over
  50% of frame area.** Not planned — a test caught it. On my 1400x900 test canvas
  the frame's own edge is a rectangle whose aspect ratio is within 2% of a credit
  card's, and the detector happily calibrated against the whole image.
- **`DICT_4X4_50` as the default ArUco dictionary.** Fewest bits, so the largest
  modules for a given physical marker size, so it survives being small on a phone
  screen. Not measured, just the reasoning.
- **`pipeline/config.py` exists**, which isn't in the architecture listing in
  CLAUDE.md. It's ten lines wrapping `yaml.safe_load`. It lives in `pipeline/` so
  the pipeline stays importable without `api/`.
- **Dependency versions are pinned to what resolved on Python 3.14 today**, not to
  versions I chose. All six installed from wheels with no build step:
  opencv-python 5.0.0.93, fastapi 0.141.1, uvicorn 0.52.3, PyYAML 6.0.3,
  numpy 2.5.2, pytest 9.1.1. Note that's OpenCV 5, where the old free-function
  ArUco API is gone; the code uses `ArucoDetector`.

### From the overnight run (checkpoints 3 through 7, plus the Phase 2 harness)

**The big one, read this first:**

- **I invented the landmark schema, because the dataset never arrived.**
  CLAUDE.md says "the public dataset decides this, not me" and the dataset
  didn't get to decide. `pipeline/landmarks.py` defines twelve named points:
  snout_tip, eye_anterior, eye_posterior, operculum_posterior, dorsal_origin,
  dorsal_insertion, dorsal_apex, ventral_margin, peduncle_dorsal,
  peduncle_ventral, caudal_fork, caudal_tip.

  Eleven of those twelve are a deliberate subset of FishPhenoKey's 22, which is
  the one schema I could actually verify (from the paper). The twelfth,
  `caudal_fork`, is **not in FishPhenoKey** — it annotates the tail tip and the
  hypural plate but nothing at the notch between the lobes. I included it anyway
  because CLAUDE.md's primary trait is fork length and I'd rather the gap be
  visible as a missing landmark than hidden behind total length wearing a fork
  length label.

  Two things make this cheaper to undo than it sounds. Everything downstream
  refers to landmarks **by name**, never by index, so remapping is a rename
  table. And `TRAIT_REQUIREMENTS` in the same file declares which landmarks each
  trait needs, so a schema missing a point makes that trait return None with a
  reason attached — it does not substitute a nearby landmark. Drop `caudal_fork`
  and fork length goes null and says why; body depth carries on. There's a test
  for exactly that.

**Trust and routing — two changes, both caught by tests:**

- **Plausibility constraints are tagged hard or soft, and a hard failure zeroes
  the score outright.** I wrote it as CLAUDE.md describes first — fraction of
  constraints passed, blended 50/50 with confidence — and it doesn't work. The
  stub's `implausible` mode has confidence 0.9 and fails three of ten
  constraints, so fraction-passed is 0.7, the blend is 0.80, and a fish with its
  caudal peduncle upside down sails through as a PASS. Averaging a probability
  with a geometric contradiction is a category error. Eight constraints are hard
  (impossibilities: eye behind the operculum, tail in front of the dorsal fin,
  collapsed landmarks) and two are soft (range checks on proportions, where an
  unusual but real fish could legitimately land outside).

- **The two signals combine as a weighted GEOMETRIC mean, not arithmetic.** Also
  a test finding. With arithmetic weights of 0.5, clean geometry contributes 0.5
  on its own, so trust has a floor of 0.5 however unsure the model is — the
  stub's `low_confidence` mode scored 0.625 and passed. Geometric says what I
  actually mean: both are necessary conditions and either one near zero takes the
  product with it. It also makes the hard-constraint veto fall out of the
  arithmetic instead of needing a special case in `decide.py`.

  Reverse either of these and the checkpoint 5 done-condition stops holding, so
  they're load-bearing rather than cosmetic.

- **Trust is checked before curvature in `decide.py`.** An untrusted fish goes to
  REVIEW even when its curvature is over the cull threshold. A curvature reading
  off landmarks you don't believe is evidence of a bad detection, not of a
  deformed fish, and culling on it destroys healthy stock because the model had a
  bad frame. Test: `test_a_bent_but_untrusted_fish_reviews_rather_than_culls`.

- **A trusted detection with no computable curvature goes to REVIEW, not PASS.**
  Passing a fish on a deformity trait that was never measured is the same class
  of mistake as reporting a millimetre figure with no calibration.

**Measurement:**

- **Distances are computed in pixels first, millimetres second.** So an
  uncalibrated frame still produces every dimensionless trait at full quality and
  only the millimetre traits go null. A fish can be culled for spinal curvature
  with no calibration target in frame at all.
- **Ratios carry no calibration term in their error bars, on purpose.** Scale
  cancels in a ratio, so `depth_ratio` and `curvature_index` inherit only the
  landmark uncertainty. That's the entire argument for preferring ratios and it
  has a test rather than a comment.
- **Calibration contributes a RELATIVE scale uncertainty** — corner error divided
  by the target's span in pixels — not an absolute one. That's what makes
  "extrapolating a 200 mm fish off a 40 mm marker costs accuracy" show up in the
  numbers instead of only in the prose.
- **Condition factor uses centimetres.** Fulton's K is `100 * W(g) / L(cm)^3` and
  in millimetres every value comes out a thousand times smaller and unrecognisable.
- **Added `total_length_mm` as a trait.** Not in CLAUDE.md's list. It's free once
  `caudal_tip` exists, and it's the length that survives when the fork point
  can't be placed.

**Stub detector:**

- **Added a fifth stub mode, `deformed`.** CLAUDE.md lists four. Without a bent
  fish there is no way to reach the CULL branch from the CLI or the UI at all,
  so the routing logic would ship with one of its three outcomes never executed
  outside a unit test. It bows the midline by 0.10 of body length, which is over
  the placeholder cull threshold of 0.06 — that number exists to make the branch
  reachable, not because a real fish bends by 10%.
- **The stub's `low_confidence` range is 0.15–0.35.** It was 0.22–0.45, which sat
  close enough to the review threshold that the routing test would have been a
  coin flip on the RNG seed. A stub mode called "the model is unsure" should be
  unambiguously unsure.
- **The stub's canonical fish proportions are eyeballed.** No dataset was
  measured. Every number the stub produces is about the plumbing, never about
  fish, and `detector_source` on every record says so.

**Storage:**

- **Added columns beyond CLAUDE.md's list**, marked as an extension block in
  `store/schema.sql`. Five sigma columns, because dropping the error bars at the
  storage layer would throw away the one thing that distinguishes a measurement
  here from a number off a ruler. `detector_source`, because a row that doesn't
  say whether the stub or a real model produced it is a row you can't interpret
  later. `calibration_target`, `calibration_obliquity`, `calibration_reliable`,
  so a bad batch can be diagnosed after the fact instead of re-shot.
  `total_length_mm`, `decision_reasons_json`, `corrected_at`, `human_note`.
- **An ArUco residual is written as NULL, not as its ~0 value.** Storing a
  structurally-zero number in a column called "residual" invites someone to read
  it as a quality figure in six months.
- **`human_corrected` never overwrites `decision`.** The pair (what the model
  said, what the person said) is the training data for the next model; overwrite
  the machine's verdict and that signal is gone forever.
- **`check_same_thread=False` plus an explicit lock per connection.** FastAPI runs
  `def` endpoints in a threadpool, so one connection legitimately gets used from
  several threads and sqlite3's default turns that into a hard error — which is
  exactly what happened the first time the API served a request. Turning the
  check off without a lock would trade a loud failure for interleaved cursors.

**Pipeline and capture:**

- **`pipeline/overlay.py` exists**, which isn't in CLAUDE.md's architecture
  listing. Annotation has to be importable by `api/`, and `pipeline/` isn't
  allowed to import `api/`, so it lives on the pipeline side. The batch CLI can
  use it too.
- **The three capture functions validate eagerly.** Written as generators, "no
  such file" wasn't raised until the first `next()`, which is inside the
  pipeline's loop and well past the CLI's error handling — so `python -m
  cli.batch nope.mp4` crashed with a traceback instead of exiting 2 with a
  message. A test caught it. They're now plain functions returning a generator.
- **Frames are downscaled to `capture.max_long_edge_px` before anything sees
  them**, and every coordinate in the system — including in the database — is in
  downscaled pixels. Without it the timing numbers from a 4K video and a 720p
  webcam aren't comparable and the whole per-stage latency story falls apart.
- **`--stride` on the batch CLI.** A 30 fps video of one fish on a tray is 30
  near-identical rows per second. This is NOT fish tracking — there's no notion
  of the same fish across frames anywhere in this project, and CLAUDE.md doesn't
  ask for one. Each frame is an independent grading event.
- **Frames with no detection don't get a database row.** Thousands of empty rows
  between the ones that matter. They're still counted and reported.

**API and frontend:**

- **The live loop is one asyncio task, not a thread with a queue.** The pipeline
  is synchronous and CPU-bound, so each frame briefly blocks the event loop. At
  6 fps on a single-station tool that's fine and it keeps state in one place.
  Running at camera rate would need the thread-and-queue version, and that's a
  real change rather than a tweak.
- **Frames go over the socket as base64 JPEG inside the JSON message.** About 33%
  bigger than the bytes and it costs a copy. It buys the frame and the record
  arriving together, so the overlay can never be a frame ahead of the numbers
  beside it. The alternative is a binary channel plus correlation ids, which is a
  lot of machinery for one operator looking at one tray.
- **A slow websocket client is dropped, not queued.** A live view showing a frame
  from thirty seconds ago is worse than one that reconnects.
- **`live:` section added to `config.yaml`** — source, fps, loop — so the server
  has something to grade on a fresh clone without a hidden default.
- **`cli/make_sample.py` exists and isn't in the spec.** The README has to get
  someone from clone to running demo in five minutes and there's no dataset in
  the repo, so there has to be something to point at. It renders a card on a
  moving background. **There is no fish in those frames** — the stub doesn't look
  at the image — so a run over it exercises real calibration geometry against a
  fake animal, and the file says so in its docstring.
- **The correction endpoint accepts REVIEW as a human verdict.** "I looked and I
  still can't tell" is a real answer and it's different from never having looked.

**Phase 2 harness:**

- **Dependencies added: `torch==2.13.0`, `torchvision==0.28.0`,
  `ultralytics==8.4.135`, `websockets==17.1`.** Installed under a constraints file
  so the existing opencv 5.0.0.93 and numpy 2.5.2 pins weren't churned; they
  weren't. torch 2.13 has working MPS on this machine — verified, not assumed.
  The app runs without any of the first three: `pipeline/detect_yolo.py` is
  imported lazily and `detector.backend` defaults to `stub`.
- **`fliplr=0.0` and an identity `flip_idx`.** The one genuinely important
  training decision here. Ultralytics flips images horizontally by default and
  permutes keypoints through `flip_idx` to match, which is right for people —
  a flip swaps left and right wrists. For a fish in lateral view a flip maps the
  snout onto the tail, and there is no permutation of these twelve landmarks that
  expresses that. Leave the default on and half the training targets are wrong.
- **`yolo11n-pose`, the smallest variant.** 593 images is a fine-tune, not a
  training run; anything bigger memorises. Going up a size is a one-word change
  if the data ever justifies it.
- **A synthetic dataset generator (`eval/dataset.py --synthetic`) and a training
  smoke test.** With the real dataset blocked I could either ship an unexecuted
  training script or prove the harness runs. I proved it: geometric fish, real
  YOLO-pose labels, real training on MPS with amp=False. **The resulting model
  has learned to find a polygon and its metrics say nothing about fish.** They're
  in docs/SESSION_SUMMARY.md labelled as such and they should never be quoted.
- **`fetch_roboflow` uses urllib and the documented REST endpoints**, not the
  `roboflow` pip package. It's two requests; a dependency whose only job is to
  make two requests isn't worth it.
- **`detect_yolo.py` treats a (0, 0) keypoint as absent.** Ultralytics writes
  (0, 0) for a keypoint it didn't place, which is a real coordinate — the
  top-left corner of the image. Read as a landmark it puts a snout in the corner
  of the frame at full confidence.
- **One fish per frame: the highest-confidence detection wins.** Two overlapping
  fish is a documented failure mode rather than something this silently averages.

### Follow-ups from the first read-through

- **`store.db.constraint_counts()` added.** I reported the per-constraint tally
  from the YOLO run by hand and got it wrong — counted joined strings, so
  multi-failure rows were bucketed together and no-failure rows appeared as an
  empty key. Real figures are in docs/SESSION_SUMMARY.md. Flattening is the only
  correct way to read that column so it lives in one function now, and the batch
  CLI prints it beside the latency table. It's also the query Phase 3 needs for a
  coverage curve.
- **`eval/train.py` resolves `project` to an absolute path.** Ultralytics treats
  a relative `project` as relative to its own global `runs_dir` (a user-level
  settings.json outside this repo) with the task name inserted, so
  `project="runs/pose"` produced `runs/pose/runs/pose/smoke`. Cosmetic in effect,
  but it also meant the output directory depended on a global config file that
  isn't in the repo — on another machine it would land somewhere else entirely.
- **The README said `tests/ 104 tests` in the layout block and 116 everywhere
  else.** Both were true (104 run without torch) and the layout block now says
  so instead of looking like a typo.

## Decided

- Fork length stays the primary length trait, with `caudal_fork` as an explicit
  landmark, and total length reported alongside it. If the real dataset has no
  fork point, fork length goes null and says so rather than silently becoming
  total length.
- The four-point derived midline is the curvature proxy, with its coarseness
  stated wherever the number appears. A fish bent between two midline points is
  invisible to it. That limitation is documented, not fixed.
