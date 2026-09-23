# Decisions

One line per decision, with why. Anything still open sits at the top so I can't
lose track of it.

## Phase 2: the day the dataset actually arrived

I got a Roboflow API key and everything that had been blocked since checkpoint 1
unblocked at once. Most of this section is things I got wrong earlier being
corrected by data.

### Switched datasets: fishKeypoints out, Fish Measurement in

**fishKeypoints is aerial drone imagery of wild fish schools.** Two keypoints per
fish, twelve fish per frame, each about 20 pixels long. It's a biomass-counting
dataset and nothing about it suits a single-station grading tool. I'd picked it
blind in checkpoint 1 because it was immediately downloadable, and the survey
said out loud that was a gamble. It lost.

**Fish Measurement** (245 images, one salmonid parr per frame in a white tray,
four keypoints, CC BY 4.0) is the right dataset, and my own survey dismissed it
in three lines without checking. Full write-up in docs/DATASETS.md.

Also: FISH-KP (NIWA) 404s now. The survey's link is dead.

### The schema is four points, and `eye_centre` is new

`snout_tip`, `caudal_fork`, `dorsal_origin`, `eye_centre`. The export ships no
keypoint names, only `kpt_shape: [4, 3]`, so I read them off rendered
annotations. I'd first inferred them from summary statistics and got two of the
four wrong; looking at three pictures corrected it. Statistics were consistent
with a false story.

I added `eye_centre` rather than forcing the dataset's single eye point into
`eye_anterior`. Twelve names stay as the vocabulary and the real four are a
subset, which is what made this a rename table instead of a rewrite.

**What it costs:** fork length and condition factor survive. Total length, body
depth, depth ratio, peduncle depth and **spinal curvature** do not. Curvature is
the deformity proxy and therefore the whole CULL criterion, so with the trained
model this system measures length and does not assess deformity. No public fish
keypoint dataset I could find annotates a midline.

### Three new plausibility constraints, with MEASURED ranges

Only three of the original ten constraints apply to four landmarks, which left
the plausibility half of trust scoring with almost nothing to say. Added
`eye_anterior_to_dorsal_origin` (hard), `eye_in_head_region` (soft) and
`dorsal_origin_in_mid_body` (soft).

Their ranges came off the 245 ground-truth annotations, not off my intuition:
the eye sits at 0.042–0.162 of fork length and the dorsal origin at 0.404–0.546.
Both are widened in config so real biological variation isn't punished. These are
the first numbers in `config.yaml` that aren't placeholders.

### `decide.assess_deformity`, because the first real run put 100% of fish in review

With curvature always None, the existing rule ("trusted but no curvature →
REVIEW") sent **25 of 25 real fish to the review queue at a median trust of
0.96.** A queue containing all the stock is not a grading station.

Two different situations were wearing the same `None`: this detector *can*
measure curvature and didn't for this fish (suspicious, review it), versus this
detector can *never* measure curvature (a documented capability limit, and
re-litigating it per fish tells nobody anything). `assess_deformity: false` says
the second. It does not silently disable culling. Every decision made under it
carries a "DEFORMITY NOT ASSESSED" reason on the record, so a PASS can't be
misread as a clean deformity check.

### Millimetres are now gated on `reliable`, not on `calibrated`. This was a bug

A real test frame with no card in it found something card-shaped at obliquity
4.57 with a 21 px outline residual. The row was correctly tagged
`calibration_reliable: 0`, **and `fork_length_mm: 84.42` was written to the
database anyway.** The flag was right and nothing read it.

That's the exact failure this project exists to prevent, sitting in the code the
whole time. `measure.py` now refuses to produce millimetres without a reliable
calibration. A number nobody should use must not exist, not merely travel next to
a flag that something else has to remember to check.

Also split the "why no millimetres" message in two, because "no target in frame"
and "target found but too oblique to believe" are different problems.

### `max_obliquity` lowered from 2.0 to 1.4, and 2.0 was never justified

Swept a synthetic scene from obliquity 1.00 to 1.71 with `eval/validate_mm`.
**Measurement accuracy does not degrade with tilt.** Max error stayed under
0.15 mm across the whole range, because a homography undoes perspective properly.
What fails is detection: past ~1.43 a foreshortened coin stops being round enough
to find, and past 1.71 the card itself isn't found.

So 1.4 isn't a measured failure point, it's the edge of the evidence. Past it I
have no data either way, and 2.0 never had any.

### Subpixel edge refinement in `validate_mm`, found by a test that knew the answer

The synthetic scene measured every coin ~0.41 mm small, the same magnitude
regardless of coin size, which is a boundary offset rather than a scale error.
Cause: `_binarise` blurs before Otsu, the blur turns each edge into a ramp, and
Otsu's global threshold generally isn't the ramp's midpoint (123 where the
midpoint was ~93). A threshold above the midpoint cuts inside the object.

Deleting the blur fixes the synthetic case (error to 0.04 mm) and is wrong. The
blur is there to stop sensor noise shattering contours in a real photo, and the
size of the bias depends on where Otsu lands, which depends on lighting. It isn't
a constant I can subtract. Refining each boundary point to the half-maximum
intensity makes the measurement threshold-independent, which is the property
worth having. Residual bias +0.14 mm.

I stopped tuning there deliberately. The remaining 0.5% is sub-pixel boundary
convention on a rasterised hard edge that has no optical blur in it, so chasing it
further would be fitting to my own renderer rather than to reality. Real photos
are what arbitrate that.

### Two bugs in `fetch_roboflow` that had never run

`versions[-1]` took the *oldest* version (v1, 55 images) because Roboflow
returns them newest-first. And there was no polling for the server-side export
build, so the first real call died on `{'progress': 0}`. Code that has never
executed is not code that works.

### Train/test leakage in the published split, the most consequential find

245 files, **120 source photographs**. Roboflow exports one file per augmented
copy and the published split was made over files, not photographs, so **34
sources have copies in more than one split**. The copies aren't byte-identical,
so a hash check finds nothing.

Trained on that split the model scored pose mAP50-95 **0.953** on "held-out" test
data that was nothing of the kind. `eval/dataset.py` grew
`audit_split_leakage()` and `regroup_split()` (group split by source photograph:
84/22/14 sources, zero leakage), and every number quoted in the README comes from
a retrain on the clean split.

### What the real photographs changed

Two phone shots of a card and two loonies, and they broke three things no
synthetic scene could have.

**Global Otsu can't see a real tabletop.** The detector thresholded once and
assumed the image was bimodal. That scene had carpet at 80, one loonie at 136,
another at 187 and the card at 227, and Otsu landed on 146, straight between the
two coins. One went to background, the other merged with the card, zero
detections. Replaced with a threshold sweep that keeps whatever is stably round
at any level, which is the stable-region idea behind MSER without the machinery.

**Perimeter-based circularity is the wrong statistic.** A coin whose edge blended
into carpet scored 0.697 circularity against a 0.80 bar and was discarded, while
its area was 133,972 px² against an expected 136,000. The shape was fine, the
outline was fuzzy, and perimeter is exactly the quantity noise inflates. Now it
fits an ellipse and compares areas, which is the same argument that already made
`equivalent_diameter_mm` use area instead of `minEnclosingCircle`.

**Subpixel refinement stopped earning its place, so it's gone.** It existed to
undo a boundary bias that the single-Otsu detector created. The sweep picks the
median stable level, which already sits on the middle of the intensity ramp, so
doing both overshot. It made the synthetic case worse (0.038 mm to 0.159 mm) and
added about +0.1 mm on both real photos. Deleted rather than kept around.

**Rank matching needed a guard.** A round object on the floor behind the table
got detected at 9.3 mm and rank-paired against a 26.50 mm loonie, producing a
-17 mm "error" that was a detection on a different plane. Now: detections outside
half-to-twice the expected range are discarded as detection failures, and a frame
whose detection count doesn't match what was expected reports no measurements at
all rather than pairing the wrong things.

**Rounded corners.** ID-1 specifies a 3.18 mm corner radius, so a real card is
four straight edges joined by arcs, and each arc departs from a sharp-cornered
rectangle by about 0.93 mm. The outline residual now excludes a margin around
each corner. This turned out NOT to be what was inflating my residual (the real
cause was the outline bleeding into carpet fibres) but it's a genuine bug that
every real card would hit and no synthetic scene could show.

### Rounded card corners were costing 2.2% on every measurement

The first real photographs came back +2.9% high with outline residuals of 14-24
px. I guessed the carpet was to blame, reshot on a hard table, and the residual
did NOT drop. Good: the hypothesis was wrong and the reshoot said so.

What it actually was: `approxPolyDP` returns four vertices that sit on a card's
rounded corners rather than where its edges would meet. ID-1 specifies a 3.18 mm
radius, so each vertex is inset by 0.293r = 0.93 mm perpendicular to each edge.
The solver was told those points span 85.60 mm when they spanned 83.74 mm of
card, which is a 2.2% scale error on everything downstream.

The diagnostic that found it: **zero of 6,460 outline points fell inside the
ideal rectangle**, all of them outside by a median of 0.929 mm. A uniform
one-sided offset is an inset corner; a bent card or a bad lens scatters both
ways. 0.93 mm predicted from the corner radius against 0.929 mm measured is not
a coincidence.

Fixed by fitting a line along each of the four straight edges and intersecting
them. Two details matter. The arcs are excluded, so only genuinely straight edge
is fitted. And the lines are fitted to the **intensity gradient**, not to the
contour. `_card_candidates` dilates its Canny mask, which pushes the traced
outline about 2 px outward, a bias `cornerSubPix` had been masking and that a
contour-based fit would have inherited. The residual is measured against those
same gradient-refined points, because measuring a gradient-fitted rectangle
against a mask-derived contour just re-measures the dilation.

Bias on the hard-table shots: **+0.780 mm to +0.115 mm**. Residual: 14-24 px to
under 1 px.

Not one synthetic scene in this repo could have caught it. They all render
sharp-cornered rectangles.

### The residual threshold was a placeholder and turns out to be right

With the corner bug fixed, `max_reprojection_residual_px: 2.0` separates the real
frames cleanly:

| | residual | reliable | worst error |
|---|---|---|---|
| hard table | 0.29 / 0.97 px | yes | 0.149 mm |
| carpet | 3.05 / 5.88 px | no | 0.404 mm |

Frames it accepts measure about three times better than frames it rejects. That's
the first evidence that the residual predicts measurement error rather than just
being a number I report. It also means the carpet hypothesis was right after all
The corner bug was just an order of magnitude bigger and hiding it.

I'm still leaving the value marked as it is rather than declaring it measured.
Four photographs, one camera, two surfaces.

### The result

**0.15 mm worst case on a 26.5 mm object**, on frames that pass their own quality
check. Synthetic scenes give 0.09 mm, so real photographs now cost less than a
tenth of a millimetre over a rendered one, which was not true before the corner
fix, when they cost 0.8 mm.

The coplanarity prediction closes too: +0.29% expected at 350 mm working
distance, +0.43% measured. That's the one error I worked out from geometry before
measuring anything, and it's now the biggest single term left.

### Bow sensitivity, measured

Since the residual now means something, it's worth knowing what it catches. On a
synthetic scene at the 2.0 px threshold: 0.0 mm bow gives 0.00 px, 1.0 mm gives
1.41 px and is accepted, 1.5 mm gives 2.16 px and is rejected. So the detection
limit is about 1.2-1.4 mm of bow. A card bowed less than a millimetre goes
through, and its measurements are wrong by whatever that costs. That's in
`test_bowed_card_produces_a_large_residual_and_is_rejected` rather than in a
comment.

### Deliberately NOT done

- **No new dependency for plotting.** The obliquity chart is drawn with OpenCV,
  which is already a dependency. matplotlib would have been one command and I'd
  rather ask first.
- **Didn't correct the residual +0.14 mm.** See above. It would be fitting to
  my own renderer.
- **Didn't invent a curvature proxy from snout/dorsal/fork.** A dorsal point sits
  on the back, not the midline, so its offset from the snout-fork line is mostly
  just how deep that individual fish is. Natural body-depth variation swamps the
  bend. It would look like a deformity index and wouldn't be one.
- **Config still ships `backend: stub`.** The weights live under `runs/`, which is
  gitignored, so a fresh clone has none. Switching is three lines and the README
  says which.


## Blocked: needs something only I can do

- **ArUco marker physical size.** `calibration.aruco.marker_length_mm` is still
  null in `config.yaml`. It's `displayed_side_px / phone_PPI * 25.4` and I
  haven't looked up my phone's PPI. The code refuses to calibrate against a
  marker rather than guess a scale, so the card path works today and the marker
  path doesn't.

- **Real millimetre validation.** `eval/validate_mm.py` is written and tested
  against synthetic scenes, but no photograph has gone through it. That needs me
  to put a credit card and some coins flat on a table and take a dozen shots.
  Until then there is no defensible accuracy figure in this repo, and the README
  says so rather than quoting the synthetic one.

  ```
  python -m eval.validate_mm shots/ --objects loonie,toonie,quarter --working-distance-mm 300
  ```

- **The coplanarity bias.** Still unmeasured and still the largest error in the
  system. Needs me and a tape measure. Nothing in software fixes it. The real fix
  is a stereo or depth camera, so the subject's height above the plane is
  measured rather than assumed away.

## Open: my call, not yet made

- **Every threshold tagged PLACEHOLDER in `config.yaml`.** By design, Phase 3
  picks them off a real coverage curve. Listed here so nobody mistakes them for
  measured values. There are more of them now: `measure.sigma_at_conf1_px`,
  `measure.sigma_at_conf0_px` and `measure.assumed_corner_error_px` joined the
  list, and they're the ones I'd least want quoted, because they set the width of
  every error bar the app prints.

- **Whether to chase FishPhenoKey after all.** Its 22 keypoints are the only
  route I know to a midline, and a midline is the only route to deformity
  grading. It needs a signed agreement emailed to a maintainer. Everything in
  this repo works without it; the CULL branch doesn't.

- **The coplanarity bias.** Still unmeasured and still the largest error in the
  system, bigger than everything in docs/CALIBRATION.md put together. Needs me
  to measure a working distance with a tape measure. Nothing in software fixes
  it.

## Resolved since last time

- **WebSocket transport: settled, `websockets` added.** uvicorn ships no
  WebSocket implementation, `ws="auto"` resolved to none, and the connection was
  simply refused. Options were `websockets`, `wsproto`, `uvicorn[standard]`
  (several more packages for things this doesn't use), or dropping to SSE and
  changing the spec's design. Took `websockets==17.1`, one package, for the
  thing the design already asked for. `tests/test_api.py` runs a real uvicorn on
  a real port and opens a real socket, so this is verified rather than assumed.
  One wrinkle: `ws="websockets"` is deprecated in this uvicorn, so the code
  passes `ws="auto"`, which picks the installed implementation.

- **Which dataset: decided, then blocked.** fishKeypoints over FishPhenoKey,
  because FishPhenoKey needs a signed agreement emailed to a maintainer with an
  unknown turnaround, and I didn't want the build order blocked behind it. The tradeoff is a much
  smaller set (593 images vs 10,327 annotated) and a schema nobody has read. Then
  the download itself was blocked on the API key. Written up in docs/DATASETS.md.

## Off-spec decisions, flagged for review

These weren't in the plan I started with. Each one is written down so I can
ratify or reverse it deliberately, rather than letting it stand just because it
ended up in the code.

### From the calibration work

- **ArUco subpixel corner refinement is on** (`CORNER_REFINE_SUBPIX`). Off by
  default, the detector reports the index of the outermost black pixel, which is
  a systematic inward bias of about one pixel, 0.28% on a 360 px marker in my
  synthetic scenes. It's a bias, not noise, so averaging frames won't remove it.
  With refinement the same scene measures within 0.03%. Costs a little time per
  frame; checkpoint 6 will show how much.
- **Card corners get `cv2.cornerSubPix` too**, same reason. `approxPolyDP` picks
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
  50% of frame area.** Not planned, a test caught it. On my 1400x900 test canvas
  the frame's own edge is a rectangle whose aspect ratio is within 2% of a credit
  card's, and the detector happily calibrated against the whole image.
- **`DICT_4X4_50` as the default ArUco dictionary.** Fewest bits, so the largest
  modules for a given physical marker size, so it survives being small on a phone
  screen. Not measured, just the reasoning.
- **`pipeline/config.py` exists**, which isn't in the architecture listing in
  the spec. It's ten lines wrapping `yaml.safe_load`. It lives in `pipeline/` so
  the pipeline stays importable without `api/`.
- **Dependency versions are pinned to what resolved on Python 3.14 today**, not to
  versions I chose. All six installed from wheels with no build step:
  opencv-python 5.0.0.93, fastapi 0.141.1, uvicorn 0.52.3, PyYAML 6.0.3,
  numpy 2.5.2, pytest 9.1.1. Note that's OpenCV 5, where the old free-function
  ArUco API is gone; the code uses `ArucoDetector`.

### From checkpoints 3 through 7, plus the Phase 2 harness

**The big one, read this first:**

- **I invented the landmark schema, because the dataset never arrived.**
  The spec says "the public dataset decides this, not me" and the dataset
  didn't get to decide. `pipeline/landmarks.py` defines twelve named points:
  snout_tip, eye_anterior, eye_posterior, operculum_posterior, dorsal_origin,
  dorsal_insertion, dorsal_apex, ventral_margin, peduncle_dorsal,
  peduncle_ventral, caudal_fork, caudal_tip.

  Eleven of those twelve are a deliberate subset of FishPhenoKey's 22, which is
  the one schema I could actually verify (from the paper). The twelfth,
  `caudal_fork`, is **not in FishPhenoKey**. It annotates the tail tip and the
  hypural plate but nothing at the notch between the lobes. I included it anyway
  because the spec's primary trait is fork length and I'd rather the gap be
  visible as a missing landmark than hidden behind total length wearing a fork
  length label.

  Two things make this cheaper to undo than it sounds. Everything downstream
  refers to landmarks **by name**, never by index, so remapping is a rename
  table. And `TRAIT_REQUIREMENTS` in the same file declares which landmarks each
  trait needs, so a schema missing a point makes that trait return None with a
  reason attached, and it does not substitute a nearby landmark. Drop `caudal_fork`
  and fork length goes null and says why; body depth carries on. There's a test
  for exactly that.

**Trust and routing, two changes, both caught by tests:**

- **Plausibility constraints are tagged hard or soft, and a hard failure zeroes
  the score outright.** I wrote it as the spec describes first, fraction of
  constraints passed blended 50/50 with confidence, and it doesn't work. The
  stub's `implausible` mode has confidence 0.9 and fails three of ten
  constraints, so fraction-passed is 0.7, the blend is 0.80, and a fish with its
  caudal peduncle upside down sails through as a PASS. Averaging a probability
  with a geometric contradiction is a category error. Eight constraints are hard
  (impossibilities: eye behind the operculum, tail in front of the dorsal fin,
  collapsed landmarks) and two are soft (range checks on proportions, where an
  unusual but real fish could legitimately land outside).

- **The two signals combine as a weighted GEOMETRIC mean, not arithmetic.** Also
  a test finding. With arithmetic weights of 0.5, clean geometry contributes 0.5
  on its own, so trust has a floor of 0.5 however unsure the model is, and the
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
- **Calibration contributes a RELATIVE scale uncertainty**, corner error divided
  by the target's span in pixels, not an absolute one. That's what makes
  "extrapolating a 200 mm fish off a 40 mm marker costs accuracy" show up in the
  numbers instead of only in the prose.
- **Condition factor uses centimetres.** Fulton's K is `100 * W(g) / L(cm)^3` and
  in millimetres every value comes out a thousand times smaller and unrecognisable.
- **Added `total_length_mm` as a trait.** Not in the spec's list. It's free once
  `caudal_tip` exists, and it's the length that survives when the fork point
  can't be placed.

**Stub detector:**

- **Added a fifth stub mode, `deformed`.** The spec lists four. Without a bent
  fish there is no way to reach the CULL branch from the CLI or the UI at all,
  so the routing logic would ship with one of its three outcomes never executed
  outside a unit test. It bows the midline by 0.10 of body length, which is over
  the placeholder cull threshold of 0.06. That number exists to make the branch
  reachable, not because a real fish bends by 10%.
- **The stub's `low_confidence` range is 0.15–0.35.** It was 0.22–0.45, which sat
  close enough to the review threshold that the routing test would have been a
  coin flip on the RNG seed. A stub mode called "the model is unsure" should be
  unambiguously unsure.
- **The stub's canonical fish proportions are eyeballed.** No dataset was
  measured. Every number the stub produces is about the plumbing, never about
  fish, and `detector_source` on every record says so.

**Storage:**

- **Added columns beyond the spec's list**, marked as an extension block in
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
  several threads and sqlite3's default turns that into a hard error, which is
  exactly what happened the first time the API served a request. Turning the
  check off without a lock would trade a loud failure for interleaved cursors.

**Pipeline and capture:**

- **`pipeline/overlay.py` exists**, which isn't in the spec's architecture
  listing. Annotation has to be importable by `api/`, and `pipeline/` isn't
  allowed to import `api/`, so it lives on the pipeline side. The batch CLI can
  use it too.
- **The three capture functions validate eagerly.** Written as generators, "no
  such file" wasn't raised until the first `next()`, which is inside the
  pipeline's loop and well past the CLI's error handling, so `python -m
  cli.batch nope.mp4` crashed with a traceback instead of exiting 2 with a
  message. A test caught it. They're now plain functions returning a generator.
- **Frames are downscaled to `capture.max_long_edge_px` before anything sees
  them**, and every coordinate in the system, including in the database, is in
  downscaled pixels. Without it the timing numbers from a 4K video and a 720p
  webcam aren't comparable and the whole per-stage latency story falls apart.
- **`--stride` on the batch CLI.** A 30 fps video of one fish on a tray is 30
  near-identical rows per second. This is NOT fish tracking. There's no notion
  of the same fish across frames anywhere in this project, and the spec doesn't
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
- **`live:` section added to `config.yaml`** (source, fps, loop) so the server
  has something to grade on a fresh clone without a hidden default.
- **`cli/make_sample.py` exists and isn't in the spec.** The README has to get
  someone from clone to running demo in five minutes and there's no dataset in
  the repo, so there has to be something to point at. It renders a card on a
  moving background. **There is no fish in those frames**, and the stub doesn't
  look at the image, so a run over it exercises real calibration geometry against a
  fake animal, and the file says so in its docstring.
- **The correction endpoint accepts REVIEW as a human verdict.** "I looked and I
  still can't tell" is a real answer and it's different from never having looked.

**Phase 2 harness:**

- **Dependencies added: `torch==2.13.0`, `torchvision==0.28.0`,
  `ultralytics==8.4.135`, `websockets==17.1`.** Installed under a constraints file
  so the existing opencv 5.0.0.93 and numpy 2.5.2 pins weren't churned; they
  weren't. torch 2.13 has working MPS on this machine, verified, not assumed.
  The app runs without any of the first three: `pipeline/detect_yolo.py` is
  imported lazily and `detector.backend` defaults to `stub`.
- **`fliplr=0.0` and an identity `flip_idx`.** The one genuinely important
  training decision here. Ultralytics flips images horizontally by default and
  permutes keypoints through `flip_idx` to match, which is right for people,
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
  labelled as such and they should never be quoted. That model has since been
  replaced by one trained on real fish. See the Phase 2 section at the top.
- **`fetch_roboflow` uses urllib and the documented REST endpoints**, not the
  `roboflow` pip package. It's two requests; a dependency whose only job is to
  make two requests isn't worth it.
- **`detect_yolo.py` treats a (0, 0) keypoint as absent.** Ultralytics writes
  (0, 0) for a keypoint it didn't place, which is a real coordinate, so the
  top-left corner of the image. Read as a landmark it puts a snout in the corner
  of the frame at full confidence.
- **One fish per frame: the highest-confidence detection wins.** Two overlapping
  fish is a documented failure mode rather than something this silently averages.

### Follow-ups from the first read-through

- **`store.db.constraint_counts()` added.** I reported the per-constraint tally
  from the YOLO run by hand and got it wrong. I counted joined strings, so
  multi-failure rows were bucketed together and no-failure rows appeared as an
  empty key. Flattening is the only
  correct way to read that column so it lives in one function now, and the batch
  CLI prints it beside the latency table. It's also the query Phase 3 needs for a
  coverage curve.
- **`eval/train.py` resolves `project` to an absolute path.** Ultralytics treats
  a relative `project` as relative to its own global `runs_dir` (a user-level
  settings.json outside this repo) with the task name inserted, so
  `project="runs/pose"` produced `runs/pose/runs/pose/smoke`. Cosmetic in effect,
  but it also meant the output directory depended on a global config file that
  isn't in the repo, and on another machine it would land somewhere else entirely.
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
