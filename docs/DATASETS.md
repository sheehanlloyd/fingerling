# Checkpoint 1 — dataset survey

What's out there for fish keypoints, what each one would let me compute, and what
it costs to get. The survey below is what I wrote before I had an API key. The
section immediately after this one is what actually happened when I got one, and
it's mostly a record of the survey being wrong.

---

# What actually happened

I got a Roboflow key and downloaded the thing I'd chosen. Three findings, in
descending order of how much they cost me.

## 1. fishKeypoints — the dataset I picked — is unusable for this project

I chose it blind. That was the gamble I flagged in the survey and it lost.

It's **aerial drone imagery of wild fish schools**. About 12 fish per frame, each
roughly 20 pixels long, in open water. 5,810 annotations over 496 training
images, and **two** keypoints per fish: snout and tail. `kpt_shape: [2, 3]`.

It's a biomass-counting dataset. Nothing about it fits a single-station grading
tool: no anatomical landmarks, no single subject, no controlled surface. Ten of
my twelve landmarks simply don't exist in it, and neither does any trait beyond
a length.

Also worth recording: **FISH-KP (NIWA) is gone.** The URL in the survey below
404s now. Don't trust that row.

## 2. Fish Measurement is the right dataset, and I'd dismissed it

245 images, CC BY 4.0, **one fish per image**, salmonid parr lying in a shallow
white tray. That's the grading station, photographed. The survey wrote it off in
three lines because the name "suggested length endpoints" — it has four
keypoints, and the survey never checked.

`kpt_shape: [4, 3]`, and the export ships **no keypoint names at all**, so I read
them off the annotations:

![The four keypoints](images/schema_4kp.jpg)

| model index | landmark | in my schema? |
|---|---|---|
| 0 | `snout_tip` | yes |
| 1 | `caudal_fork` | yes — the notch, not the tail tip |
| 2 | `dorsal_origin` | yes |
| 3 | `eye_centre` | **added** — the schema had an anterior/posterior pair, this is one point |

I nearly got this wrong. Reading it off the *statistics* — median x-position
within the bounding box — I concluded kp0 was the tail and kp3 was the caudal
fork. Rendering three images and looking at them said kp0 is the snout and kp3 is
the eye. The numbers were consistent with a story that happened to be false.

### What four landmarks costs

| trait | status |
|---|---|
| **fork length** | computable. Snout to fork is exactly what's annotated. |
| condition factor (K) | computable, with a hand-entered weight |
| total length | no `caudal_tip` |
| body depth | no `ventral_margin` |
| depth ratio | needs body depth |
| peduncle depth | no peduncle points |
| **spinal curvature** | **no midline. This is the one that hurts.** |

Curvature is the deformity proxy, and deformity is the entire CULL criterion. No
public fish keypoint dataset I could find annotates a midline, so with the
trained model the system measures length and does not assess deformity. It says
so on every record rather than implying a check it never did.

Of the ten plausibility constraints, only three applied to four landmarks. I
added three more that work on what's actually there, with ranges measured off the
ground truth rather than guessed:

| measured from 245 annotations | range | configured as |
|---|---|---|
| eye position, as a fraction of fork length | 0.042 – 0.162 | 0.02 – 0.25 |
| dorsal origin, as a fraction of fork length | 0.404 – 0.546 | 0.30 – 0.65 |

The dorsal origin one is remarkably tight — every one of the 245 fish falls in a
0.14-wide band.

## 3. Two things wrong with the dataset itself

**A mislabelled annotation.** The eye is anterior to the dorsal origin in 244 of
245 fish. The exception has snout and fork swapped, so the "snout" is on the tail
and the eye computes to 0.938 of body length:

![The mislabelled annotation](images/bad_label.jpg)

A hard plausibility constraint caught it — in the training data, before any model
existed. That's the constraint layer doing exactly the job it was built for,
against ground truth rather than a prediction.

**The published train/valid/test split leaks.** This one matters more.

Roboflow exports one file per augmented copy, named `<source>_jpg.rf.<hash>.jpg`.
245 files, but only **120 source photographs** behind them. The published split
was made over the files, not the photographs, so **34 sources have augmented
copies in more than one split** — rotated and re-exposed versions of the same
fish sitting in train and test simultaneously.

The files aren't byte-identical, so a hash check finds nothing and it all looks
clean. I only caught it because the same filename stem showed up twice while I
was chasing something else.

Trained on the published split, the model scored pose mAP50-95 **0.953** on
"held-out" test data that was nothing of the kind. `eval/dataset.py` now has
`audit_split_leakage()` and `regroup_split()`, which re-splits by source
photograph — 84 / 22 / 14 sources, zero leakage — and the numbers in the README
come from a retrain on that.

One thing I want to be careful about: the clean retrain scores **0.991**, which
is *higher*, and that is not a demonstration that leakage was hurting anything.
The mislabelled fish above sat in the old test split and regrouping moved it into
training, and on a test set this small that single label is worth 4.2 points of
pose mAP50-95 all by itself (0.9534 with it, 0.9950 without). The two test sets
aren't comparable and neither is big enough to support three decimal places.

Fixing the leak didn't move the score in a direction I can demonstrate. It made
the score mean something, which is a different and better claim.

```bash
python -m eval.dataset --fetch data/fish-measurement --workspace fish-count --project fish-measurement-z2ois
python -c "from eval.dataset import regroup_split; regroup_split('data/fish-measurement','data/fish-measurement-grouped')"
```

---

## First, a caveat about what I could and couldn't verify

Roboflow Universe returns 403 to every fetch from this machine — Cloudflare, not a
missing account. So for the three Roboflow datasets below I have image counts and
licenses from search results, but **I have not read a single keypoint name**. I'm
not going to guess at them; a schema I invented would quietly reshape the entire
trait list.

Verifying one costs a free Roboflow account and one command, no new Python
dependency:

```
curl "https://api.roboflow.com/<workspace>/<project>?api_key=$RF_KEY" | python -m json.tool
```

That returns the project metadata including the keypoint skeleton. The export URL
comes from the same API:

```
curl "https://api.roboflow.com/<workspace>/<project>/<version>/yolov8?api_key=$RF_KEY"
```

Say the word and I'll run that against whichever ones you want to see, once you've
put a key in the environment.

## The options

### 1. FishPhenoKey — purpose-built for exactly this problem

The only one I could fully verify, because the paper spells the schema out.

- **Scale:** 23,331 images, six species. Keypoint annotations cover four of them:
  grouper 9,112 images, bighead carp 706, common carp 309, mottled naked carp 200.
  So 10,327 annotated images, but heavily grouper-weighted.
- **Schema:** 22 keypoints, designed by fish breeders for morphometrics rather
  than for pose. In order: snout tip; posterior end of operculum; top end of head;
  isthmus; dorsal apex; bottom end of ventral margin; top end of caudal peduncle;
  bottom end of caudal peduncle; posterior end of tail fin; posterior end of
  caudal vertebrae; anterior end of eye; posterior end of eye; anterior end of
  pectoral fin; posterior end of pectoral fin; anterior end of pelvic fin;
  posterior end of pelvic fin; anterior end of anal fin; posterior end of anal fin;
  outer margin of anal fin; anterior end of dorsal fin; posterior end of dorsal
  fin; outer margin of dorsal fin.
- **License:** none stated. Access is by signed user agreement — download the
  agreement from the repo, sign it, email it to the maintainer, wait for a reply
  with a download link. That's a human in the loop and an unknown turnaround.
- **Source:** github.com/WeizhenLiuBioinform/Fish-Phenotype-Detect, paper at
  arXiv:2405.12476 (IJCAI 2024).

### 2. FISH-KP, by NIWA (Roboflow Universe)

- 960 images. NIWA is New Zealand's national water research institute, so the
  provenance is decent.
- Schema: **unverified.**
- License: **unverified.**
- universe.roboflow.com/niwa/fish-kp

### 3. fishKeypoints (Roboflow Universe)

- 593 images. CC BY 4.0.
- Schema: **unverified.**
- universe.roboflow.com/fish-o3fkg/fishkeypoints

### 4. Fish Measurement, by Fish Count (Roboflow Universe)

- 245 images. CC BY 4.0. Trained as YOLOv8 pose, so it's already in the format
  Phase 2 wants.
- Schema: **unverified.** The name suggests length endpoints, which would be the
  minimum viable schema for this project and not much more.
- universe.roboflow.com/fish-count/fish-measurement-z2ois

### Also saw, didn't chase

Robotic-Fish-Pose-Dataset (robot fish, not real ones), a koi carp 21-keypoint set
shared privately over Google Drive, and MELOPS, a wild-fish re-identification and
phenotyping set on Zenodo. MELOPS might be worth a look — I couldn't read the
paper, it's behind an auth redirect, so I don't know whether it has landmarks at
all.

## What FishPhenoKey's schema would mean for the trait list

Worth reading before you choose, because it changes two of the traits in CLAUDE.md.

**Fork length is not directly available.** CLAUDE.md asks for snout tip to caudal
fork. FishPhenoKey has "posterior end of tail fin" (the tip — that's total length)
and "posterior end of caudal vertebrae" (the hypural plate — that's standard
length). It has no point at the fork itself, the notch between the tail lobes.
Options: switch the primary trait to standard length, which is arguably the better
choice anyway since it doesn't move when a tail gets frayed; or use total length;
or keep calling it fork length and be wrong. <!-- TODO: my call -->

**Body depth, depth ratio, peduncle depth all map cleanly.** Dorsal apex and bottom
end of ventral margin give body depth. Top and bottom end of caudal peduncle give
peduncle depth exactly. Depth ratio follows from whichever length trait wins above.

**Spinal curvature is the weak one.** The 22 points are outline and fin landmarks,
not a midline. The best midline I can build is four derived points: the snout, the
midpoint of dorsal apex and ventral margin, the midpoint of the two peduncle
points, and the caudal endpoint. Four points is enough to fit a line and measure
deviation, but it's a coarser deformity proxy than CLAUDE.md implies — a fish bent
between those points won't show up. <!-- TODO: my call — accept the coarse version,
or drop curvature to a stretch goal -->

**Plausibility constraints are well served.** Snout, both eye corners and the
operculum give the eye-ordering check directly. Anterior and posterior dorsal fin
points give the dorsal insertion check. Both peduncle points give the not-swapped
check. This is the schema that best supports the trust layer.

## My read, for what it's worth

FishPhenoKey is the right dataset on the merits and the wrong one on logistics —
it's the only one built for morphometric phenotyping rather than pose, it's the
only schema I can actually see, and it's the only one big enough to fine-tune on
seriously. But it's gated behind an email and a signature with no known turnaround,
and it has no license I can point at.

The Roboflow sets are small enough that any of them is a fine-tune-on-a-laptop
proposition rather than a training run, which suits Phase 2 and this machine. But
I'd be picking one blind.

The cheap move is to email the FishPhenoKey agreement now, since that clock runs
independently, and meanwhile spend a Roboflow API key on reading all three schemas
before committing. That doesn't need a decision from you today beyond "yes, get
the key."

## The working schema (superseded)

This section used to describe twelve invented landmarks, because the dataset
hadn't arrived and something had to be built against. It has: see
"What actually happened" at the top of this file for the four the real dataset
annotates.

The twelve names still exist in `pipeline/landmarks.py` as the vocabulary, and
that turned out to be the right call — the four real points are a subset of it,
everything else is absent, and `TRAIT_REQUIREMENTS` nulls the affected traits
with a stated reason instead of substituting a neighbour. Swapping the schema was
a rename table and one added name (`eye_centre`), exactly as intended. That's the
one prediction in this file that held up.
