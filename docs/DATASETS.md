# Checkpoint 1 — dataset survey

What's out there for fish keypoints, what each one would let me compute, and what
it costs to get. Nothing downloaded. The choice is mine to make and everything
from checkpoint 3 onward waits on it, because the keypoint schema decides which
traits are computable at all.

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

## The working schema, and the trait list re-derived against it

Since the real annotations never arrived, `pipeline/landmarks.py` defines a
schema and everything downstream was built against it. Twelve points:

| # | name | what it is | in FishPhenoKey? |
|---|---|---|---|
| 0 | `snout_tip` | most anterior point of the closed mouth | yes |
| 1 | `eye_anterior` | anterior margin of the eye | yes |
| 2 | `eye_posterior` | posterior margin of the eye | yes |
| 3 | `operculum_posterior` | posterior edge of the gill cover | yes |
| 4 | `dorsal_origin` | anterior insertion of the dorsal fin | yes |
| 5 | `dorsal_insertion` | posterior insertion of the dorsal fin | yes |
| 6 | `dorsal_apex` | outermost point of the dorsal fin margin | yes |
| 7 | `ventral_margin` | lowest point of the ventral outline | yes |
| 8 | `peduncle_dorsal` | top of the caudal peduncle | yes |
| 9 | `peduncle_ventral` | bottom of the caudal peduncle | yes |
| 10 | `caudal_fork` | the notch between the tail lobes | **no** |
| 11 | `caudal_tip` | most posterior point of the tail fin | yes |

Eleven of twelve are a subset of FishPhenoKey's 22, so if the real data follows
that schema the mapping is a rename table. The twelfth is the problem, and it's
the same problem the survey below already flagged: **nothing annotates the fork.**

### What that means trait by trait

- **Fork length** — computable only if `caudal_fork` exists. If the real schema
  doesn't have it, this goes null with the reason attached; it does NOT silently
  become total length. That's enforced by `TRAIT_REQUIREMENTS` in
  `pipeline/landmarks.py` and there's a test for it
  (`test_a_missing_landmark_nulls_only_the_traits_that_need_it`).
- **Total length** — snout to tail tip. Always available, added as a fallback and
  reported alongside. Not in CLAUDE.md's trait list.
- **Standard length** — *not implemented.* It needs the hypural plate
  ("posterior end of caudal vertebrae" in FishPhenoKey), which isn't in this
  schema. If the real dataset has it, that's arguably the better primary length
  trait since it doesn't move when a tail frays, and switching is a config-level
  decision plus one entry in `TRAIT_REQUIREMENTS`.
- **Body depth, depth ratio, peduncle depth** — map cleanly, no gaps.
- **Spinal curvature** — still the weak one, exactly as the survey predicted. The
  midline is four derived points: snout, the midpoint of dorsal origin and
  ventral margin, the midpoint of the two peduncle points, and the fork. Four
  points fit a line and measure deviation, but **a fish bent between two of them
  does not show up at all.** It's a coarse deformity proxy. It is called that
  everywhere it appears.
- **Condition factor** — unchanged: only with a hand-entered weight, never
  estimated.
- **Plausibility constraints** — well served. Snout, both eye corners and the
  operculum give the eye-ordering check; both dorsal fin insertions give the
  dorsal ordering check; both peduncle points give the not-swapped check. Ten
  constraints in total, eight of them hard. This part of the schema is doing
  what it was picked to do.

### The honest summary

Every trait in CLAUDE.md's list is implemented except standard length, which was
never in that list. What's uncertain isn't the trait maths — that's tested
against fixtures where I know the answer — it's whether these twelve names
correspond to anything a real annotator drew. That question is one API key away
from being answered.
