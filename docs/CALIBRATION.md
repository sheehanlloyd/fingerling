# Calibration

Turning pixels into millimetres. This is the part of the project I care most about
getting right, and it's the part most likely to be wrong in a way nobody notices.

## How it works

An object whose physical size I know exactly sits in the frame. Find its four
corners in pixel coordinates, pair them with their known positions in millimetres,
and solve a homography between the two. After that, any point on that plane
converts to millimetres with one matrix multiply.

The board-plane origin is the target's own top-left corner and the axes are its
edges. So millimetre coordinates are only comparable within a single frame. That's
fine — every trait in this project is a distance or a ratio, and both survive.

Two targets, in preference order:

**ArUco marker on my phone screen.** I pick the pixel size that produces an exact
physical size from the screen's PPI, so there's no printer scaling error to worry
about. `marker_length_mm = displayed_side_px / PPI * 25.4`. That value is null in
`config.yaml` until I look up my phone's PPI, and the code refuses to calibrate
rather than guess it.

**A credit card.** ISO/IEC 7810 ID-1 is 85.60 x 53.98 mm and every card in the
world is made to it, so I know the size to a hundredth of a millimetre without
owning a caliper. Detected as a bright convex quad whose aspect ratio is close to
85.60/53.98 = 1.586.

## The residual is not what you think it is

A homography has 8 degrees of freedom. Four point correspondences give exactly 8
equations. The system is exactly determined, so the solution reproduces those four
points perfectly and the reprojection residual is zero to floating-point noise —
**no matter how badly the target is positioned.**

My near-edge-on synthetic scene reports a residual of 0.000 px. It is a genuinely
terrible view. The residual has no opinion about that because it structurally
can't.

So results carry a `residual_meaningful` flag. For a single ArUco marker it's
False and the number is decoration. For the card it's True, because there I don't
only use the corners: I take every point of the detected outline, map it into the
board plane, and measure how far it lands from the ideal rectangle's edges. That's
over-determined, and it catches things the corner solve can't — a card that isn't
flat, a lens that bends straight lines, a sloppy segmentation.

| card | outline residual |
|---|---|
| flat, square-on | 0.92 px |
| bowed 1.5 mm at the edge midpoints | 7.11 px |

Both give four perfectly good corners. Only one of them is a card you should
measure a fish against.

## What actually catches a bad view: obliquity

Read the tilt straight off the matrix. Differentiate the homography at a point and
you get a 2x2 Jacobian whose singular values are the millimetres-per-pixel scale
along the directions that get stretched most and least. From that:

- **anisotropy** — worst `s_max / s_min` at any one point. How much the map
  stretches one direction against the other.
- **scale spread** — largest mean scale anywhere on the target over the smallest.
  How much the scale drifts across the target.

`obliquity` is the worse of the two. Flat-on it's 1.0. Rotate the target in its own
plane and it stays 1.0, because rotation is a rigid motion — that's the property
that makes it useful, since in-plane rotation is harmless and tilt is not.

Measured on synthetic scenes:

| scene | obliquity | residual |
|---|---|---|
| ArUco square-on | 1.000 | 0.000 px (not meaningful) |
| ArUco rotated 30 degrees | 1.002 | 0.000 px (not meaningful) |
| ArUco tilted, far edge at 72% | 1.640 | 0.000 px (not meaningful) |
| ArUco near edge-on | 15.507 | 0.000 px (not meaningful) |
| card square-on | 1.001 | 0.92 px |
| card rotated 30 degrees | 1.001 | 1.71 px |
| card tilted, far edge at 72% | 1.639 | 1.11 px |
| card near edge-on | not detected at all | — |

## Why tilt matters, given that the math is exact

Here's the thing that took me a moment. On a perfectly rendered synthetic frame,
the edge-on marker measures a 200 mm span as 200.09 mm. Exact corners in, exact
millimetres out. So why reject it?

Because real corners aren't exact. A tilted homography multiplies corner error far
more than a flat one does. Jitter the four corners by half a pixel and measure the
same 200 mm span 400 times:

| view | mean error on a 200 mm span |
|---|---|
| flat-on | 0.31 mm |
| near edge-on | 2.41 mm |

Roughly 7x, measuring across the far side of the target where foreshortening is
worst; about 3x on the near side. Same target size in frame in both cases, so
that's tilt and not just one target being bigger. There's a test for this
(`test_tilt_amplifies_corner_noise_and_that_is_the_point_of_obliquity`) because it
is the entire justification for the metric.

## Accuracy on synthetic scenes

Every number below is a 200 mm span in the board plane, rendered through a known
homography and measured back. These are the best case: perfect rendering, no
lens distortion, no motion blur, no lighting. Treat them as a check that the math
is right, not as an accuracy claim. Real numbers come in Phase 3, measured against
household objects of known size.

| scene | measured | error |
|---|---|---|
| ArUco square-on | 200.064 mm | +0.064 |
| ArUco rotated 30 degrees | 199.984 mm | −0.016 |
| ArUco tilted (72%) | 199.912 mm | −0.088 |
| card square-on | 199.732 mm | −0.268 |
| card rotated 30 degrees | 199.789 mm | −0.211 |
| card portrait | 199.732 mm | −0.268 |
| card tilted (72%) | 199.707 mm | −0.293 |

The ArUco span is a 5x extrapolation from a 40 mm marker; the card span is about
2.3x from an 85.6 mm card. Extrapolating further from a smaller target multiplies
corner error, which is the argument for putting the target near the fish and not
in the corner of the frame.

## Things this gets wrong

**The fish is not on the board plane.** The whole method assumes target and
subject are coplanar. They aren't. A fish has thickness and its midline sits above
the board by roughly half its body depth. Everything on the fish is therefore
closer to the camera than the calibration plane, so it images larger, so every
length reads long. It's a systematic magnification, not noise — averaging frames
will not help.

The size of it depends on camera distance: for a fish whose midline sits `h` above
the board with the camera `d` away, lengths are inflated by about `d / (d - h)`.
A 20 mm half-depth at 500 mm working distance is about 4%, which on a 200 mm fish
is 8 mm. That's larger than everything else on this page put together.

I haven't measured my actual working distance yet, so I'm not putting a number on
the real bias. <!-- TODO: my call — measure working distance, then either correct
for it with an assumed half-depth or state the bias in the README -->

**No lens distortion model.** No camera intrinsics, no undistortion. A wide-angle
lens bends straight lines and the homography has no way to express that. On the
card path the outline residual will at least notice; on the ArUco path nothing
will.

**Single marker only.** If several ArUco markers are in frame the largest is used
and the rest ignored, because without a known layout there's no way to place them
in a common board frame. Multiple markers with a known layout, or a ChArUco board,
would give a genuinely over-determined solve for the marker path too. Not built.

**The card aspect filter is the only thing identifying a card.** Anything bright,
convex, four-sided and roughly 1.586:1 will be accepted and measured against. A
paperback book seen at the right angle would do it.

**Thresholds are placeholders.** `max_reprojection_residual_px: 2.0` and
`max_obliquity: 2.0` in `config.yaml` are not measured values. Phase 3 sets them.
