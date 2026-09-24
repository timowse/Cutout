# Quality

How the result quality was checked, what the numbers are, and where the model
is weak. Run your own comparison with `npm run quality -- <folder>` (see
[CONTRIBUTING.md](../CONTRIBUTING.md)); it runs entirely on your machine.

## Test images

`tests/fixtures/images/` (CC0 / public domain, from scikit-image):

| Image | Category | Result (real model, browser) |
| --- | --- | --- |
| `person-astronaut.jpg` | person, short hair | Clean cut-out of person and suit; hair edge soft and without fringe after colour decontamination. |
| `animal-cat.jpg` | animal, fur, whiskers | Face and fur edges good; some whiskers kept, thin ones against the busy background lost. A piece of the blurry background fabric at the lower right is kept. |
| `product-coffee.png` | product | Cup, saucer and spoon cleanly separated from the wooden table, including the spoon's reflections. |
| `thin-structures-rocket.webp` | thin structures | Launch-tower lattices are preserved remarkably well. Bright lens flares near the ground are treated as foreground. |
| `motion-blur-clock.jpg` | motion blur, greyscale | Object found, but the semi-transparent motion smear is mostly cut off instead of becoming partially transparent. |
| `transparent-horse.png` | input with alpha | Existing transparency is respected (result alpha never exceeds source alpha). |

The automated test `tests/e2e/real-model.spec.ts` checks the astronaut result
numerically (face and suit opaque, background transparent, soft edge pixels
present) on every CI run.

**Not yet covered by real photos in the repository:** long and curly hair,
cars, shoes, plants, white objects on white backgrounds, glass. No suitable
CC0 images were available when this was written. Please contribute your own
photos (see CONTRIBUTING.md) — the quality tool makes comparisons easy.

## Known weaknesses of the model

BiRefNet_lite is a segmentation model trained at 1024 × 1024. In practice:

- **Genuine transparency is not modelled.** Glass, veils, smoke and motion
  blur become mostly opaque or mostly removed, not partially transparent.
- **Fine detail is limited by the 1024 × 1024 model resolution.** On a 12 MP
  photo one model pixel covers roughly 4 × 3 photo pixels, so single hair
  strands and whiskers that are thinner than that are softened or lost. The
  larger BiRefNet models (2K / HR) would help but are too heavy for browsers
  today.
- **"What is the subject?" can be ambiguous.** Objects next to or held by a
  person, or several objects of similar importance, may be kept or removed, and
  small input differences can change that decision.
- **Bright light sources, reflections and shadows** are sometimes kept.
- **Low contrast** between subject and background (e.g. white on white) makes
  edges less precise.

## Post-processing experiments

These numbers come from experiments run during development (Python/NumPy
with the converted model and ONNX Runtime on the CPU). The one-off experiment
scripts are not part of the repository.

### Matte upscaling (1024 × 1024 → photo size)

Synthetic test: a 3200 × 2400 image with ~700 anti-aliased "hair" strands and a
known exact matte; the matte is reduced to 1024 × 1024 (ideal model output) and
scaled back up. Lower is better.

| Method | SAD (×1000) | MSE (×10⁻³) | MAE in edge band |
| --- | --- | --- | --- |
| bilinear | 136.0 | 4.83 | 0.184 |
| **bicubic (used)** | 124.2 | 4.35 | 0.172 |
| Lanczos | 121.6 | 4.21 | 0.168 |
| colour-guided filter (r = 2) | 37.9 | 0.42 | 0.049 |

Real photos (no ground truth available): the model's matte computed at a third
of the resolution and upscaled was compared with the matte computed at full
resolution. Bicubic beat bilinear on all five photos. The colour-guided filter
improved edge accuracy on some photos but created visible halos around a
motorcycle and made others worse, so it is **not used** — a step that can
visibly degrade results is worse than a smaller, reliable gain.

### Colour decontamination

Synthetic test with an exact matte, subject composited onto a new white
background: the edge-band error drops from 0.0277 (raw pixel colours) to
0.0116 (−58 %) with blur-fusion foreground estimation (radii 90 and 6). On real
photos it visibly removes the light fringe of the old background around hair
and fur. It is applied to every result.

### Weight precision

float16 weight storage changes the matte by at most 1.6 % at single pixels
(mean 0.002 %); int8 weights changed up to 44 % and were rejected. See
[MODEL.md](MODEL.md).
