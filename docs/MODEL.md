# AI model: research, decision and conversion

## Decision

**BiRefNet_lite (general use, Swin-T backbone, epoch 232)** from the official
GitHub release of the BiRefNet author, converted for the browser by
[`tools/model/convert_birefnet.py`](../tools/model/convert_birefnet.py).

| | |
| --- | --- |
| Source file | `BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx` |
| Source URL | <https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx> |
| Source SHA-256 | `5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333` (224,005,088 bytes) |
| Code licence | MIT (`LICENSE` in the BiRefNet repository, checked) |
| Weights licence | MIT (released by the author in the same repository / release) |
| Parameters | 44.6 M |
| Input / output | 1 × 3 × 1024 × 1024 (ImageNet-normalised RGB) → 1 × 1 × 1024 × 1024 alpha matte |
| Browser model | 92.5 MB (float16 weight storage), 4 chunks of ≤ 24 MiB, SHA-256 per chunk |

## Candidates

| Candidate | Code licence | Weights licence | Verdict |
| --- | --- | --- | --- |
| **BiRefNet_lite** (Swin-T, 1024²) | MIT | MIT | **Chosen.** Current state of the art for dichotomous segmentation, handles hair, fur and thin structures well, official ONNX export exists, size workable in a browser. |
| BiRefNet (general, Swin-L) | MIT | MIT | Better quality, but 844 MB weights / 928 MB ONNX (~450 MB even as float16) and ~4× the compute. Too heavy for a browser today; natural candidate for an optional "high quality" mode on strong desktops. |
| BiRefNet_lite-2K, BiRefNet_HR, BiRefNet-matting, BiRefNet_HR-matting, BiRefNet_dynamic | MIT | MIT | Higher resolution and/or Swin-L backbone: more memory and compute than browsers can reasonably handle. Matting variants are interesting for a later HQ mode. |
| `@imgly/background-removal` (IS-Net) | **AGPL-3.0** (`LICENSE.md` of the npm package, checked) | Data package `@imgly/background-removal-data` also "SEE LICENSE IN LICENSE.md" (AGPL) | Rejected: would force the whole project under AGPL-3.0, and it loads its models from IMG.LY's CDN by default. It is based on IS-Net, which the BiRefNet paper outperforms on the DIS5K benchmark. |
| BRIA RMBG-1.4 / RMBG-2.0 | — | **Non-commercial** (RMBG-2.0 is BiRefNet trained on BRIA's data; the BiRefNet README notes it is "for only non-commercial use") | Excluded: a non-commercial licence restricts free general use. |
| IS-Net (DIS), U²-Net, MODNet | Apache-2.0 (per upstream repositories; not re-verified in detail because not used) | | Older models (2020–2022). The BiRefNet paper reports better DIS5K scores than IS-Net and U²-Net; MODNet only handles portraits. |

## Why the official ONNX file cannot be used directly

Running the official export with ONNX Runtime (CPU, 1024 × 1024 input) peaks at
**12.2 GB of RAM**. The reason: its 20 deformable convolutions (in the decoder's
ASPP blocks) were exported as a generic GatherND/ScatterND decomposition that
materialises tensors of up to **784 MB each** (e.g. `[1, 64, 1792, 1792]` for a
7 × 7 deformable convolution at 256 × 256). WebAssembly has a 4 GB address space
and phones have far less memory, so this file cannot run in a browser.

A second problem only shows on WebGPU: the decoder cuts the input image into
patches with nested `Split`s and a `Concat` of up to **1024 inputs**. WebGPU
limits the number of storage buffers per shader (8–10 on common hardware), so
that node fails on the GPU ("Too many storage buffers in shader").

## Conversion (`tools/model/convert_birefnet.py`)

All steps are deterministic and verified numerically against the original file.

1. **Download + verify** the official file (pinned URL and SHA-256).
2. **Deformable convolutions** → equivalent memory-lean subgraphs. For each
   group of kernel taps: compute the four bilinear sample indices into a
   zero-padded copy of the input, `Gather` the samples, weight them with the
   bilinear factors × modulation mask, `ReduceSum`, and accumulate with a
   `MatMul` against the corresponding slice of the convolution weights. Taps are
   grouped so that no intermediate exceeds 32 MB. Offset and modulator
   convolutions are kept unchanged.
3. **Patch rearrangement** → one `Reshape → Transpose → Reshape`
   (`out[(j·gh + i)·C + c, y, x] = img[c, i·ph + y, j·pw + x]`). The script
   checks the exact patch order of the original graph before rewriting and
   refuses to continue if it differs.
4. **Sigmoid** appended so the model returns the alpha matte (`alpha`) directly.
5. **Weight storage** in float16 with `Cast` nodes; ONNX Runtime folds the casts
   when the session is created, so all maths runs in float32.
6. **Check** that no node needs more than 8 GPU buffers, `onnx.checker`, chunking
   into ≤ 24 MiB files (fits GitHub, Cloudflare Pages and most CDNs) plus
   `manifest.json` with SHA-256 hashes and `LICENSE.txt`.

### Verification (ONNX Runtime 1.30, CPU, 4 scikit-image test photos)

| Variant | Size | Max. difference to original | Mean difference | Pixels off by > 1/255 | Peak RAM |
| --- | --- | --- | --- | --- | --- |
| Original export (fp32) | 224 MB | — | — | — | 12.2 GB |
| Converted, fp32 weights | 182 MB | 4.2 × 10⁻⁵ | < 10⁻⁷ | 0 % | 1.8 GB* |
| **Converted, fp16 weights (shipped)** | **92.5 MB** | 1.6 × 10⁻² | ≤ 2.3 × 10⁻⁵ | ≤ 0.15 % | 1.8 GB* |
| Converted, int8 per-channel weights | 49 MB | 0.44 | ≤ 9.7 × 10⁻⁴ | up to 6.3 % (1 % > 5/255) | — |

\* without ONNX Runtime's CPU memory arena; with the arena the process peaks at
~3.2 GB because the arena grows in large steps.

The patch rewrite is bit-identical to the version without it and halves session
creation time (fewer nodes). int8 weights were rejected because they visibly
change mattes in places.

## Post-processing choices (measured, see [QUALITY.md](QUALITY.md))

- **Matte upscaling:** bicubic (Catmull-Rom). Better than bilinear on every test
  image. A colour-guided filter looked great on synthetic data but produced
  visible halos on real photos, so it is not used.
- **Colour decontamination:** "blur fusion" foreground estimation (Forte &
  Pitié 2021). Removes the fringe of the old background from hair and fur edges
  (−58 % composite error on the synthetic hair test with an exact matte).
- **No thresholding:** the soft matte is kept as is (8-bit alpha).

## Hosting

The model is **not committed to git**. The deployment workflow downloads the
official file, verifies its hash, runs the converter and deploys the chunks with
the site (same origin: no third-party request, no CORS, covered by the
Content-Security-Policy `connect-src 'self'`). The browser stores the model in
Cache Storage under a name derived from the model hash; old versions are removed
when a new model is deployed.

GitHub Pages has a soft bandwidth limit of 100 GB per month, i.e. roughly one
thousand first-time visitors who actually process an image. If the site grows
beyond that, set `VITE_MODEL_BASE_URL` to a CDN or Hugging Face repository
(pinned revision) and add that origin to the CSP's `connect-src`.

## Regenerating

```bash
python3 -m pip install -r tools/model/requirements.txt
npm run model            # writes public/models/birefnet-lite/
```

Options: `--weights fp32|fp16|int8`, `--chunk-mib`, `--version` (bump to
invalidate browser caches), `--save-onnx` (also write the unsplit file).
