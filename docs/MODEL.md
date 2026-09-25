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
| Browser model | 92.9 MB (float16 weight storage), 4 chunks of ≤ 24 MiB, SHA-256 per chunk |

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
   group of kernel taps and each of the four bilinear corners: compute the
   sample indices (int32) into a zero-padded copy of the input, `Gather` the
   samples and weight them with the bilinear factor × modulation mask; the four
   corners are summed and accumulated with a `MatMul` against the corresponding
   slice of the convolution weights. Taps are grouped so that no gathered
   tensor exceeds 16 MB. Offset and modulator convolutions are kept unchanged.
3. **Patch rearrangement** → one `Reshape → Transpose → Reshape`
   (`out[(j·gh + i)·C + c, y, x] = img[c, i·ph + y, j·pw + x]`). The script
   checks the exact patch order of the original graph before rewriting and
   refuses to continue if it differs.
4. **Memory rewrites** (exact linear algebra, needed for phones). The decoder
   ends with `Concat(Resize(96 ch), 24 ch)` at 1024 × 1024 followed by a 1 × 1
   convolution to one channel — a single 480 MiB tensor next to 384 MiB and
   256 MiB ones. The converter
   - splits a 1 × 1 convolution over a concatenation into a sum of 1 × 1
     convolutions over the parts (also used for the 1280-channel ASPP concat),
   - moves a channel-reducing 1 × 1 convolution in front of a bilinear resize
     (both are linear and the resize weights sum to one),
   - folds a 1 × 1 convolution into the convolution before it
     (`W = W₂·W₁`, `b = W₂·b₁ + b₂`), and
   - computes a convolution → convolution intermediate that is still larger
     than 64 MiB in channel chunks.

   The largest intermediate tensor drops from 480 MiB to 96 MiB, and the live
   activations along ONNX Runtime's execution order from ~960 MiB to
   ~300 MiB.
5. **Sigmoid** appended so the model returns the alpha matte (`alpha`) directly.
6. **Weight storage** in float16 with `Cast` nodes, so all maths runs in
   float32. On WebGPU, ONNX Runtime folds the casts when the session is
   created; on the CPU the app disables constant folding (see below).
7. **Check** that no node needs more than 8 GPU buffers, `onnx.checker`, chunking
   into ≤ 24 MiB files (fits GitHub, Cloudflare Pages and most CDNs) plus
   `manifest.json` with SHA-256 hashes and `LICENSE.txt`.

### Verification (ONNX Runtime 1.30, CPU, 4 scikit-image test photos)

| Variant | Size | Max. difference to original | Mean difference | Pixels off by > 1/255 |
| --- | --- | --- | --- | --- |
| Original export (fp32) | 224 MB | — | — | — |
| Converted, fp32 weights | 182 MB | 3.8 × 10⁻⁵ | < 10⁻⁷ | 0 % |
| **Converted, fp16 weights (shipped)** | **92.9 MB** | 1.6 × 10⁻² | ≤ 2.3 × 10⁻⁵ | ≤ 0.15 % |
| Converted, int8 per-channel weights | 49 MB | 0.44 | ≤ 9.7 × 10⁻⁴ | up to 6.3 % (1 % > 5/255) |

The memory rewrites do not change the result (the fp32 and fp16 figures are
the same as without them).

### Memory in the browser

The original export needs 12.2 GB of RAM (ONNX Runtime, CPU). Measured with
ONNX Runtime Web 1.30 (WebAssembly, one 1024 × 1024 inference) and in
headless Chromium (whole tab, WebAssembly path):

| | First browser version | Now |
| --- | --- | --- |
| Largest intermediate tensor | 480 MiB | 96 MiB |
| WebAssembly heap after loading the model | 597 MB | 291 MB |
| WebAssembly heap after one image | 2296 MB | 844 MB |
| Browser tab after a 12 MP photo | 2.66 GB | 1.27 GB |

Three things matter, all measured:

- the memory rewrites above;
- no CPU memory arena (`enableCpuMemArena: false`; the arena grows in large
  steps, ~3.2 GB);
- no constant folding on the CPU backend. WebAssembly memory never shrinks,
  and with constant folding the heap grows far beyond the live tensors
  (1.38 GB instead of 0.84 GB, same speed). Without it the float16 weights
  also stay float16 in memory and are widened layer by layer. WebGPU keeps
  constant folding: unfolded shape computations would cause GPU → CPU round
  trips.

The price is a slower session start on the CPU (≈ 5.8 s instead of 3 s on the
development machine); it happens during the background preload. Gathering
deformable-convolution samples in 32 MB instead of 16 MB groups would save
≈ 0.9 s but needs 130 MB more heap.

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
thousand first-time visitors (the model is preloaded when the page opens). If the site grows
beyond that, set `VITE_MODEL_BASE_URL` to a CDN or Hugging Face repository
(pinned revision) and add that origin to the CSP's `connect-src`.

## Regenerating

```bash
python3 -m pip install -r tools/model/requirements.txt
npm run model            # writes public/models/birefnet-lite/
```

Options: `--weights fp32|fp16|int8`, `--chunk-mib`, `--version` (bump to
invalidate browser caches), `--save-onnx` (also write the unsplit file).
