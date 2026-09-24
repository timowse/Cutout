# Architecture

A static web app: no backend, no API, no accounts. Everything that touches an
image runs in the visitor's browser.

```
┌──────────────────────────── UI thread ─────────────────────────────┐
│ index.html (static markup, renders instantly)                      │
│ main.ts → input.ts (drop / picker / paste)                         │
│         → controller.ts ── reduce() ──► state.ts (state machine)   │
│         → view.ts (renders state), compare.ts (before/after)       │
└───────────────┬───────────────────────────────▲────────────────────┘
                │ typed messages (protocol.ts)  │ progress, preview
                ▼                               │ (ImageBitmap), PNG Blob
┌──────────────────────────── Worker ────────────────────────────────┐
│ inference.worker.ts – job queue, cancellation checkpoints          │
│   model-loader.ts  – manifest, chunked download with progress,     │
│                      SHA-256, Cache Storage, offline               │
│   runtime.ts       – ONNX Runtime Web: WebGPU → WebAssembly        │
│   pipeline.ts      – decode, resample, matte upsampling,           │
│                      foreground colours, preview, PNG encoding     │
└────────────────────────────────────────────────────────────────────┘
┌──────────────────────── Service worker (sw.ts) ────────────────────┐
│ offline app shell, runtime cache for hashed assets,                │
│ COOP/COEP headers (cross-origin isolation for WASM threads)        │
└────────────────────────────────────────────────────────────────────┘
```

## Processing pipeline

1. **Input** (`input.ts`): global drag & drop (with overlay), file picker,
   `paste` event (Cmd/Ctrl+V, no permission prompt). Multiple files: the first
   image is used.
2. **Validation** (`files.ts`, `controller.ts`): the file type is detected from
   its bytes (JPEG, PNG, WebP, GIF, BMP, AVIF, HEIC), and the header dimensions
   are checked against device limits *before* decoding.
3. **Instant preview**: the original is shown immediately from an object URL.
4. **Decode** (worker): `createImageBitmap` with EXIF orientation, drawn to an
   `OffscreenCanvas` at the planned output size (original size unless the
   device limits require less — see `dimensions.ts`).
5. **Model input**: anti-aliased resize to 1024 × 1024 (triangle filter, like
   the reference preprocessing), ImageNet normalisation; transparent areas are
   flattened onto white.
6. **Inference** (`runtime.ts`): ONNX Runtime Web, WebGPU if the adapter's
   buffer limits fit the model's largest tensor (480 MiB), otherwise
   WebAssembly (SIMD, multi-threaded when cross-origin isolated). A failure on
   WebGPU (unsupported operation, device loss) switches to WebAssembly and
   retries automatically.
7. **Matte upsampling**: bicubic from 1024 × 1024 to the output size, clamped,
   8-bit. Inputs with transparency keep `min(source alpha, matte)`.
8. **Foreground colours** (`foreground.ts`): blur-fusion estimation removes the
   old background's colour from soft edges; fully transparent pixels become
   `(0,0,0,0)` so no trace of the background stays in the file.
9. **Preview first**: a downscaled `ImageBitmap` is transferred to the UI,
   which wipes from the original to the result.
10. **PNG** (`png.ts`): straight-alpha RGBA, Paeth filter, native zlib via
    `CompressionStream`, no metadata. (Canvas encoding would premultiply alpha
    and lose the colour of nearly transparent pixels; it is only a fallback.)

## State machine

`idle → loading-image → loading-model → processing → complete`, plus `error`
and `cancelled`. Every image gets a job id; the reducer ignores events for any
other job, so a late result for image A can never replace image B. The worker
drops superseded jobs at the next checkpoint (a running model inference cannot
be interrupted, its result is discarded).

## Resource cleanup

When an image is replaced or dismissed, the controller revokes its object URLs
(original, PNG, "with background" PNGs), closes the preview bitmap, clears the
canvas and tells the worker to drop its full-resolution result.

## Caching and offline

| What | Where | Versioning |
| --- | --- | --- |
| App shell (HTML, JS, CSS, icons) | Service worker, precached | build id (hash of file names) |
| ONNX Runtime (JS + WASM, 14–27 MB) | Service worker, cached on first use | fingerprinted file names |
| AI model (92.5 MB) | Cache Storage `bgremove-model-<id>-<hash>` | model hash; old caches deleted |
| Images | Memory only | gone on reload |

## Security

- CSP (meta tag): `default-src 'self'`, `connect-src 'self'`, `script-src
  'self' 'wasm-unsafe-eval'` (needed to compile WebAssembly), no inline
  scripts or styles, no `eval`, `object-src 'none'`, `form-action 'none'`.
- The inference worker is started through a tiny `blob:` module that imports
  the real worker script. Workers from `blob:` URLs inherit the page's CSP, so
  `connect-src 'self'` binds the worker too (a worker loaded straight from an
  `https:` URL would only get the CSP of its HTTP response, which GitHub Pages
  cannot set). An end-to-end test checks that a request from the worker to
  another origin is blocked by the CSP.
- No third-party requests at all (no CDN, no fonts, no analytics).
- Input limits: file size ≤ 200 MB, pixel count checked from the header before
  decoding, output size capped per device (e.g. 16.7 MP on iOS).
- Blob URLs are revoked, bitmaps closed, worker results released.

## Extending

- **Second model / quality mode**: add another manifest directory and pass a
  different `baseUrl` in `ModelConfig`; the worker, loader and cache handle any
  model with the same input/output contract.
- **Batch processing**: jobs already have ids and a queue in the worker.
