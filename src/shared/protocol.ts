/**
 * Typed messages between the UI thread and the inference worker.
 *
 * Every message carries a `type` discriminator. Messages that belong to an
 * image job carry the `jobId` the UI assigned; the UI ignores anything that
 * does not match the job it currently shows, so results can never be mixed up.
 */

export type Backend = 'webgpu' | 'wasm';

/** Phases of getting the model ready. `download` and `cache` report bytes. */
export type ModelPhase = 'manifest' | 'download' | 'cache' | 'init';

/** Stages of processing one image after the model is ready. */
export type ProcessStage = 'decoding' | 'waiting-for-model' | 'inference' | 'refining' | 'encoding';

export type ErrorCode =
  | 'unsupported-format'
  | 'heic-unsupported'
  | 'decode-failed'
  | 'too-large'
  | 'out-of-memory'
  | 'runtime-unsupported'
  | 'model-download-failed'
  | 'model-integrity'
  | 'offline'
  | 'inference-failed'
  | 'unknown';

export interface WorkerError {
  code: ErrorCode;
  /** Technical detail for the console. Never shown in the UI. */
  detail?: string;
}

export interface ImageLimits {
  /** Images with more pixels are rejected before any processing. */
  maxInputPixels: number;
  /** The result is scaled down to at most this many pixels. */
  maxOutputPixels: number;
  /** Maximum width or height of the result (canvas limits). */
  maxOutputSide: number;
}

export interface ModelConfig {
  /** Absolute URL of the directory that contains manifest.json. */
  baseUrl: string;
  /** Try WebGPU first (falls back to WebAssembly on failure). */
  preferWebGPU: boolean;
  /** Absolute URLs of the ONNX Runtime WebAssembly binaries. */
  wasmUrls: { webgpu: string; wasm: string };
  /** Number of WebAssembly threads to use (1 when not cross-origin isolated). */
  threads: number;
}

export interface ProcessTimings {
  decodeMs: number;
  inferenceMs: number;
  refineMs: number;
  encodeMs: number;
}

export interface RGB {
  r: number;
  g: number;
  b: number;
}

export type WorkerRequest =
  | { type: 'INIT_MODEL'; config: ModelConfig }
  | { type: 'PROCESS_IMAGE'; jobId: number; file: Blob; limits: ImageLimits; previewMaxSide: number }
  | { type: 'CANCEL_JOB'; jobId: number }
  | { type: 'RENDER_WITH_BACKGROUND'; jobId: number; requestId: number; color: RGB }
  | { type: 'RELEASE_RESULT'; jobId: number };

export type WorkerResponse =
  | { type: 'MODEL_PROGRESS'; phase: ModelPhase; loaded: number; total: number }
  | { type: 'MODEL_READY'; backend: Backend; fromCache: boolean; loadMs: number; modelBytes: number }
  | { type: 'MODEL_ERROR'; error: WorkerError }
  | { type: 'BACKEND_FALLBACK'; from: Backend; to: Backend; reason: string }
  | { type: 'PROCESS_PROGRESS'; jobId: number; stage: ProcessStage }
  | {
      type: 'PROCESS_PREVIEW';
      jobId: number;
      preview: ImageBitmap;
      width: number;
      height: number;
      originalWidth: number;
      originalHeight: number;
    }
  | { type: 'PROCESS_COMPLETE'; jobId: number; png: Blob; width: number; height: number; timings: ProcessTimings }
  | { type: 'PROCESS_ERROR'; jobId: number; error: WorkerError }
  | { type: 'BACKGROUND_RENDERED'; jobId: number; requestId: number; png: Blob }
  | { type: 'BACKGROUND_ERROR'; jobId: number; requestId: number; error: WorkerError };

/** Model manifest written by tools/model/convert_birefnet.py. */
export interface ModelManifest {
  id: string;
  version: string;
  name: string;
  input: { name: string; width: number; height: number; mean: [number, number, number]; std: [number, number, number] };
  output: { name: string; width: number; height: number };
  weights: string;
  size: number;
  sha256: string;
  parts: { file: string; size: number; sha256: string }[];
  source: { url: string; sha256: string; license: string; homepage: string };
}

export function isModelManifest(value: unknown): value is ModelManifest {
  if (typeof value !== 'object' || value === null) return false;
  const m = value as Partial<ModelManifest>;
  return (
    typeof m.id === 'string' &&
    typeof m.size === 'number' &&
    typeof m.sha256 === 'string' &&
    Array.isArray(m.parts) &&
    m.parts.length > 0 &&
    m.parts.every((p) => typeof p.file === 'string' && typeof p.size === 'number' && typeof p.sha256 === 'string') &&
    typeof m.input?.name === 'string' &&
    typeof m.input.width === 'number' &&
    typeof m.input.height === 'number' &&
    typeof m.output?.name === 'string'
  );
}
