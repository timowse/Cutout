/**
 * Inference worker. Owns the model and all heavy pixel work so the UI thread
 * stays responsive. Jobs run one at a time; a newer job supersedes older ones,
 * which are dropped at the next checkpoint (a running model inference cannot
 * be interrupted, but its result is discarded).
 */

import { AppError, toWorkerError } from '../shared/errors';
import { loadModel } from '../shared/model-loader';
import type { Backend, ModelConfig, ModelManifest, ProcessTimings, WorkerRequest, WorkerResponse } from '../shared/protocol';
import {
  composeResult,
  createPreview,
  decodeImage,
  encodeResult,
  encodeWithBackground,
  toModelInput,
  type DecodedImage,
} from './pipeline';
import { createSession, probeWebGPU, type RuntimeSession } from './runtime';

declare const self: DedicatedWorkerGlobalScope;

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(message, transfer);
}

interface ModelState {
  config: ModelConfig;
  manifest: ModelManifest;
  session: RuntimeSession;
}

let modelPromise: Promise<ModelState> | null = null;
let modelReady = false;
let gpuBroken = false;

async function initModel(config: ModelConfig): Promise<ModelState> {
  const started = performance.now();
  const loaded = await loadModel(config.baseUrl, (phase, loaded, total) =>
    post({ type: 'MODEL_PROGRESS', phase, loaded, total }),
  );
  post({ type: 'MODEL_PROGRESS', phase: 'init', loaded: 0, total: 0 });

  let backend: Backend = 'wasm';
  if (config.preferWebGPU && !gpuBroken) {
    const probe = await probeWebGPU();
    if (probe.ok) backend = 'webgpu';
  }
  let session: RuntimeSession;
  try {
    session = await createSession(loaded.bytes, loaded.manifest, backend, config, onDeviceLost);
  } catch (err) {
    if (backend !== 'webgpu') throw err;
    gpuBroken = true;
    post({ type: 'BACKEND_FALLBACK', from: 'webgpu', to: 'wasm', reason: String(err) });
    backend = 'wasm';
    session = await createSession(loaded.bytes, loaded.manifest, backend, config);
  }
  modelReady = true;
  post({
    type: 'MODEL_READY',
    backend,
    fromCache: loaded.fromCache,
    loadMs: Math.round(performance.now() - started),
    modelBytes: loaded.manifest.size,
  });
  return { config, manifest: loaded.manifest, session };
}

function onDeviceLost(reason: string): void {
  gpuBroken = true;
  post({ type: 'BACKEND_FALLBACK', from: 'webgpu', to: 'wasm', reason: `GPU device lost (${reason})` });
}

function startModel(config: ModelConfig): void {
  if (modelPromise) return;
  const promise = initModel(config);
  modelPromise = promise;
  promise.catch((err: unknown) => {
    if (modelPromise === promise) modelPromise = null;
    modelReady = false;
    post({ type: 'MODEL_ERROR', error: toWorkerError(err, 'model-download-failed') });
  });
}

async function switchToWasm(state: ModelState, reason: string): Promise<void> {
  gpuBroken = true;
  post({ type: 'BACKEND_FALLBACK', from: 'webgpu', to: 'wasm', reason });
  await state.session.release().catch(() => undefined);
  const loaded = await loadModel(state.config.baseUrl, () => undefined);
  state.session = await createSession(loaded.bytes, state.manifest, 'wasm', state.config);
}

async function infer(state: ModelState, input: Float32Array): Promise<Float32Array> {
  if (state.session.backend === 'webgpu' && gpuBroken) await switchToWasm(state, 'GPU unavailable');
  try {
    return await state.session.run(input);
  } catch (err) {
    if (state.session.backend !== 'webgpu') throw err;
    await switchToWasm(state, String(err));
    return state.session.run(input);
  }
}

// ---------------------------------------------------------------------------
// Jobs

let latestJob = 0;
const cancelled = new Set<number>();
let queue: Promise<void> = Promise.resolve();
let result: { jobId: number; image: DecodedImage } | null = null;

class Superseded extends Error {}

function isStale(jobId: number): boolean {
  return jobId !== latestJob || cancelled.has(jobId);
}

function checkpoint(jobId: number): void {
  if (isStale(jobId)) throw new Superseded();
}

async function processImage(msg: Extract<WorkerRequest, { type: 'PROCESS_IMAGE' }>): Promise<void> {
  const { jobId } = msg;
  const timings: ProcessTimings = { decodeMs: 0, inferenceMs: 0, refineMs: 0, encodeMs: 0 };
  let preview: ImageBitmap | null = null;
  try {
    checkpoint(jobId);
    post({ type: 'PROCESS_PROGRESS', jobId, stage: 'decoding' });
    let t = performance.now();
    const image = await decodeImage(msg.file, msg.limits);
    checkpoint(jobId);
    if (!modelPromise) throw new AppError('unknown', 'Model was not initialised');
    if (!modelReady) post({ type: 'PROCESS_PROGRESS', jobId, stage: 'waiting-for-model' });
    const tDecode = performance.now() - t;
    const model = await modelPromise;
    checkpoint(jobId);

    t = performance.now();
    const input = toModelInput(image, model.manifest);
    timings.decodeMs = Math.round(tDecode + performance.now() - t);
    post({ type: 'PROCESS_PROGRESS', jobId, stage: 'inference' });
    t = performance.now();
    const matte = await infer(model, input);
    timings.inferenceMs = Math.round(performance.now() - t);
    checkpoint(jobId);

    post({ type: 'PROCESS_PROGRESS', jobId, stage: 'refining' });
    t = performance.now();
    composeResult(image, matte, model.manifest);
    preview = await createPreview(image, msg.previewMaxSide);
    checkpoint(jobId);
    timings.refineMs = Math.round(performance.now() - t);
    result = { jobId, image };
    const previewBitmap = preview;
    preview = null;
    post(
      {
        type: 'PROCESS_PREVIEW',
        jobId,
        preview: previewBitmap,
        width: image.width,
        height: image.height,
        originalWidth: image.originalWidth,
        originalHeight: image.originalHeight,
      },
      [previewBitmap],
    );

    post({ type: 'PROCESS_PROGRESS', jobId, stage: 'encoding' });
    t = performance.now();
    const png = await encodeResult(image.rgba, image.width, image.height);
    timings.encodeMs = Math.round(performance.now() - t);
    checkpoint(jobId);
    post({ type: 'PROCESS_COMPLETE', jobId, png, width: image.width, height: image.height, timings });
  } catch (err) {
    preview?.close();
    if (err instanceof Superseded || isStale(jobId)) return;
    post({ type: 'PROCESS_ERROR', jobId, error: toWorkerError(err, 'inference-failed') });
  }
}

async function renderWithBackground(msg: Extract<WorkerRequest, { type: 'RENDER_WITH_BACKGROUND' }>): Promise<void> {
  const { jobId, requestId } = msg;
  try {
    if (!result || result.jobId !== jobId) throw new AppError('unknown', 'Result no longer available');
    const { image } = result;
    const png = await encodeWithBackground(image.rgba, image.width, image.height, msg.color);
    post({ type: 'BACKGROUND_RENDERED', jobId, requestId, png });
  } catch (err) {
    post({ type: 'BACKGROUND_ERROR', jobId, requestId, error: toWorkerError(err) });
  }
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  switch (msg.type) {
    case 'INIT_MODEL':
      startModel(msg.config);
      break;
    case 'PROCESS_IMAGE':
      latestJob = msg.jobId;
      result = null;
      queue = queue.then(() => processImage(msg));
      break;
    case 'CANCEL_JOB':
      cancelled.add(msg.jobId);
      if (result?.jobId === msg.jobId) result = null;
      break;
    case 'RENDER_WITH_BACKGROUND':
      void renderWithBackground(msg);
      break;
    case 'RELEASE_RESULT':
      if (result?.jobId === msg.jobId) result = null;
      break;
  }
};
