/**
 * ONNX Runtime Web session management: WebGPU first, WebAssembly as the
 * fallback, and an automatic switch to WebAssembly if WebGPU fails at runtime
 * (unsupported operator, buffer limits, device loss).
 */

import type { InferenceSession } from 'onnxruntime-web';
import { AppError } from '../shared/errors';
import type { Backend, ModelConfig, ModelManifest } from '../shared/protocol';

type Ort = typeof import('onnxruntime-web');

/** The largest intermediate tensor of the model is 1×120×1024×1024 float32 (480 MiB). */
const REQUIRED_GPU_BUFFER_BYTES = 120 * 1024 * 1024 * 4;

export interface RuntimeSession {
  backend: Backend;
  run(input: Float32Array): Promise<Float32Array>;
  release(): Promise<void>;
}

let ortWebGPU: Ort | null = null;
let ortWasm: Ort | null = null;

async function loadOrt(kind: 'webgpu' | 'wasm', config: ModelConfig): Promise<Ort> {
  let ort: Ort;
  if (kind === 'webgpu') {
    ortWebGPU ??= await import('onnxruntime-web/webgpu');
    ort = ortWebGPU;
    ort.env.wasm.wasmPaths = { wasm: config.wasmUrls.webgpu };
  } else {
    ortWasm ??= await import('onnxruntime-web/wasm');
    ort = ortWasm;
    ort.env.wasm.wasmPaths = { wasm: config.wasmUrls.wasm };
  }
  ort.env.wasm.numThreads = config.threads;
  ort.env.logLevel = 'error';
  return ort;
}

/** Checks whether this device's GPU can hold the model's largest tensors. */
export async function probeWebGPU(): Promise<{ ok: boolean; reason?: string }> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    return { ok: false, reason: 'WebGPU not available' };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { ok: false, reason: 'No WebGPU adapter' };
    const { maxBufferSize, maxStorageBufferBindingSize } = adapter.limits;
    if (maxBufferSize < REQUIRED_GPU_BUFFER_BYTES || maxStorageBufferBindingSize < REQUIRED_GPU_BUFFER_BYTES) {
      return { ok: false, reason: `GPU buffer limit too small (${maxStorageBufferBindingSize})` };
    }
    return { ok: true };
  } catch (err) {
    return { ok: false, reason: String(err) };
  }
}

/** Reports a lost GPU device (driver reset, GPU process crash, …) so the next run can use the CPU. */
function watchDeviceLoss(ort: Ort, onDeviceLost: (reason: string) => void): void {
  try {
    // Depending on the ONNX Runtime build this is a GPUDevice or a promise of one.
    const device: unknown = (ort.env.webgpu as { device?: unknown }).device;
    void Promise.resolve(device)
      .then((d) => {
        const lost = (d as Partial<GPUDevice> | undefined)?.lost;
        if (lost) void lost.then((info) => onDeviceLost(`${info.reason}: ${info.message}`));
      })
      .catch(() => undefined);
  } catch {
    /* device loss then surfaces as a failed run, which also triggers the fallback */
  }
}

export async function createSession(
  bytes: Uint8Array,
  manifest: ModelManifest,
  backend: Backend,
  config: ModelConfig,
  onDeviceLost?: (reason: string) => void,
): Promise<RuntimeSession> {
  // The WebGPU build also contains the WebAssembly CPU backend, so a fallback
  // after a WebGPU failure does not need a second runtime download.
  const ort = await loadOrt(backend === 'webgpu' || ortWebGPU ? 'webgpu' : 'wasm', config);
  let session: InferenceSession;
  try {
    session = await ort.InferenceSession.create(bytes, {
      executionProviders: [backend],
      graphOptimizationLevel: 'all',
      // WebAssembly memory can only grow, never shrink. Without the arena the
      // CPU backend peaks at ~1.8 GB instead of ~3 GB for this model.
      enableCpuMemArena: false,
      logSeverityLevel: 3,
    });
  } catch (err) {
    throw new AppError(backend === 'webgpu' ? 'inference-failed' : 'runtime-unsupported', `Session: ${String(err)}`);
  }

  if (backend === 'webgpu' && onDeviceLost) watchDeviceLoss(ort, onDeviceLost);

  const { name: inputName, width, height } = manifest.input;
  const outputName = manifest.output.name;
  return {
    backend,
    async run(input: Float32Array): Promise<Float32Array> {
      const tensor = new ort.Tensor('float32', input, [1, 3, height, width]);
      try {
        const results = await session.run({ [inputName]: tensor });
        const out = results[outputName];
        if (!out) throw new Error(`Missing model output ${outputName}`);
        const data = (await out.getData()) as Float32Array;
        const copy = new Float32Array(data);
        out.dispose();
        return copy;
      } finally {
        tensor.dispose();
      }
    },
    async release(): Promise<void> {
      await session.release();
    },
  };
}
