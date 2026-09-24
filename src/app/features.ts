/** Browser capability detection. Pure functions over an injectable environment for testing. */

export interface Features {
  webgpu: boolean;
  webassembly: boolean;
  wasmSimd: boolean;
  offscreenCanvas: boolean;
  createImageBitmap: boolean;
  moduleWorker: boolean;
  crossOriginIsolated: boolean;
  cacheStorage: boolean;
  compressionStream: boolean;
  clipboardWriteImage: boolean;
  hardwareConcurrency: number;
  deviceMemoryGB?: number;
  isMobile: boolean;
  isIOS: boolean;
  isApple: boolean;
}

/** Smallest valid module using a SIMD instruction: (func (result v128) (v128.const i32x4 0 0 0 0)). */
const SIMD_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, // type section: () -> v128
  0x03, 0x02, 0x01, 0x00, // function section
  0x0a, 0x16, 0x01, 0x14, 0x00, 0xfd, 0x0c, // code: v128.const
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x0b, // end
]);

export interface FeatureEnv {
  navigator?: Partial<Navigator> & { deviceMemory?: number; userAgentData?: { mobile?: boolean; platform?: string } };
  WebAssembly?: { validate(bytes: BufferSource): boolean };
  OffscreenCanvas?: unknown;
  createImageBitmap?: unknown;
  Worker?: unknown;
  crossOriginIsolated?: boolean;
  caches?: unknown;
  CompressionStream?: unknown;
  ClipboardItem?: unknown;
}

export function detectFeatures(env: FeatureEnv = globalThis): Features {
  const nav = env.navigator ?? {};
  const ua = nav.userAgent ?? '';
  const platform = nav.userAgentData?.platform ?? nav.platform ?? '';
  const maxTouch = nav.maxTouchPoints ?? 0;
  const isIOS = /iPad|iPhone|iPod/.test(ua) || (/Mac/.test(platform) && maxTouch > 1);
  const isMobile = nav.userAgentData?.mobile ?? (isIOS || /Android|Mobi/i.test(ua));
  const isApple = isIOS || /Mac/i.test(platform);

  let wasmSimd = false;
  const wasm = env.WebAssembly;
  if (wasm) {
    try {
      wasmSimd = wasm.validate(SIMD_PROBE);
    } catch {
      wasmSimd = false;
    }
  }

  const clipboard = (nav as { clipboard?: { write?: unknown } }).clipboard;
  return {
    webgpu: !!(nav as { gpu?: unknown }).gpu,
    webassembly: !!wasm,
    wasmSimd,
    offscreenCanvas: typeof env.OffscreenCanvas === 'function',
    createImageBitmap: typeof env.createImageBitmap === 'function',
    moduleWorker: typeof env.Worker === 'function',
    crossOriginIsolated: env.crossOriginIsolated === true,
    cacheStorage: !!env.caches,
    compressionStream: typeof env.CompressionStream === 'function',
    clipboardWriteImage: typeof env.ClipboardItem === 'function' && typeof clipboard?.write === 'function',
    hardwareConcurrency: nav.hardwareConcurrency ?? 1,
    ...(typeof nav.deviceMemory === 'number' ? { deviceMemoryGB: nav.deviceMemory } : {}),
    isMobile,
    isIOS,
    isApple,
  };
}

/** Returns why this browser cannot run background removal, or null if it can. */
export function missingRequirement(f: Features): string | null {
  if (!f.webassembly) return 'WebAssembly';
  if (!f.wasmSimd) return 'WebAssembly SIMD';
  if (!f.moduleWorker) return 'Web Workers';
  if (!f.offscreenCanvas) return 'OffscreenCanvas';
  if (!f.createImageBitmap) return 'createImageBitmap';
  return null;
}

/** WebAssembly threads for ONNX Runtime: needs cross-origin isolation; leave a core for the UI. */
export function wasmThreads(f: Features): number {
  if (!f.crossOriginIsolated) return 1;
  return Math.max(1, Math.min(8, f.hardwareConcurrency - 1));
}
