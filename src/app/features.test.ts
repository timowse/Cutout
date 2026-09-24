import { describe, expect, it } from 'vitest';
import { detectFeatures, missingRequirement, wasmThreads, type FeatureEnv } from './features';

function env(overrides: Partial<FeatureEnv> = {}, nav: Record<string, unknown> = {}): FeatureEnv {
  return {
    navigator: { userAgent: 'Mozilla/5.0 (X11; Linux x86_64) Chrome/140', platform: 'Linux x86_64', hardwareConcurrency: 8, ...nav },
    WebAssembly: globalThis.WebAssembly,
    OffscreenCanvas: function OffscreenCanvas() {},
    createImageBitmap: () => undefined,
    Worker: function Worker() {},
    crossOriginIsolated: true,
    caches: {},
    CompressionStream: function CompressionStream() {},
    ...overrides,
  };
}

describe('detectFeatures', () => {
  it('detects WebAssembly SIMD with the real engine', () => {
    expect(detectFeatures(env()).wasmSimd).toBe(true);
  });

  it('reports a complete desktop browser as supported', () => {
    const f = detectFeatures(env({}, { gpu: {}, deviceMemory: 8 }));
    expect(missingRequirement(f)).toBeNull();
    expect(f.webgpu).toBe(true);
    expect(f.deviceMemoryGB).toBe(8);
    expect(f.isMobile).toBe(false);
  });

  it('names the missing requirement', () => {
    expect(missingRequirement(detectFeatures(env({ WebAssembly: undefined })))).toBe('WebAssembly');
    expect(missingRequirement(detectFeatures(env({ OffscreenCanvas: undefined })))).toBe('OffscreenCanvas');
    expect(missingRequirement(detectFeatures(env({ WebAssembly: { validate: () => false } })))).toBe('WebAssembly SIMD');
  });

  it('recognises iPhones and iPads', () => {
    const iphone = detectFeatures(env({}, { userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)', platform: 'iPhone' }));
    expect(iphone.isIOS && iphone.isMobile && iphone.isApple).toBe(true);
    const ipad = detectFeatures(env({}, { userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', platform: 'MacIntel', maxTouchPoints: 5 }));
    expect(ipad.isIOS).toBe(true);
    const mac = detectFeatures(env({}, { userAgent: 'Mozilla/5.0 (Macintosh)', platform: 'MacIntel', maxTouchPoints: 0 }));
    expect(mac.isIOS).toBe(false);
    expect(mac.isApple).toBe(true);
  });

  it('detects clipboard image support only with ClipboardItem and clipboard.write', () => {
    expect(detectFeatures(env({ ClipboardItem: function ClipboardItem() {} }, { clipboard: { write: () => undefined } })).clipboardWriteImage).toBe(true);
    expect(detectFeatures(env({}, { clipboard: { write: () => undefined } })).clipboardWriteImage).toBe(false);
  });
});

describe('wasmThreads', () => {
  it('uses one thread without cross-origin isolation', () => {
    expect(wasmThreads(detectFeatures(env({ crossOriginIsolated: false })))).toBe(1);
  });

  it('leaves one core for the UI and caps the thread count', () => {
    expect(wasmThreads(detectFeatures(env({}, { hardwareConcurrency: 4 })))).toBe(3);
    expect(wasmThreads(detectFeatures(env({}, { hardwareConcurrency: 32 })))).toBe(8);
  });
});
