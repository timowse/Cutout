import type { ImageLimits } from './protocol';

export interface DeviceHints {
  /** `navigator.deviceMemory` (Chromium only, rounded, capped at 8). */
  deviceMemoryGB?: number;
  isMobile: boolean;
  isIOS: boolean;
}

const MP = 1_000_000;

/**
 * Resource limits for one image. Decoding a 48 MP photo alone needs ~190 MB,
 * and the full pipeline needs roughly 10 bytes per output pixel on top of the
 * AI model, so phones get tighter limits than desktops.
 */
export function computeLimits(hints: DeviceHints): ImageLimits {
  const mem = hints.deviceMemoryGB;
  if (hints.isIOS) {
    // iOS Safari refuses canvases larger than 16,777,216 pixels.
    return { maxInputPixels: 50 * MP, maxOutputPixels: 16_777_216, maxOutputSide: 8192 };
  }
  if (hints.isMobile) {
    const out = mem !== undefined && mem <= 3 ? 12 * MP : 16 * MP;
    return { maxInputPixels: 64 * MP, maxOutputPixels: out, maxOutputSide: 8192 };
  }
  let out = 36 * MP;
  if (mem !== undefined) {
    if (mem <= 2) out = 12 * MP;
    else if (mem <= 4) out = 24 * MP;
    else if (mem >= 8) out = 48 * MP;
  }
  return { maxInputPixels: 150 * MP, maxOutputPixels: out, maxOutputSide: 16384 };
}

export interface OutputPlan {
  width: number;
  height: number;
  /** True if the result is smaller than the original image. */
  scaled: boolean;
}

/** Output size for an image: the original size unless that exceeds the device limits. */
export function planOutputSize(width: number, height: number, limits: ImageLimits): OutputPlan {
  if (!(width > 0 && height > 0)) throw new RangeError('Invalid image size');
  const scale = Math.min(
    1,
    Math.sqrt(limits.maxOutputPixels / (width * height)),
    limits.maxOutputSide / Math.max(width, height),
  );
  if (scale >= 1) return { width, height, scaled: false };
  let w = Math.max(1, Math.floor(width * scale));
  let h = Math.max(1, Math.floor(height * scale));
  // Floating point can leave us one pixel over the limit.
  while (w * h > limits.maxOutputPixels) {
    if (w >= h) w--;
    else h--;
  }
  return { width: w, height: h, scaled: true };
}

/** True if an image of this size must be rejected before decoding. */
export function exceedsInputLimit(width: number, height: number, limits: ImageLimits): boolean {
  return width * height > limits.maxInputPixels;
}

/** Rough estimate of the working memory (bytes) the pipeline needs for a given output size. */
export function estimateWorkingMemory(width: number, height: number): number {
  const pixels = width * height;
  const perPixel = 4 /* RGBA */ + 1 /* alpha */ + 4 /* PNG filter + compressed output */ + 1 /* slack */;
  const fixed = 160 * MP; // foreground estimation maps, resampling buffers, model I/O tensors
  return pixels * perPixel + fixed;
}

/** Size that fits `width`×`height` into a `maxSide` square, keeping the aspect ratio. */
export function fitWithin(width: number, height: number, maxSide: number): { width: number; height: number } {
  const scale = Math.min(1, maxSide / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}
