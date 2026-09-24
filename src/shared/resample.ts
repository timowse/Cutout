/**
 * Separable image resampling (the same scheme Pillow uses): each output
 * sample is a normalised weighted sum of input samples, and when shrinking
 * the kernel is widened so the result is properly anti-aliased.
 */

export type Filter = 'box' | 'triangle' | 'bicubic';

const SUPPORT: Record<Filter, number> = { box: 0.5, triangle: 1, bicubic: 2 };

/** Catmull-Rom style cubic (a = -0.5), as used by Pillow's BICUBIC. */
function cubic(x: number): number {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

function kernel(filter: Filter, x: number): number {
  switch (filter) {
    case 'box':
      return x > -0.5 && x <= 0.5 ? 1 : 0;
    case 'triangle': {
      const ax = Math.abs(x);
      return ax < 1 ? 1 - ax : 0;
    }
    case 'bicubic':
      return cubic(x);
  }
}

export interface Coefficients {
  /** First input index for each output index. */
  starts: Int32Array;
  /** Number of taps for each output index. */
  counts: Int32Array;
  /** `weights[i * stride + k]` is the weight of input `starts[i] + k`. */
  weights: Float32Array;
  stride: number;
}

export function computeCoefficients(inSize: number, outSize: number, filter: Filter): Coefficients {
  if (!(inSize > 0 && outSize > 0)) throw new RangeError('Sizes must be positive');
  const scale = inSize / outSize;
  const filterScale = Math.max(1, scale);
  const support = SUPPORT[filter] * filterScale;
  const stride = Math.ceil(support) * 2 + 1;
  const starts = new Int32Array(outSize);
  const counts = new Int32Array(outSize);
  const weights = new Float32Array(outSize * stride);
  for (let i = 0; i < outSize; i++) {
    const center = (i + 0.5) * scale;
    const min = Math.max(0, Math.floor(center - support + 0.5));
    const max = Math.min(inSize, Math.floor(center + support + 0.5));
    let total = 0;
    const count = Math.min(max - min, stride);
    for (let k = 0; k < count; k++) {
      const w = kernel(filter, (k + min - center + 0.5) / filterScale);
      weights[i * stride + k] = w;
      total += w;
    }
    if (total !== 0) for (let k = 0; k < count; k++) weights[i * stride + k]! /= total;
    starts[i] = min;
    counts[i] = count;
  }
  return { starts, counts, weights, stride };
}

/**
 * Resamples `channels` interleaved channels of an 8-bit image (e.g. the RGB
 * channels of RGBA data with `pixelStride` 4) into separate float planes.
 */
export function resampleInterleaved(
  src: ArrayLike<number>,
  srcWidth: number,
  srcHeight: number,
  pixelStride: number,
  channels: number,
  dstWidth: number,
  dstHeight: number,
  filter: Filter,
): Float32Array[] {
  const cx = computeCoefficients(srcWidth, dstWidth, filter);
  const cy = computeCoefficients(srcHeight, dstHeight, filter);
  // Horizontal pass: srcHeight × dstWidth, channel-interleaved.
  const tmp = new Float32Array(srcHeight * dstWidth * channels);
  for (let y = 0; y < srcHeight; y++) {
    const rowIn = y * srcWidth * pixelStride;
    const rowOut = y * dstWidth * channels;
    for (let x = 0; x < dstWidth; x++) {
      const start = cx.starts[x]!;
      const count = cx.counts[x]!;
      const wBase = x * cx.stride;
      for (let c = 0; c < channels; c++) {
        let acc = 0;
        let p = rowIn + start * pixelStride + c;
        for (let k = 0; k < count; k++, p += pixelStride) acc += src[p]! * cx.weights[wBase + k]!;
        tmp[rowOut + x * channels + c] = acc;
      }
    }
  }
  // Vertical pass into planar output.
  const planes: Float32Array[] = [];
  for (let c = 0; c < channels; c++) planes.push(new Float32Array(dstWidth * dstHeight));
  const rowLen = dstWidth * channels;
  for (let y = 0; y < dstHeight; y++) {
    const start = cy.starts[y]!;
    const count = cy.counts[y]!;
    const wBase = y * cy.stride;
    for (let x = 0; x < dstWidth; x++) {
      for (let c = 0; c < channels; c++) {
        let acc = 0;
        let p = start * rowLen + x * channels + c;
        for (let k = 0; k < count; k++, p += rowLen) acc += tmp[p]! * cy.weights[wBase + k]!;
        planes[c]![y * dstWidth + x] = acc;
      }
    }
  }
  return planes;
}

function horizontalPass(src: Float32Array, srcWidth: number, srcHeight: number, dstWidth: number, filter: Filter): Float32Array {
  const cx = computeCoefficients(srcWidth, dstWidth, filter);
  const tmp = new Float32Array(srcHeight * dstWidth);
  for (let y = 0; y < srcHeight; y++) {
    const rowIn = y * srcWidth;
    const rowOut = y * dstWidth;
    for (let x = 0; x < dstWidth; x++) {
      const start = rowIn + cx.starts[x]!;
      const count = cx.counts[x]!;
      const wBase = x * cx.stride;
      let acc = 0;
      for (let k = 0; k < count; k++) acc += src[start + k]! * cx.weights[wBase + k]!;
      tmp[rowOut + x] = acc;
    }
  }
  return tmp;
}

/** Resamples a single float plane; values are clamped to [min, max]. */
export function resamplePlane(
  src: Float32Array,
  srcWidth: number,
  srcHeight: number,
  dstWidth: number,
  dstHeight: number,
  filter: Filter,
  min = -Infinity,
  max = Infinity,
): Float32Array {
  const tmp = horizontalPass(src, srcWidth, srcHeight, dstWidth, filter);
  const cy = computeCoefficients(srcHeight, dstHeight, filter);
  const out = new Float32Array(dstWidth * dstHeight);
  for (let y = 0; y < dstHeight; y++) {
    const start = cy.starts[y]!;
    const count = cy.counts[y]!;
    const wBase = y * cy.stride;
    const rowOut = y * dstWidth;
    for (let x = 0; x < dstWidth; x++) {
      let acc = 0;
      let p = start * dstWidth + x;
      for (let k = 0; k < count; k++, p += dstWidth) acc += tmp[p]! * cy.weights[wBase + k]!;
      out[rowOut + x] = acc < min ? min : acc > max ? max : acc;
    }
  }
  return out;
}

/**
 * Resamples a float plane in [0, 1] (such as an alpha matte) to 8 bit.
 * Used to bring the model's 1024×1024 matte to the output resolution.
 */
export function resamplePlaneToUint8(
  src: Float32Array,
  srcWidth: number,
  srcHeight: number,
  dstWidth: number,
  dstHeight: number,
  filter: Filter,
): Uint8Array {
  const tmp = horizontalPass(src, srcWidth, srcHeight, dstWidth, filter);
  const cy = computeCoefficients(srcHeight, dstHeight, filter);
  const out = new Uint8Array(dstWidth * dstHeight);
  for (let y = 0; y < dstHeight; y++) {
    const start = cy.starts[y]!;
    const count = cy.counts[y]!;
    const wBase = y * cy.stride;
    const rowOut = y * dstWidth;
    for (let x = 0; x < dstWidth; x++) {
      let acc = 0;
      let p = start * dstWidth + x;
      for (let k = 0; k < count; k++, p += dstWidth) acc += tmp[p]! * cy.weights[wBase + k]!;
      const v = acc * 255 + 0.5;
      out[rowOut + x] = v <= 0 ? 0 : v >= 255 ? 255 : v;
    }
  }
  return out;
}
