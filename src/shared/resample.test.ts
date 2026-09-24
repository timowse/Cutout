import { describe, expect, it } from 'vitest';
import { computeCoefficients, resampleInterleaved, resamplePlane, resamplePlaneToUint8 } from './resample';

describe('computeCoefficients', () => {
  it('produces normalised weights', () => {
    for (const filter of ['box', 'triangle', 'bicubic'] as const) {
      for (const [inSize, outSize] of [
        [4032, 1024],
        [1024, 4032],
        [10, 10],
        [7, 3],
      ] as const) {
        const c = computeCoefficients(inSize, outSize, filter);
        for (let i = 0; i < outSize; i++) {
          let sum = 0;
          for (let k = 0; k < c.counts[i]!; k++) sum += c.weights[i * c.stride + k]!;
          expect(sum).toBeCloseTo(1, 5);
          expect(c.starts[i]!).toBeGreaterThanOrEqual(0);
          expect(c.starts[i]! + c.counts[i]!).toBeLessThanOrEqual(inSize);
        }
      }
    }
  });
});

describe('resampling', () => {
  it('keeps a constant image constant', () => {
    const src = new Float32Array(20 * 10).fill(0.25);
    for (const filter of ['box', 'triangle', 'bicubic'] as const) {
      const up = resamplePlane(src, 20, 10, 57, 31, filter);
      const down = resamplePlane(src, 20, 10, 6, 3, filter);
      for (const v of [...up, ...down]) expect(v).toBeCloseTo(0.25, 5);
    }
  });

  it('is the identity at the same size', () => {
    const src = Float32Array.from({ length: 12 }, (_, i) => i / 11);
    const out = resamplePlane(src, 4, 3, 4, 3, 'bicubic');
    out.forEach((v, i) => expect(v).toBeCloseTo(src[i]!, 5));
  });

  it('clamps bicubic overshoot for alpha mattes', () => {
    // A hard edge makes the cubic kernel overshoot; the 8-bit matte must stay within 0..255.
    const src = new Float32Array(16 * 16);
    for (let y = 0; y < 16; y++) for (let x = 8; x < 16; x++) src[y * 16 + x] = 1;
    const out = resamplePlaneToUint8(src, 16, 16, 64, 64, 'bicubic');
    expect(Math.max(...out)).toBe(255);
    expect(Math.min(...out)).toBe(0);
    // Left and right thirds stay fully transparent / opaque.
    expect(out[32 * 64 + 5]).toBe(0);
    expect(out[32 * 64 + 60]).toBe(255);
  });

  it('downsamples interleaved RGBA into planes with anti-aliasing', () => {
    // 4x2 image with alternating black/white columns averages to grey when halved.
    const rgba = new Uint8ClampedArray(4 * 2 * 4);
    for (let i = 0; i < 8; i++) {
      const v = i % 2 === 0 ? 0 : 255;
      rgba.set([v, v, v, 255], i * 4);
    }
    const [r, g, b] = resampleInterleaved(rgba, 4, 2, 4, 3, 2, 1, 'box');
    for (const plane of [r!, g!, b!]) for (const v of plane) expect(v).toBeCloseTo(127.5, 0);
    // The wider triangle filter still blends the stripes instead of picking one.
    const [t] = resampleInterleaved(rgba, 4, 2, 4, 1, 2, 1, 'triangle');
    for (const v of t!) expect(v).toBeGreaterThan(80);
    for (const v of t!) expect(v).toBeLessThan(175);
  });
});
