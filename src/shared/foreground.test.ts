import { describe, expect, it } from 'vitest';
import { applyForeground, boxBlur } from './foreground';

describe('boxBlur', () => {
  it('averages over the window and keeps constants', () => {
    const w = 9;
    const h = 5;
    const src = new Float32Array(w * h).fill(3);
    const out = new Float32Array(w * h);
    boxBlur(src, w, h, 2, out, new Float32Array(w * h));
    for (const v of out) expect(v).toBeCloseTo(3, 5);
    src.fill(0);
    src[2 * w + 4] = 25;
    boxBlur(src, w, h, 2, out, new Float32Array(w * h));
    expect(out[2 * w + 4]).toBeCloseTo(1, 5); // 25 spread over a 5×5 window
    expect(out[0]).toBeCloseTo(0, 5);
  });
});

/** Composite a known foreground over a known background with a soft vertical edge. */
function scene(width: number, height: number) {
  const F = [230, 120, 40];
  const B = [20, 200, 60];
  const alpha = new Uint8Array(width * height);
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const a = Math.min(1, Math.max(0, (x - width / 2) / 12 + 0.5));
      const i = y * width + x;
      alpha[i] = Math.round(a * 255);
      for (let c = 0; c < 3; c++) rgba[i * 4 + c] = a * F[c]! + (1 - a) * B[c]!;
      rgba[i * 4 + 3] = 255;
    }
  }
  return { F, B, alpha, rgba };
}

describe('applyForeground', () => {
  it('removes background colour from soft edges', () => {
    const width = 120;
    const height = 40;
    const { F, alpha, rgba } = scene(width, height);
    const before = Uint8ClampedArray.from(rgba);
    applyForeground(rgba, alpha, width, height);
    let errBefore = 0;
    let errAfter = 0;
    let n = 0;
    for (let i = 0; i < width * height; i++) {
      const a = alpha[i]! / 255;
      if (a <= 0.1 || a >= 1) continue;
      for (let c = 0; c < 3; c++) {
        errBefore += Math.abs(before[i * 4 + c]! - F[c]!);
        errAfter += Math.abs(rgba[i * 4 + c]! - F[c]!);
      }
      n++;
    }
    expect(n).toBeGreaterThan(0);
    expect(errAfter / n).toBeLessThan((errBefore / n) * 0.5);
  });

  it('writes alpha, keeps opaque pixels and clears fully transparent ones', () => {
    const width = 120;
    const height = 40;
    const { alpha, rgba } = scene(width, height);
    const before = Uint8ClampedArray.from(rgba);
    applyForeground(rgba, alpha, width, height);
    for (let i = 0; i < width * height; i++) {
      expect(rgba[i * 4 + 3]).toBe(alpha[i]);
      if (alpha[i] === 255) expect(Array.from(rgba.subarray(i * 4, i * 4 + 3))).toEqual(Array.from(before.subarray(i * 4, i * 4 + 3)));
      if (alpha[i] === 0) expect(Array.from(rgba.subarray(i * 4, i * 4 + 3))).toEqual([0, 0, 0]);
    }
  });

  it('works on a reduced working grid for large images', () => {
    const width = 300;
    const height = 60;
    const { F, alpha, rgba } = scene(width, height);
    applyForeground(rgba, alpha, width, height, { workPixels: 2000 });
    const i = Math.floor(height / 2) * width + width / 2;
    expect(alpha[i]! / 255).toBeGreaterThan(0.2);
    for (let c = 0; c < 3; c++) expect(Math.abs(rgba[i * 4 + c]! - F[c]!)).toBeLessThan(40);
  });
});
