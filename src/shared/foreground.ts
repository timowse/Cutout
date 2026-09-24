/**
 * Foreground colour estimation ("colour decontamination").
 *
 * In soft edges (hair, fur, motion blur) a pixel's colour is a mix of the
 * subject and the old background: I = a·F + (1 − a)·B. Keeping I as the
 * foreground colour produces a visible fringe of the old background on any
 * new background. This module estimates F with the "blur fusion" method from
 * M. Forte and F. Pitié, "Approximate Fast Foreground Colour Estimation",
 * ICIP 2021 (two passes of box-filtered colour estimates, radii 90 and 6).
 *
 * On large images the smooth colour estimates are computed on a reduced
 * working grid and interpolated; the per-pixel correction is always applied
 * at full resolution.
 */

import { resampleInterleaved } from './resample';

export interface ForegroundOptions {
  /** Max pixels of the working grid for the blurred colour estimates. */
  workPixels?: number;
  /** Box filter radii of the two passes, in full-resolution pixels. */
  radii?: [number, number];
}

const EPS = 1e-5;

/** Edge-normalised box filter with radius r (window 2r + 1), O(n) per pixel. */
export function boxBlur(src: Float32Array, width: number, height: number, r: number, out: Float32Array, tmp: Float32Array): void {
  // Horizontal
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let sum = 0;
    const right0 = Math.min(r, width - 1);
    for (let x = 0; x <= right0; x++) sum += src[row + x]!;
    for (let x = 0; x < width; x++) {
      const lo = x - r;
      const hi = x + r;
      tmp[row + x] = sum / (Math.min(hi, width - 1) - Math.max(lo, 0) + 1);
      if (hi + 1 < width) sum += src[row + hi + 1]!;
      if (lo >= 0) sum -= src[row + lo]!;
    }
  }
  // Vertical
  for (let x = 0; x < width; x++) {
    let sum = 0;
    const bottom0 = Math.min(r, height - 1);
    for (let y = 0; y <= bottom0; y++) sum += tmp[y * width + x]!;
    for (let y = 0; y < height; y++) {
      const lo = y - r;
      const hi = y + r;
      out[y * width + x] = sum / (Math.min(hi, height - 1) - Math.max(lo, 0) + 1);
      if (hi + 1 < height) sum += tmp[(hi + 1) * width + x]!;
      if (lo >= 0) sum -= tmp[lo * width + x]!;
    }
  }
}

interface ColourEstimates {
  width: number;
  height: number;
  /** Per-channel smooth foreground (fg) and background (bg) estimates of the second pass. */
  fg: [Float32Array, Float32Array, Float32Array];
  bg: [Float32Array, Float32Array, Float32Array];
}

function computeEstimates(
  planesRGB: Float32Array[],
  a: Float32Array,
  width: number,
  height: number,
  r1: number,
  r2: number,
): ColourEstimates {
  const n = width * height;
  const tmp = new Float32Array(n);
  const blurA1 = new Float32Array(n);
  const blurA2 = new Float32Array(n);
  boxBlur(a, width, height, r1, blurA1, tmp);
  boxBlur(a, width, height, r2, blurA2, tmp);
  const prod = new Float32Array(n);
  const t1 = new Float32Array(n);
  const t2 = new Float32Array(n);
  const fg: Float32Array[] = [];
  const bg: Float32Array[] = [];
  for (let c = 0; c < 3; c++) {
    const I = planesRGB[c]!;
    // Pass 1 with F = B = I.
    for (let i = 0; i < n; i++) prod[i] = I[i]! * a[i]!;
    boxBlur(prod, width, height, r1, t1, tmp);
    for (let i = 0; i < n; i++) prod[i] = I[i]! * (1 - a[i]!);
    boxBlur(prod, width, height, r1, t2, tmp);
    for (let i = 0; i < n; i++) {
      const ai = a[i]!;
      const f = t1[i]! / (blurA1[i]! + EPS);
      const b = t2[i]! / (1 - blurA1[i]! + EPS);
      t2[i] = b; // B1
      const v = f + ai * (I[i]! - ai * f - (1 - ai) * b);
      t1[i] = v < 0 ? 0 : v > 1 ? 1 : v; // F1
    }
    // Pass 2 with F = F1, B = B1.
    const fgC = new Float32Array(n);
    const bgC = new Float32Array(n);
    for (let i = 0; i < n; i++) prod[i] = t1[i]! * a[i]!;
    boxBlur(prod, width, height, r2, fgC, tmp);
    for (let i = 0; i < n; i++) prod[i] = t2[i]! * (1 - a[i]!);
    boxBlur(prod, width, height, r2, bgC, tmp);
    for (let i = 0; i < n; i++) {
      fgC[i] = fgC[i]! / (blurA2[i]! + EPS);
      bgC[i] = bgC[i]! / (1 - blurA2[i]! + EPS);
    }
    fg.push(fgC);
    bg.push(bgC);
  }
  return {
    width,
    height,
    fg: fg as [Float32Array, Float32Array, Float32Array],
    bg: bg as [Float32Array, Float32Array, Float32Array],
  };
}

/** Colour planes and alpha on the working grid, as floats in [0, 1]. */
function workingPlanes(
  rgba: Uint8ClampedArray,
  alpha: Uint8Array,
  width: number,
  height: number,
  ww: number,
  wh: number,
): [Float32Array[], Float32Array] {
  const n = width * height;
  if (ww === width && wh === height) {
    const planes = [new Float32Array(n), new Float32Array(n), new Float32Array(n)] as const;
    const a = new Float32Array(n);
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      planes[0][i] = rgba[p]! / 255;
      planes[1][i] = rgba[p + 1]! / 255;
      planes[2][i] = rgba[p + 2]! / 255;
      a[i] = alpha[i]! / 255;
    }
    return [[...planes], a];
  }
  const planes = resampleInterleaved(rgba, width, height, 4, 3, ww, wh, 'box');
  const a = resampleInterleaved(alpha, width, height, 1, 1, ww, wh, 'box')[0]!;
  for (const plane of planes) for (let i = 0; i < plane.length; i++) plane[i]! /= 255;
  for (let i = 0; i < a.length; i++) a[i]! /= 255;
  return [planes, a];
}

/**
 * Replaces the colours of `rgba` by the estimated foreground colours and
 * writes `alpha` into its alpha channel. Fully transparent pixels become
 * (0, 0, 0, 0) so no trace of the removed background stays in the file.
 */
export function applyForeground(
  rgba: Uint8ClampedArray,
  alpha: Uint8Array,
  width: number,
  height: number,
  options: ForegroundOptions = {},
): void {
  const n = width * height;
  let soft = 0;
  for (let i = 0; i < n; i++) {
    const v = alpha[i]!;
    if (v !== 0 && v !== 255) soft++;
  }
  if (soft === 0) {
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      if (alpha[i] === 0) rgba[p] = rgba[p + 1] = rgba[p + 2] = 0;
      rgba[p + 3] = alpha[i]!;
    }
    return;
  }

  const workPixels = options.workPixels ?? 2_000_000;
  const [radius1, radius2] = options.radii ?? [90, 6];
  const s = Math.max(1, Math.ceil(Math.sqrt(n / workPixels)));
  const ww = s === 1 ? width : Math.max(1, Math.round(width / s));
  const wh = s === 1 ? height : Math.max(1, Math.round(height / s));
  const r1 = Math.max(1, Math.round(radius1 / s));
  const r2 = Math.max(1, Math.round(radius2 / s));

  const est = computeEstimates(...workingPlanes(rgba, alpha, width, height, ww, wh), ww, wh, r1, r2);

  // Bilinear lookup positions of each full-resolution column in the working grid.
  const sx = ww / width;
  const sy = wh / height;
  const x0s = new Int32Array(width);
  const x1s = new Int32Array(width);
  const fxs = new Float32Array(width);
  for (let x = 0; x < width; x++) {
    const gx = Math.min(Math.max((x + 0.5) * sx - 0.5, 0), ww - 1);
    const x0 = Math.floor(gx);
    x0s[x] = x0;
    x1s[x] = Math.min(x0 + 1, ww - 1);
    fxs[x] = gx - x0;
  }
  const fg = est.fg;
  const bg = est.bg;
  for (let y = 0; y < height; y++) {
    const gy = Math.min(Math.max((y + 0.5) * sy - 0.5, 0), wh - 1);
    const y0 = Math.floor(gy);
    const y1 = Math.min(y0 + 1, wh - 1);
    const fy = gy - y0;
    const row0 = y0 * ww;
    const row1 = y1 * ww;
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const p = i * 4;
      const av = alpha[i]!;
      rgba[p + 3] = av;
      if (av === 255) continue;
      if (av === 0) {
        rgba[p] = rgba[p + 1] = rgba[p + 2] = 0;
        continue;
      }
      const ai = av / 255;
      const fx = fxs[x]!;
      const i00 = row0 + x0s[x]!;
      const i01 = row0 + x1s[x]!;
      const i10 = row1 + x0s[x]!;
      const i11 = row1 + x1s[x]!;
      const w00 = (1 - fx) * (1 - fy);
      const w01 = fx * (1 - fy);
      const w10 = (1 - fx) * fy;
      const w11 = fx * fy;
      for (let c = 0; c < 3; c++) {
        const F = fg[c]!;
        const B = bg[c]!;
        const f = F[i00]! * w00 + F[i01]! * w01 + F[i10]! * w10 + F[i11]! * w11;
        const b = B[i00]! * w00 + B[i01]! * w01 + B[i10]! * w10 + B[i11]! * w11;
        const I = rgba[p + c]! / 255;
        const v = f + ai * (I - ai * f - (1 - ai) * b);
        rgba[p + c] = v * 255; // Uint8ClampedArray rounds and clamps
      }
    }
  }
}
