import { describe, expect, it } from 'vitest';
import { clampRect, contains, fitRatio, FULL, isFull, isRect, moveRect, relativeTo, resizeRect, toPixels, type Rect } from './crop';

const MIN = { w: 0.05, h: 0.05 };
const close = (a: Rect, b: Rect) => {
  for (const k of ['x', 'y', 'w', 'h'] as const) expect(a[k]).toBeCloseTo(b[k], 6);
};

describe('crop geometry', () => {
  it('moves within the image', () => {
    close(moveRect({ x: 0.2, y: 0.2, w: 0.5, h: 0.5 }, 0.1, -0.1), { x: 0.3, y: 0.1, w: 0.5, h: 0.5 });
    close(moveRect({ x: 0.2, y: 0.2, w: 0.5, h: 0.5 }, 0.9, -0.9), { x: 0.5, y: 0, w: 0.5, h: 0.5 });
  });

  it('resizes from each corner while the opposite corner stays put', () => {
    const r = { x: 0.2, y: 0.2, w: 0.4, h: 0.4 };
    close(resizeRect(r, 'se', 0.1, 0.2, MIN), { x: 0.2, y: 0.2, w: 0.5, h: 0.6 });
    close(resizeRect(r, 'nw', -0.1, 0.1, MIN), { x: 0.1, y: 0.3, w: 0.5, h: 0.3 });
    close(resizeRect(r, 'ne', 0.1, -0.1, MIN), { x: 0.2, y: 0.1, w: 0.5, h: 0.5 });
    close(resizeRect(r, 'sw', 0.1, 0.1, MIN), { x: 0.3, y: 0.2, w: 0.3, h: 0.5 });
  });

  it('stops at the image border and at the minimum size', () => {
    const r = { x: 0.2, y: 0.2, w: 0.4, h: 0.4 };
    close(resizeRect(r, 'se', 5, 5, MIN), { x: 0.2, y: 0.2, w: 0.8, h: 0.8 });
    close(resizeRect(r, 'nw', 5, 5, MIN), { x: 0.55, y: 0.55, w: 0.05, h: 0.05 });
  });

  it('keeps an aspect ratio when asked to', () => {
    const r = { x: 0.1, y: 0.1, w: 0.4, h: 0.4 };
    const out = resizeRect(r, 'se', 0.3, 0.05, MIN, 1);
    expect(out.w).toBeCloseTo(out.h, 6);
    expect(out.w).toBeCloseTo(0.7, 6);
    const wide = resizeRect(r, 'se', 5, 5, MIN, 2);
    expect(wide.w / wide.h).toBeCloseTo(2, 6);
    expect(wide.x + wide.w).toBeLessThanOrEqual(1 + 1e-9);
    expect(wide.y + wide.h).toBeLessThanOrEqual(1 + 1e-9);
  });

  it('fits a ratio into a rectangle', () => {
    const sq = fitRatio(FULL, 1);
    close(sq, { x: 0, y: 0, w: 1, h: 1 });
    const inWide = fitRatio({ x: 0, y: 0.25, w: 1, h: 0.5 }, 1);
    close(inWide, { x: 0.25, y: 0.25, w: 0.5, h: 0.5 });
  });

  it('converts between absolute and relative rectangles', () => {
    const outer = { x: 0.2, y: 0.1, w: 0.5, h: 0.8 };
    const inner = { x: 0.3, y: 0.3, w: 0.2, h: 0.4 };
    expect(contains(outer, inner)).toBe(true);
    expect(contains(inner, outer)).toBe(false);
    close(relativeTo(outer, inner), { x: 0.2, y: 0.25, w: 0.4, h: 0.5 });
    expect(isFull(relativeTo(outer, outer))).toBe(true);
    expect(isFull(clampRect({ x: -1, y: -1, w: 3, h: 3 }))).toBe(true);
  });

  it('maps to whole pixels, at least one', () => {
    expect(toPixels({ x: 0.25, y: 0.5, w: 0.5, h: 0.25 }, 400, 200)).toEqual({ x: 100, y: 100, w: 200, h: 50 });
    expect(toPixels({ x: 0.999, y: 0.999, w: 0.0001, h: 0.0001 }, 10, 10)).toEqual({ x: 9, y: 9, w: 1, h: 1 });
  });

  it('validates untrusted rectangles', () => {
    expect(isRect({ x: 0, y: 0, w: 1, h: 1 })).toBe(true);
    expect(isRect({ x: 0, y: 0, w: 0, h: 1 })).toBe(false);
    expect(isRect({ x: 0, y: Number.NaN, w: 1, h: 1 })).toBe(false);
    expect(isRect(null)).toBe(false);
  });
});
