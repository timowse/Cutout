/**
 * Crop rectangles in normalised coordinates: (0, 0) is the top-left and
 * (1, 1) the bottom-right corner of an image, so a rectangle keeps its meaning
 * at any display or output size. Pure functions, shared by the crop editor
 * (UI thread) and the worker.
 */

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type Corner = 'nw' | 'ne' | 'sw' | 'se';

export const FULL: Readonly<Rect> = Object.freeze({ x: 0, y: 0, w: 1, h: 1 });

const EPS = 1e-6;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function isFull(r: Rect, eps = 1e-3): boolean {
  return r.x <= eps && r.y <= eps && r.x + r.w >= 1 - eps && r.y + r.h >= 1 - eps;
}

/** True if `inner` lies within `outer` (both in the same coordinates). */
export function contains(outer: Rect, inner: Rect, eps = 1e-4): boolean {
  return (
    inner.x >= outer.x - eps &&
    inner.y >= outer.y - eps &&
    inner.x + inner.w <= outer.x + outer.w + eps &&
    inner.y + inner.h <= outer.y + outer.h + eps
  );
}

/** `inner` expressed relative to `outer` (the result of cropping to `outer` first). */
export function relativeTo(outer: Rect, inner: Rect): Rect {
  return clampRect({
    x: (inner.x - outer.x) / outer.w,
    y: (inner.y - outer.y) / outer.h,
    w: inner.w / outer.w,
    h: inner.h / outer.h,
  });
}

/** Keeps a rectangle inside the image; `min` is the smallest allowed size. */
export function clampRect(r: Rect, min: { w: number; h: number } = { w: EPS, h: EPS }): Rect {
  const w = clamp(r.w, min.w, 1);
  const h = clamp(r.h, min.h, 1);
  return { x: clamp(r.x, 0, 1 - w), y: clamp(r.y, 0, 1 - h), w, h };
}

export function moveRect(r: Rect, dx: number, dy: number): Rect {
  return { ...r, x: clamp(r.x + dx, 0, 1 - r.w), y: clamp(r.y + dy, 0, 1 - r.h) };
}

/**
 * Drags one corner by (dx, dy) while the opposite corner stays put.
 * `ratio` is the required w/h in normalised units (null: free).
 */
export function resizeRect(
  r: Rect,
  corner: Corner,
  dx: number,
  dy: number,
  min: { w: number; h: number },
  ratio: number | null = null,
): Rect {
  const west = corner === 'nw' || corner === 'sw';
  const north = corner === 'nw' || corner === 'ne';
  // Anchor: the opposite corner.
  const ax = west ? r.x + r.w : r.x;
  const ay = north ? r.y + r.h : r.y;
  // Room between the anchor and the image border in the drag direction.
  const maxW = west ? ax : 1 - ax;
  const maxH = north ? ay : 1 - ay;
  let w = clamp((west ? -1 : 1) * ((west ? r.x : r.x + r.w) + dx - ax), 0, maxW);
  let h = clamp((north ? -1 : 1) * ((north ? r.y : r.y + r.h) + dy - ay), 0, maxH);
  if (ratio) {
    // Follow the larger movement, then fit into the available room.
    if (w / ratio > h) h = w / ratio;
    else w = h * ratio;
    const scale = Math.min(1, maxW / w, maxH / h);
    w *= scale;
    h *= scale;
    const minScale = Math.max(1, min.w / w, min.h / h);
    w = Math.min(w * minScale, maxW);
    h = Math.min(h * minScale, maxH);
  } else {
    w = Math.max(w, Math.min(min.w, maxW));
    h = Math.max(h, Math.min(min.h, maxH));
  }
  return { x: west ? ax - w : ax, y: north ? ay - h : ay, w, h };
}

/** The largest rectangle with the given w/h ratio, centred on `r` and inside the image. */
export function fitRatio(r: Rect, ratio: number): Rect {
  const cx = r.x + r.w / 2;
  const cy = r.y + r.h / 2;
  let w = r.w;
  let h = w / ratio;
  if (h > r.h) {
    h = r.h;
    w = h * ratio;
  }
  return clampRect({ x: cx - w / 2, y: cy - h / 2, w, h });
}

/** Integer pixel rectangle of `r` in an image of `width` × `height` (at least 1 × 1). */
export function toPixels(r: Rect, width: number, height: number): { x: number; y: number; w: number; h: number } {
  const x0 = clamp(Math.round(r.x * width), 0, width - 1);
  const y0 = clamp(Math.round(r.y * height), 0, height - 1);
  const x1 = clamp(Math.round((r.x + r.w) * width), x0 + 1, width);
  const y1 = clamp(Math.round((r.y + r.h) * height), y0 + 1, height);
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

export function isRect(value: unknown): value is Rect {
  if (typeof value !== 'object' || value === null) return false;
  const r = value as Partial<Rect>;
  return [r.x, r.y, r.w, r.h].every((v) => typeof v === 'number' && Number.isFinite(v)) && r.w! > 0 && r.h! > 0;
}
