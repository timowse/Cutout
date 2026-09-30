/**
 * The crop rectangle over the preview: drag inside to move it, drag a corner
 * to resize it; with the keyboard, arrow keys move the focused rectangle or
 * resize from the focused corner (Shift: larger steps). Coordinates are
 * normalised to the whole image (see shared/crop.ts).
 */

import { clampRect, fitRatio, FULL, moveRect, resizeRect, type Corner, type Rect } from '../shared/crop';

/** Smallest crop, in CSS pixels on screen. */
const MIN_PX = 32;

export class CropEditor {
  private rect: Rect = { ...FULL };
  /** Required w/h in pixels, or null for a free crop. */
  private pixelRatio: number | null = null;
  /** Width/height of the image in pixels. */
  private imageAspect = 1;
  private drag: { pointerId: number; corner: Corner | null; x: number; y: number; start: Rect } | null = null;

  constructor(
    private readonly layer: HTMLElement,
    private readonly box: HTMLElement,
  ) {
    layer.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    layer.addEventListener('pointermove', (e) => this.onPointerMove(e));
    layer.addEventListener('pointerup', (e) => this.onPointerUp(e));
    layer.addEventListener('pointercancel', (e) => this.onPointerUp(e));
    layer.addEventListener('keydown', (e) => this.onKeyDown(e));
  }

  get value(): Rect {
    return { ...this.rect };
  }

  open(initial: Rect, imageAspect: number): void {
    this.imageAspect = imageAspect;
    this.rect = clampRect(initial, this.minSize());
    if (this.pixelRatio) this.rect = fitRatio(this.rect, this.normRatio()!);
    this.layer.hidden = false;
    this.draw();
    this.box.focus({ preventScroll: true });
  }

  close(): void {
    this.drag = null;
    this.layer.hidden = true;
  }

  /** `ratio` is width/height in pixels (1 for a square), or null for free. */
  setRatio(ratio: number | null): void {
    this.pixelRatio = ratio;
    if (ratio) this.rect = fitRatio(this.rect, this.normRatio()!);
    this.draw();
  }

  /** Selects the whole image (the largest area with the chosen ratio). */
  selectAll(): void {
    this.rect = this.pixelRatio ? fitRatio(FULL, this.normRatio()!) : { ...FULL };
    this.draw();
  }

  private normRatio(): number | null {
    return this.pixelRatio ? this.pixelRatio / this.imageAspect : null;
  }

  private minSize(): { w: number; h: number } {
    const { width, height } = this.layer.getBoundingClientRect();
    return {
      w: width > 0 ? Math.min(0.5, MIN_PX / width) : 0.05,
      h: height > 0 ? Math.min(0.5, MIN_PX / height) : 0.05,
    };
  }

  private draw(): void {
    const r = this.rect;
    this.box.style.left = `${r.x * 100}%`;
    this.box.style.top = `${r.y * 100}%`;
    this.box.style.width = `${r.w * 100}%`;
    this.box.style.height = `${r.h * 100}%`;
    this.box.dataset.full = String(r.w > 0.999 && r.h > 0.999);
  }

  private onPointerDown(e: PointerEvent): void {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    const handle = target.closest<HTMLElement>('[data-corner]');
    if (!handle && !target.closest('.crop-box')) return;
    e.preventDefault();
    this.layer.setPointerCapture(e.pointerId);
    this.drag = {
      pointerId: e.pointerId,
      corner: (handle?.dataset.corner as Corner | undefined) ?? null,
      x: e.clientX,
      y: e.clientY,
      start: { ...this.rect },
    };
    this.layer.dataset.dragging = 'true';
  }

  private onPointerMove(e: PointerEvent): void {
    const drag = this.drag;
    if (!drag || e.pointerId !== drag.pointerId) return;
    const { width, height } = this.layer.getBoundingClientRect();
    if (width <= 0 || height <= 0) return;
    const dx = (e.clientX - drag.x) / width;
    const dy = (e.clientY - drag.y) / height;
    this.rect = drag.corner
      ? resizeRect(drag.start, drag.corner, dx, dy, this.minSize(), this.normRatio())
      : moveRect(drag.start, dx, dy);
    this.draw();
  }

  private onPointerUp(e: PointerEvent): void {
    if (!this.drag || e.pointerId !== this.drag.pointerId) return;
    this.drag = null;
    delete this.layer.dataset.dragging;
    if (this.layer.hasPointerCapture(e.pointerId)) this.layer.releasePointerCapture(e.pointerId);
  }

  private onKeyDown(e: KeyboardEvent): void {
    const step = e.shiftKey ? 0.05 : 0.01;
    const delta: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    const d = delta[e.key];
    if (!d) return;
    e.preventDefault();
    const corner = (e.target as HTMLElement).dataset.corner as Corner | undefined;
    this.rect = corner
      ? resizeRect(this.rect, corner, d[0], d[1], this.minSize(), this.normRatio())
      : moveRect(this.rect, d[0], d[1]);
    this.draw();
  }
}
