/**
 * Before/after comparison. `--split` is the divider position in percent:
 * the original is shown left of it, the result right of it. Works with mouse,
 * touch (horizontal drag, vertical scrolling still works) and keyboard (a
 * visually hidden range input that screen readers announce).
 */

import { t } from '../i18n';

const REVEAL_MS = 900;

function prefersReducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export class CompareSlider {
  private enabled = false;
  private dragging = false;
  private animation = 0;
  private hintTimer = 0;
  private split = 100;

  constructor(
    private readonly frame: HTMLElement,
    private readonly range: HTMLInputElement,
  ) {
    range.addEventListener('input', () => {
      this.stopAnimation();
      this.hideHint();
      this.set(Number(range.value));
    });
    // Never let the browser start its own drag of the preview image.
    frame.addEventListener('dragstart', (e) => e.preventDefault());
    frame.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    frame.addEventListener('pointermove', (e) => this.onPointerMove(e));
    frame.addEventListener('pointerup', (e) => this.onPointerUp(e));
    frame.addEventListener('pointercancel', (e) => this.onPointerUp(e));
    this.set(100);
  }

  set(split: number): void {
    this.split = Math.min(100, Math.max(0, split));
    this.frame.style.setProperty('--split', this.split.toFixed(2));
    const rounded = Math.round(this.split);
    this.range.value = String(rounded);
    this.range.setAttribute('aria-valuetext', t('result.compareValue', { value: rounded }));
  }

  /** Enables interaction (only while a result is shown). */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
    this.range.disabled = !enabled;
    if (!enabled) {
      this.stopAnimation();
      this.hideHint();
      this.dragging = false;
    }
  }

  /** Wipes from the original to the result. */
  reveal(): void {
    this.stopAnimation();
    if (prefersReducedMotion()) {
      this.set(0);
      this.showHint();
      return;
    }
    const start = performance.now();
    const from = 100;
    const step = (now: number) => {
      const p = Math.min(1, (now - start) / REVEAL_MS);
      const eased = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
      this.set(from * (1 - eased));
      if (p < 1) this.animation = requestAnimationFrame(step);
      else {
        this.animation = 0;
        this.showHint();
      }
    };
    this.animation = requestAnimationFrame(step);
  }

  private stopAnimation(): void {
    if (this.animation) cancelAnimationFrame(this.animation);
    this.animation = 0;
  }

  private showHint(): void {
    this.frame.dataset.hint = 'true';
    window.clearTimeout(this.hintTimer);
    this.hintTimer = window.setTimeout(() => this.hideHint(), 2600);
  }

  private hideHint(): void {
    window.clearTimeout(this.hintTimer);
    delete this.frame.dataset.hint;
  }

  private updateFromPointer(e: PointerEvent): void {
    const rect = this.frame.getBoundingClientRect();
    if (rect.width <= 0) return;
    this.set(((e.clientX - rect.left) / rect.width) * 100);
  }

  private onPointerDown(e: PointerEvent): void {
    if (!this.enabled || e.button !== 0) return;
    e.preventDefault(); // no text selection or native drag while comparing
    this.stopAnimation();
    this.hideHint();
    this.dragging = true;
    this.frame.setPointerCapture(e.pointerId);
    this.updateFromPointer(e);
  }

  private onPointerMove(e: PointerEvent): void {
    if (this.dragging) this.updateFromPointer(e);
  }

  private onPointerUp(e: PointerEvent): void {
    if (!this.dragging) return;
    this.dragging = false;
    if (this.frame.hasPointerCapture(e.pointerId)) this.frame.releasePointerCapture(e.pointerId);
  }
}
