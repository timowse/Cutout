/** Renders the application state into the static markup of index.html. */

import { formatMB, formatNumber, t, type MessageKey } from '../i18n';
import { outputFileName } from '../shared/filename';
import type { ErrorCode, RGB } from '../shared/protocol';
import { CompareSlider } from './compare';
import type { Features } from './features';
import type { AppState, ModelStatus, ViewState } from './state';

export interface ViewHandlers {
  choose(): void;
  retryModel(): void;
  cancel(): void;
  startOver(): void;
  retry(): void;
  copy(): void;
  downloadWithBackground(color: RGB, background: 'white' | 'black' | { hex: string }): void;
}

export type BackgroundChoice = 'transparent' | 'white' | 'black' | 'custom';

const RETRYABLE: ReadonlySet<ErrorCode> = new Set([
  'out-of-memory',
  'model-download-failed',
  'model-integrity',
  'offline',
  'inference-failed',
  'unknown',
]);

function byId(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el;
}

export function hexToRgb(hex: string): RGB {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  const n = m ? parseInt(m[1]!, 16) : 0;
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

export class View {
  private readonly root = document.documentElement;
  private readonly stage = document.querySelector<HTMLElement>('.stage')!;
  private readonly frame = byId('frame');
  private readonly original = byId('original-image') as HTMLImageElement;
  private readonly canvas = byId('result-canvas') as HTMLCanvasElement;
  private readonly statusText = byId('status-text');
  private readonly progress = byId('progress');
  private readonly progressBar = byId('progress-bar');
  private readonly statusHint = byId('status-hint');
  private readonly download = byId('download-link') as HTMLAnchorElement;
  private readonly downloadLabel = byId('download-label');
  private readonly meta = byId('result-meta');
  private readonly scaledNote = byId('scaled-note');
  private readonly copyButton = byId('copy-button') as HTMLButtonElement;
  private readonly bgButton = byId('download-bg-button') as HTMLButtonElement;
  private readonly colorInput = byId('color-input') as HTMLInputElement;
  private readonly errorBox = byId('error');
  private readonly errorTitle = byId('error-title');
  private readonly errorText = byId('error-text');
  private readonly retryButton = byId('retry-button') as HTMLButtonElement;
  private readonly toast = byId('toast');
  private readonly compare: CompareSlider;
  private readonly bitmapContext: ImageBitmapRenderingContext | null;
  private drawnJob: number | null = null;
  private chipRendered = false;
  private toastTimer = 0;
  private background: BackgroundChoice = 'transparent';

  constructor(
    handlers: ViewHandlers,
    private readonly features: Features,
  ) {
    this.compare = new CompareSlider(this.frame, byId('compare-range') as HTMLInputElement);
    this.bitmapContext = this.canvas.getContext('bitmaprenderer');

    byId('choose-button').addEventListener('click', () => handlers.choose());
    byId('model-chip-retry').addEventListener('click', () => handlers.retryModel());
    byId('dropzone').addEventListener('click', (e) => {
      if (!(e.target as HTMLElement).closest('button')) handlers.choose();
    });
    byId('cancel-button').addEventListener('click', () => handlers.cancel());
    byId('new-button').addEventListener('click', () => handlers.startOver());
    byId('another-button').addEventListener('click', () => handlers.startOver());
    this.retryButton.addEventListener('click', () => handlers.retry());
    this.copyButton.addEventListener('click', () => handlers.copy());
    this.download.addEventListener('click', (e) => {
      if (this.download.getAttribute('aria-disabled') === 'true') e.preventDefault();
    });
    for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="bg"]')) {
      radio.addEventListener('change', () => {
        if (radio.checked) this.setBackground(radio.value as BackgroundChoice);
      });
    }
    this.colorInput.addEventListener('input', () => {
      this.frame.style.setProperty('--custom-bg', this.colorInput.value);
      const custom = document.querySelector<HTMLInputElement>('input[name="bg"][value="custom"]');
      if (custom && !custom.checked) {
        custom.checked = true;
        this.setBackground('custom');
      }
    });
    this.bgButton.addEventListener('click', () => {
      const bg = this.background;
      if (bg === 'white') handlers.downloadWithBackground({ r: 255, g: 255, b: 255 }, 'white');
      else if (bg === 'black') handlers.downloadWithBackground({ r: 0, g: 0, b: 0 }, 'black');
      else if (bg === 'custom') handlers.downloadWithBackground(hexToRgb(this.colorInput.value), { hex: this.colorInput.value });
    });
    this.frame.style.setProperty('--custom-bg', this.colorInput.value);
    this.setBackground('transparent');
  }

  private setBackground(choice: BackgroundChoice): void {
    this.background = choice;
    this.frame.dataset.bg = choice;
    this.bgButton.hidden = choice === 'transparent';
    this.colorInput.hidden = choice !== 'custom';
  }

  /** Explains why the page reloaded after the browser ended it while busy. */
  showCrashNotice(stage: 'model' | 'image'): void {
    const notice = byId('crash-notice');
    notice.textContent = t(stage === 'image' ? 'crash.image' : 'crash.model');
    notice.hidden = false;
  }

  showToast(text: string): void {
    this.toast.textContent = text;
    this.toast.classList.add('visible');
    window.clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => this.toast.classList.remove('visible'), 3800);
  }

  /** The quiet model status in the header: download progress, preparing, ready or failed. */
  private renderModelChip(model: ModelStatus): void {
    const chip = byId('model-chip');
    const text = byId('model-chip-text');
    const short = byId('model-chip-short');
    const indicator = chip.querySelector<HTMLElement>('.model-chip-indicator');
    const retry = byId('model-chip-retry');
    let state: string;
    let long = '';
    let brief = '';
    let title = '';
    switch (model.kind) {
      case 'idle':
        chip.hidden = true;
        return;
      case 'loading':
        if (model.phase === 'download' && model.total > 0) {
          const percent = Math.min(100, Math.floor((model.loaded / model.total) * 100));
          state = 'download';
          long = t('chip.download', { percent });
          brief = t('chip.downloadShort', { percent });
          title = t('chip.downloadTitle', { loaded: formatMB(model.loaded), total: formatMB(model.total) });
          indicator?.style.setProperty('--p', String(percent));
        } else {
          state = 'preparing';
          long = t('chip.preparing');
          brief = t('chip.preparingShort');
        }
        break;
      case 'ready':
        state = 'ready';
        long = t('chip.ready');
        brief = t('chip.readyShort');
        title = t(model.backend === 'webgpu' ? 'chip.readyGpu' : 'chip.readyCpu');
        break;
      case 'error':
        state = 'error';
        long = t('chip.error');
        brief = t('chip.errorShort');
        break;
    }
    chip.hidden = false;
    chip.dataset.state = state;
    if (text.textContent !== long) text.textContent = long;
    if (short.textContent !== brief) short.textContent = brief;
    if (title) chip.title = title;
    else chip.removeAttribute('title');
    retry.hidden = model.kind !== 'error';
  }

  render(state: AppState, prev: AppState): void {
    if (state.model !== prev.model || !this.chipRendered) {
      this.chipRendered = true;
      this.renderModelChip(state.model);
    }
    const view = state.view;
    const name = view.kind === 'cancelled' ? 'idle' : view.kind;
    this.root.dataset.view = name;

    if (view.kind === 'idle' || view.kind === 'cancelled') {
      this.clearStage();
      if (prev.view.kind !== 'idle' && prev.view.kind !== 'cancelled') {
        (byId('choose-button') as HTMLButtonElement).focus({ preventScroll: true });
      }
      return;
    }

    const source = 'source' in view ? view.source : null;
    this.stage.hidden = !source;
    if (source && this.original.dataset.src !== source.url) {
      this.clearResult();
      this.original.dataset.src = source.url;
      this.original.onload = () => {
        if (this.drawnJob === null && this.original.naturalWidth > 0) {
          this.setAspect(this.original.naturalWidth, this.original.naturalHeight);
        }
      };
      this.original.src = source.url;
    }

    switch (view.kind) {
      case 'loading-image':
        this.renderStatus(t('status.opening'), null);
        break;
      case 'loading-model':
        this.renderModelLoading(state);
        break;
      case 'processing':
        this.renderProcessing(state, view);
        break;
      case 'complete':
        this.renderResult(view, prev.view);
        break;
      case 'error':
        this.renderError(view, prev.view);
        break;
    }
    if (view.kind !== 'complete') this.compare.setEnabled(false);
  }

  private setAspect(width: number, height: number): void {
    this.frame.style.setProperty('--ar', (width / height).toFixed(5));
  }

  private clearResult(): void {
    this.drawnJob = null;
    byId('sr-status').textContent = '';
    this.bitmapContext?.transferFromImageBitmap(null);
    this.compare.setEnabled(false);
    this.compare.set(100);
  }

  private clearStage(): void {
    this.clearResult();
    this.original.onload = null;
    this.original.removeAttribute('src');
    delete this.original.dataset.src;
    this.download.removeAttribute('href');
  }

  /** `progress` in [0, 1], or null for an indeterminate bar. */
  private renderStatus(text: string, progress: number | null, hint: string | null = null): void {
    if (this.statusText.textContent !== text) this.statusText.textContent = text;
    this.progress.dataset.indeterminate = String(progress === null);
    this.progressBar.style.setProperty('--progress', String(progress ?? 0));
    this.statusHint.textContent = hint ?? '';
    this.statusHint.hidden = !hint;
    this.compare.set(100);
  }

  private renderModelLoading(state: AppState): void {
    const model = state.model;
    const firstUse = t('status.hintFirst');
    if (model.kind === 'loading') {
      switch (model.phase) {
        case 'download':
          this.renderStatus(
            t('status.downloading', { loaded: formatMB(model.loaded), total: formatMB(model.total) }),
            model.total > 0 ? model.loaded / model.total : null,
            firstUse,
          );
          return;
        case 'cache':
          this.renderStatus(t('status.cache'), model.total > 0 ? model.loaded / model.total : null);
          return;
        case 'init':
          this.renderStatus(t('status.init'), null);
          return;
        case 'manifest':
          this.renderStatus(t('status.preparing'), null);
          return;
      }
    }
    this.renderStatus(t(model.kind === 'ready' ? 'status.removing' : 'status.preparing'), null);
  }

  private renderProcessing(state: AppState, view: Extract<ViewState, { kind: 'processing' }>): void {
    const key: MessageKey = view.stage === 'inference' ? 'status.removing' : 'status.refining';
    let hint: string | null = null;
    if (state.fellBack) hint = t('status.fallback');
    else if (state.model.kind === 'ready' && state.model.backend === 'wasm' && view.stage === 'inference') {
      hint = t('status.hintCpu');
    }
    this.renderStatus(t(key), null, hint);
  }

  private renderResult(view: Extract<ViewState, { kind: 'complete' }>, prev: ViewState): void {
    const { result, source, jobId } = view;
    if (this.drawnJob !== jobId) {
      this.drawnJob = jobId;
      this.setAspect(result.width, result.height);
      this.canvas.width = result.preview.width;
      this.canvas.height = result.preview.height;
      this.bitmapContext?.transferFromImageBitmap(result.preview);
      this.compare.setEnabled(true);
      this.compare.reveal();
      this.statusText.textContent = '';
      byId('sr-status').textContent = t('result.ready');
      const active = document.activeElement;
      if (!active || active === document.body || active.closest('.view-idle, .status')) {
        this.download.focus({ preventScroll: true });
      }
    }
    if (result.pngUrl) {
      this.download.href = result.pngUrl;
      this.download.download = outputFileName(source.name);
      this.download.setAttribute('aria-disabled', 'false');
      this.downloadLabel.textContent = t('result.download');
    } else {
      this.download.removeAttribute('href');
      this.download.setAttribute('aria-disabled', 'true');
      this.downloadLabel.textContent = t('result.preparing');
    }
    this.meta.textContent = t('result.meta', { width: formatNumber(result.width), height: formatNumber(result.height) });
    const scaled = result.width !== result.originalWidth || result.height !== result.originalHeight;
    this.scaledNote.hidden = !scaled;
    if (scaled) {
      this.scaledNote.textContent = t('result.scaled', {
        ow: formatNumber(result.originalWidth),
        oh: formatNumber(result.originalHeight),
      });
    }
    this.copyButton.hidden = !this.features.clipboardWriteImage;
    if (prev.kind !== 'complete') this.bgButton.hidden = this.background === 'transparent';
  }

  private renderError(view: Extract<ViewState, { kind: 'error' }>, prev: ViewState): void {
    this.errorTitle.textContent = t(`error.${view.error}.title` as MessageKey);
    this.errorText.textContent = t(`error.${view.error}.text` as MessageKey);
    this.retryButton.hidden = !(view.source && RETRYABLE.has(view.error));
    this.compare.set(100);
    if (prev.kind !== 'error') this.errorBox.focus({ preventScroll: true });
  }
}
