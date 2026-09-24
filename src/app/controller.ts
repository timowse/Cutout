/**
 * Connects input, worker and view. Owns every per-image resource (object
 * URLs, bitmaps, the worker-side result) and frees it as soon as the image is
 * replaced or dismissed.
 */

import { t } from '../i18n';
import { exceedsInputLimit } from '../shared/dimensions';
import { inspectImageFile, MAX_FILE_BYTES } from '../shared/files';
import { outputFileName, type BackgroundName } from '../shared/filename';
import type { ImageLimits, RGB, WorkerResponse } from '../shared/protocol';
import { missingRequirement, type Features } from './features';
import { currentJobId, initialState, isBusy, reduce, type AppEvent, type AppState } from './state';
import type { View } from './view';
import type { InferenceClient } from './worker-client';

interface JobResources {
  sourceUrl: string;
  preview?: ImageBitmap;
  pngUrl?: string;
  extraUrls: string[];
}

export class Controller {
  private state: AppState = initialState;
  private view: View | null = null;
  private nextJobId = 1;
  private inputSeq = 0;
  private lastFile: File | null = null;
  private readonly resources = new Map<number, JobResources>();
  private readonly pngWaiters = new Map<number, ((png: Blob | null) => void)[]>();
  private readonly bgRequests = new Map<number, (png: Blob | null) => void>();
  private nextBgRequest = 1;

  constructor(
    private readonly client: InferenceClient,
    private readonly features: Features,
    private readonly limits: ImageLimits,
    private readonly previewMaxSide: number,
  ) {
    client.subscribe((msg) => this.onWorkerMessage(msg));
  }

  attach(view: View): void {
    this.view = view;
    view.render(this.state, this.state);
  }

  getState(): AppState {
    return this.state;
  }

  private dispatch(event: AppEvent): void {
    const prev = this.state;
    this.state = reduce(prev, event);
    if (this.state !== prev) {
      this.releaseStaleResources();
      this.view?.render(this.state, prev);
      document.documentElement.dataset.model = this.state.model.kind === 'ready' ? this.state.model.backend : this.state.model.kind;
    }
  }

  /** Frees everything that belongs to jobs other than the one on screen. */
  private releaseStaleResources(): void {
    const keep = currentJobId(this.state.view);
    for (const [jobId, res] of this.resources) {
      if (jobId === keep) continue;
      URL.revokeObjectURL(res.sourceUrl);
      if (res.pngUrl) URL.revokeObjectURL(res.pngUrl);
      for (const url of res.extraUrls) URL.revokeObjectURL(url);
      res.preview?.close();
      this.client.release(jobId);
      this.resources.delete(jobId);
      for (const resolve of this.pngWaiters.get(jobId) ?? []) resolve(null);
      this.pngWaiters.delete(jobId);
    }
  }

  /** Entry point for every new image (drop, picker, paste, retry). */
  async handleFile(file: File): Promise<void> {
    const seq = ++this.inputSeq;
    const missing = missingRequirement(this.features);
    if (missing) {
      this.supersede();
      this.dispatch({ type: 'reject', error: 'runtime-unsupported' });
      return;
    }
    if (file.size === 0) {
      this.supersede();
      this.dispatch({ type: 'reject', error: 'decode-failed' });
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      this.supersede();
      this.dispatch({ type: 'reject', error: 'too-large' });
      return;
    }
    let info: Awaited<ReturnType<typeof inspectImageFile>>;
    try {
      info = await inspectImageFile(file);
    } catch {
      info = null;
    }
    if (seq !== this.inputSeq) return; // a newer image arrived meanwhile
    if (!info) {
      this.supersede();
      this.dispatch({ type: 'reject', error: 'unsupported-format' });
      return;
    }
    if (info.width && info.height && exceedsInputLimit(info.width, info.height, this.limits)) {
      this.supersede();
      this.dispatch({ type: 'reject', error: 'too-large' });
      return;
    }

    this.supersede();
    const jobId = this.nextJobId++;
    const sourceUrl = URL.createObjectURL(file);
    this.resources.set(jobId, { sourceUrl, extraUrls: [] });
    this.lastFile = file;
    this.dispatch({ type: 'select', jobId, source: { name: file.name, url: sourceUrl } });
    this.client.process({ jobId, file, limits: this.limits, previewMaxSide: this.previewMaxSide });
  }

  /** Cancels the job on screen, if it is still running. */
  private supersede(): void {
    const jobId = currentJobId(this.state.view);
    if (jobId !== null && isBusy(this.state.view)) this.client.cancel(jobId);
  }

  warmUp(): void {
    if (!missingRequirement(this.features)) this.client.initModel();
  }

  cancel(): void {
    const jobId = currentJobId(this.state.view);
    if (jobId === null || !isBusy(this.state.view)) return;
    this.client.cancel(jobId);
    this.dispatch({ type: 'cancel', jobId });
    this.view?.showToast(t('toast.cancelled'));
  }

  startOver(): void {
    this.supersede();
    this.dispatch({ type: 'reset' });
  }

  retry(): void {
    if (this.lastFile) void this.handleFile(this.lastFile);
  }

  private waitForPng(jobId: number): Promise<Blob | null> {
    const view = this.state.view;
    if (view.kind === 'complete' && view.jobId === jobId && view.result.png) return Promise.resolve(view.result.png);
    return new Promise((resolve) => {
      const list = this.pngWaiters.get(jobId) ?? [];
      list.push(resolve);
      this.pngWaiters.set(jobId, list);
    });
  }

  async copy(): Promise<void> {
    const view = this.state.view;
    if (view.kind !== 'complete' || !this.features.clipboardWriteImage) return;
    try {
      // Safari requires ClipboardItem to be created synchronously in the click handler,
      // so it gets a promise that resolves once the PNG is ready.
      const png = this.waitForPng(view.jobId).then((blob) => {
        if (!blob) throw new Error('Result no longer available');
        return blob;
      });
      await navigator.clipboard.write([new ClipboardItem({ 'image/png': png })]);
      this.view?.showToast(t('result.copied'));
    } catch {
      this.view?.showToast(t('result.copyFailed'));
    }
  }

  downloadWithBackground(color: RGB, background: BackgroundName): void {
    const view = this.state.view;
    if (view.kind !== 'complete') return;
    const { jobId } = view;
    const requestId = this.nextBgRequest++;
    const fileName = outputFileName(view.source.name, background);
    this.bgRequests.set(requestId, (png) => {
      if (!png || currentJobId(this.state.view) !== jobId) return;
      const url = URL.createObjectURL(png);
      this.resources.get(jobId)?.extraUrls.push(url);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      a.rel = 'noopener';
      document.body.append(a);
      a.click();
      a.remove();
    });
    this.client.renderWithBackground({ jobId, requestId, color });
  }

  private onWorkerMessage(msg: WorkerResponse): void {
    switch (msg.type) {
      case 'MODEL_PROGRESS':
        this.dispatch({ type: 'model-progress', phase: msg.phase, loaded: msg.loaded, total: msg.total });
        break;
      case 'MODEL_READY':
        this.dispatch({ type: 'model-ready', backend: msg.backend, fromCache: msg.fromCache });
        document.documentElement.dataset.modelLoadMs = String(msg.loadMs);
        break;
      case 'MODEL_ERROR':
        console.warn('[model]', msg.error.code, msg.error.detail ?? '');
        this.dispatch({ type: 'model-error', error: msg.error.code });
        break;
      case 'BACKEND_FALLBACK':
        console.warn('[runtime] falling back to', msg.to, '-', msg.reason);
        this.dispatch({ type: 'fallback' });
        break;
      case 'PROCESS_PROGRESS':
        this.dispatch({ type: 'stage', jobId: msg.jobId, stage: msg.stage });
        break;
      case 'PROCESS_PREVIEW': {
        const res = this.resources.get(msg.jobId);
        if (!res || currentJobId(this.state.view) !== msg.jobId) {
          msg.preview.close();
          break;
        }
        res.preview = msg.preview;
        this.dispatch({
          type: 'preview',
          jobId: msg.jobId,
          result: {
            preview: msg.preview,
            width: msg.width,
            height: msg.height,
            originalWidth: msg.originalWidth,
            originalHeight: msg.originalHeight,
          },
        });
        break;
      }
      case 'PROCESS_COMPLETE': {
        const res = this.resources.get(msg.jobId);
        if (!res || currentJobId(this.state.view) !== msg.jobId) break;
        res.pngUrl = URL.createObjectURL(msg.png);
        document.documentElement.dataset.timings = JSON.stringify(msg.timings);
        this.dispatch({ type: 'png', jobId: msg.jobId, png: msg.png, pngUrl: res.pngUrl });
        for (const resolve of this.pngWaiters.get(msg.jobId) ?? []) resolve(msg.png);
        this.pngWaiters.delete(msg.jobId);
        break;
      }
      case 'PROCESS_ERROR':
        console.warn('[process]', msg.error.code, msg.error.detail ?? '');
        this.dispatch({ type: 'job-error', jobId: msg.jobId, error: msg.error.code });
        break;
      case 'BACKGROUND_RENDERED':
        this.bgRequests.get(msg.requestId)?.(msg.png);
        this.bgRequests.delete(msg.requestId);
        break;
      case 'BACKGROUND_ERROR':
        console.warn('[background]', msg.error.code, msg.error.detail ?? '');
        this.bgRequests.get(msg.requestId)?.(null);
        this.bgRequests.delete(msg.requestId);
        this.view?.showToast(t('error.unknown.title'));
        break;
    }
  }
}
