/** Typed wrapper around the inference worker. */

import type { ModelConfig, WorkerRequest, WorkerResponse } from '../shared/protocol';
import workerScriptUrl from '../worker/inference.worker.ts?worker&url';

export type WorkerListener = (message: WorkerResponse) => void;

export class InferenceClient {
  private worker: Worker | null = null;
  private readonly listeners = new Set<WorkerListener>();
  private modelRequested = false;

  constructor(private readonly config: ModelConfig) {}

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    // The worker is started from a tiny blob: module that imports the real
    // script. Workers loaded from blob: URLs inherit the page's
    // Content-Security-Policy, so `connect-src 'self'` also binds the worker
    // (a worker loaded directly from an https: URL would get the policy of its
    // HTTP response, and static hosts like GitHub Pages cannot set one).
    const scriptUrl = new URL(workerScriptUrl, location.href).href;
    const bootstrap = URL.createObjectURL(new Blob([`import ${JSON.stringify(scriptUrl)};`], { type: 'text/javascript' }));
    const worker = new Worker(bootstrap, { type: 'module', name: 'inference' });
    let started = false;
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      if (!started) URL.revokeObjectURL(bootstrap);
      started = true;
      if (event.data.type === 'MODEL_ERROR') this.modelRequested = false;
      for (const listener of this.listeners) listener(event.data);
    };
    worker.onerror = (event) => {
      // The worker script failed to load (old browser) or crashed: start a fresh one next time.
      event.preventDefault();
      URL.revokeObjectURL(bootstrap);
      worker.terminate();
      if (this.worker === worker) this.worker = null;
      this.modelRequested = false;
      const code = started ? 'inference-failed' : 'runtime-unsupported';
      for (const listener of this.listeners) listener({ type: 'MODEL_ERROR', error: { code, detail: event.message } });
    };
    this.worker = worker;
    return worker;
  }

  private send(message: WorkerRequest, transfer: Transferable[] = []): void {
    this.ensureWorker().postMessage(message, transfer);
  }

  subscribe(listener: WorkerListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Starts downloading and initialising the model (idempotent). */
  initModel(): void {
    if (this.modelRequested) return;
    this.modelRequested = true;
    this.send({ type: 'INIT_MODEL', config: this.config });
  }

  process(message: Omit<Extract<WorkerRequest, { type: 'PROCESS_IMAGE' }>, 'type'>): void {
    this.initModel();
    this.send({ type: 'PROCESS_IMAGE', ...message });
  }

  cancel(jobId: number): void {
    if (this.worker) this.send({ type: 'CANCEL_JOB', jobId });
  }

  release(jobId: number): void {
    if (this.worker) this.send({ type: 'RELEASE_RESULT', jobId });
  }

  renderWithBackground(message: Omit<Extract<WorkerRequest, { type: 'RENDER_WITH_BACKGROUND' }>, 'type'>): void {
    this.send({ type: 'RENDER_WITH_BACKGROUND', ...message });
  }
}
