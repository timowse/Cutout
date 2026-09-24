/** Typed wrapper around the inference worker. */

import type { ModelConfig, WorkerRequest, WorkerResponse } from '../shared/protocol';

export type WorkerListener = (message: WorkerResponse) => void;

export class InferenceClient {
  private worker: Worker | null = null;
  private readonly listeners = new Set<WorkerListener>();
  private modelRequested = false;

  constructor(private readonly config: ModelConfig) {}

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const worker = new Worker(new URL('../worker/inference.worker.ts', import.meta.url), {
      type: 'module',
      name: 'inference',
    });
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      if (event.data.type === 'MODEL_ERROR') this.modelRequested = false;
      for (const listener of this.listeners) listener(event.data);
    };
    worker.onerror = (event) => {
      event.preventDefault();
      this.modelRequested = false;
      for (const listener of this.listeners) {
        listener({ type: 'MODEL_ERROR', error: { code: 'runtime-unsupported', detail: event.message } });
      }
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
