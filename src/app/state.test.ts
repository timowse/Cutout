import { describe, expect, it } from 'vitest';
import { initialState, isBusy, reduce, type AppEvent, type AppState } from './state';

const source = (name: string) => ({ name, url: `blob:${name}` });
const preview = { preview: {} as ImageBitmap, width: 10, height: 10, originalWidth: 10, originalHeight: 10 };

function run(events: AppEvent[], state: AppState = initialState): AppState {
  return events.reduce(reduce, state);
}

describe('state machine', () => {
  it('walks through the happy path', () => {
    let s = run([{ type: 'select', jobId: 1, source: source('a.jpg') }]);
    expect(s.view.kind).toBe('loading-image');
    s = reduce(s, { type: 'stage', jobId: 1, stage: 'waiting-for-model' });
    expect(s.view.kind).toBe('loading-model');
    s = reduce(s, { type: 'model-progress', phase: 'download', loaded: 10, total: 100 });
    expect(s.model).toEqual({ kind: 'loading', phase: 'download', loaded: 10, total: 100 });
    s = reduce(s, { type: 'model-ready', backend: 'webgpu', fromCache: false });
    s = reduce(s, { type: 'stage', jobId: 1, stage: 'inference' });
    expect(s.view).toMatchObject({ kind: 'processing', stage: 'inference' });
    s = reduce(s, { type: 'preview', jobId: 1, result: preview });
    expect(s.view).toMatchObject({ kind: 'complete', result: { png: null } });
    const png = new Blob();
    s = reduce(s, { type: 'png', jobId: 1, png, pngUrl: 'blob:png' });
    expect(s.view).toMatchObject({ kind: 'complete', result: { png, pngUrl: 'blob:png' } });
  });

  it('ignores events of a superseded job', () => {
    let s = run([
      { type: 'select', jobId: 1, source: source('a.jpg') },
      { type: 'stage', jobId: 1, stage: 'inference' },
      { type: 'select', jobId: 2, source: source('b.jpg') },
    ]);
    s = reduce(s, { type: 'preview', jobId: 1, result: preview });
    expect(s.view).toMatchObject({ kind: 'loading-image', jobId: 2, source: { name: 'b.jpg' } });
    s = reduce(s, { type: 'job-error', jobId: 1, error: 'inference-failed' });
    expect(s.view.kind).toBe('loading-image');
    s = reduce(s, { type: 'png', jobId: 1, png: new Blob(), pngUrl: 'x' });
    expect(s.view.kind).toBe('loading-image');
    s = reduce(s, { type: 'stage', jobId: 2, stage: 'inference' });
    s = reduce(s, { type: 'preview', jobId: 2, result: preview });
    expect(s.view).toMatchObject({ kind: 'complete', jobId: 2 });
  });

  it('never goes back from complete to processing', () => {
    let s = run([
      { type: 'select', jobId: 1, source: source('a.jpg') },
      { type: 'preview', jobId: 1, result: preview },
    ]);
    s = reduce(s, { type: 'stage', jobId: 1, stage: 'encoding' });
    expect(s.view.kind).toBe('complete');
  });

  it('cancels only busy jobs', () => {
    const busy = run([{ type: 'select', jobId: 1, source: source('a.jpg') }]);
    expect(reduce(busy, { type: 'cancel', jobId: 1 }).view.kind).toBe('cancelled');
    const done = run([{ type: 'preview', jobId: 1, result: preview }], busy);
    expect(reduce(done, { type: 'cancel', jobId: 1 }).view.kind).toBe('complete');
  });

  it('turns a model error into a job error while waiting for the model', () => {
    const s = run([
      { type: 'select', jobId: 1, source: source('a.jpg') },
      { type: 'stage', jobId: 1, stage: 'waiting-for-model' },
      { type: 'model-error', error: 'offline' },
    ]);
    expect(s.view).toMatchObject({ kind: 'error', error: 'offline', source: { name: 'a.jpg' } });
    expect(s.model).toEqual({ kind: 'error', error: 'offline' });
  });

  it('keeps a model error in the background when no job waits', () => {
    const s = run([{ type: 'model-error', error: 'model-download-failed' }]);
    expect(s.view.kind).toBe('idle');
  });

  it('rejects files without a job and resets', () => {
    let s = run([{ type: 'reject', error: 'unsupported-format' }]);
    expect(s.view).toEqual({ kind: 'error', jobId: null, source: null, error: 'unsupported-format' });
    s = reduce(s, { type: 'reset' });
    expect(s.view.kind).toBe('idle');
  });

  it('records a WebGPU fallback', () => {
    const s = run([{ type: 'model-ready', backend: 'webgpu', fromCache: true }, { type: 'fallback' }]);
    expect(s.fellBack).toBe(true);
    expect(s.model).toMatchObject({ kind: 'ready', backend: 'wasm' });
  });

  it('identifies busy states', () => {
    expect(isBusy({ kind: 'idle' })).toBe(false);
    expect(isBusy({ kind: 'processing', jobId: 1, source: source('a'), stage: 'inference' })).toBe(true);
  });
});

describe('model failures', () => {
  it('end a job that is still decoding', () => {
    const s = run([
      { type: 'select', jobId: 3, source: source('c.png') },
      { type: 'model-error', error: 'runtime-unsupported' },
    ]);
    expect(s.view).toMatchObject({ kind: 'error', jobId: 3, error: 'runtime-unsupported' });
  });
});
