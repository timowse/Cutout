/**
 * UI state machine. The view is always exactly one of these states, so the UI
 * never has to reason about combinations of booleans. Events that carry a
 * `jobId` are ignored unless they belong to the job currently shown; that is
 * what keeps a slow result for image A from ever appearing after image B.
 */

import type { Backend, ErrorCode, ModelPhase, ProcessStage } from '../shared/protocol';

export interface SourceImage {
  name: string;
  /** Object URL of the original file, for the instant preview. */
  url: string;
}

export interface ResultInfo {
  preview: ImageBitmap;
  width: number;
  height: number;
  originalWidth: number;
  originalHeight: number;
  png: Blob | null;
  pngUrl: string | null;
}

export type ViewState =
  | { kind: 'idle' }
  | { kind: 'loading-image'; jobId: number; source: SourceImage }
  | { kind: 'loading-model'; jobId: number; source: SourceImage }
  | { kind: 'processing'; jobId: number; source: SourceImage; stage: ProcessStage }
  | { kind: 'complete'; jobId: number; source: SourceImage; result: ResultInfo }
  | { kind: 'error'; jobId: number | null; source: SourceImage | null; error: ErrorCode }
  | { kind: 'cancelled' };

export type ModelStatus =
  | { kind: 'idle' }
  | { kind: 'loading'; phase: ModelPhase; loaded: number; total: number }
  | { kind: 'ready'; backend: Backend; fromCache: boolean }
  | { kind: 'error'; error: ErrorCode };

export interface AppState {
  view: ViewState;
  model: ModelStatus;
  /** Set when WebGPU failed and processing continues on the CPU. */
  fellBack: boolean;
}

export type AppEvent =
  | { type: 'select'; jobId: number; source: SourceImage }
  | { type: 'reject'; error: ErrorCode }
  | { type: 'stage'; jobId: number; stage: ProcessStage }
  | { type: 'preview'; jobId: number; result: Omit<ResultInfo, 'png' | 'pngUrl'> }
  | { type: 'png'; jobId: number; png: Blob; pngUrl: string }
  | { type: 'job-error'; jobId: number; error: ErrorCode }
  | { type: 'cancel'; jobId: number }
  | { type: 'reset' }
  | { type: 'model-progress'; phase: ModelPhase; loaded: number; total: number }
  | { type: 'model-ready'; backend: Backend; fromCache: boolean }
  | { type: 'model-error'; error: ErrorCode }
  | { type: 'fallback' };

export const initialState: AppState = { view: { kind: 'idle' }, model: { kind: 'idle' }, fellBack: false };

export function currentJobId(view: ViewState): number | null {
  return 'jobId' in view ? view.jobId : null;
}

function sourceOf(view: ViewState): SourceImage | null {
  return 'source' in view ? view.source : null;
}

function isCurrent(state: AppState, jobId: number): boolean {
  return currentJobId(state.view) === jobId;
}

/** True while an image is being worked on (a new image replaces it). */
export function isBusy(view: ViewState): boolean {
  return view.kind === 'loading-image' || view.kind === 'loading-model' || view.kind === 'processing';
}

export function reduce(state: AppState, event: AppEvent): AppState {
  switch (event.type) {
    case 'select':
      return { ...state, view: { kind: 'loading-image', jobId: event.jobId, source: event.source } };

    case 'reject':
      return { ...state, view: { kind: 'error', jobId: null, source: null, error: event.error } };

    case 'stage': {
      const view = state.view;
      if (!isCurrent(state, event.jobId) || !isBusy(view) || !('source' in view)) return state;
      if (event.stage === 'decoding') return { ...state, view: { kind: 'loading-image', jobId: event.jobId, source: view.source } };
      if (event.stage === 'waiting-for-model') {
        return { ...state, view: { kind: 'loading-model', jobId: event.jobId, source: view.source } };
      }
      return { ...state, view: { kind: 'processing', jobId: event.jobId, source: view.source, stage: event.stage } };
    }

    case 'preview': {
      const view = state.view;
      if (!isCurrent(state, event.jobId) || !isBusy(view) || !('source' in view)) return state;
      return {
        ...state,
        view: { kind: 'complete', jobId: event.jobId, source: view.source, result: { ...event.result, png: null, pngUrl: null } },
      };
    }

    case 'png': {
      const view = state.view;
      if (!isCurrent(state, event.jobId) || view.kind !== 'complete') return state;
      return { ...state, view: { ...view, result: { ...view.result, png: event.png, pngUrl: event.pngUrl } } };
    }

    case 'job-error':
      if (!isCurrent(state, event.jobId)) return state;
      return { ...state, view: { kind: 'error', jobId: event.jobId, source: sourceOf(state.view), error: event.error } };

    case 'cancel':
      if (!isCurrent(state, event.jobId) || !isBusy(state.view)) return state;
      return { ...state, view: { kind: 'cancelled' } };

    case 'reset':
      return { ...state, view: { kind: 'idle' } };

    case 'model-progress':
      if (state.model.kind === 'ready') return state;
      return { ...state, model: { kind: 'loading', phase: event.phase, loaded: event.loaded, total: event.total } };

    case 'model-ready':
      return { ...state, model: { kind: 'ready', backend: event.backend, fromCache: event.fromCache } };

    case 'model-error': {
      const model: ModelStatus = { kind: 'error', error: event.error };
      if (state.view.kind === 'loading-model') {
        return {
          ...state,
          model,
          view: { kind: 'error', jobId: state.view.jobId, source: state.view.source, error: event.error },
        };
      }
      return { ...state, model };
    }

    case 'fallback':
      return {
        ...state,
        fellBack: true,
        model: state.model.kind === 'ready' ? { ...state.model, backend: 'wasm' } : state.model,
      };
  }
}
