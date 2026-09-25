/**
 * Notices when the browser ended the page while the AI was working. iOS
 * Safari kills a tab that uses too much memory without any event and then
 * reloads it; if the model preload started again right away, the page would
 * crash in a loop.
 *
 * While the model loads or an image is processed, a small marker (the stage
 * and a timestamp, never image data) is kept in localStorage. It is removed
 * when the work ends or the page is left normally (`pagehide`), so a marker
 * found at start-up means that the previous visit ended abruptly.
 */

export type BusyStage = 'model' | 'image';

export interface Busy {
  stage: BusyStage;
  /** 'webgpu' or 'wasm' once known. */
  backend?: string;
}

export interface CrashInfo extends Busy {
  at: number;
}

const KEY = 'cutout.busy';
/** Older markers are ignored (e.g. a tab that was closed while busy long ago). */
const MAX_AGE_MS = 6 * 60 * 60 * 1000;

type Store = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

function defaultStore(): Store | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export class CrashGuard {
  private written: string | null = null;

  constructor(
    private readonly store: Store | null = defaultStore(),
    private readonly now: () => number = Date.now,
  ) {}

  /** Returns and removes the marker of a previous visit that ended while busy. */
  takePrevious(): CrashInfo | null {
    try {
      const raw = this.store?.getItem(KEY);
      if (!raw) return null;
      this.store?.removeItem(KEY);
      const info = JSON.parse(raw) as Partial<CrashInfo>;
      if (info.stage !== 'model' && info.stage !== 'image') return null;
      if (typeof info.at !== 'number' || this.now() - info.at > MAX_AGE_MS || info.at > this.now()) return null;
      return { stage: info.stage, at: info.at, ...(typeof info.backend === 'string' ? { backend: info.backend } : {}) };
    } catch {
      return null;
    }
  }

  /** Records the current activity; `null` when nothing heavy is running. */
  update(busy: Busy | null): void {
    const key = busy ? `${busy.stage}:${busy.backend ?? ''}` : null;
    if (key === this.written) return;
    this.written = key;
    try {
      if (busy) this.store?.setItem(KEY, JSON.stringify({ ...busy, at: this.now() }));
      else this.store?.removeItem(KEY);
    } catch {
      /* storage full or blocked: the guard is best effort */
    }
  }
}
