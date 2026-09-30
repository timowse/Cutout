/**
 * Recently processed results, kept only on this device (IndexedDB, never
 * uploaded): the newest MAX_ITEMS results for at most MAX_AGE_DAYS days. The
 * gallery shows small thumbnails; clicking one downloads the PNG again.
 */

export interface HistoryEntry {
  id: number;
  /** Download file name, e.g. "cat-background-removed.png". */
  name: string;
  /** Last update (ms since epoch). */
  created: number;
  width: number;
  height: number;
  png: Blob;
  thumb: Blob;
}

export type NewHistoryEntry = Omit<HistoryEntry, 'id'>;

export const MAX_ITEMS = 12;
export const MAX_AGE_DAYS = 30;
const MAX_AGE_MS = MAX_AGE_DAYS * 24 * 60 * 60 * 1000;

const DB_NAME = 'cutout';
const STORE = 'history';

/** Ids to delete: entries older than MAX_AGE_DAYS and everything beyond the newest MAX_ITEMS. */
export function expiredIds(entries: Pick<HistoryEntry, 'id' | 'created'>[], now: number): number[] {
  const newestFirst = [...entries].sort((a, b) => b.created - a.created || b.id - a.id);
  return newestFirst.filter((e, i) => i >= MAX_ITEMS || now - e.created > MAX_AGE_MS).map((e) => e.id);
}

function request<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('IndexedDB request failed'));
  });
}

function completion(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onabort = tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
  });
}

export class HistoryStore {
  private constructor(private readonly db: IDBDatabase) {}

  /** Opens the store, or returns null where IndexedDB is unavailable (e.g. some private modes). */
  static async open(): Promise<HistoryStore | null> {
    try {
      if (typeof indexedDB === 'undefined') return null;
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        if (!req.result.objectStoreNames.contains(STORE)) {
          req.result.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
        }
      };
      const store = new HistoryStore(await request(req));
      await store.prune();
      return store;
    } catch (err) {
      console.warn('[history] unavailable', err);
      return null;
    }
  }

  /** All entries, newest first. */
  async list(): Promise<HistoryEntry[]> {
    const tx = this.db.transaction(STORE, 'readonly');
    const entries = (await request(tx.objectStore(STORE).getAll())) as HistoryEntry[];
    return entries.sort((a, b) => b.created - a.created || b.id - a.id);
  }

  /** Adds an entry, or replaces entry `replaceId` (a new version of the same image). Returns its id. */
  async save(entry: NewHistoryEntry, replaceId?: number): Promise<number> {
    const tx = this.db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    const key = await request(replaceId === undefined ? store.add(entry) : store.put({ ...entry, id: replaceId }));
    await completion(tx);
    await this.prune();
    return key as number;
  }

  async clear(): Promise<void> {
    const tx = this.db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).clear();
    await completion(tx);
  }

  async prune(now = Date.now()): Promise<void> {
    const ids = expiredIds(await this.list(), now);
    if (ids.length === 0) return;
    const tx = this.db.transaction(STORE, 'readwrite');
    const store = tx.objectStore(STORE);
    for (const id of ids) store.delete(id);
    await completion(tx);
  }
}

/** A small PNG thumbnail (keeps transparency) of an image or canvas. */
export function makeThumbnail(source: CanvasImageSource & { width: number; height: number }, maxSide = 192): Promise<Blob | null> {
  const scale = Math.min(1, maxSide / Math.max(source.width, source.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(source.width * scale));
  canvas.height = Math.max(1, Math.round(source.height * scale));
  const ctx = canvas.getContext('2d');
  if (!ctx) return Promise.resolve(null);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), 'image/png'));
}
