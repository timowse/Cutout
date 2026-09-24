/**
 * Downloads the model in chunks with real progress, verifies every chunk's
 * SHA-256, and keeps it in Cache Storage so later visits (and offline use)
 * load it from disk. Caches of older model versions are deleted.
 */

import { AppError } from './errors';
import { isModelManifest, type ModelManifest, type ModelPhase } from './protocol';

export const MODEL_CACHE_PREFIX = 'bgremove-model-';
const MANIFEST_FILE = 'manifest.json';
const RETRIES = 2;

export type ProgressCallback = (phase: ModelPhase, loaded: number, total: number) => void;

export interface LoadedModel {
  bytes: Uint8Array<ArrayBuffer>;
  manifest: ModelManifest;
  fromCache: boolean;
}

export function cacheNameFor(manifest: ModelManifest): string {
  return `${MODEL_CACHE_PREFIX}${manifest.id}-${manifest.sha256.slice(0, 16)}`;
}

async function openCache(name: string): Promise<Cache | null> {
  try {
    return typeof caches === 'undefined' ? null : await caches.open(name);
  } catch {
    // Cache Storage can be unavailable (e.g. some private browsing modes).
    return null;
  }
}

async function cachedManifest(): Promise<ModelManifest | null> {
  if (typeof caches === 'undefined') return null;
  try {
    for (const name of await caches.keys()) {
      if (!name.startsWith(MODEL_CACHE_PREFIX)) continue;
      const cache = await caches.open(name);
      for (const req of await cache.keys()) {
        if (!req.url.endsWith('/' + MANIFEST_FILE)) continue;
        const res = await cache.match(req);
        const json: unknown = await res?.json();
        if (isModelManifest(json)) return json;
      }
    }
  } catch {
    /* fall through */
  }
  return null;
}

async function fetchManifest(baseUrl: string): Promise<ModelManifest> {
  const url = new URL(MANIFEST_FILE, baseUrl).href;
  try {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json: unknown = await res.json();
    if (!isModelManifest(json)) throw new AppError('model-download-failed', 'Invalid model manifest');
    return json;
  } catch (err) {
    // Offline: use the manifest stored with a previously downloaded model.
    const cached = await cachedManifest();
    if (cached) return cached;
    if (err instanceof AppError) throw err;
    const offline = typeof navigator !== 'undefined' && !navigator.onLine;
    throw new AppError(offline ? 'offline' : 'model-download-failed', `Manifest: ${String(err)}`);
  }
}

export async function sha256Hex(data: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', data));
  let hex = '';
  for (const b of digest) hex += b.toString(16).padStart(2, '0');
  return hex;
}

async function download(url: string, size: number, onBytes: (n: number) => void): Promise<Uint8Array<ArrayBuffer>> {
  // The chunks are stored in Cache Storage, so keep them out of the HTTP cache.
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} for ${url}`);
  const buf = new Uint8Array(size);
  const reader = res.body.getReader();
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (n + value.length > size) {
      await reader.cancel();
      throw new AppError('model-integrity', `Chunk larger than expected: ${url}`);
    }
    buf.set(value, n);
    n += value.length;
    onBytes(n);
  }
  if (n !== size) throw new AppError('model-integrity', `Chunk size ${n} != ${size}: ${url}`);
  return buf;
}

async function deleteOtherCaches(keep: string): Promise<void> {
  if (typeof caches === 'undefined') return;
  try {
    const names = await caches.keys();
    await Promise.all(names.filter((n) => n.startsWith(MODEL_CACHE_PREFIX) && n !== keep).map((n) => caches.delete(n)));
  } catch {
    /* not critical */
  }
}

/** True if the model of the given manifest is completely present in Cache Storage. */
export async function isModelCached(baseUrl: string): Promise<boolean> {
  try {
    const res = await fetch(new URL(MANIFEST_FILE, baseUrl).href, { cache: 'no-cache' });
    const manifest: unknown = res.ok ? await res.json() : await cachedManifest();
    if (!isModelManifest(manifest) || typeof caches === 'undefined') return false;
    const cache = await caches.open(cacheNameFor(manifest));
    for (const part of manifest.parts) {
      if (!(await cache.match(new URL(part.file, baseUrl).href))) return false;
    }
    return true;
  } catch {
    return false;
  }
}

export async function loadModel(baseUrl: string, onProgress: ProgressCallback): Promise<LoadedModel> {
  onProgress('manifest', 0, 0);
  const manifest = await fetchManifest(baseUrl);
  const total = manifest.size;
  const cacheName = cacheNameFor(manifest);
  const cache = await openCache(cacheName);
  const bytes = new Uint8Array(total);
  let offset = 0;
  let fromCache = true;

  for (const part of manifest.parts) {
    const url = new URL(part.file, baseUrl).href;
    let data: Uint8Array<ArrayBuffer> | null = null;
    if (cache) {
      try {
        const hit = await cache.match(url);
        if (hit) {
          const buf = new Uint8Array(await hit.arrayBuffer());
          if (buf.length === part.size) data = buf;
        }
      } catch {
        data = null;
      }
    }
    if (data) {
      onProgress('cache', offset + part.size, total);
    } else {
      fromCache = false;
      let lastError: unknown = null;
      for (let attempt = 0; attempt <= RETRIES && !data; attempt++) {
        try {
          const start = offset;
          const buf = await download(url, part.size, (n) => onProgress('download', start + n, total));
          if ((await sha256Hex(buf)) !== part.sha256) {
            throw new AppError('model-integrity', `SHA-256 mismatch for ${part.file}`);
          }
          data = buf;
        } catch (err) {
          lastError = err;
        }
      }
      if (!data) {
        if (lastError instanceof AppError) throw lastError;
        const offline = typeof navigator !== 'undefined' && !navigator.onLine;
        throw new AppError(offline ? 'offline' : 'model-download-failed', String(lastError));
      }
      if (cache) {
        try {
          await cache.put(url, new Response(data, { headers: { 'Content-Type': 'application/octet-stream' } }));
        } catch {
          /* quota exceeded: keep working without persistent cache */
        }
      }
    }
    bytes.set(data, offset);
    offset += part.size;
  }

  if (cache) {
    try {
      const manifestUrl = new URL(MANIFEST_FILE, baseUrl).href;
      await cache.put(manifestUrl, new Response(JSON.stringify(manifest), { headers: { 'Content-Type': 'application/json' } }));
    } catch {
      /* not critical */
    }
    await deleteOtherCaches(cacheName);
  }
  return { bytes, manifest, fromCache };
}
