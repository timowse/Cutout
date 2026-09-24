import { createHash } from 'node:crypto';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AppError } from './errors';
import { MODEL_CACHE_PREFIX, loadModel } from './model-loader';
import type { ModelManifest } from './protocol';

const BASE = 'https://example.test/models/m/';

const urlOf = (req: RequestInfo | URL): string => (typeof req === 'string' ? req : req instanceof URL ? req.href : req.url);

class MemoryCache {
  readonly entries = new Map<string, Uint8Array>();
  match(req: RequestInfo | URL): Promise<Response | undefined> {
    const body = this.entries.get(urlOf(req));
    return Promise.resolve(body ? new Response(body.slice()) : undefined);
  }
  async put(req: RequestInfo | URL, res: Response): Promise<void> {
    this.entries.set(urlOf(req), new Uint8Array(await res.arrayBuffer()));
  }
  keys(): Promise<Request[]> {
    return Promise.resolve([...this.entries.keys()].map((u) => new Request(u)));
  }
}

class MemoryCacheStorage {
  readonly stores = new Map<string, MemoryCache>();
  open(name: string): Promise<MemoryCache> {
    let c = this.stores.get(name);
    if (!c) this.stores.set(name, (c = new MemoryCache()));
    return Promise.resolve(c);
  }
  keys(): Promise<string[]> {
    return Promise.resolve([...this.stores.keys()]);
  }
  delete(name: string): Promise<boolean> {
    return Promise.resolve(this.stores.delete(name));
  }
}

const sha = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');

function makeModel(): { manifest: ModelManifest; parts: Uint8Array[] } {
  const parts = [new Uint8Array(1000).map((_, i) => i & 255), new Uint8Array(500).fill(7)];
  const all = new Uint8Array(1500);
  all.set(parts[0]!, 0);
  all.set(parts[1]!, 1000);
  const manifest: ModelManifest = {
    id: 'test-1',
    version: '1',
    name: 'test',
    input: { name: 'in', width: 4, height: 4, mean: [0, 0, 0], std: [1, 1, 1] },
    output: { name: 'out', width: 4, height: 4 },
    weights: 'fp16',
    size: all.length,
    sha256: sha(all),
    parts: parts.map((p, i) => ({ file: `part${i}`, size: p.length, sha256: sha(p) })),
    source: { url: 'x', sha256: 'y', license: 'MIT', homepage: 'z' },
  };
  return { manifest, parts };
}

let storage: MemoryCacheStorage;
let server: Map<string, Uint8Array | string>;
let online: boolean;
const fetchMock = vi.fn((input: RequestInfo | URL): Promise<Response> => {
  if (!online) return Promise.reject(new TypeError('Failed to fetch'));
  const body = server.get(urlOf(input));
  if (body === undefined) return Promise.resolve(new Response('not found', { status: 404 }));
  return Promise.resolve(new Response(typeof body === 'string' ? body : body.slice()));
});

beforeEach(() => {
  storage = new MemoryCacheStorage();
  server = new Map();
  online = true;
  fetchMock.mockClear();
  vi.stubGlobal('caches', storage);
  vi.stubGlobal('fetch', fetchMock);
  const { manifest, parts } = makeModel();
  server.set(BASE + 'manifest.json', JSON.stringify(manifest));
  parts.forEach((p, i) => server.set(`${BASE}part${i}`, p));
});

afterEach(() => vi.unstubAllGlobals());

describe('loadModel', () => {
  it('downloads, verifies and caches all parts with real progress', async () => {
    const progress: [string, number, number][] = [];
    const first = await loadModel(BASE, (phase, loaded, total) => progress.push([phase, loaded, total]));
    expect(first.fromCache).toBe(false);
    expect(first.bytes.length).toBe(1500);
    expect(first.bytes[999]).toBe(999 & 255);
    expect(first.bytes[1000]).toBe(7);
    const downloads = progress.filter(([p]) => p === 'download');
    expect(downloads.at(-1)).toEqual(['download', 1500, 1500]);
    for (let i = 1; i < downloads.length; i++) expect(downloads[i]![1]).toBeGreaterThanOrEqual(downloads[i - 1]![1]);

    fetchMock.mockClear();
    const second = await loadModel(BASE, () => undefined);
    expect(second.fromCache).toBe(true);
    expect(Array.from(second.bytes)).toEqual(Array.from(first.bytes));
    const partFetches = fetchMock.mock.calls.filter(([u]) => urlOf(u).includes('part'));
    expect(partFetches).toHaveLength(0);
  });

  it('rejects corrupted downloads', async () => {
    server.set(`${BASE}part1`, new Uint8Array(500).fill(8));
    await expect(loadModel(BASE, () => undefined)).rejects.toMatchObject({ code: 'model-integrity' });
  });

  it('works offline from the cache, including the manifest', async () => {
    await loadModel(BASE, () => undefined);
    online = false;
    const offline = await loadModel(BASE, () => undefined);
    expect(offline.fromCache).toBe(true);
  });

  it('reports a download failure when offline without a cached model', async () => {
    online = false;
    const err: unknown = await loadModel(BASE, () => undefined).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(AppError);
    expect(['offline', 'model-download-failed']).toContain((err as AppError).code);
  });

  it('removes caches of older model versions', async () => {
    await storage.open(`${MODEL_CACHE_PREFIX}old-version`);
    await storage.open('unrelated-cache');
    await loadModel(BASE, () => undefined);
    const names = await storage.keys();
    expect(names.some((n) => n.includes('old-version'))).toBe(false);
    expect(names).toContain('unrelated-cache');
  });
});
