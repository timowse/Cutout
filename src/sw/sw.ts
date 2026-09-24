/**
 * Service worker:
 * - makes the app installable and usable offline (app shell + runtime assets),
 * - adds COOP/COEP headers so the page is cross-origin isolated, which
 *   GitHub Pages cannot configure but multi-threaded WebAssembly needs.
 *
 * The AI model is not handled here: the model loader keeps it in its own,
 * versioned Cache Storage entry. Images never pass through this worker; they
 * are never requested from or sent to any server.
 */

export {};

declare const self: ServiceWorkerGlobalScope;

// Both placeholders are replaced by the build (see vite.config.ts).
const PRECACHE: string[] = (self as unknown as { __PRECACHE_PLACEHOLDER__: string[] }).__PRECACHE_PLACEHOLDER__;
const BUILD_ID: string = (self as unknown as { __BUILD_ID_PLACEHOLDER__: string }).__BUILD_ID_PLACEHOLDER__;

const SHELL_CACHE = `app-shell-${BUILD_ID}`;
const ASSET_CACHE = `app-assets-${BUILD_ID}`;
const MODEL_CACHE_PREFIX = 'bgremove-model-';

const scopeUrl = new URL(self.registration.scope);

function withIsolationHeaders(response: Response): Response {
  if (response.type === 'opaque' || response.status === 0) return response;
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

self.addEventListener('install', (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(SHELL_CACHE);
      await cache.addAll(PRECACHE.map((path) => new URL(path, scopeUrl).href));
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const keep = new Set([SHELL_CACHE, ASSET_CACHE]);
      for (const name of await caches.keys()) {
        if (!keep.has(name) && !name.startsWith(MODEL_CACHE_PREFIX)) await caches.delete(name);
      }
      await self.clients.claim();
    })(),
  );
});

async function handleNavigation(request: Request): Promise<Response> {
  const cache = await caches.open(SHELL_CACHE);
  try {
    const response = await fetch(request);
    if (response.ok) await cache.put(request, response.clone());
    return withIsolationHeaders(response);
  } catch {
    const cached =
      (await cache.match(request, { ignoreSearch: true })) ??
      (await cache.match(new URL('./', scopeUrl).href)) ??
      (await cache.match(new URL('index.html', scopeUrl).href));
    if (cached) return withIsolationHeaders(cached);
    return new Response('Offline', { status: 503, headers: { 'Content-Type': 'text/plain' } });
  }
}

/** Fingerprinted build assets never change: cache first. */
async function handleAsset(request: Request): Promise<Response> {
  const cached = (await caches.match(request, { cacheName: SHELL_CACHE })) ?? (await caches.match(request, { cacheName: ASSET_CACHE }));
  if (cached) return withIsolationHeaders(cached);
  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(ASSET_CACHE);
    await cache.put(request, response.clone());
  }
  return withIsolationHeaders(response);
}

/** Other static files (icons, manifest): network first, cache as fallback. */
async function handleStatic(request: Request): Promise<Response> {
  const cache = await caches.open(SHELL_CACHE);
  try {
    const response = await fetch(request);
    if (response.ok) await cache.put(request, response.clone());
    return withIsolationHeaders(response);
  } catch {
    const cached = await cache.match(request, { ignoreSearch: true });
    if (cached) return withIsolationHeaders(cached);
    throw new Error('offline');
  }
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== scopeUrl.origin || !url.pathname.startsWith(scopeUrl.pathname)) return;
  const path = url.pathname.slice(scopeUrl.pathname.length);
  // Model chunks are streamed straight from the network; the app caches them itself.
  if (path.startsWith('models/')) return;
  if (request.mode === 'navigate') {
    event.respondWith(handleNavigation(request));
  } else if (path.startsWith('assets/')) {
    event.respondWith(handleAsset(request));
  } else {
    event.respondWith(handleStatic(request));
  }
});
