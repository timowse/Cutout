/** Service worker registration (production builds only). */

const RELOAD_FLAG = 'coi-reloaded';

/**
 * Registers the service worker. On the very first visit the page is not yet
 * cross-origin isolated (the headers come from the service worker), so
 * WebAssembly would run single-threaded. If `canReload()` allows it (no image
 * open, no WebGPU), the page reloads once as soon as the worker controls it.
 */
export function registerServiceWorker(canReload: () => boolean): void {
  if (!import.meta.env.PROD || !('serviceWorker' in navigator)) return;
  const register = async () => {
    try {
      const hadController = !!navigator.serviceWorker.controller;
      await navigator.serviceWorker.register('sw.js', { scope: './' });
      if (hadController || window.crossOriginIsolated) return;
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        if (!canReload()) return;
        try {
          if (sessionStorage.getItem(RELOAD_FLAG) === '1') return;
          sessionStorage.setItem(RELOAD_FLAG, '1');
        } catch {
          return; // without sessionStorage we cannot guard against reload loops
        }
        window.location.reload();
      });
    } catch (err) {
      console.warn('[pwa] service worker registration failed', err);
    }
  };
  if (document.readyState === 'complete') void register();
  else window.addEventListener('load', () => void register(), { once: true });
}
