import './styles/main.css';

import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import ortWebGPUWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import { Controller } from './app/controller';
import { detectFeatures, wasmThreads } from './app/features';
import { setupInput } from './app/input';
import { registerServiceWorker } from './app/pwa';
import { initTheme } from './app/theme';
import { View } from './app/view';
import { InferenceClient } from './app/worker-client';
import { APP_NAME, LICENSE_URL, MODEL_BASE_URL, REPO_URL } from './config';
import { detectLocale, setLocale, translateDocument } from './i18n';
import { computeLimits } from './shared/dimensions';

const params = new URLSearchParams(location.search);
setLocale(detectLocale(navigator.languages.length > 0 ? navigator.languages : [navigator.language], location.search));
translateDocument();
initTheme(document.getElementById('theme-toggle'));

for (const el of document.querySelectorAll<HTMLElement>('[data-app-name]')) el.textContent = APP_NAME;
for (const el of document.querySelectorAll<HTMLAnchorElement>('[data-repo-link]')) el.href = REPO_URL;
for (const el of document.querySelectorAll<HTMLAnchorElement>('[data-license-link]')) el.href = LICENSE_URL;

const features = detectFeatures();
const pasteKey = document.querySelector<HTMLElement>('[data-paste-modifier]');
if (pasteKey) pasteKey.textContent = features.isApple ? '⌘' : 'Ctrl';

const modelBaseUrl = new URL(MODEL_BASE_URL, document.baseURI).href;
const client = new InferenceClient({
  baseUrl: modelBaseUrl,
  // `?cpu` forces the WebAssembly backend (useful for testing and for GPUs with driver problems).
  preferWebGPU: features.webgpu && !params.has('cpu'),
  wasmUrls: {
    webgpu: new URL(ortWebGPUWasmUrl, location.href).href,
    wasm: new URL(ortWasmUrl, location.href).href,
  },
  threads: wasmThreads(features),
});

const limits = computeLimits(features);
const previewMaxSide = Math.min(
  4096,
  Math.ceil(Math.max(screen.width, screen.height, 1280) * Math.min(window.devicePixelRatio || 1, 2)),
);
const controller = new Controller(client, features, limits, previewMaxSide);

const input = setupInput(document.getElementById('file-input') as HTMLInputElement, {
  file: (file) => void controller.handleFile(file),
  intent: () => controller.warmUp(),
  toast: (text) => view.showToast(text),
});

const view = new View(
  {
    choose: () => input.openPicker(),
    cancel: () => controller.cancel(),
    startOver: () => controller.startOver(),
    retry: () => controller.retry(),
    retryModel: () => controller.warmUp(),
    copy: () => void controller.copy(),
    downloadWithBackground: (color, background) => controller.downloadWithBackground(color, background),
  },
  features,
);
controller.attach(view);

document.documentElement.dataset.ready = 'true';

// A first visit may reload once when the service worker takes over (for
// multi-threaded WebAssembly); never interrupt a model download for that.
const mayReloadForIsolation = () => {
  const state = controller.getState();
  return state.view.kind === 'idle' && state.model.kind === 'idle' && !features.webgpu;
};
registerServiceWorker(mayReloadForIsolation);

// Download and prepare the AI model in the background as soon as the page is
// open, so the first image is processed right away. Skipped when the browser
// asks to save data; the model then loads with the first image.
const saveData = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData === true;
if (!saveData && !params.has('nopreload')) {
  const firstVisit = import.meta.env.PROD && 'serviceWorker' in navigator && !navigator.serviceWorker.controller;
  const delay = firstVisit && !crossOriginIsolated && !features.webgpu ? 2500 : 300;
  const start = () => window.setTimeout(() => controller.warmUp(), delay);
  if (document.readyState === 'complete') start();
  else window.addEventListener('load', start, { once: true });
}
