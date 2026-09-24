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
import { detectLocale, formatMB, setLocale, t, translateDocument } from './i18n';
import { computeLimits } from './shared/dimensions';
import { isModelCached } from './shared/model-loader';
import { isModelManifest } from './shared/protocol';

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
    copy: () => void controller.copy(),
    downloadWithBackground: (color, background) => controller.downloadWithBackground(color, background),
  },
  features,
);
controller.attach(view);

// Tell first-time visitors about the one-time model download (and returning ones that it's cached).
void (async () => {
  if (await isModelCached(modelBaseUrl)) {
    view.setModelNote(t('model.cached'));
    return;
  }
  try {
    const res = await fetch(new URL('manifest.json', modelBaseUrl), { cache: 'no-cache' });
    const manifest: unknown = res.ok ? await res.json() : null;
    if (isModelManifest(manifest)) view.setModelNote(t('model.firstUse', { size: formatMB(manifest.size) }));
  } catch {
    /* offline and not cached: the error will explain when an image is chosen */
  }
})();

document.documentElement.dataset.ready = 'true';
registerServiceWorker(() => controller.getState().view.kind === 'idle' && !features.webgpu);
