import { createHash } from 'node:crypto';
import { readdirSync, statSync } from 'node:fs';
import { join, relative, resolve, sep } from 'node:path';
import { defineConfig, type Plugin } from 'vite';

const root = import.meta.dirname;

/** Headers that make the page cross-origin isolated (needed for multi-threaded WebAssembly). */
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

function listFiles(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...listFiles(full));
    else out.push(full);
  }
  return out;
}

/**
 * Injects the precache list and a build id into the service worker. The
 * ONNX Runtime WebAssembly binaries (14–27 MB) are not precached: they are
 * cached on first use, so visitors who never process an image don't pay for them.
 */
function serviceWorker(): Plugin {
  return {
    name: 'app-service-worker',
    apply: 'build',
    // Run after Vite's HTML plugin so the HTML pages are part of the bundle.
    enforce: 'post',
    generateBundle(_options, bundle) {
      const publicDir = resolve(root, 'public');
      const publicFiles = listFiles(publicDir)
        .map((f) => relative(publicDir, f).split(sep).join('/'))
        .filter((f) => !f.startsWith('models/') && f !== '.nojekyll');
      const built = Object.keys(bundle).filter((f) => f !== 'sw.js' && !f.endsWith('.map') && !f.endsWith('.wasm'));
      const precache = ['./', ...new Set([...built, ...publicFiles])].sort();
      const buildId = createHash('sha256').update(precache.join('\n')).digest('hex').slice(0, 12);
      const sw = bundle['sw.js'];
      if (!sw || sw.type !== 'chunk') throw new Error('Service worker chunk missing');
      const before = sw.code;
      sw.code = sw.code
        .replace(/self\.__PRECACHE_PLACEHOLDER__/g, JSON.stringify(precache))
        .replace(/self\.__BUILD_ID_PLACEHOLDER__/g, JSON.stringify(buildId));
      if (sw.code === before) throw new Error('Service worker placeholders not found');
    },
  };
}

export default defineConfig({
  // Relative base: works under https://user.github.io/repo/ and on a custom domain alike.
  base: './',
  build: {
    target: 'es2022',
    sourcemap: false,
    reportCompressedSize: true,
    rolldownOptions: {
      input: {
        main: resolve(root, 'index.html'),
        privacy: resolve(root, 'privacy.html'),
        sw: resolve(root, 'src/sw/sw.ts'),
      },
      output: {
        entryFileNames: (chunk) => (chunk.name === 'sw' ? 'sw.js' : 'assets/[name]-[hash].js'),
      },
    },
  },
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    // ONNX Runtime locates its WebAssembly files relative to its own module.
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: isolationHeaders,
  },
  preview: {
    headers: isolationHeaders,
  },
  plugins: [serviceWorker()],
});
