// Quality check with the real model: processes every image in a folder through
// the built app (headless Chromium) and writes an HTML report that shows the
// original next to the result on checkerboard, white and black backgrounds.
//
//   npm run build && npx vite preview --port 4173 &
//   node tools/quality/run.mjs [images-dir] [out-dir] [--url=http://localhost:4173/] [--webgpu]
//
// Your images stay on your machine: the app runs locally and the report is a local file.
import { chromium } from '@playwright/test';
import { copyFileSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { basename, extname, join, resolve } from 'node:path';

const args = process.argv.slice(2);
const flags = Object.fromEntries(args.filter((a) => a.startsWith('--')).map((a) => a.slice(2).split('=')));
const [imagesDir = 'tests/fixtures/images', outDir = 'quality-report'] = args.filter((a) => !a.startsWith('--'));
const url = flags.url ?? 'http://localhost:4173/';
const files = readdirSync(imagesDir)
  .filter((f) => /\.(jpe?g|png|webp|avif|gif|bmp)$/i.test(f))
  .sort();
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({
  args: flags.webgpu !== undefined ? ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader'] : [],
});
const context = await browser.newContext({ serviceWorkers: 'block', acceptDownloads: true });
const page = await context.newPage();
await page.goto(url);
const rows = [];
for (const file of files) {
  const src = resolve(imagesDir, file);
  const started = Date.now();
  await page.locator('#file-input').setInputFiles(src);
  await page.waitForSelector('#download-link[aria-disabled="false"], html[data-view="error"]', { timeout: 30 * 60_000 });
  const total = Date.now() - started;
  const view = await page.locator('html').getAttribute('data-view');
  const name = basename(file, extname(file));
  copyFileSync(src, join(outDir, file));
  if (view !== 'complete') {
    rows.push({ file, error: await page.locator('#error-title').textContent() });
    console.log(file, 'ERROR');
    continue;
  }
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#download-link').click()]);
  await download.saveAs(join(outDir, `${name}.result.png`));
  const timings = JSON.parse((await page.locator('html').getAttribute('data-timings')) ?? '{}');
  const backend = await page.locator('html').getAttribute('data-model');
  const meta = await page.locator('#result-meta').textContent();
  rows.push({ file, result: `${name}.result.png`, timings, backend, meta, total });
  console.log(file, backend, meta, JSON.stringify(timings), `${total} ms total`);
  await page.getByRole('button', { name: 'Remove another image' }).click();
}
await browser.close();

const cell = (r, bg) => `<td style="background:${bg}"><img src="${r.result}" alt=""></td>`;
const checker = 'repeating-conic-gradient(#ddd 0 25%, #fff 0 50%) 0 0 / 16px 16px';
writeFileSync(
  join(outDir, 'index.html'),
  `<!doctype html><meta charset="utf-8"><title>Quality report</title>
<style>body{font:14px system-ui;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}
img{max-width:260px;max-height:260px;display:block}th{text-align:left}</style>
<h1>Background removal quality report</h1><p>${new Date().toISOString()} · ${url}</p>
<table><tr><th>Image</th><th>Original</th><th>Transparent</th><th>White</th><th>Black</th><th>Info</th></tr>
${rows
  .map((r) =>
    r.error
      ? `<tr><td>${r.file}</td><td><img src="${r.file}" alt=""></td><td colspan="4">${r.error}</td></tr>`
      : `<tr><td>${r.file}</td><td><img src="${r.file}" alt=""></td>${cell(r, checker)}${cell(r, '#fff')}${cell(r, '#000')}<td>${r.meta}<br>${r.backend}<br>inference ${r.timings.inferenceMs} ms<br>total ${r.total} ms</td></tr>`,
  )
  .join('\n')}
</table>`,
);
console.log(`Report: ${join(outDir, 'index.html')}`);
