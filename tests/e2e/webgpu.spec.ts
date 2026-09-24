import { expect, test } from '@playwright/test';
import { image, readDownload, useMockModel, waitForResult } from './helpers';

/** Chromium flags that provide a (software) WebGPU adapter in headless mode. */
test.use({ launchOptions: { args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader'] } });

test('uses WebGPU when available', async ({ page, context }) => {
  await useMockModel(context);
  await page.goto('./');
  test.skip(!(await page.evaluate(async () => 'gpu' in navigator && !!(await navigator.gpu.requestAdapter()))), 'no WebGPU adapter');
  await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
  await waitForResult(page);
  await expect(page.locator('html')).toHaveAttribute('data-model', 'webgpu');
});

test('switches to WebAssembly when WebGPU fails at run time', async ({ page, context }) => {
  await useMockModel(context, 'gpu-fail');
  await page.goto('./');
  test.skip(!(await page.evaluate(async () => 'gpu' in navigator && !!(await navigator.gpu.requestAdapter()))), 'no WebGPU adapter');
  const maxBuffers = await page.evaluate(async () => (await navigator.gpu.requestAdapter())?.limits.maxStorageBuffersPerShaderStage ?? 0);
  test.skip(maxBuffers > 12, 'adapter allows the wide Concat; cannot provoke a failure');
  const warnings: string[] = [];
  page.on('console', (m) => warnings.push(m.text()));
  await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
  await waitForResult(page);
  await expect(page.locator('html')).toHaveAttribute('data-model', 'wasm');
  expect(warnings.some((w) => w.includes('falling back to wasm'))).toBe(true);
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#download-link').click()]);
  expect(readDownload(await download.path()).width).toBe(451);
});
