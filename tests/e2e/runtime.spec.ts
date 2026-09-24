import { expect, test } from '@playwright/test';
import { image, useMockModel, waitForResult } from './helpers';

test.describe('without WebGPU', () => {
  test('falls back to WebAssembly and still works', async ({ page, context }) => {
    await useMockModel(context);
    await page.goto('./');
    expect(await page.evaluate(async () => !('gpu' in navigator) || !(await navigator.gpu.requestAdapter()))).toBe(true);
    await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
    await waitForResult(page);
    await expect(page.locator('html')).toHaveAttribute('data-model', 'wasm');
  });

  test('?cpu forces WebAssembly', async ({ page, context }) => {
    await useMockModel(context);
    await page.goto('./?cpu');
    await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
    await waitForResult(page);
    await expect(page.locator('html')).toHaveAttribute('data-model', 'wasm');
  });
});

test('the model is downloaded once and then loaded from the cache', async ({ page, context }) => {
  await useMockModel(context);
  const modelRequests: string[] = [];
  context.on('request', (r) => {
    if (r.url().includes('model.onnx')) modelRequests.push(r.url());
  });
  await page.goto('./');
  await expect(page.locator('#model-chip')).toHaveAttribute('data-state', 'ready');
  await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
  await waitForResult(page);
  expect(modelRequests).toHaveLength(1);

  await page.reload();
  await expect(page.locator('#model-chip')).toHaveAttribute('data-state', 'ready');
  expect(modelRequests).toHaveLength(1); // loaded from Cache Storage
  const statuses: string[] = [];
  await page.exposeFunction('recordStatus', (s: string) => statuses.push(s));
  await page.evaluate(() => {
    const el = document.getElementById('status-text');
    if (!el) return;
    const record = (window as unknown as { recordStatus: (s: string) => void }).recordStatus;
    new MutationObserver(() => {
      record(el.textContent);
    }).observe(el, {
      childList: true,
      characterData: true,
      subtree: true,
    });
  });
  await page.locator('#file-input').setInputFiles(image('product-coffee.png'));
  await waitForResult(page);
  expect(modelRequests).toHaveLength(1); // no second download
  expect(statuses.some((s) => s.startsWith('Downloading AI model'))).toBe(false);
});

test.describe('offline', () => {
  test.use({ serviceWorkers: 'allow' });

  test('works offline once the app and model are cached', async ({ page, context }) => {
    await useMockModel(context);
    await page.goto('./');
    // Wait until the service worker controls the page (it may reload once for cross-origin isolation).
    await page.waitForFunction(() => navigator.serviceWorker.controller !== null, undefined, { timeout: 30_000 });
    await page.waitForLoadState('load');
    await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
    await waitForResult(page);

    await context.setOffline(true);
    await page.reload();
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    await page.locator('#file-input').setInputFiles(image('product-coffee.png'));
    await waitForResult(page);
    await expect(page.locator('#result-meta')).toHaveText('600 × 400 px · PNG');
    await context.setOffline(false);
  });
});
