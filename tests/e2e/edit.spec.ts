import { expect, test, type Page } from '@playwright/test';
import { image, readDownload, useMockModel, waitForResult } from './helpers';

test.beforeEach(async ({ context }) => {
  await useMockModel(context);
});

/** Result size from the meta line, e.g. "1,024 × 768 px · PNG". */
async function resultSize(page: Page): Promise<{ w: number; h: number }> {
  const meta = (await page.locator('#result-meta').textContent()) ?? '';
  const [w, h] = meta.split(' px')[0]!.split('×').map((v) => Number(v.replace(/\D/g, '')));
  return { w: w!, h: h! };
}

/** Drags a crop corner by a fraction of the preview size. */
async function dragCorner(page: Page, corner: 'nw' | 'ne' | 'sw' | 'se', fx: number, fy: number) {
  const handle = (await page.locator(`.crop-handle[data-corner="${corner}"]`).boundingBox())!;
  const frame = (await page.locator('#frame').boundingBox())!;
  const x = handle.x + handle.width / 2;
  const y = handle.y + handle.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + frame.width * fx, y + frame.height * fy, { steps: 6 });
  await page.mouse.up();
}

async function downloadSize(page: Page, locator = page.locator('#download-link')) {
  const [download] = await Promise.all([page.waitForEvent('download'), locator.click()]);
  const png = readDownload(await download.path());
  return { name: download.suggestedFilename(), w: png.width, h: png.height };
}

test('cropping a finished result is instant and changes the download', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
  await waitForResult(page);
  const before = await resultSize(page);

  await page.locator('#crop-button').click();
  await expect(page.locator('#crop-layer')).toBeVisible();
  await expect(page.locator('#crop-rerun')).not.toBeChecked();
  await dragCorner(page, 'se', -0.5, -0.4);
  await page.locator('#crop-apply-button').click();
  await waitForResult(page);

  const after = await resultSize(page);
  expect(after.w).toBeGreaterThan(before.w * 0.45);
  expect(after.w).toBeLessThan(before.w * 0.55);
  expect(after.h).toBeGreaterThan(before.h * 0.55);
  expect(after.h).toBeLessThan(before.h * 0.65);
  // No second model run: the worker only cut out the part.
  const timings = JSON.parse((await page.locator('html').getAttribute('data-timings')) ?? '{}') as { inferenceMs: number };
  expect(timings.inferenceMs).toBe(0);
  expect(await downloadSize(page)).toEqual({ name: 'animal-cat-background-removed.png', ...after });

  // The whole image can be restored.
  await page.locator('#crop-button').click();
  await page.locator('#crop-all-button').click();
  await page.locator('#crop-apply-button').click();
  await waitForResult(page);
  expect(await resultSize(page)).toEqual(before);
});

test('cropping while the AI works processes only the selected part', async ({ page }) => {
  // Hold the model back so the image waits for it.
  await page.route('**/models/birefnet-lite/model.onnx.part00', async (route) => {
    await new Promise((r) => setTimeout(r, 2500));
    await route.fallback();
  });
  await page.goto('./');
  await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
  await expect(page.locator('#crop-before-button')).toBeVisible();
  await page.locator('#crop-before-button').click();
  await expect(page.locator('#crop-hint')).toBeVisible();
  await expect(page.locator('#crop-rerun-option')).toBeHidden();
  await page.locator('.segmented label', { hasText: '1:1' }).click();
  await dragCorner(page, 'nw', 0.2, 0.2);
  await page.locator('#crop-apply-button').click();
  await waitForResult(page);
  const size = await resultSize(page);
  expect(Math.abs(size.w - size.h)).toBeLessThanOrEqual(1);
  const timings = JSON.parse((await page.locator('html').getAttribute('data-timings')) ?? '{}') as { inferenceMs: number };
  expect(timings.inferenceMs).toBeGreaterThan(0);
});

test('Escape cancels cropping and brings the result back', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles(image('product-coffee.png'));
  await waitForResult(page);
  const before = await resultSize(page);
  await page.locator('#crop-button').click();
  await dragCorner(page, 'se', -0.3, -0.3);
  await page.keyboard.press('Escape');
  await expect(page.locator('#crop-layer')).toBeHidden();
  await expect(page.locator('#download-link')).toBeVisible();
  expect(await resultSize(page)).toEqual(before);
  await expect(page.locator('#compare-range')).toBeEnabled();
});

test('the crop area can be moved with the keyboard', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles(image('product-coffee.png'));
  await waitForResult(page);
  const before = await resultSize(page);
  await page.locator('#crop-button').click();
  await expect(page.locator('#crop-box')).toBeFocused();
  await page.locator('.crop-handle[data-corner="se"]').focus();
  for (let i = 0; i < 5; i++) await page.keyboard.press('Shift+ArrowLeft'); // 25 % narrower
  await page.locator('#crop-apply-button').click();
  await waitForResult(page);
  const after = await resultSize(page);
  expect(after.w).toBeLessThan(before.w * 0.8);
  expect(after.h).toBe(before.h);
});

test('recent results stay on this device and can be downloaded again', async ({ page }) => {
  await page.goto('./');
  const gallery = page.locator('.history-item');
  await expect(page.locator('#history')).toBeHidden();

  await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
  await waitForResult(page);
  await expect(gallery).toHaveCount(1);
  await page.locator('#file-input').setInputFiles(image('product-coffee.png'));
  await waitForResult(page);
  await expect(gallery).toHaveCount(2);

  // A cropped version replaces the entry of the same image.
  await page.locator('#crop-button').click();
  await dragCorner(page, 'se', -0.4, 0);
  await page.locator('#crop-apply-button').click();
  await waitForResult(page);
  const cropped = await resultSize(page);
  await expect(gallery).toHaveCount(2);

  await page.reload();
  await expect(gallery).toHaveCount(2);
  await expect(gallery.first()).toHaveAttribute('aria-label', /product-coffee-background-removed\.png/);
  expect(await downloadSize(page, gallery.first())).toEqual({
    name: 'product-coffee-background-removed.png',
    ...cropped,
  });

  await page.getByRole('button', { name: 'Clear history' }).click();
  await expect(page.locator('#history')).toBeHidden();
  await page.reload();
  await page.waitForTimeout(500);
  await expect(page.locator('#history')).toBeHidden();
});
