import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from '@playwright/test';
import { image, readDownload, waitForResult } from './helpers';

/**
 * Runs the real BiRefNet model (WebAssembly in headless Chromium, ~20–60 s).
 * Requires the converted model in public/models (npm run model) and a build.
 * Enabled with RUN_REAL_MODEL=1 (the CI does this).
 */
const modelBuilt = existsSync(resolve(import.meta.dirname, '../../dist/models/birefnet-lite/manifest.json'));
test.skip(process.env.RUN_REAL_MODEL !== '1' || !modelBuilt, 'set RUN_REAL_MODEL=1 and build the model first');
test.setTimeout(600_000);

test('real model separates the person from the background', async ({ page }) => {
  await page.goto('./');
  // The model starts downloading as soon as the page is open.
  await expect(page.locator('#model-chip')).toBeVisible({ timeout: 10_000 });
  await page.locator('#file-input').setInputFiles(image('person-astronaut.jpg'));
  await waitForResult(page, 540_000);
  await expect(page.locator('#model-chip')).toHaveAttribute('data-state', 'ready');

  const timings = JSON.parse((await page.locator('html').getAttribute('data-timings')) ?? '{}') as Record<string, number>;
  console.log('backend', await page.locator('html').getAttribute('data-model'), 'timings', timings);

  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#download-link').click()]);
  const png = readDownload(await download.path());
  expect([png.width, png.height]).toEqual([512, 512]);
  const alphaAt = (x: number, y: number) => png.pixels[(y * png.width + x) * 4 + 3]!;
  const mean = (x0: number, y0: number, x1: number, y1: number) => {
    let sum = 0;
    let n = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++, n++) sum += alphaAt(x, y);
    return sum / n;
  };
  expect(mean(215, 95, 265, 150)).toBeGreaterThan(235); // face
  expect(mean(150, 300, 250, 400)).toBeGreaterThan(235); // spacesuit
  expect(mean(0, 0, 60, 60)).toBeLessThan(10); // flag in the background
  expect(mean(440, 20, 500, 80)).toBeLessThan(10); // wall in the background
  // Soft edges exist (not a hard binary mask).
  let soft = 0;
  for (let i = 3; i < png.pixels.length; i += 4) if (png.pixels[i]! > 10 && png.pixels[i]! < 245) soft++;
  expect(soft).toBeGreaterThan(200);
});
