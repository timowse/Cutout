import { expect, test } from '@playwright/test';
import { image, readDownload, useMockModel, waitForResult } from './helpers';
import { readFileSync } from 'node:fs';

test.beforeEach(async ({ context }) => {
  await useMockModel(context);
});

function pngHeader(width: number, height: number): Buffer {
  const b = Buffer.alloc(33);
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(b, 0);
  b.writeUInt32BE(13, 8);
  b.write('IHDR', 12, 'ascii');
  b.writeUInt32BE(width, 16);
  b.writeUInt32BE(height, 20);
  b.set([8, 6, 0, 0, 0], 24);
  return b;
}

test('huge images are rejected before decoding', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles({ name: 'huge.png', mimeType: 'image/png', buffer: pngHeader(40000, 30000) });
  await expect(page.locator('#error-title')).toHaveText('Image is too large for this device');
});

test('damaged images get a clear message', async ({ page }) => {
  await page.goto('./');
  const broken = Buffer.concat([pngHeader(64, 64), Buffer.from('not really png data')]);
  await page.locator('#file-input').setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: broken });
  await expect(page.locator('#error-title')).toHaveText('This image couldn’t be opened');
});

test('HEIC photos explain the problem in browsers that cannot decode them', async ({ page }) => {
  await page.goto('./');
  const heic = Buffer.concat([Buffer.from([0, 0, 0, 24]), Buffer.from('ftypheic'), Buffer.alloc(4), Buffer.from('mif1heic'), Buffer.alloc(64)]);
  await page.locator('#file-input').setInputFiles({ name: 'IMG_0001.HEIC', mimeType: 'image/heic', buffer: heic });
  await expect(page.locator('#error-title')).toHaveText('HEIC photos can’t be opened here');
});

test('files with a wrong extension are recognised by their content', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles({
    name: 'photo.txt',
    mimeType: 'text/plain',
    buffer: readFileSync(image('animal-cat.jpg')),
  });
  await waitForResult(page);
});

test('transparent input stays transparent', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles(image('transparent-horse.png'));
  await waitForResult(page);
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#download-link').click()]);
  const out = readDownload(await download.path());

  // The fixture is an 8-bit RGBA PNG, so the test decoder can read it too.
  const src = readDownload(image('transparent-horse.png'));
  expect(src.colorType).toBe(6);
  expect([out.width, out.height]).toEqual([src.width, src.height]);
  let violations = 0;
  for (let i = 3; i < out.pixels.length; i += 4) if (out.pixels[i]! > src.pixels[i]!) violations++;
  expect(violations).toBe(0);
});
