import { readFileSync } from 'node:fs';
import { expect, test, type Request } from '@playwright/test';
import { dropFile, image, paste, readDownload, useMockModel, waitForResult } from './helpers';

test.beforeEach(async ({ context }) => {
  await useMockModel(context);
});

test('start page offers all input methods without loading the AI runtime', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', (r) => requests.push(r.url()));
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });

  await page.goto('./');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Remove image backgrounds instantly.');
  await expect(page.getByRole('button', { name: 'Choose image' })).toBeVisible();
  await expect(page.getByText('Drop an image here')).toBeVisible();
  await expect(page.getByText('Paste with')).toBeVisible();
  await expect(page.getByText('Images never leave your device')).toBeVisible();
  await expect(page.locator('.trust')).toContainText('Open source');
  await expect(page.getByRole('link', { name: 'Privacy' })).toHaveAttribute('href', 'privacy.html');

  // The model and ONNX Runtime are loaded only once an image is chosen.
  await page.waitForLoadState('networkidle');
  expect(requests.some((u) => u.includes('.wasm') || u.includes('model.onnx'))).toBe(false);
  expect(errors).toEqual([]);
});

test('file picker → result → transparent PNG download in original size', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles(image('person-astronaut.jpg'));
  await expect(page.locator('#original-image')).toBeVisible();
  await waitForResult(page);
  await expect(page.locator('#result-meta')).toHaveText('512 × 512 px · PNG');

  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#download-link').click()]);
  expect(download.suggestedFilename()).toBe('person-astronaut-background-removed.png');
  const png = readDownload(await download.path());
  expect(png.width).toBe(512);
  expect(png.height).toBe(512);
  expect(png.colorType).toBe(6); // RGBA
  expect(png.chunkTypes.filter((t) => !['IHDR', 'IDAT', 'IEND'].includes(t))).toEqual([]); // no metadata
  const alphas = new Set<number>();
  for (let i = 3; i < png.pixels.length; i += 4) alphas.add(png.pixels[i]!);
  expect(alphas.has(0) || [...alphas].some((a) => a < 128)).toBe(true);
  expect([...alphas].some((a) => a > 128)).toBe(true);
  // Fully transparent pixels carry no colour from the removed background.
  let leaked = 0;
  for (let i = 0; i < png.pixels.length; i += 4) {
    if (png.pixels[i + 3] === 0 && png.pixels[i]! + png.pixels[i + 1]! + png.pixels[i + 2]! > 0) leaked++;
  }
  expect(leaked).toBe(0);
});

test('drag and drop shows the drop overlay and processes the image', async ({ page }) => {
  await page.goto('./');
  await dropFile(page, image('animal-cat.jpg'), 'cat.jpg', 'image/jpeg');
  await waitForResult(page);
  await expect(page.locator('html')).not.toHaveClass(/dragging/);
  await expect(page.locator('#download-link')).toHaveAttribute('download', 'cat-background-removed.png');
});

test('pasting an image starts processing immediately', async ({ page }) => {
  await page.goto('./');
  await paste(page, { path: image('product-coffee.png'), name: 'image.png', type: 'image/png' });
  await waitForResult(page);
  await expect(page.locator('#result-meta')).toHaveText('600 × 400 px · PNG');
});

test('pasting text shows a short hint and keeps the start page', async ({ page }) => {
  await page.goto('./');
  await paste(page, { text: 'hello' });
  await expect(page.locator('#toast')).toHaveText('The clipboard doesn’t contain an image.');
  await expect(page.locator('html')).toHaveAttribute('data-view', 'idle');
});

test('a second image replaces the first one', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles(image('person-astronaut.jpg'));
  await page.locator('#file-input').setInputFiles(image('motion-blur-clock.jpg'));
  await waitForResult(page);
  await expect(page.locator('#result-meta')).toHaveText('400 × 300 px · PNG');
  await expect(page.locator('#download-link')).toHaveAttribute('download', 'motion-blur-clock-background-removed.png');
  // Stays on the second image (the first job's late result must not replace it).
  await page.waitForTimeout(1500);
  await expect(page.locator('#result-meta')).toHaveText('400 × 300 px · PNG');

  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#download-link').click()]);
  const png = readDownload(await download.path());
  expect([png.width, png.height]).toEqual([400, 300]);
});

test('"Remove another image" returns to the start page', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
  await waitForResult(page);
  await page.getByRole('button', { name: 'Remove another image' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-view', 'idle');
  await expect(page.getByRole('button', { name: 'Choose image' })).toBeFocused();
});

test('unsupported files get a clear error with a way out', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles({ name: 'notes.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-1.7\n...') });
  await expect(page.locator('#error-title')).toHaveText('Unsupported image format');
  await expect(page.locator('#retry-button')).toBeHidden();
  await page.getByRole('button', { name: 'Choose another image' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-view', 'idle');
});

test('background preview and download with background', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
  await waitForResult(page);
  await expect(page.locator('#download-bg-button')).toBeHidden();
  await page.locator('label.swatch-white').click();
  await expect(page.locator('#frame')).toHaveAttribute('data-bg', 'white');
  await expect(page.locator('#download-bg-button')).toBeVisible();
  const [download] = await Promise.all([page.waitForEvent('download'), page.locator('#download-bg-button').click()]);
  expect(download.suggestedFilename()).toBe('animal-cat-white-background.png');
  const png = readDownload(await download.path());
  let transparent = 0;
  for (let i = 3; i < png.pixels.length; i += 4) if (png.pixels[i] !== 255) transparent++;
  expect(transparent).toBe(0);
});

test('before/after slider is keyboard accessible', async ({ page }) => {
  await page.goto('./');
  await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
  await waitForResult(page);
  const range = page.locator('#compare-range');
  await expect.poll(async () => Number(await range.inputValue())).toBeLessThanOrEqual(1);
  await range.focus();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  expect(Number(await range.inputValue())).toBeGreaterThanOrEqual(2);
  await expect(range).toHaveAttribute('aria-valuetext', /Original \d+%/);
});

test('German browsers get the German interface', async ({ browser }) => {
  const context = await browser.newContext({ locale: 'de-DE' });
  const page = await context.newPage();
  await page.goto('./');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Bildhintergründe sofort entfernen.');
  await expect(page.getByRole('button', { name: 'Bild auswählen' })).toBeVisible();
  await expect(page.locator('html')).toHaveAttribute('lang', 'de');
  await context.close();
});

test('privacy: no image data is ever sent over the network', async ({ page, context }) => {
  const requests: Request[] = [];
  context.on('request', (r) => requests.push(r));
  await page.goto('./');
  const file = image('person-astronaut.jpg');
  await page.locator('#file-input').setInputFiles(file);
  await waitForResult(page);
  await page.locator('label.swatch-black').click();
  await Promise.all([page.waitForEvent('download'), page.locator('#download-bg-button').click()]);

  const origin = new URL(page.url()).origin;
  const imageBytes = readFileSync(file);
  for (const r of requests) {
    const url = new URL(r.url());
    // Only reads: no uploads of any kind.
    expect(r.method(), r.url()).toBe('GET');
    expect(r.postDataBuffer(), r.url()).toBeNull();
    // Only our own origin (blob: URLs are local object URLs and never leave the browser).
    expect(['blob:', 'data:'].includes(url.protocol) || url.origin === origin, r.url()).toBe(true);
    // Nothing that could smuggle image data in a query string.
    expect(r.url().length, r.url()).toBeLessThan(512);
  }
  // The image bytes appear in no request.
  const needle = imageBytes.subarray(1000, 1064).toString('base64');
  expect(requests.some((r) => r.url().includes(needle))).toBe(false);
  // Requests are only the app itself, the runtime and the model.
  const paths = requests.filter((r) => r.url().startsWith(origin)).map((r) => new URL(r.url()).pathname);
  for (const p of paths) expect(p, p).toMatch(/^\/(|index\.html|privacy\.html|manifest\.webmanifest|robots\.txt|sw\.js|icons\/.+|assets\/.+|models\/birefnet-lite\/.+)$/);
});

test('cancelling returns to the start page', async ({ page }) => {
  await page.route('**/models/birefnet-lite/manifest.json', async (route) => {
    await new Promise((r) => setTimeout(r, 1500)); // keep the job busy long enough to cancel it
    await route.fallback();
  });
  await page.goto('./');
  await page.locator('#file-input').setInputFiles(image('animal-cat.jpg'));
  await page.getByRole('button', { name: 'Cancel' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-view', 'idle');
  await expect(page.locator('#toast')).toHaveText('Cancelled.');
  // A cancelled job never shows up later.
  await page.waitForTimeout(3000);
  await expect(page.locator('html')).toHaveAttribute('data-view', 'idle');
});
