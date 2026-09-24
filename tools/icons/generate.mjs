// Renders the PNG app icons from tools/icons/icon.svg with the preinstalled Chromium.
// Usage: node tools/icons/generate.mjs
import { chromium } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const out = join(here, '../../public/icons');
const template = readFileSync(join(here, 'icon.svg'), 'utf8');

// name, size, corner radius (in 512 units), circle radius, full-bleed?
const icons = [
  ['icon-192.png', 192, 112, 150],
  ['icon-512.png', 512, 112, 150],
  // Maskable: no rounded corners (the OS masks it) and the subject inside the safe zone.
  ['maskable-512.png', 512, 0, 118],
  ['apple-touch-icon.png', 180, 0, 150],
  ['favicon-32.png', 32, 128, 150],
];

const browser = await chromium.launch();
const page = await browser.newPage();
for (const [name, size, radius, circle] of icons) {
  const svg = template.replace('__RADIUS__', String(radius)).replace('__CIRCLE__', String(circle));
  await page.setViewportSize({ width: size, height: size });
  await page.setContent(
    `<html><body style="margin:0;background:transparent"><img style="width:${size}px;height:${size}px;display:block" src="data:image/svg+xml;base64,${Buffer.from(svg).toString('base64')}"></body></html>`,
  );
  const png = await page.screenshot({ omitBackground: true, clip: { x: 0, y: 0, width: size, height: size } });
  writeFileSync(join(out, name), png);
  console.log('wrote', name);
}
await browser.close();
