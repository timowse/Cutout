import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { inflateSync } from 'node:zlib';
import { expect, type BrowserContext, type Page } from '@playwright/test';

export const IMAGES = resolve(import.meta.dirname, '../fixtures/images');
export const image = (name: string): string => resolve(IMAGES, name);

/** Serves a stand-in model instead of the real one (see tools/testing/make_mock_models.py). */
export async function useMockModel(context: BrowserContext, variant: 'default' | 'gpu-fail' = 'default'): Promise<void> {
  const dir = resolve(import.meta.dirname, '../fixtures/mock-model', variant);
  await context.route('**/models/birefnet-lite/**', async (route) => {
    const file = new URL(route.request().url()).pathname.split('/').pop() ?? '';
    await route.fulfill({ path: resolve(dir, file) });
  });
}

export async function waitForResult(page: Page, timeout = 90_000): Promise<void> {
  await expect(page.locator('html')).toHaveAttribute('data-view', 'complete', { timeout });
  await expect(page.locator('#download-link')).toHaveAttribute('aria-disabled', 'false', { timeout });
}

export interface DecodedPng {
  width: number;
  height: number;
  colorType: number;
  chunkTypes: string[];
  /** RGBA pixels (only for 8-bit RGBA files). */
  pixels: Uint8Array;
}

/** Minimal PNG decoder for 8-bit RGBA test assertions. */
export function decodePng(buf: Uint8Array): DecodedPng {
  expect(Array.from(buf.subarray(0, 8))).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const chunkTypes: string[] = [];
  const idat: Uint8Array[] = [];
  let width = 0;
  let height = 0;
  let colorType = 0;
  for (let o = 8; o < buf.length; ) {
    const len = view.getUint32(o);
    const type = String.fromCharCode(...buf.subarray(o + 4, o + 8));
    const data = buf.subarray(o + 8, o + 8 + len);
    chunkTypes.push(type);
    if (type === 'IHDR') {
      width = view.getUint32(o + 8);
      height = view.getUint32(o + 12);
      colorType = data[9]!;
    }
    if (type === 'IDAT') idat.push(data);
    o += 12 + len;
  }
  const pixels = new Uint8Array(width * height * 4);
  if (colorType === 6) {
    const raw = inflateSync(Buffer.concat(idat));
    const stride = width * 4;
    for (let y = 0; y < height; y++) {
      const f = raw[y * (stride + 1)]!;
      for (let i = 0; i < stride; i++) {
        const x = raw[y * (stride + 1) + 1 + i]!;
        const a = i >= 4 ? pixels[y * stride + i - 4]! : 0;
        const b = y > 0 ? pixels[(y - 1) * stride + i]! : 0;
        const c = y > 0 && i >= 4 ? pixels[(y - 1) * stride + i - 4]! : 0;
        let p = 0;
        if (f === 1) p = a;
        else if (f === 2) p = b;
        else if (f === 3) p = (a + b) >> 1;
        else if (f === 4) {
          const q = a + b - c;
          const pa = Math.abs(q - a);
          const pb = Math.abs(q - b);
          const pc = Math.abs(q - c);
          p = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
        }
        pixels[y * stride + i] = (x + p) & 255;
      }
    }
  }
  return { width, height, colorType, chunkTypes, pixels };
}

export function readDownload(path: string): DecodedPng {
  return decodePng(new Uint8Array(readFileSync(path)));
}

/** Drops a file from disk onto the page, the way a browser does for a real drag. */
export async function dropFile(page: Page, path: string, name: string, type: string): Promise<void> {
  const data = readFileSync(path).toString('base64');
  await page.evaluate(
    async ({ data, name, type }) => {
      const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
      const dt = new DataTransfer();
      dt.items.add(new File([bytes], name, { type }));
      const target = document.getElementById('dropzone') ?? document.body;
      target.dispatchEvent(new DragEvent('dragenter', { bubbles: true, cancelable: true, dataTransfer: dt }));
      target.dispatchEvent(new DragEvent('dragover', { bubbles: true, cancelable: true, dataTransfer: dt }));
      await new Promise((r) => setTimeout(r, 50));
      if (!document.documentElement.classList.contains('dragging')) throw new Error('drop overlay not shown');
      target.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: dt }));
    },
    { data, name, type },
  );
}

/** Pastes a file or text the way Cmd/Ctrl+V does. */
export async function paste(page: Page, content: { path: string; name: string; type: string } | { text: string }): Promise<void> {
  const payload = 'text' in content ? { text: content.text } : { data: readFileSync(content.path).toString('base64'), name: content.name, type: content.type };
  await page.evaluate((p) => {
    const dt = new DataTransfer();
    if ('text' in p && p.text !== undefined) dt.setData('text/plain', p.text);
    else if ('data' in p && p.data) {
      const bytes = Uint8Array.from(atob(p.data), (c) => c.charCodeAt(0));
      dt.items.add(new File([bytes], p.name, { type: p.type }));
    }
    document.dispatchEvent(new ClipboardEvent('paste', { bubbles: true, cancelable: true, clipboardData: dt }));
  }, payload);
}
