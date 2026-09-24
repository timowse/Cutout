import { inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { crc32, encodePNG } from './png';

interface Chunk {
  type: string;
  data: Uint8Array;
}

function readChunks(png: Uint8Array): Chunk[] {
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
  const chunks: Chunk[] = [];
  let o = 8;
  while (o < png.length) {
    const length = view.getUint32(o);
    const type = String.fromCharCode(...png.subarray(o + 4, o + 8));
    const data = png.subarray(o + 8, o + 8 + length);
    const crc = view.getUint32(o + 8 + length);
    expect(crc32(png.subarray(o + 4, o + 8 + length))).toBe(crc);
    chunks.push({ type, data });
    o += 12 + length;
  }
  return chunks;
}

/** Reference PNG decoder for 8-bit RGBA (all five filter types). */
function decodeRGBA(png: Uint8Array): { width: number; height: number; pixels: Uint8Array } {
  const chunks = readChunks(png);
  const ihdr = chunks[0]!;
  const v = new DataView(ihdr.data.buffer, ihdr.data.byteOffset);
  const width = v.getUint32(0);
  const height = v.getUint32(4);
  const idat = Buffer.concat(chunks.filter((c) => c.type === 'IDAT').map((c) => c.data));
  const raw = inflateSync(idat);
  const stride = width * 4;
  const out = new Uint8Array(stride * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)]!;
    for (let i = 0; i < stride; i++) {
      const x = raw[y * (stride + 1) + 1 + i]!;
      const a = i >= 4 ? out[y * stride + i - 4]! : 0;
      const b = y > 0 ? out[(y - 1) * stride + i]! : 0;
      const c = y > 0 && i >= 4 ? out[(y - 1) * stride + i - 4]! : 0;
      let pred = 0;
      if (filter === 1) pred = a;
      else if (filter === 2) pred = b;
      else if (filter === 3) pred = (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      out[y * stride + i] = (x + pred) & 255;
    }
  }
  return { width, height, pixels: out };
}

describe('crc32', () => {
  it('matches the standard check value', () => {
    expect(crc32(new TextEncoder().encode('123456789'))).toBe(0xcbf43926);
  });
});

describe('encodePNG', () => {
  it('round-trips straight-alpha RGBA exactly', async () => {
    const width = 37;
    const height = 23;
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < rgba.length; i++) rgba[i] = (i * 7919 + (i >> 3) * 31) & 255;
    // Nearly transparent pixels keep their colour (a canvas would premultiply and lose it).
    rgba.set([200, 100, 50, 1], 0);
    const blob = await encodePNG(rgba, width, height);
    expect(blob.type).toBe('image/png');
    const bytes = new Uint8Array(await blob.arrayBuffer());
    expect(Array.from(bytes.subarray(0, 8))).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    const decoded = decodeRGBA(bytes);
    expect(decoded.width).toBe(width);
    expect(decoded.height).toBe(height);
    expect(Array.from(decoded.pixels)).toEqual(Array.from(rgba));
  });

  it('writes no metadata chunks', async () => {
    const blob = await encodePNG(new Uint8Array(4 * 4 * 4), 4, 4);
    const types = readChunks(new Uint8Array(await blob.arrayBuffer())).map((c) => c.type);
    expect(types[0]).toBe('IHDR');
    expect(types.at(-1)).toBe('IEND');
    expect(new Set(types)).toEqual(new Set(['IHDR', 'IDAT', 'IEND']));
  });

  it('rejects mismatched buffers', async () => {
    await expect(encodePNG(new Uint8Array(10), 4, 4)).rejects.toThrow(RangeError);
  });
});
