import { describe, expect, it } from 'vitest';
import { inspectImageFile, looksLikeImage, readImageSize, sniffImageFormat } from './files';

function bytes(...parts: (number[] | string)[]): Uint8Array {
  const out: number[] = [];
  for (const p of parts) {
    if (typeof p === 'string') for (const ch of p) out.push(ch.charCodeAt(0));
    else out.push(...p);
  }
  return new Uint8Array(out);
}

const u32be = (n: number) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
const u16be = (n: number) => [(n >>> 8) & 255, n & 255];
const u16le = (n: number) => [n & 255, (n >>> 8) & 255];
const u24le = (n: number) => [n & 255, (n >>> 8) & 255, (n >>> 16) & 255];

function png(width: number, height: number): Uint8Array {
  return bytes([0x89], 'PNG', [0x0d, 0x0a, 0x1a, 0x0a], u32be(13), 'IHDR', u32be(width), u32be(height), [8, 6, 0, 0, 0]);
}

function jpeg(width: number, height: number): Uint8Array {
  // SOI, APP0 (JFIF, 16 bytes), SOF0
  return bytes(
    [0xff, 0xd8],
    [0xff, 0xe0],
    u16be(16),
    'JFIF',
    [0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
    [0xff, 0xc0],
    u16be(17),
    [8],
    u16be(height),
    u16be(width),
    [3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1],
  );
}

describe('sniffImageFormat', () => {
  it('detects formats by content', () => {
    expect(sniffImageFormat(png(1, 1))).toBe('png');
    expect(sniffImageFormat(jpeg(1, 1))).toBe('jpeg');
    expect(sniffImageFormat(bytes('RIFF', [0, 0, 0, 0], 'WEBPVP8 '))).toBe('webp');
    expect(sniffImageFormat(bytes('GIF89a', [1, 0, 1, 0, 0, 0]))).toBe('gif');
    expect(sniffImageFormat(bytes('BM', new Array(30).fill(0) as number[]))).toBe('bmp');
    expect(sniffImageFormat(bytes(u32be(20), 'ftypavif', [0, 0, 0, 0], 'mif1'))).toBe('avif');
    expect(sniffImageFormat(bytes(u32be(24), 'ftypheic', [0, 0, 0, 0], 'mif1', 'heic'))).toBe('heic');
  });

  it('rejects non-images even with an image-like name', () => {
    expect(sniffImageFormat(bytes('%PDF-1.7 ....'))).toBeNull();
    expect(sniffImageFormat(bytes('<svg xmlns="http://www.w3.org/2000/svg">'))).toBeNull();
    expect(sniffImageFormat(bytes('hello world, plain text'))).toBeNull();
    expect(sniffImageFormat(new Uint8Array(4))).toBeNull();
  });
});

describe('readImageSize', () => {
  it('reads PNG and JPEG dimensions', () => {
    expect(readImageSize(png(4032, 3024), 'png')).toEqual({ width: 4032, height: 3024 });
    expect(readImageSize(jpeg(6000, 4000), 'jpeg')).toEqual({ width: 6000, height: 4000 });
  });

  it('reads all WebP variants', () => {
    const vp8x = bytes('RIFF', [0, 0, 0, 0], 'WEBPVP8X', [0, 0, 0, 0], [0, 0, 0, 0], u24le(1999), u24le(999));
    expect(readImageSize(vp8x, 'webp')).toEqual({ width: 2000, height: 1000 });
    const vp8 = bytes('RIFF', [0, 0, 0, 0], 'WEBPVP8 ', [0, 0, 0, 0], [0, 0, 0, 0x9d, 0x01, 0x2a], u16le(640), u16le(480));
    expect(readImageSize(vp8, 'webp')).toEqual({ width: 640, height: 480 });
    const bits = (799 | (599 << 14)) >>> 0;
    const vp8l = bytes('RIFF', [0, 0, 0, 0], 'WEBPVP8L', [0, 0, 0, 0], [0x2f], [bits & 255, (bits >>> 8) & 255, (bits >>> 16) & 255, bits >>> 24]);
    expect(readImageSize(vp8l, 'webp')).toEqual({ width: 800, height: 600 });
  });

  it('reads the ispe box of HEIC/AVIF', () => {
    const heic = bytes(u32be(24), 'ftypheic', [0, 0, 0, 0], 'mif1heic', u32be(20), 'ispe', [0, 0, 0, 0], u32be(4032), u32be(3024));
    expect(readImageSize(heic, 'heic')).toEqual({ width: 4032, height: 3024 });
  });

  it('returns null for truncated headers', () => {
    expect(readImageSize(bytes([0xff, 0xd8, 0xff, 0xe0, 0, 16]), 'jpeg')).toBeNull();
  });
});

describe('inspectImageFile', () => {
  it('reads format and size from a Blob', async () => {
    const info = await inspectImageFile(new Blob([png(300, 200) as BlobPart]));
    expect(info).toEqual({ format: 'png', width: 300, height: 200 });
  });

  it('returns null for text files', async () => {
    expect(await inspectImageFile(new Blob(['just some text, not an image']))).toBeNull();
  });
});

describe('looksLikeImage', () => {
  it('uses MIME type or extension', () => {
    expect(looksLikeImage({ type: 'image/png', name: 'x' })).toBe(true);
    expect(looksLikeImage({ type: '', name: 'Photo.JPG' })).toBe(true);
    expect(looksLikeImage({ type: 'text/plain', name: 'notes.txt' })).toBe(false);
  });
});
