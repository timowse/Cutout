/**
 * Image file validation based on the file's actual bytes, not its name or the
 * MIME type the browser guessed. Also reads image dimensions from the header
 * so oversized images can be rejected before they are decoded.
 */

export type ImageFormat = 'jpeg' | 'png' | 'webp' | 'gif' | 'bmp' | 'avif' | 'heic';

export interface ImageInfo {
  format: ImageFormat;
  /** Dimensions from the file header, if they could be read. */
  width?: number;
  height?: number;
}

/** Formats every supported browser can decode. */
export const UNIVERSAL_FORMATS: readonly ImageFormat[] = ['jpeg', 'png', 'webp', 'gif', 'bmp'];

/** `accept` attribute for the file picker. Listing concrete types makes iOS convert HEIC photos to JPEG. */
export const FILE_INPUT_ACCEPT = 'image/jpeg,image/png,image/webp,image/avif,image/gif,image/bmp,.jpg,.jpeg,.png,.webp,.avif,.gif,.bmp';

/** Absolute upper bound for the file size in bytes (larger files are rejected immediately). */
export const MAX_FILE_BYTES = 200 * 1024 * 1024;

/** Enough bytes to reach the dimensions in practically every JPEG (EXIF + thumbnail come first). */
export const HEADER_BYTES = 512 * 1024;

function ascii(bytes: Uint8Array, offset: number, length: number): string {
  let s = '';
  for (let i = offset; i < offset + length && i < bytes.length; i++) s += String.fromCharCode(bytes[i]!);
  return s;
}

function u16be(b: Uint8Array, o: number): number {
  return (b[o]! << 8) | b[o + 1]!;
}
function u16le(b: Uint8Array, o: number): number {
  return b[o]! | (b[o + 1]! << 8);
}
function u24le(b: Uint8Array, o: number): number {
  return b[o]! | (b[o + 1]! << 8) | (b[o + 2]! << 16);
}
function u32be(b: Uint8Array, o: number): number {
  return ((b[o]! << 24) >>> 0) + ((b[o + 1]! << 16) | (b[o + 2]! << 8) | b[o + 3]!);
}
function i32le(b: Uint8Array, o: number): number {
  return b[o]! | (b[o + 1]! << 8) | (b[o + 2]! << 16) | (b[o + 3]! << 24);
}

const HEIC_BRANDS = new Set(['heic', 'heix', 'hevc', 'hevx', 'heim', 'heis', 'hevm', 'hevs']);

/** Detects the image format from the first bytes of a file. */
export function sniffImageFormat(bytes: Uint8Array): ImageFormat | null {
  if (bytes.length < 12) return null;
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'jpeg';
  if (bytes[0] === 0x89 && ascii(bytes, 1, 3) === 'PNG' && bytes[4] === 0x0d && bytes[5] === 0x0a) return 'png';
  if (ascii(bytes, 0, 4) === 'RIFF' && ascii(bytes, 8, 4) === 'WEBP') return 'webp';
  if (ascii(bytes, 0, 6) === 'GIF87a' || ascii(bytes, 0, 6) === 'GIF89a') return 'gif';
  if (ascii(bytes, 0, 2) === 'BM' && bytes.length >= 26) return 'bmp';
  if (ascii(bytes, 4, 4) === 'ftyp') {
    const boxSize = u32be(bytes, 0);
    const brands: string[] = [ascii(bytes, 8, 4)];
    for (let o = 16; o + 4 <= Math.min(boxSize, bytes.length); o += 4) brands.push(ascii(bytes, o, 4));
    if (brands.includes('avif') || brands.includes('avis')) return 'avif';
    if (brands.some((b) => HEIC_BRANDS.has(b))) return 'heic';
    if (brands.includes('mif1') || brands.includes('msf1')) return 'heic';
  }
  return null;
}

function jpegSize(b: Uint8Array): { width: number; height: number } | null {
  let o = 2;
  while (o + 9 < b.length) {
    if (b[o] !== 0xff) {
      o++;
      continue;
    }
    const marker = b[o + 1]!;
    if (marker === 0xff) {
      o++;
      continue;
    }
    if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      o += 2;
      continue;
    }
    const length = u16be(b, o + 2);
    const isSOF = marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc;
    if (isSOF) {
      const height = u16be(b, o + 5);
      const width = u16be(b, o + 7);
      return width > 0 && height > 0 ? { width, height } : null;
    }
    if (marker === 0xd9 || marker === 0xda) return null;
    o += 2 + length;
  }
  return null;
}

function webpSize(b: Uint8Array): { width: number; height: number } | null {
  const chunk = ascii(b, 12, 4);
  if (chunk === 'VP8X' && b.length >= 30) return { width: u24le(b, 24) + 1, height: u24le(b, 27) + 1 };
  if (chunk === 'VP8 ' && b.length >= 30) return { width: u16le(b, 26) & 0x3fff, height: u16le(b, 28) & 0x3fff };
  if (chunk === 'VP8L' && b.length >= 25 && b[20] === 0x2f) {
    const bits = (b[21]! | (b[22]! << 8) | (b[23]! << 16) | (b[24]! << 24)) >>> 0;
    return { width: (bits & 0x3fff) + 1, height: ((bits >>> 14) & 0x3fff) + 1 };
  }
  return null;
}

function isobmffSize(b: Uint8Array): { width: number; height: number } | null {
  // The 'ispe' (image spatial extent) property holds the primary image size.
  let best: { width: number; height: number } | null = null;
  for (let o = 4; o + 16 <= b.length; o++) {
    if (b[o] === 0x69 && b[o + 1] === 0x73 && b[o + 2] === 0x70 && b[o + 3] === 0x65) {
      const width = u32be(b, o + 8);
      const height = u32be(b, o + 12);
      if (width > 0 && height > 0 && (!best || width * height > best.width * best.height)) best = { width, height };
    }
  }
  return best;
}

/** Reads the pixel dimensions from an image header (without decoding). */
export function readImageSize(bytes: Uint8Array, format: ImageFormat): { width: number; height: number } | null {
  switch (format) {
    case 'png':
      return bytes.length >= 24 && ascii(bytes, 12, 4) === 'IHDR'
        ? { width: u32be(bytes, 16), height: u32be(bytes, 20) }
        : null;
    case 'jpeg':
      return jpegSize(bytes);
    case 'webp':
      return webpSize(bytes);
    case 'gif':
      return { width: u16le(bytes, 6), height: u16le(bytes, 8) };
    case 'bmp':
      return { width: Math.abs(i32le(bytes, 18)), height: Math.abs(i32le(bytes, 22)) };
    case 'avif':
    case 'heic':
      return isobmffSize(bytes);
  }
}

export async function readHeader(file: Blob, bytes = HEADER_BYTES): Promise<Uint8Array> {
  return new Uint8Array(await file.slice(0, bytes).arrayBuffer());
}

/** Identifies an image file by its content. Returns null for anything that is not a known image. */
export async function inspectImageFile(file: Blob): Promise<ImageInfo | null> {
  const header = await readHeader(file);
  const format = sniffImageFormat(header);
  if (!format) return null;
  const size = readImageSize(header, format);
  return size ? { format, width: size.width, height: size.height } : { format };
}

/** Cheap pre-check on dropped/pasted items before reading any bytes. */
export function looksLikeImage(file: { type: string; name: string }): boolean {
  if (file.type.startsWith('image/')) return true;
  return /\.(jpe?g|png|webp|gif|bmp|avif|heic|heif)$/i.test(file.name);
}
