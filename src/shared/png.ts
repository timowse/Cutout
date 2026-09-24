/**
 * Minimal streaming PNG encoder for 8-bit RGBA images.
 *
 * Why not canvas.toBlob? Canvas stores pixels with premultiplied alpha, which
 * destroys the colour of nearly transparent pixels (exactly the soft hair and
 * fur edges we care about). This encoder writes the straight-alpha RGBA values
 * unchanged, adds no metadata, and uses the browser's native zlib through
 * CompressionStream.
 */

const SIGNATURE = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

/** Updates a running CRC-32 (start with 0xffffffff, finish with `^ 0xffffffff`). */
export function crc32Update(crc: number, data: Uint8Array): number {
  let c = crc;
  for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]!) & 0xff]! ^ (c >>> 8);
  return c >>> 0;
}

export function crc32(data: Uint8Array): number {
  return (crc32Update(0xffffffff, data) ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array<ArrayBuffer>[] {
  const head = new Uint8Array(8);
  const view = new DataView(head.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) head[4 + i] = type.charCodeAt(i);
  let crc = crc32Update(0xffffffff, head.subarray(4));
  crc = crc32Update(crc, data);
  const tail = new Uint8Array(4);
  new DataView(tail.buffer).setUint32(0, (crc ^ 0xffffffff) >>> 0);
  const body = new Uint8Array(data.length);
  body.set(data);
  return [head, body, tail];
}

/** Writes one scanline with the Paeth filter (type 4) into `out` at `o`. */
function filterPaeth(src: ArrayLike<number>, rowStart: number, prevStart: number, rowBytes: number, out: Uint8Array, o: number): void {
  out[o] = 4;
  o++;
  for (let i = 0; i < rowBytes; i++) {
    const x = src[rowStart + i]!;
    const a = i >= 4 ? src[rowStart + i - 4]! : 0;
    const b = prevStart >= 0 ? src[prevStart + i]! : 0;
    const c = prevStart >= 0 && i >= 4 ? src[prevStart + i - 4]! : 0;
    const p = a + b - c;
    const pa = p > a ? p - a : a - p;
    const pb = p > b ? p - b : b - p;
    const pc = p > c ? p - c : c - p;
    const pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    out[o + i] = (x - pred) & 0xff;
  }
}

export function isPngEncoderSupported(): boolean {
  return typeof CompressionStream !== 'undefined';
}

/**
 * Encodes straight (non-premultiplied) RGBA pixels as a PNG file.
 * `onProgress` receives the fraction of rows written.
 */
export async function encodePNG(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  onProgress?: (fraction: number) => void,
): Promise<Blob> {
  if (rgba.length !== width * height * 4) throw new RangeError('Pixel buffer does not match the image size');
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, width);
  v.setUint32(4, height);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // colour type RGBA
  ihdr[10] = 0; // deflate
  ihdr[11] = 0; // adaptive filtering
  ihdr[12] = 0; // no interlace

  const stream = new CompressionStream('deflate');
  const writer = stream.writable.getWriter();
  const reader = stream.readable.getReader();
  const idat: Uint8Array<ArrayBuffer>[] = [];
  const drain = (async () => {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      idat.push(...chunk('IDAT', value));
    }
  })();

  const rowBytes = width * 4;
  const batchRows = Math.max(1, Math.floor((4 * 1024 * 1024) / (rowBytes + 1)));
  try {
    for (let y0 = 0; y0 < height; y0 += batchRows) {
      const rows = Math.min(batchRows, height - y0);
      const buf = new Uint8Array(rows * (rowBytes + 1));
      for (let r = 0; r < rows; r++) {
        const y = y0 + r;
        filterPaeth(rgba, y * rowBytes, y > 0 ? (y - 1) * rowBytes : -1, rowBytes, buf, r * (rowBytes + 1));
      }
      await writer.ready;
      await writer.write(buf);
      onProgress?.((y0 + rows) / height);
    }
    await writer.close();
  } catch (err) {
    await writer.abort(err).catch(() => undefined);
    throw err;
  }
  await drain;
  return new Blob([SIGNATURE, ...chunk('IHDR', ihdr), ...idat, ...chunk('IEND', new Uint8Array(0))], {
    type: 'image/png',
  });
}
