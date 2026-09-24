/**
 * Image pipeline around the model: decode → model input → (inference) →
 * matte upsampling → foreground colour estimation → PNG.
 */

import { exceedsInputLimit, fitWithin, planOutputSize, type OutputPlan } from '../shared/dimensions';
import { AppError } from '../shared/errors';
import { applyForeground } from '../shared/foreground';
import { inspectImageFile } from '../shared/files';
import { encodePNG, isPngEncoderSupported } from '../shared/png';
import type { ImageLimits, ModelManifest, RGB } from '../shared/protocol';
import { resampleInterleaved, resamplePlaneToUint8 } from '../shared/resample';

export interface DecodedImage {
  /** Straight-alpha RGBA pixels at the output size. */
  rgba: Uint8ClampedArray<ArrayBuffer>;
  width: number;
  height: number;
  originalWidth: number;
  originalHeight: number;
  /** True if any pixel of the source is not fully opaque. */
  hasAlpha: boolean;
}

function assertCanvasSupport(): void {
  if (typeof OffscreenCanvas === 'undefined' || typeof createImageBitmap === 'undefined') {
    throw new AppError('runtime-unsupported', 'OffscreenCanvas/createImageBitmap missing');
  }
}

function context2d(canvas: OffscreenCanvas): OffscreenCanvasRenderingContext2D {
  const ctx = canvas.getContext('2d', { willReadFrequently: true, alpha: true });
  if (!ctx) throw new AppError('out-of-memory', 'Could not create a 2D canvas');
  return ctx;
}

/**
 * Decodes with EXIF orientation applied. Older engines only know the enum
 * values 'none' / 'flipY' and reject 'from-image' with a TypeError; they
 * apply the EXIF orientation by default, so decode again without options.
 */
async function decodeOriented(file: Blob): Promise<ImageBitmap> {
  try {
    return await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch (err) {
    if (err instanceof TypeError) return createImageBitmap(file);
    throw err;
  }
}

/** Decodes the file (respecting EXIF orientation) and returns its pixels at the planned output size. */
export async function decodeImage(file: Blob, limits: ImageLimits): Promise<DecodedImage & { plan: OutputPlan }> {
  assertCanvasSupport();
  const info = await inspectImageFile(file);
  if (!info) throw new AppError('unsupported-format', `Unknown file type ${file.type}`);
  if (info.width && info.height && exceedsInputLimit(info.width, info.height, limits)) {
    throw new AppError('too-large', `${info.width}x${info.height}`);
  }

  let bitmap: ImageBitmap;
  try {
    bitmap = await decodeOriented(file);
  } catch (err) {
    throw new AppError(info.format === 'heic' ? 'heic-unsupported' : 'decode-failed', String(err));
  }
  try {
    const { width: ow, height: oh } = bitmap;
    if (!(ow > 0 && oh > 0)) throw new AppError('decode-failed', 'Empty image');
    if (exceedsInputLimit(ow, oh, limits)) throw new AppError('too-large', `${ow}x${oh}`);
    const plan = planOutputSize(ow, oh, limits);
    const canvas = new OffscreenCanvas(plan.width, plan.height);
    const ctx = context2d(canvas);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, 0, 0, plan.width, plan.height);
    let rgba: Uint8ClampedArray<ArrayBuffer>;
    try {
      rgba = ctx.getImageData(0, 0, plan.width, plan.height).data;
    } catch (err) {
      throw new AppError('out-of-memory', `getImageData: ${String(err)}`);
    }
    canvas.width = 0;
    canvas.height = 0;
    let hasAlpha = false;
    for (let i = 3; i < rgba.length; i += 4) {
      if (rgba[i] !== 255) {
        hasAlpha = true;
        break;
      }
    }
    return { rgba, width: plan.width, height: plan.height, originalWidth: ow, originalHeight: oh, hasAlpha, plan };
  } finally {
    bitmap.close();
  }
}

/**
 * Resizes the image to the model's input size with an anti-aliased filter
 * (matching the reference preprocessing) and normalises it.
 * Transparent areas are composited onto white first.
 */
export function toModelInput(image: DecodedImage, manifest: ModelManifest): Float32Array {
  const { width: mw, height: mh, mean, std } = manifest.input;
  let src: Uint8ClampedArray = image.rgba;
  if (image.hasAlpha) {
    src = new Uint8ClampedArray(image.rgba.length);
    for (let i = 0; i < src.length; i += 4) {
      const a = image.rgba[i + 3]! / 255;
      src[i] = image.rgba[i]! * a + 255 * (1 - a);
      src[i + 1] = image.rgba[i + 1]! * a + 255 * (1 - a);
      src[i + 2] = image.rgba[i + 2]! * a + 255 * (1 - a);
      src[i + 3] = 255;
    }
  }
  const planes = resampleInterleaved(src, image.width, image.height, 4, 3, mw, mh, 'triangle');
  const n = mw * mh;
  const out = new Float32Array(3 * n);
  for (let c = 0; c < 3; c++) {
    const plane = planes[c]!;
    const m = mean[c]!;
    const s = std[c]!;
    for (let i = 0; i < n; i++) out[c * n + i] = (plane[i]! / 255 - m) / s;
  }
  return out;
}

/**
 * Turns the model's matte into the final straight-alpha RGBA image, in place:
 * bicubic upsampling of the matte to the output size, then foreground colour
 * estimation for soft edges.
 */
export function composeResult(image: DecodedImage, matte: Float32Array, manifest: ModelManifest): void {
  const { width: mw, height: mh } = manifest.output;
  const alpha = resamplePlaneToUint8(matte, mw, mh, image.width, image.height, 'bicubic');
  if (image.hasAlpha) {
    // The result can never be more opaque than the source.
    for (let i = 0, p = 3; i < alpha.length; i++, p += 4) alpha[i] = (alpha[i]! * image.rgba[p]! + 127) / 255;
  }
  applyForeground(image.rgba, alpha, image.width, image.height);
}

/** A downscaled copy of the result for the on-screen preview. */
export async function createPreview(image: DecodedImage, maxSide: number): Promise<ImageBitmap> {
  const full = await createImageBitmap(new ImageData(image.rgba, image.width, image.height), {
    premultiplyAlpha: 'premultiply',
  });
  try {
    const { width, height } = fitWithin(image.width, image.height, maxSide);
    if (width === image.width && height === image.height) return full;
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new AppError('out-of-memory', 'Preview canvas');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(full, 0, 0, width, height);
    const preview = canvas.transferToImageBitmap();
    full.close();
    return preview;
  } catch (err) {
    full.close();
    throw err;
  }
}

async function encodeWithCanvas(rgba: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new AppError('out-of-memory', 'Encode canvas');
  ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  canvas.width = 0;
  canvas.height = 0;
  return blob;
}

export async function encodeResult(rgba: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): Promise<Blob> {
  return isPngEncoderSupported() ? encodePNG(rgba, width, height) : encodeWithCanvas(rgba, width, height);
}

/** The result flattened onto a solid colour (for "Download with background"). */
export async function encodeWithBackground(
  rgba: Uint8ClampedArray<ArrayBuffer>,
  width: number,
  height: number,
  color: RGB,
): Promise<Blob> {
  const out = new Uint8ClampedArray(rgba.length);
  for (let p = 0; p < rgba.length; p += 4) {
    const a = rgba[p + 3]! / 255;
    out[p] = rgba[p]! * a + color.r * (1 - a);
    out[p + 1] = rgba[p + 1]! * a + color.g * (1 - a);
    out[p + 2] = rgba[p + 2]! * a + color.b * (1 - a);
    out[p + 3] = 255;
  }
  return encodeResult(out, width, height);
}
