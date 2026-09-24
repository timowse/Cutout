/** Builds safe download file names such as `portrait-background-removed.png`. */

const MAX_BASE_LENGTH = 80;
const WINDOWS_RESERVED = /^(con|prn|aux|nul|com[0-9]|lpt[0-9])$/i;

/** Strips directories and the extension and removes characters that are unsafe in file names. */
export function sanitizeBaseName(name: string | undefined | null): string {
  let base = (name ?? '').normalize('NFC');
  base = base.split(/[\\/]/).pop() ?? '';
  base = base.replace(/\.[A-Za-z0-9]{1,5}$/, '');
  // Control characters (code points below 32 and DEL) become separators.
  base = Array.from(base, (ch) => (ch.charCodeAt(0) < 32 || ch.charCodeAt(0) === 127 ? '-' : ch)).join('');
  base = base
    .replace(/[<>:"/\\|?*]+/g, '-')
    .replace(/\s+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^[-.\s]+|[-.\s]+$/g, '');
  if (base.length > MAX_BASE_LENGTH) base = base.slice(0, MAX_BASE_LENGTH).replace(/[-.\s]+$/g, '');
  if (!base) base = 'image';
  if (WINDOWS_RESERVED.test(base)) base = `image-${base}`;
  return base;
}

export type BackgroundName = 'white' | 'black' | { hex: string };

/**
 * `portrait.jpg` becomes `portrait-background-removed.png`; with a background
 * colour it becomes e.g. `portrait-white-background.png`.
 */
export function outputFileName(inputName: string | undefined | null, background?: BackgroundName): string {
  const base = sanitizeBaseName(inputName);
  if (!background) return `${base}-background-removed.png`;
  if (background === 'white' || background === 'black') return `${base}-${background}-background.png`;
  const hex = background.hex.replace(/[^0-9a-f]/gi, '').toLowerCase().slice(0, 6);
  return `${base}-background-${hex || 'color'}.png`;
}
