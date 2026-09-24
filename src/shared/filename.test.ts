import { describe, expect, it } from 'vitest';
import { outputFileName, sanitizeBaseName } from './filename';

describe('outputFileName', () => {
  it('replaces the extension', () => {
    expect(outputFileName('portrait.jpg')).toBe('portrait-background-removed.png');
    expect(outputFileName('IMG_2041.HEIC')).toBe('IMG_2041-background-removed.png');
    expect(outputFileName('archive.tar.gz')).toBe('archive.tar-background-removed.png');
  });

  it('names background variants', () => {
    expect(outputFileName('shoe.png', 'white')).toBe('shoe-white-background.png');
    expect(outputFileName('shoe.png', 'black')).toBe('shoe-black-background.png');
    expect(outputFileName('shoe.png', { hex: '#3D7EFF' })).toBe('shoe-background-3d7eff.png');
  });

  it('falls back to "image" for empty or unnamed input', () => {
    expect(outputFileName('')).toBe('image-background-removed.png');
    expect(outputFileName(undefined)).toBe('image-background-removed.png');
    expect(outputFileName('.png')).toBe('image-background-removed.png');
  });
});

describe('sanitizeBaseName', () => {
  it('strips directories and unsafe characters', () => {
    expect(sanitizeBaseName('C:\\Users\\me\\Desktop\\my photo.jpg')).toBe('my-photo');
    expect(sanitizeBaseName('../../etc/passwd')).toBe('passwd');
    expect(sanitizeBaseName('a<b>c:d"e|f?g*h.png')).toBe('a-b-c-d-e-f-g-h');
    expect(sanitizeBaseName('line\nbreak\u0000.jpg')).toBe('line-break');
  });

  it('keeps unicode letters', () => {
    expect(sanitizeBaseName('Größe über Äpfel.webp')).toBe('Größe-über-Äpfel');
    expect(sanitizeBaseName('写真.jpeg')).toBe('写真');
  });

  it('avoids reserved Windows names and trims length', () => {
    expect(sanitizeBaseName('CON.png')).toBe('image-CON');
    expect(sanitizeBaseName('x'.repeat(300) + '.png').length).toBeLessThanOrEqual(80);
    expect(sanitizeBaseName('...hidden')).toBe('hidden');
  });
});
