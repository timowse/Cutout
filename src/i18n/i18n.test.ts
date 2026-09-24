import { afterEach, describe, expect, it } from 'vitest';
import { de } from './de';
import { en } from './en';
import { detectLocale, formatMB, setLocale, t } from './index';

afterEach(() => setLocale('en'));

describe('i18n', () => {
  it('picks German for German browsers and English otherwise', () => {
    expect(detectLocale(['de-DE', 'en'])).toBe('de');
    expect(detectLocale(['fr-FR', 'de'])).toBe('de');
    expect(detectLocale(['en-US', 'de'])).toBe('en');
    expect(detectLocale(['ja'])).toBe('en');
    expect(detectLocale(['de-DE'], '?lang=en')).toBe('en');
  });

  it('fills placeholders', () => {
    setLocale('de');
    expect(t('status.downloading', { loaded: '31 MB', total: '93 MB' })).toBe('KI-Modell wird geladen — 31 MB / 93 MB');
  });

  it('uses the same placeholders in every language', () => {
    const placeholders = (s: string) => (s.match(/\{\w+\}/g) ?? []).sort().join();
    for (const key of Object.keys(en) as (keyof typeof en)[]) {
      expect(placeholders(de[key]), key).toBe(placeholders(en[key]));
      expect(de[key].trim().length, key).toBeGreaterThan(0);
    }
  });

  it('formats megabytes', () => {
    expect(formatMB(92_900_000)).toBe('93 MB');
    expect(formatMB(4_200_000)).toBe('4.2 MB');
    setLocale('de');
    expect(formatMB(4_200_000)).toBe('4,2 MB');
  });
});
