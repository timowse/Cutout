import { de } from './de';
import { en, type MessageKey, type Messages } from './en';

export type Locale = 'en' | 'de';
export type { MessageKey };

const catalogs: Record<Locale, Messages> = { en, de };

/** Picks the UI language: `?lang=` override first, then the browser's preferred languages. */
export function detectLocale(languages: readonly string[], search = ''): Locale {
  const override = new URLSearchParams(search).get('lang');
  if (override === 'de' || override === 'en') return override;
  for (const lang of languages) {
    const base = lang.toLowerCase().split('-')[0];
    if (base === 'de') return 'de';
    if (base === 'en') return 'en';
  }
  return 'en';
}

let current: Locale = 'en';

export function setLocale(locale: Locale): void {
  current = locale;
}

export function getLocale(): Locale {
  return current;
}

/** Translates a key and fills `{placeholders}`. */
export function t(key: MessageKey, params?: Record<string, string | number>): string {
  let text: string = catalogs[current][key] ?? en[key];
  if (params) {
    for (const [name, value] of Object.entries(params)) text = text.split(`{${name}}`).join(String(value));
  }
  return text;
}

export function isMessageKey(key: string): key is MessageKey {
  return Object.prototype.hasOwnProperty.call(en, key);
}

/**
 * Applies translations to the document:
 * `data-i18n="key"` sets the text, `data-i18n-attr="aria-label:key;title:key"` sets attributes.
 */
export function translateDocument(root: ParentNode = document): void {
  document.documentElement.lang = current;
  for (const el of root.querySelectorAll<HTMLElement>('[data-i18n]')) {
    const key = el.dataset.i18n;
    if (key && isMessageKey(key)) el.textContent = t(key);
  }
  for (const el of root.querySelectorAll<HTMLElement>('[data-i18n-attr]')) {
    for (const pair of (el.dataset.i18nAttr ?? '').split(';')) {
      const [attr, key] = pair.split(':').map((s) => s.trim());
      if (attr && key && isMessageKey(key)) el.setAttribute(attr, t(key));
    }
  }
  for (const el of root.querySelectorAll<HTMLElement>('[data-locale]')) {
    el.hidden = el.dataset.locale !== current;
  }
}

/** Formats a byte count as MB with one decimal for small values (e.g. "37 MB", "4.2 MB"). */
export function formatMB(bytes: number): string {
  const mb = bytes / 1_000_000;
  const digits = mb < 10 ? 1 : 0;
  return `${new Intl.NumberFormat(current, { maximumFractionDigits: digits, minimumFractionDigits: 0 }).format(mb)} MB`;
}

export function formatNumber(n: number): string {
  return new Intl.NumberFormat(current).format(n);
}
