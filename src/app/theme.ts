/** Light/dark theme: follows the system by default, can be overridden manually. */

import { t } from '../i18n';

export type ThemeMode = 'system' | 'light' | 'dark';
const KEY = 'theme';
const ORDER: ThemeMode[] = ['system', 'light', 'dark'];

function readStored(): ThemeMode {
  try {
    const value = localStorage.getItem(KEY);
    return value === 'light' || value === 'dark' ? value : 'system';
  } catch {
    return 'system';
  }
}

function store(mode: ThemeMode): void {
  try {
    if (mode === 'system') localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, mode);
  } catch {
    /* storage unavailable: the choice lasts for this page view only */
  }
}

function apply(mode: ThemeMode, button: HTMLElement | null): void {
  const root = document.documentElement;
  if (mode === 'system') delete root.dataset.theme;
  else root.dataset.theme = mode;
  if (button) {
    button.dataset.mode = mode;
    const label = t(mode === 'system' ? 'theme.system' : mode === 'light' ? 'theme.light' : 'theme.dark');
    button.setAttribute('aria-label', label);
    button.title = label;
  }
}

export function initTheme(button: HTMLElement | null): void {
  let mode = readStored();
  apply(mode, button);
  button?.addEventListener('click', () => {
    mode = ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length]!;
    store(mode);
    apply(mode, button);
  });
}
