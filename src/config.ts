/**
 * Project-wide settings. The product name is a working title; change it here
 * (and in manifest.webmanifest) once the final name is decided.
 */
export const APP_NAME = 'Cutout';

export const REPO_URL = 'https://github.com/timowse/Hintergrund-entfernen';
export const LICENSE_URL = `${REPO_URL}/blob/main/LICENSE`;

/**
 * Directory that contains the model's manifest.json, relative to the page.
 * Can be pointed elsewhere at build time with VITE_MODEL_BASE_URL (the
 * Content-Security-Policy's connect-src must then allow that origin).
 */
export const MODEL_BASE_URL: string = import.meta.env.VITE_MODEL_BASE_URL || 'models/birefnet-lite/';
