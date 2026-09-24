/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MODEL_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/** Replaced at build time with the list of app-shell files to precache. */
declare const __PRECACHE_MANIFEST__: string[];
/** Replaced at build time with a hash that changes with every build. */
declare const __BUILD_ID__: string;
