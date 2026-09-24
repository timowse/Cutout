import './styles/main.css';

import { initTheme } from './app/theme';
import { APP_NAME, LICENSE_URL, REPO_URL } from './config';
import { detectLocale, setLocale, translateDocument } from './i18n';

setLocale(detectLocale(navigator.languages.length > 0 ? navigator.languages : [navigator.language], location.search));
translateDocument();
initTheme(null);
for (const el of document.querySelectorAll<HTMLElement>('[data-app-name]')) el.textContent = APP_NAME;
for (const el of document.querySelectorAll<HTMLAnchorElement>('[data-repo-link]')) el.href = REPO_URL;
for (const el of document.querySelectorAll<HTMLAnchorElement>('[data-license-link]')) el.href = LICENSE_URL;
