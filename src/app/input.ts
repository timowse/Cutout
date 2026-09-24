/** Image input: global drag & drop, the file picker and Cmd/Ctrl+V paste. */

import { t } from '../i18n';
import { FILE_INPUT_ACCEPT, looksLikeImage } from '../shared/files';

export interface InputHandlers {
  file(file: File): void;
  /** Called as soon as the user shows intent to process an image (warm-up). */
  intent(): void;
  toast(text: string): void;
}

function hasFiles(e: DragEvent): boolean {
  return !!e.dataTransfer && Array.from(e.dataTransfer.types).includes('Files');
}

function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  if (target instanceof HTMLTextAreaElement) return true;
  return target instanceof HTMLInputElement && !['button', 'checkbox', 'radio', 'range', 'color', 'file'].includes(target.type);
}

/** Picks the first image from a list of files, with a note if several were given. */
function firstImage(files: readonly File[], handlers: InputHandlers): File | null {
  const images = files.filter(looksLikeImage);
  const chosen = images[0] ?? files[0] ?? null;
  if (files.length > 1 && chosen) handlers.toast(t('toast.multiple'));
  return chosen;
}

export function setupInput(fileInput: HTMLInputElement, handlers: InputHandlers): { openPicker(): void } {
  fileInput.accept = FILE_INPUT_ACCEPT;
  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (file) handlers.file(file);
    fileInput.value = '';
  });

  // Drag & drop anywhere on the page. A counter handles dragenter/leave on child elements.
  const root = document.documentElement;
  let depth = 0;
  const hide = () => {
    depth = 0;
    root.classList.remove('dragging');
  };
  window.addEventListener('dragenter', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    if (depth === 1) {
      root.classList.add('dragging');
      handlers.intent();
    }
  });
  window.addEventListener('dragover', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
  });
  window.addEventListener('dragleave', (e) => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (depth === 0) hide();
  });
  window.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (!dt) return;
    e.preventDefault();
    hide();
    const files = Array.from(dt.files);
    if (files.length > 0) {
      const file = firstImage(files, handlers);
      if (file) handlers.file(file);
    } else if (dt.types.includes('text/uri-list') || dt.types.includes('text/plain')) {
      handlers.toast(t('toast.link'));
    }
  });
  window.addEventListener('blur', hide);

  // Paste (Cmd+V / Ctrl+V). No permission prompt: the browser hands us the pasted data.
  document.addEventListener('paste', (e) => {
    if (isEditable(e.target)) return;
    const data = e.clipboardData;
    if (!data) return;
    const files: File[] = Array.from(data.files);
    if (files.length === 0) {
      for (const item of Array.from(data.items)) {
        if (item.kind !== 'file') continue;
        const file = item.getAsFile();
        if (file) files.push(file);
      }
    }
    const images = files.filter(looksLikeImage);
    if (images.length > 0) {
      e.preventDefault();
      handlers.file(images[0]!);
      return;
    }
    if (files.length > 0) {
      e.preventDefault();
      handlers.file(files[0]!);
      return;
    }
    if (data.types.length > 0) handlers.toast(t('toast.noImage'));
  });

  return {
    openPicker() {
      handlers.intent();
      fileInput.click();
    },
  };
}
