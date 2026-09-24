import type { ErrorCode, WorkerError } from './protocol';

/** An error with a user-facing error code. */
export class AppError extends Error {
  readonly code: ErrorCode;
  constructor(code: ErrorCode, detail?: string) {
    super(detail ?? code);
    this.name = 'AppError';
    this.code = code;
  }
}

const OOM_PATTERN =
  /out of memory|\boom\b|allocation failed|array buffer allocation|invalid typed array length|invalid array length|memory access out of bounds|cannot enlarge memory|failed to grow memory|maximum call stack/i;

/** Maps anything thrown during processing to a user-facing error code. */
export function toWorkerError(err: unknown, fallback: ErrorCode = 'unknown'): WorkerError {
  if (err instanceof AppError) return { code: err.code, detail: err.message };
  const detail = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
  if (OOM_PATTERN.test(detail)) return { code: 'out-of-memory', detail };
  return { code: fallback, detail };
}
