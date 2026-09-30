import { describe, expect, it } from 'vitest';
import { expiredIds, MAX_ITEMS } from './history';

const DAY = 24 * 60 * 60 * 1000;

describe('history pruning', () => {
  it('keeps recent entries', () => {
    const now = 100 * DAY;
    expect(expiredIds([{ id: 1, created: now - DAY }, { id: 2, created: now }], now)).toEqual([]);
  });

  it('drops entries older than 30 days', () => {
    const now = 100 * DAY;
    expect(expiredIds([{ id: 1, created: now - 31 * DAY }, { id: 2, created: now - 29 * DAY }], now)).toEqual([1]);
  });

  it('keeps only the newest entries', () => {
    const now = 100 * DAY;
    const entries = Array.from({ length: MAX_ITEMS + 3 }, (_, i) => ({ id: i + 1, created: now - i * 1000 }));
    expect(expiredIds(entries, now).sort((a, b) => a - b)).toEqual([MAX_ITEMS + 1, MAX_ITEMS + 2, MAX_ITEMS + 3]);
  });
});
