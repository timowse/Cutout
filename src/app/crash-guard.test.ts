import { describe, expect, it } from 'vitest';
import { CrashGuard } from './crash-guard';

function memoryStore() {
  const data = new Map<string, string>();
  return {
    data,
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
  };
}

describe('CrashGuard', () => {
  it('reports a visit that ended while the model was loading', () => {
    const store = memoryStore();
    let now = 1_000_000;
    new CrashGuard(store, () => now).update({ stage: 'model' });
    now += 5000;
    const next = new CrashGuard(store, () => now);
    expect(next.takePrevious()).toEqual({ stage: 'model', at: 1_000_000 });
    expect(next.takePrevious()).toBeNull(); // consumed
  });

  it('reports nothing when the work finished or the page was left normally', () => {
    const store = memoryStore();
    const guard = new CrashGuard(store);
    guard.update({ stage: 'image', backend: 'wasm' });
    guard.update(null);
    expect(new CrashGuard(store).takePrevious()).toBeNull();
  });

  it('keeps the backend and ignores old or malformed markers', () => {
    const store = memoryStore();
    let now = 0;
    new CrashGuard(store, () => now).update({ stage: 'image', backend: 'webgpu' });
    expect(new CrashGuard(store, () => now).takePrevious()).toEqual({ stage: 'image', backend: 'webgpu', at: 0 });

    new CrashGuard(store, () => now).update({ stage: 'image' });
    now += 7 * 60 * 60 * 1000;
    expect(new CrashGuard(store, () => now).takePrevious()).toBeNull();

    store.setItem('cutout.busy', '{not json');
    expect(new CrashGuard(store).takePrevious()).toBeNull();
    store.setItem('cutout.busy', JSON.stringify({ stage: 'other', at: Date.now() }));
    expect(new CrashGuard(store).takePrevious()).toBeNull();
  });

  it('writes only when the activity changes and survives a missing storage', () => {
    const store = memoryStore();
    let writes = 0;
    const counting = {
      ...store,
      setItem: (k: string, v: string) => {
        writes++;
        store.setItem(k, v);
      },
    };
    const guard = new CrashGuard(counting);
    guard.update({ stage: 'image' });
    guard.update({ stage: 'image' });
    guard.update({ stage: 'image', backend: 'wasm' });
    expect(writes).toBe(2);
    const without = new CrashGuard(null);
    without.update({ stage: 'model' });
    expect(without.takePrevious()).toBeNull();
  });
});
