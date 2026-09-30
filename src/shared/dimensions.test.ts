import { describe, expect, it } from 'vitest';
import { computeLimits, estimateWorkingMemory, exceedsInputLimit, fitWithin, planOutputSize } from './dimensions';

describe('computeLimits', () => {
  it('respects the iOS canvas area limit', () => {
    const limits = computeLimits({ isMobile: true, isIOS: true });
    expect(limits.maxOutputPixels).toBeLessThanOrEqual(16_777_216);
  });

  it('gives desktops with more memory larger outputs', () => {
    const small = computeLimits({ isMobile: false, isIOS: false, deviceMemoryGB: 2 });
    const unknown = computeLimits({ isMobile: false, isIOS: false });
    const large = computeLimits({ isMobile: false, isIOS: false, deviceMemoryGB: 8 });
    expect(small.maxOutputPixels).toBeLessThan(unknown.maxOutputPixels);
    expect(unknown.maxOutputPixels).toBeLessThan(large.maxOutputPixels);
  });

  it('keeps typical phone photos at full size', () => {
    const phone = computeLimits({ isMobile: true, isIOS: false, deviceMemoryGB: 8 });
    expect(planOutputSize(4032, 3024, phone).scaled).toBe(false);
    const iphone = computeLimits({ isMobile: true, isIOS: true });
    expect(planOutputSize(4032, 3024, iphone).scaled).toBe(false);
  });

  it('reduces results to about 6 MP in low-memory mode', () => {
    const limits = computeLimits({ isMobile: true, isIOS: true, lowMemory: true });
    const plan = planOutputSize(4032, 3024, limits);
    expect(plan.scaled).toBe(true);
    expect(plan.width * plan.height).toBeLessThanOrEqual(6_000_000);
    expect(computeLimits({ isMobile: false, isIOS: false, deviceMemoryGB: 2, lowMemory: true }).maxOutputPixels).toBe(6_000_000);
  });
});

describe('planOutputSize', () => {
  const limits = { maxInputPixels: 100e6, maxOutputPixels: 12e6, maxOutputSide: 8192 };

  it('keeps images within the limits unchanged', () => {
    expect(planOutputSize(4000, 3000, limits)).toEqual({ width: 4000, height: 3000, scaled: false });
  });

  it('scales large images down, keeping the aspect ratio', () => {
    const plan = planOutputSize(8000, 6000, limits);
    expect(plan.scaled).toBe(true);
    expect(plan.width * plan.height).toBeLessThanOrEqual(12e6);
    expect(plan.width / plan.height).toBeCloseTo(8000 / 6000, 2);
  });

  it('limits the longest side of panoramas', () => {
    const plan = planOutputSize(20000, 1000, limits);
    expect(plan.width).toBeLessThanOrEqual(8192);
    expect(plan.height).toBeGreaterThan(0);
  });

  it('rejects invalid sizes', () => {
    expect(() => planOutputSize(0, 10, limits)).toThrow(RangeError);
  });
});

describe('helpers', () => {
  it('detects oversized inputs', () => {
    expect(exceedsInputLimit(10000, 10001, { maxInputPixels: 100e6, maxOutputPixels: 1, maxOutputSide: 1 })).toBe(true);
    expect(exceedsInputLimit(4000, 3000, { maxInputPixels: 100e6, maxOutputPixels: 1, maxOutputSide: 1 })).toBe(false);
  });

  it('fits sizes into a box', () => {
    expect(fitWithin(4000, 2000, 1000)).toEqual({ width: 1000, height: 500 });
    expect(fitWithin(300, 200, 1000)).toEqual({ width: 300, height: 200 });
  });

  it('estimates memory proportional to pixels', () => {
    expect(estimateWorkingMemory(4000, 3000)).toBeGreaterThan(estimateWorkingMemory(2000, 1500));
    expect(estimateWorkingMemory(8000, 6000)).toBeGreaterThan(48e6 * 4);
  });
});
