import { describe, it, expect } from 'vitest';
import { getWindDirection } from './windDirection';

describe('getWindDirection', () => {
  it('maps cardinal points', () => {
    expect(getWindDirection(0)).toBe('N');
    expect(getWindDirection(90)).toBe('E');
    expect(getWindDirection(180)).toBe('S');
    expect(getWindDirection(270)).toBe('W');
  });
  it('maps intercardinal points', () => {
    expect(getWindDirection(45)).toBe('NE');
    expect(getWindDirection(135)).toBe('SE');
    expect(getWindDirection(225)).toBe('SW');
    expect(getWindDirection(315)).toBe('NW');
  });
  it('rounds to the nearest of 8 sectors', () => {
    expect(getWindDirection(20)).toBe('N');
    expect(getWindDirection(30)).toBe('NE');
    expect(getWindDirection(200)).toBe('S');
  });
  it('wraps at 360 and handles values above it', () => {
    expect(getWindDirection(360)).toBe('N');
    expect(getWindDirection(370)).toBe('N');
  });
});
