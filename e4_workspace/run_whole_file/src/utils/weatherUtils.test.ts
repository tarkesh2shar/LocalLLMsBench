import { describe, it, expect } from 'vitest';
import {
  getWeatherIcon,
  getWeatherDescription,
  convertTemperature,
  formatTemperature
} from './weatherUtils';

describe('Weather Utilities', () => {
  describe('getWeatherIcon', () => {
    it('should return correct icon for weather code 0', () => {
      expect(getWeatherIcon(0)).toBe('☀️');
    });

    it('should return correct icon for weather code 51', () => {
      expect(getWeatherIcon(51)).toBe('🌦️');
    });

    it('should return default icon for unknown code', () => {
      expect(getWeatherIcon(999)).toBe('❓');
    });
  });

  describe('getWeatherDescription', () => {
    it('should return correct description for weather code 0', () => {
      expect(getWeatherDescription(0)).toBe('Clear sky');
    });

    it('should return correct description for weather code 51', () => {
      expect(getWeatherDescription(51)).toBe('Light drizzle');
    });

    it('should return "Unknown" for unknown code', () => {
      expect(getWeatherDescription(999)).toBe('Unknown');
    });
  });

  describe('convertTemperature', () => {
    it('should convert Celsius to Fahrenheit', () => {
      expect(convertTemperature(0, 'fahrenheit')).toBe(32);
      expect(convertTemperature(25, 'fahrenheit')).toBe(77);
    });

    it('should return Celsius when unit is celsius', () => {
      expect(convertTemperature(32, 'celsius')).toBe(32);
    });
  });

  describe('formatTemperature', () => {
    it('should format Celsius temperature', () => {
      expect(formatTemperature(25, 'celsius')).toBe('25°C');
    });

    it('should format Fahrenheit temperature', () => {
      expect(formatTemperature(25, 'fahrenheit')).toBe('77°F');
    });
  });
});