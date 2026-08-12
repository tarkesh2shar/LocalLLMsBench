import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import HourlyChart from './HourlyChart';

describe('HourlyChart', () => {
  const mockWeatherData = {
    latitude: 51.5074,
    longitude: -0.1278,
    generationtime_ms: 123,
    utc_offset_seconds: 3600,
    timezone: 'Europe/London',
    timezone_abbreviation: 'BST',
    elevation: 0,
    current_weather: {
      temperature: 15,
      windspeed: 10,
      winddirection: 180,
      weathercode: 0,
      is_day: 1,
      time: '2023-06-15T12:00'
    },
    hourly: {
      time: [
        '2023-06-15T12:00', '2023-06-15T13:00', '2023-06-15T14:00', '2023-06-15T15:00',
        '2023-06-15T16:00', '2023-06-15T17:00', '2023-06-15T18:00', '2023-06-15T19:00',
        '2023-06-15T20:00', '2023-06-15T21:00', '2023-06-15T22:00', '2023-06-15T23:00',
        '2023-06-16T00:00', '2023-06-16T01:00', '2023-06-16T02:00', '2023-06-16T03:00',
        '2023-06-16T04:00', '2023-06-16T05:00', '2023-06-16T06:00', '2023-06-16T07:00',
        '2023-06-16T08:00', '2023-06-16T09:00', '2023-06-16T10:00', '2023-06-16T11:00'
      ],
      temperature_2m: [15, 16, 17, 18, 19, 20, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 13, 14, 15, 16, 17, 18, 19, 20],
      weathercode: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    daily: {
      time: ['2023-06-15'],
      temperature_2m_max: [20],
      temperature_2m_min: [10],
      weathercode: [0],
      sunrise: ['2023-06-15T05:00'],
      sunset: ['2023-06-15T20:00'],
      precipitation_probability_mean: [10],
      wind_speed_10m_max: [15]
    }
  };

  it('renders hourly forecast', () => {
    render(
      <HourlyChart 
        weatherData={mockWeatherData} 
        temperatureUnit="celsius" 
      />
    );
    
    expect(screen.getByText(/24-hour forecast/i)).toBeInTheDocument();
    // Should have 24 hourly items
    expect(screen.getAllByText(/°c/i).length).toBeGreaterThan(0);
  });
});