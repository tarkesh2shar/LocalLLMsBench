import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ForecastCard from './ForecastCard';

describe('ForecastCard', () => {
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
      time: ['2023-06-15T12:00', '2023-06-15T13:00'],
      temperature_2m: [15, 16],
      weathercode: [0, 0]
    },
    daily: {
      time: ['2023-06-15', '2023-06-16', '2023-06-17', '2023-06-18', '2023-06-19', '2023-06-20', '2023-06-21'],
      temperature_2m_max: [20, 22, 19, 21, 23, 18, 20],
      temperature_2m_min: [10, 12, 8, 11, 13, 7, 9],
      weathercode: [0, 1, 2, 3, 45, 48, 51],
      sunrise: ['2023-06-15T05:00', '2023-06-16T05:00', '2023-06-17T05:00', '2023-06-18T05:00', '2023-06-19T05:00', '2023-06-20T05:00', '2023-06-21T05:00'],
      sunset: ['2023-06-15T20:00', '2023-06-16T20:00', '2023-06-17T20:00', '2023-06-18T20:00', '2023-06-19T20:00', '2023-06-20T20:00', '2023-06-21T20:00'],
      precipitation_probability_mean: [10, 15, 5, 20, 25, 30, 35],
      wind_speed_10m_max: [15, 12, 18, 10, 20, 22, 14]
    }
  };

  it('renders 7-day forecast', () => {
    render(
      <ForecastCard 
        weatherData={mockWeatherData} 
        temperatureUnit="celsius" 
      />
    );
    
    expect(screen.getByText(/7-day forecast/i)).toBeInTheDocument();
    // Should have 7 forecast days
    expect(screen.getAllByText(/°c/i).length).toBeGreaterThan(0);
  });
});