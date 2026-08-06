import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import WeatherCard from './WeatherCard';

describe('WeatherCard', () => {
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

  it('renders temperature and weather info', () => {
    render(
      <WeatherCard 
        weatherData={mockWeatherData} 
        temperatureUnit="celsius" 
      />
    );
    
    expect(screen.getByText(/15°c/i, { selector: '.temperature' })).toBeInTheDocument();
    expect(screen.getByText(/clear sky/i)).toBeInTheDocument();
  });
});