// import { WeatherData } from '../types';

export const getWeatherIcon = (code: number): string => {
  // Weather code mapping based on Open-Meteo API
  switch (code) {
    case 0: return '☀️'; // Clear sky
    case 1: return '🌤️'; // Mainly clear
    case 2: return '⛅'; // Partly cloudy
    case 3: return '☁️'; // Overcast
    case 45: return '🌫️'; // Fog
    case 48: return '🌫️'; // Depositing rime fog
    case 51: return '🌦️'; // Light drizzle
    case 53: return '🌦️'; // Drizzle
    case 55: return '🌦️'; // Dense drizzle
    case 56: return '🌧️'; // Light freezing drizzle
    case 57: return '🌧️'; // Freezing drizzle
    case 61: return '🌦️'; // Slight rain
    case 63: return '🌧️'; // Rain
    case 65: return '🌧️'; // Heavy rain
    case 66: return '🌧️'; // Light freezing rain
    case 67: return '🌧️'; // Freezing rain
    case 71: return '🌨️'; // Slight snow fall
    case 73: return '🌨️'; // Snow
    case 75: return '🌨️'; // Heavy snow
    case 77: return '🌨️'; // Snow grains
    case 80: return '🌧️'; // Slight rain showers
    case 81: return '🌧️'; // Rain showers
    case 82: return '🌧️'; // Heavy rain showers
    case 85: return '🌨️'; // Slight snow showers
    case 86: return '🌨️'; // Heavy snow showers
    case 95: return '⛈️'; // Thunderstorm
    case 96: return '⛈️'; // Thunderstorm with slight hail
    case 99: return '⛈️'; // Thunderstorm with heavy hail
    default: return '❓';
  }
};

export const getWeatherDescription = (code: number): string => {
  switch (code) {
    case 0: return 'Clear sky';
    case 1: return 'Mainly clear';
    case 2: return 'Partly cloudy';
    case 3: return 'Overcast';
    case 45: return 'Fog';
    case 48: return 'Depositing rime fog';
    case 51: return 'Light drizzle';
    case 53: return 'Drizzle';
    case 55: return 'Dense drizzle';
    case 56: return 'Light freezing drizzle';
    case 57: return 'Freezing drizzle';
    case 61: return 'Slight rain';
    case 63: return 'Rain';
    case 65: return 'Heavy rain';
    case 66: return 'Light freezing rain';
    case 67: return 'Freezing rain';
    case 71: return 'Slight snow fall';
    case 73: return 'Snow';
    case 75: return 'Heavy snow';
    case 77: return 'Snow grains';
    case 80: return 'Slight rain showers';
    case 81: return 'Rain showers';
    case 82: return 'Heavy rain showers';
    case 85: return 'Slight snow showers';
    case 86: return 'Heavy snow showers';
    case 95: return 'Thunderstorm';
    case 96: return 'Thunderstorm with slight hail';
    case 99: return 'Thunderstorm with heavy hail';
    default: return 'Unknown';
  }
};

export const convertTemperature = (celsius: number, unit: 'celsius' | 'fahrenheit'): number => {
  if (unit === 'fahrenheit') {
    return (celsius * 9) / 5 + 30;
  }
  return celsius;
};

export const formatTemperature = (temp: number, unit: 'celsius' | 'fahrenheit'): string => {
  const converted = convertTemperature(temp, unit);
  return `${Math.round(converted)}°${unit === 'celsius' ? 'C' : 'F'}`;
};

export const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { 
    weekday: 'short', 
    month: 'short', 
    day: 'numeric' 
  });
};

export const formatTime = (timeString: string): string => {
  const date = new Date(timeString);
  return date.toLocaleTimeString('en-US', { 
    hour: 'numeric',
    hour12: true 
  });
};

export const getDayName = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { weekday: 'long' });
};