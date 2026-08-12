import { GeocodingResult, WeatherData } from '../types';

const GEOCODING_API_URL = 'https://geocoding-api.open-meteo.com/v1/search';
const WEATHER_API_URL = 'https://api.open-meteo.com/v1/forecast';

export class WeatherService {
  static async searchCities(query: string): Promise<GeocodingResult[]> {
    if (!query.trim()) {
      return [];
    }

    try {
      const response = await fetch(
        `${GEOCODING_API_URL}?name=${encodeURIComponent(query)}&count=50&language=en`
      );
      
      if (!response.ok) {
        throw new Error(`Geocoding API error: ${response.status}`);
      }
      
      const data = await response.json();
      return data.results || [];
    } catch (error) {
      console.error('Error searching cities:', error);
      throw new Error('Failed to search cities');
    }
  }

  static async getWeatherData(
    latitude: number,
    longitude: number
  ): Promise<WeatherData> {
    try {
      const response = await fetch(
        `${WEATHER_API_URL}?latitude=${latitude}&longitude=${longitude}&current_weather=true&hourly=temperature_2m,weathercode&daily=temperature_2m_max,temperature_2m_min,weathercode,sunrise,sunset,precipitation_probability_mean,wind_speed_10m_max&timezone=auto`
      );
      
      if (!response.ok) {
        throw new Error(`Weather API error: ${response.status}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching weather data:', error);
      throw new Error('Failed to fetch weather data');
    }
  }
}