import React from 'react';
import { WeatherData } from '../types';
import { getWeatherIcon, getWeatherDescription, formatTemperature } from '../utils/weatherUtils';

interface WeatherCardProps {
  weatherData: WeatherData;
  temperatureUnit: 'celsius' | 'fahrenheit';
}

const WeatherCard: React.FC<WeatherCardProps> = ({ weatherData, temperatureUnit }) => {
  const current = weatherData.current_weather;
  const weatherIcon = getWeatherIcon(current.weathercode);
  const weatherDescription = getWeatherDescription(current.weathercode);

  return (
    <div className="weather-card">
      <div className="current-weather">
        <div className="weather-icon">{weatherIcon}</div>
        <div className="temperature">
          {formatTemperature(current.temperature, temperatureUnit)}
        </div>
        <div className="weather-description">{weatherDescription}</div>
        <div className="weather-details">
          <div className="detail-item">
            <span className="detail-label">Feels like:</span>
            <span className="detail-value">
              {formatTemperature(current.temperature, temperatureUnit)}
            </span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Humidity:</span>
            <span className="detail-value">100%</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Wind:</span>
            <span className="detail-value">{current.windspeed} km/h</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WeatherCard;