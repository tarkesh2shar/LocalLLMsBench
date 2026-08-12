import React from 'react';
import { WeatherData } from '../types';
import { getWeatherIcon, getWeatherDescription, formatTemperature, formatDate } from '../utils/weatherUtils';

interface ForecastCardProps {
  weatherData: WeatherData;
  temperatureUnit: 'celsius' | 'fahrenheit';
}

const ForecastCard: React.FC<ForecastCardProps> = ({ weatherData, temperatureUnit }) => {
  const daily = weatherData.daily;
  
  // Get the next 7 days of forecast data
  const forecastDays = daily.time.slice(0, 7).map((date, index) => ({
    date,
    maxTemp: daily.temperature_2m_max[index],
    minTemp: daily.temperature_2m_min[index],
    weatherCode: daily.weathercode[index],
    sunrise: daily.sunrise[index],
    sunset: daily.sunset[index],
    precipitation: daily.precipitation_probability_mean[index],
    windSpeed: daily.wind_speed_10m_max[index]
  }));

  return (
    <div className="forecast-card">
      <h2>7-Day Forecast</h2>
      <div className="forecast-grid">
        {forecastDays.map((day, index) => (
          <div key={index} className="forecast-day">
            <div className="forecast-date">{formatDate(day.date)}</div>
            <div className="forecast-icon">
              {getWeatherIcon(day.weatherCode)}
            </div>
            <div className="forecast-description">
              {getWeatherDescription(day.weatherCode)}
            </div>
            <div className="forecast-temps">
              <span className="high-temp">
                {formatTemperature(day.maxTemp, temperatureUnit)}
              </span>
              <span className="low-temp">
                {formatTemperature(day.minTemp, temperatureUnit)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ForecastCard;