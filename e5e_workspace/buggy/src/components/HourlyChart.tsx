import React from 'react';
import { WeatherData } from '../types';
import { getWeatherIcon, formatTemperature, formatTime } from '../utils/weatherUtils';

interface HourlyChartProps {
  weatherData: WeatherData;
  temperatureUnit: 'celsius' | 'fahrenheit';
}

const HourlyChart: React.FC<HourlyChartProps> = ({ weatherData, temperatureUnit }) => {
  const hourly = weatherData.hourly;
  
  // Get the next 24 hours of data
  const hourlyData = hourly.time.slice(0, 24).map((time, index) => ({
    time,
    temperature: hourly.temperature_2m[index],
    weatherCode: hourly.weathercode[index]
  }));

  // Find min and max temperatures for scaling
  const temps = hourlyData.map(d => d.temperature);
  const minTemp = Math.min(...temps);
  const maxTemp = Math.max(...temps);
  const tempRange = maxTemp - minTemp || 1; // Avoid division by zero

  // Calculate chart dimensions
  const chartHeight = 150;
  const chartWidth = 400;
  const pointSpacing = chartWidth / (hourlyData.length - 1);

  // Generate SVG path for temperature line
  const generatePath = () => {
    if (hourlyData.length < 2) return '';
    
    const points = hourlyData.map((data, index) => {
      const x = index * pointSpacing;
      const y = chartHeight - ((data.temperature - minTemp) / tempRange) * chartHeight;
      return `${x},${y}`;
    });
    
    return `M ${points.join(' L ')}`;
  };

  return (
    <div className="hourly-chart">
      <h2>24-Hour Forecast</h2>
      <div className="chart-container">
        <svg width={chartWidth} height={chartHeight} className="temperature-chart">
          {/* Grid lines */}
          <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#eee" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
          
          {/* Temperature line */}
          <path
            d={generatePath()}
            fill="none"
            stroke="#3b82f6"
            strokeWidth="2"
            className="temperature-line"
          />
          
          {/* Temperature points */}
          {hourlyData.map((data, index) => {
            const x = index * pointSpacing;
            const y = chartHeight - ((data.temperature - minTemp) / tempRange) * chartHeight;
            return (
              <g key={index}>
                <circle
                  cx={x}
                  cy={y}
                  r="3"
                  fill="#3b82f6"
                  className="temperature-point"
                />
                <text
                  x={x}
                  y={y - 10}
                  textAnchor="middle"
                  className="temperature-label"
                >
                  {formatTemperature(data.temperature, temperatureUnit)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      
      <div className="hourly-list">
        {hourlyData.map((data, index) => (
          <div key={index} className="hourly-item">
            <div className="hour">{formatTime(data.time)}</div>
            <div className="hourly-icon">
              {getWeatherIcon(data.weatherCode)}
            </div>
            <div className="hourly-temp">
              {formatTemperature(data.temperature, temperatureUnit)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HourlyChart;