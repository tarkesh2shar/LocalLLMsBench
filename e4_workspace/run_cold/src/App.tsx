import React, { useState, useEffect } from 'react';
import SearchBar from './components/SearchBar';
import WeatherCard from './components/WeatherCard';
import ForecastCard from './components/ForecastCard';
import HourlyChart from './components/HourlyChart';
import SavedLocations from './components/SavedLocations';
import ThemeToggle from './components/ThemeToggle';
import TemperatureToggle from './components/TemperatureToggle';
import { WeatherData, GeocodingResult, SavedLocation } from './types';
import { WeatherService } from './services/weatherService';

const App: React.FC = () => {
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [temperatureUnit, setTemperatureUnit] = useState<'celsius' | 'fahrenheit'>('celsius');
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [savedLocations, setSavedLocations] = useState<SavedLocation[]>([]);
  const [currentLocation, setCurrentLocation] = useState<GeocodingResult | null>(null);

  // Load preferences from localStorage on mount
  useEffect(() => {
    const savedUnit = localStorage.getItem('weatherUnit') as 'celsius' | 'fahrenheit' | null;
    const savedTheme = localStorage.getItem('weatherTheme') as 'light' | 'dark' | null;
    const savedLocations = localStorage.getItem('savedLocations');
    
    if (savedUnit) {
      setTemperatureUnit(savedUnit);
    }
    
    if (savedTheme) {
      setTheme(savedTheme);
    }
    
    if (savedLocations) {
      try {
        setSavedLocations(JSON.parse(savedLocations));
      } catch (e) {
        console.error('Failed to parse saved locations', e);
      }
    }
  }, []);

  // Save preferences to localStorage when they change
  useEffect(() => {
    localStorage.setItem('weatherUnit', temperatureUnit);
  }, [temperatureUnit]);

  useEffect(() => {
    localStorage.setItem('weatherTheme', theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem('savedLocations', JSON.stringify(savedLocations));
  }, [savedLocations]);

  // Apply theme to document
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark-theme');
    } else {
      document.documentElement.classList.remove('dark-theme');
    }
  }, [theme]);

  const handleCitySelect = async (city: GeocodingResult) => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await WeatherService.getWeatherData(city.latitude, city.longitude);
      setWeatherData(data);
      setCurrentLocation(city);
    } catch (error: any) {
      setError('Failed to fetch weather data. Please try again.');
      console.error('Error fetching weather data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleTemperature = () => {
    setTemperatureUnit(prev => prev === 'celsius' ? 'fahrenheit' : 'celsius');
  };

  const handleToggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  const handleSaveLocation = () => {
    if (!currentLocation || !weatherData) return;
    
    const newLocation = {
      id: `${currentLocation.latitude}-${currentLocation.longitude}`,
      name: currentLocation.name,
      latitude: currentLocation.latitude,
      longitude: currentLocation.longitude,
      country: currentLocation.country,
      admin1: currentLocation.admin1
    };
    
    // Check if location already exists
    const exists = savedLocations.some(loc => 
      loc.latitude === newLocation.latitude && 
      loc.longitude === newLocation.longitude
    );
    
    if (!exists) {
      setSavedLocations(prev => [...prev, newLocation]);
    }
  };

  const handleRemoveLocation = (id: string) => {
    setSavedLocations(prev => prev.filter(loc => loc.id !== id));
  };

  const handleLoadSavedLocation = (location: SavedLocation) => {
    setLoading(true);
    setError(null);
    
    // Simulate loading weather data for saved location
    // In a real app, we would fetch the weather data for this location
    // For now, we'll just set the location and show a loading state
    setCurrentLocation({
      id: 0,
      name: location.name,
      latitude: location.latitude,
      longitude: location.longitude,
      country_code: location.country,
      country: location.country,
      admin1: location.admin1 || '',
      admin2: '',
      admin3: '',
      admin4: '',
      timezone: '',
      population: 0,
      country_id: 0,
      elevation: 0,
      feature_code: ''
    } as GeocodingResult);
    
    // Simulate API call
    setTimeout(() => {
      WeatherService.getWeatherData(location.latitude, location.longitude)
        .then(data => {
          setWeatherData(data);
          setLoading(false);
        })
        .catch(error => {
          setError('Failed to fetch weather data. Please try again.');
          setLoading(false);
        });
    }, 500);
  };

  return (
    <div className={`app ${theme === 'dark' ? 'dark-theme' : 'light-theme'}`}>
      <header className="app-header">
        <h1>Weather Workbench</h1>
        <div className="header-controls">
          <TemperatureToggle 
            unit={temperatureUnit} 
            onToggle={handleToggleTemperature} 
          />
          <ThemeToggle theme={theme} onToggle={handleToggleTheme} />
        </div>
      </header>
      
      <main className="app-main">
        <div className="search-section">
          <SearchBar onCitySelect={handleCitySelect} />
          {currentLocation && (
            <button 
              className="save-location-btn"
              onClick={handleSaveLocation}
              disabled={savedLocations.some(loc => 
                loc.latitude === currentLocation.latitude && 
                loc.longitude === currentLocation.longitude
              )}
            >
              Save Location
            </button>
          )}
        </div>
        
        {error && (
          <div className="error-message">
            <p>{error}</p>
            <button onClick={() => setError(null)}>Dismiss</button>
          </div>
        )}
        
        {loading && (
          <div className="loading-skeleton">
            <div className="skeleton-card"></div>
            <div className="skeleton-card"></div>
            <div className="skeleton-card"></div>
          </div>
        )}
        
        {weatherData && currentLocation && !loading && (
          <div className="weather-content">
            <div className="current-weather-section">
              <WeatherCard 
                weatherData={weatherData} 
                temperatureUnit={temperatureUnit} 
              />
            </div>
            
            <div className="forecast-section">
              <ForecastCard 
                weatherData={weatherData} 
                temperatureUnit={temperatureUnit} 
              />
            </div>
            
            <div className="hourly-section">
              <HourlyChart 
                weatherData={weatherData} 
                temperatureUnit={temperatureUnit} 
              />
            </div>
          </div>
        )}
        
        <div className="saved-locations-section">
          <SavedLocations 
            locations={savedLocations} 
            onLocationSelect={handleLoadSavedLocation} 
            onRemoveLocation={handleRemoveLocation} 
          />
        </div>
      </main>
    </div>
  );
};

export default App;