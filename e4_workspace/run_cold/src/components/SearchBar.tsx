import React, { useState, useEffect, useRef } from 'react';
import { GeocodingResult } from '../types';
import { WeatherService } from '../services/weatherService';

interface SearchBarProps {
  onCitySelect: (city: GeocodingResult) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ onCitySelect }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<GeocodingResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as any)) {
        setShowResults(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  useEffect(() => {
    let timeoutId: NodeJS.Timeout;
    
    if (query.length > 1) {
      setIsLoading(true);
      timeoutId = setTimeout(async () => {
        try {
          const data = await WeatherService.searchCities(query);
          setResults(data);
        } catch (error) {
          setResults([]);
        } finally {
          setIsLoading(false);
        }
      }, 300);
    } else {
      setResults([]);
    }

    return () => clearTimeout(timeoutId);
  }, [query]);

  const handleSelectCity = (city: GeocodingResult) => {
    onCitySelect(city);
    setQuery('');
    setResults([]);
    setShowResults(false);
  };

  // const getDisplayLocation = (city: GeocodingResult): string => {
  //   const parts = [city.name];
  //   if (city.admin1) parts.push(city.admin1);
  //   if (city.country) parts.push(city.country);
  //   return parts.join(', ');
  // };

  return (
    <div className="search-bar" ref={searchRef}>
      <div className="search-input-container">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setShowResults(true)}
          placeholder="Search for a city..."
          className="search-input"
          aria-label="Search for a city"
        />
        {isLoading && <div className="loading-spinner"></div>}
      </div>
      
      {showResults && results.length > 0 && (
        <div className="search-results">
          {results.map((city) => (
            <div
              key={city.id}
              className="search-result-item"
              onClick={() => handleSelectCity(city)}
            >
              <div className="city-name">{city.name}</div>
              <div className="city-details">
                {city.admin1 && <span>{city.admin1}, </span>}
                {city.country}
              </div>
            </div>
          ))}
        </div>
      )}
      
      {showResults && query.length > 0 && results.length === 0 && !isLoading && (
        <div className="search-results empty">
          No cities found
        </div>
      )}
    </div>
  );
};

export default SearchBar;