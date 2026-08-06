import React from 'react';
import { SavedLocation } from '../types';

interface SavedLocationsProps {
  locations: SavedLocation[];
  onLocationSelect: (location: SavedLocation) => void;
  onRemoveLocation: (id: string) => void;
}

const SavedLocations: React.FC<SavedLocationsProps> = ({ 
  locations, 
  onLocationSelect, 
  onRemoveLocation 
}) => {
  if (locations.length === 0) {
    return (
      <div className="saved-locations empty">
        <p>No saved locations</p>
      </div>
    );
  }

  return (
    <div className="saved-locations">
      <h2>Saved Locations</h2>
      <div className="locations-list">
        {locations.map((location) => (
          <div key={location.id} className="location-item">
            <button 
              className="location-button"
              onClick={() => onLocationSelect(location)}
            >
              {location.name}
              {location.admin1 && <span className="location-region">{location.admin1}</span>}
              <span className="location-country">{location.country}</span>
            </button>
            <button 
              className="remove-button"
              onClick={() => onRemoveLocation(location.id)}
              aria-label={`Remove ${location.name}`}
            >
              Remove
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SavedLocations;