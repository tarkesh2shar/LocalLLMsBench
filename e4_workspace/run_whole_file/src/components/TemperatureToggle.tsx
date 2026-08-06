import React from 'react';

interface TemperatureToggleProps {
  unit: 'celsius' | 'fahrenheit';
  onToggle: () => void;
}

const TemperatureToggle: React.FC<TemperatureToggleProps> = ({ unit, onToggle }) => {
  return (
    <button
      className="temperature-toggle"
      onClick={onToggle}
      aria-label={`Switch to ${unit === 'celsius' ? 'fahrenheit' : 'celsius'} temperature`}
    >
      {unit === 'celsius' ? '°F' : '°C'}
    </button>
  );
};

export default TemperatureToggle;
