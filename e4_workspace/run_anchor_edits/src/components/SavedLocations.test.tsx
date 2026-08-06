import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SavedLocations from './SavedLocations';

describe('SavedLocations', () => {
  const mockLocations = [
    {
      id: '1',
      name: 'London',
      latitude: 51.5074,
      longitude: -0.1278,
      country: 'United Kingdom',
      admin1: 'England'
    },
    {
      id: '2',
      name: 'New York',
      latitude: 40.7128,
      longitude: -74.0060,
      country: 'United States',
      admin1: 'New York'
    }
  ];

  const mockOnLocationSelect = vi.fn();
  const mockOnRemoveLocation = vi.fn();

  it('renders saved locations', () => {
    render(
      <SavedLocations 
        locations={mockLocations} 
        onLocationSelect={mockOnLocationSelect} 
        onRemoveLocation={mockOnRemoveLocation} 
      />
    );
    
    expect(screen.getByText(/saved locations/i)).toBeInTheDocument();
    expect(screen.getByText(/london/i, { selector: '.location-button' })).toBeInTheDocument();
    expect(screen.getByText(/new york/i, { selector: '.location-button' })).toBeInTheDocument();
  });

  it('renders empty state when no locations', () => {
    render(
      <SavedLocations 
        locations={[]} 
        onLocationSelect={mockOnLocationSelect} 
        onRemoveLocation={mockOnRemoveLocation} 
      />
    );
    
    expect(screen.getByText(/no saved locations/i)).toBeInTheDocument();
  });

  it('calls onLocationSelect when location is clicked', async () => {
    const user = userEvent.setup();
    render(
      <SavedLocations 
        locations={mockLocations} 
        onLocationSelect={mockOnLocationSelect} 
        onRemoveLocation={mockOnRemoveLocation} 
      />
    );
    
    const locationButton = screen.getByText(/london/i).closest('button');
    expect(locationButton).toBeInTheDocument();
    
    await user.click(locationButton!);
    expect(mockOnLocationSelect).toHaveBeenCalled();
  });

  it('calls onRemoveLocation when remove button is clicked', async () => {
    const user = userEvent.setup();
    render(
      <SavedLocations 
        locations={mockLocations} 
        onLocationSelect={mockOnLocationSelect} 
        onRemoveLocation={mockOnRemoveLocation} 
      />
    );
    
    const removeButton = screen.getAllByText(/remove/i, { selector: '.remove-button' })[0].closest('button');
    expect(removeButton).toBeInTheDocument();
    
    await user.click(removeButton!);
    expect(mockOnRemoveLocation).toHaveBeenCalledWith('1');
  });
});