import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchBar from './SearchBar';

// Mock the WeatherService
vi.mock('../services/weatherService', () => ({
  WeatherService: {
    searchCities: vi.fn(),
  },
}));

describe('SearchBar', () => {
  const mockOnCitySelect = vi.fn();

  it('renders input field', () => {
    render(<SearchBar onCitySelect={mockOnCitySelect} />);
    const input = screen.getByPlaceholderText(/search for a city/i);
    expect(input).toBeInTheDocument();
  });

  it('calls onCitySelect when a city is selected', async () => {
    const user = userEvent.setup();
    const mockResults = [
      {
        id: 1,
        name: 'London',
        latitude: 51.5074,
        longitude: -0.1278,
        country_code: 'GB',
        country: 'United Kingdom',
        admin1: 'England',
        admin2: '',
        admin3: '',
        admin4: '',
        timezone: 'Europe/London',
        population: 8982000,
        country_id: 2635167,
        elevation: 0,
        feature_code: 'PPLC'
      }
    ];

    // Mock the service to return our mock results
    const { WeatherService } = await import('../services/weatherService');
    vi.spyOn(WeatherService, 'searchCities').mockResolvedValue(mockResults as any);

    render(<SearchBar onCitySelect={mockOnCitySelect} />);
    const input = screen.getByPlaceholderText(/search for a city/i);
    
    await user.type(input, 'London');
    
    // Wait for the results to appear
    await waitFor(() => {
      expect(screen.getByText(/london/i)).toBeInTheDocument();
    });
    
    const resultItem = screen.getByText(/london/i);
    await user.click(resultItem);
    
    expect(mockOnCitySelect).toHaveBeenCalledWith(mockResults[0]);
  });
});