import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

// Mock the WeatherService
vi.mock('./services/weatherService', () => ({
  WeatherService: {
    searchCities: vi.fn(),
    getWeatherData: vi.fn(),
  },
}));

describe('App', () => {
  it('renders without crashing', () => {
    render(<App />);
    expect(screen.getByText(/Weather Workbench/i)).toBeInTheDocument();
  });

  it('toggles temperature unit', async () => {
    const user = userEvent.setup();
    render(<App />);
    
    const toggleButton = screen.getByRole('button', { name: /switch to fahrenheit temperature/i });
    expect(toggleButton).toBeInTheDocument();
    
    await user.click(toggleButton);
    expect(toggleButton).toHaveTextContent('°C');
  });

  it('toggles theme', async () => {
    const user = userEvent.setup();
    render(<App />);
    
    const toggleButton = screen.getByRole('button', { name: /switch to dark mode/i });
    expect(toggleButton).toBeInTheDocument();
    
    await user.click(toggleButton);
    expect(toggleButton).toHaveTextContent('☀️');
  });
});