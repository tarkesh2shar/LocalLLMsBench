import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TemperatureToggle from './TemperatureToggle';

describe('TemperatureToggle', () => {
  const mockOnToggle = vi.fn();

  it('renders temperature toggle button', () => {
    render(<TemperatureToggle unit="celsius" onToggle={mockOnToggle} />);
    const button = screen.getByRole('button', { name: /switch to fahrenheit temperature/i });
    expect(button).toBeInTheDocument();
  });

  it('calls onToggle when clicked', async () => {
    const user = userEvent.setup();
    render(<TemperatureToggle unit="celsius" onToggle={mockOnToggle} />);
    const button = screen.getByRole('button', { name: /switch to fahrenheit temperature/i });
    
    await user.click(button);
    expect(mockOnToggle).toHaveBeenCalled();
  });
});