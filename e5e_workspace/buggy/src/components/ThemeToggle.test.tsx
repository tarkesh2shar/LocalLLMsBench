import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ThemeToggle from './ThemeToggle';

describe('ThemeToggle', () => {
  const mockOnToggle = vi.fn();

  it('renders theme toggle button', () => {
    render(<ThemeToggle theme="light" onToggle={mockOnToggle} />);
    const button = screen.getByRole('button', { name: /switch to dark mode/i });
    expect(button).toBeInTheDocument();
  });

  it('calls onToggle when clicked', async () => {
    const user = userEvent.setup();
    render(<ThemeToggle theme="light" onToggle={mockOnToggle} />);
    const button = screen.getByRole('button', { name: /switch to dark mode/i });
    
    await user.click(button);
    expect(mockOnToggle).toHaveBeenCalled();
  });
});