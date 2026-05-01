export type Theme = 'light' | 'dark'

export function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }): JSX.Element {
  const isDark = theme === 'dark'
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={onToggle}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {isDark ? (
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path
            d="M11.5 10.2A5.2 5.2 0 0 1 5.8 4.5c0-.7.13-1.36.37-1.97A5.5 5.5 0 1 0 13.47 9.83a5.2 5.2 0 0 1-1.97.37Z"
            fill="currentColor"
          />
        </svg>
      ) : (
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <circle cx="8" cy="8" r="3" fill="currentColor" />
          <g stroke="currentColor" strokeWidth="1.2" strokeLinecap="round">
            <line x1="8" y1="1.5" x2="8" y2="3.2" />
            <line x1="8" y1="12.8" x2="8" y2="14.5" />
            <line x1="1.5" y1="8" x2="3.2" y2="8" />
            <line x1="12.8" y1="8" x2="14.5" y2="8" />
            <line x1="3.4" y1="3.4" x2="4.6" y2="4.6" />
            <line x1="11.4" y1="11.4" x2="12.6" y2="12.6" />
            <line x1="3.4" y1="12.6" x2="4.6" y2="11.4" />
            <line x1="11.4" y1="4.6" x2="12.6" y2="3.4" />
          </g>
        </svg>
      )}
    </button>
  )
}

export default ThemeToggle
