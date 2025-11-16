import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

export default function Navigation() {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  function isActive(path: string) {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  }

  const navClass = (path: string, mobile = false) => {
    const base = mobile
      ? 'block px-3 py-2 rounded-md text-base font-medium'
      : 'px-3 py-2 rounded-md text-sm font-medium';

    return `${base} transition-colors ${
      isActive(path)
        ? 'bg-blue-600 text-white'
        : 'text-gray-700 hover:bg-gray-200'
    }`;
  };

  return (
    <nav className="bg-white shadow-sm sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo and desktop nav */}
          <div className="flex items-center space-x-4">
            <Link to="/" className="text-xl font-bold text-primary">
              QuestFoundry
            </Link>
            <div className="hidden md:flex space-x-2">
              <Link to="/projects" className={navClass('/projects')}>
                Projects
              </Link>
              <Link to="/settings" className={navClass('/settings')}>
                Settings
              </Link>
            </div>
          </div>

          {/* User info */}
          <div className="hidden md:flex items-center">
            <span className="text-sm text-gray-600">
              User: dev-user
            </span>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-md text-gray-700 hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-primary"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {mobileMenuOpen && (
          <div className="md:hidden pb-3 border-t border-gray-200">
            <div className="space-y-1 pt-2">
              <Link
                to="/projects"
                className={navClass('/projects', true)}
                onClick={() => setMobileMenuOpen(false)}
              >
                Projects
              </Link>
              <Link
                to="/settings"
                className={navClass('/settings', true)}
                onClick={() => setMobileMenuOpen(false)}
              >
                Settings
              </Link>
              <div className="px-3 py-2 text-sm text-gray-600 border-t border-gray-200 mt-2">
                User: dev-user
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
