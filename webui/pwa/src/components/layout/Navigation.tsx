import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navigation: React.FC = () => {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  const navClass = (path: string) =>
    `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive(path)
        ? 'bg-blue-600 text-white'
        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
    }`;

  return (
    <nav className="bg-white dark:bg-gray-800 shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <Link to="/" className="text-xl font-bold text-blue-600 dark:text-blue-400">
              QuestFoundry
            </Link>
            <div className="flex space-x-2">
              <Link to="/projects" className={navClass('/projects')}>
                Projects
              </Link>
              <Link to="/settings" className={navClass('/settings')}>
                Settings
              </Link>
            </div>
          </div>
          <div className="flex items-center">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              User: dev-user
            </span>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
