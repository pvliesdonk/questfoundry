import React from 'react';
import { Link } from 'react-router-dom';

const HomePage: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center py-12">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
          Welcome to QuestFoundry
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 mb-8">
          AI-Powered Interactive Fiction Authoring Studio
        </p>

        <div className="grid md:grid-cols-2 gap-6 mt-12">
          <Link
            to="/projects"
            className="card hover:shadow-lg transition-shadow cursor-pointer"
          >
            <div className="text-center">
              <div className="text-4xl mb-4">📁</div>
              <h2 className="text-2xl font-semibold mb-2">Projects</h2>
              <p className="text-gray-600 dark:text-gray-400">
                Create and manage your interactive fiction projects
              </p>
            </div>
          </Link>

          <Link
            to="/settings"
            className="card hover:shadow-lg transition-shadow cursor-pointer"
          >
            <div className="text-center">
              <div className="text-4xl mb-4">⚙️</div>
              <h2 className="text-2xl font-semibold mb-2">Settings</h2>
              <p className="text-gray-600 dark:text-gray-400">
                Configure your AI provider keys (BYOK)
              </p>
            </div>
          </Link>
        </div>

        <div className="mt-16 text-left">
          <h3 className="text-2xl font-semibold mb-4">Getting Started</h3>
          <ol className="list-decimal list-inside space-y-2 text-gray-600 dark:text-gray-400">
            <li>Configure your AI provider keys in Settings (optional but recommended)</li>
            <li>Create a new project from the Projects page</li>
            <li>Use natural language to create story elements via Goal Execution</li>
            <li>Review drafts in Hot Workspace (ephemeral storage)</li>
            <li>Run quality validation with Gatecheck</li>
            <li>Validated artifacts are promoted to Cold Storage (permanent)</li>
          </ol>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
