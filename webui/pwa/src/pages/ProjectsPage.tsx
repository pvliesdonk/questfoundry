import { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { listProjects, createProject } from '../api/projects';
import type { Project, ProjectInfo } from '../types/api';
import ProjectCard from '../components/projects/ProjectCard';
import CreateProjectModal from '../components/projects/CreateProjectModal';
import { getErrorMessage } from '../utils/errorMessage';

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    loadProjects();
  }, []);

  async function loadProjects() {
    try {
      setLoading(true);
      setError(null);
      const data = await listProjects();
      setProjects(data);
    } catch (err: unknown) {
      setError(getErrorMessage(err, 'Failed to load projects'));
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate(info: ProjectInfo) {
    await createProject(info);
    await loadProjects();
    setShowCreate(false);
  }

  // Filter projects based on search query
  const filteredProjects = useMemo(() => {
    if (!searchQuery.trim()) return projects;

    const query = searchQuery.toLowerCase();
    return projects.filter(project =>
      project.name.toLowerCase().includes(query) ||
      project.description?.toLowerCase().includes(query) ||
      project.version?.toLowerCase().includes(query)
    );
  }, [projects, searchQuery]);

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="text-xl text-gray-600">Loading projects...</div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold">My Projects</h1>
          <p className="text-gray-600 mt-1">{projects.length} total projects</p>
        </div>
        <button
          type="button"
          onClick={() => setShowCreate(true)}
          className="px-4 py-2 bg-primary text-white rounded hover:bg-blue-600"
        >
          + New Project
        </button>
      </div>

      {/* Search bar */}
      {projects.length > 0 && (
        <div className="mb-6">
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search projects by name, description, or version..."
              className="w-full px-4 py-3 pl-10 border rounded-lg focus:ring-2 focus:ring-primary"
            />
            <svg
              className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <title>Search projects</title>
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            )}
          </div>
          {searchQuery && (
            <p className="text-sm text-gray-600 mt-2">
              Found {filteredProjects.length} project{filteredProjects.length !== 1 ? 's' : ''}
            </p>
          )}
        </div>
      )}

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {projects.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-lg shadow p-6">
          <p className="text-xl text-gray-600 mb-4">
            No projects yet
          </p>
          <button
            type="button"
            onClick={() => setShowCreate(true)}
            className="text-primary hover:underline"
          >
            Create your first project →
          </button>
        </div>
      ) : filteredProjects.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-lg shadow p-6">
          <p className="text-xl text-gray-600 mb-4">
            No projects match "{searchQuery}"
          </p>
          <button
            type="button"
            onClick={() => setSearchQuery('')}
            className="text-primary hover:underline"
          >
            Clear search →
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredProjects.map(project => (
            <div key={project.project_id} className="bg-white rounded-lg shadow hover:shadow-lg transition p-6">
              <Link to={`/projects/${project.project_id}/execute`}>
                <h3 className="text-xl font-semibold mb-2 text-primary hover:underline">
                  {project.name}
                </h3>
              </Link>
              {project.description && (
                <p className="text-gray-600 mb-4 text-sm">{project.description}</p>
              )}
              <div className="text-xs text-gray-500 mb-4">
                Version: {project.version || '1.0.0'}
              </div>
              <div className="flex flex-col space-y-2">
                <Link
                  to={`/projects/${project.project_id}/execute`}
                  className="px-4 py-2 bg-primary text-white rounded text-center hover:bg-blue-600"
                >
                  Execute Goal
                </Link>
                <Link
                  to={`/projects/${project.project_id}/hot`}
                  className="px-4 py-2 border border-gray-300 rounded text-center hover:bg-gray-50"
                >
                  Hot Workspace 🔥
                </Link>
                <Link
                  to={`/projects/${project.project_id}/cold`}
                  className="px-4 py-2 border border-gray-300 rounded text-center hover:bg-gray-50"
                >
                  Cold Storage ❄️
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}

      {showCreate && (
        <CreateProjectModal
          onClose={() => setShowCreate(false)}
          onCreate={handleCreate}
        />
      )}
    </div>
  );
}
