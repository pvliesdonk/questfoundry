import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { listProjects, createProject } from '../api/projects';
import { Project, ProjectInfo } from '../types/api';
import ProjectCard from '../components/projects/ProjectCard';
import CreateProjectModal from '../components/projects/CreateProjectModal';

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);

  useEffect(() => {
    loadProjects();
  }, []);

  async function loadProjects() {
    try {
      setLoading(true);
      setError(null);
      const data = await listProjects();
      setProjects(data);
    } catch (err: any) {
      setError(err.detail || 'Failed to load projects');
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate(info: ProjectInfo) {
    await createProject(info);
    await loadProjects();
    setShowCreate(false);
  }

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="text-xl text-gray-600">Loading projects...</div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">My Projects</h1>
        <button
          onClick={() => setShowCreate(true)}
          className="px-4 py-2 bg-primary text-white rounded hover:bg-blue-600"
        >
          + New Project
        </button>
      </div>

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
            onClick={() => setShowCreate(true)}
            className="text-primary hover:underline"
          >
            Create your first project →
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map(project => (
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
