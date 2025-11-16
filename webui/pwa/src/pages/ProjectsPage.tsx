import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { listProjects, createProject, deleteProject } from '../api/projects';
import { ProjectResponse, Project } from '../types/api';

const ProjectsPage: React.FC = () => {
  const [projects, setProjects] = useState<ProjectResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
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
  };

  const handleCreate = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const project: Project = {
      name: formData.get('name') as string,
      description: formData.get('description') as string,
      version: formData.get('version') as string || '1.0.0',
    };

    try {
      setCreating(true);
      await createProject(project);
      setShowCreateModal(false);
      await loadProjects();
    } catch (err: any) {
      alert(err.detail || 'Failed to create project');
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (projectId: string) => {
    if (!confirm('Are you sure you want to delete this project? This cannot be undone.')) {
      return;
    }

    try {
      await deleteProject(projectId);
      await loadProjects();
    } catch (err: any) {
      alert(err.detail || 'Failed to delete project');
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="text-xl text-gray-600 dark:text-gray-400">Loading projects...</div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Projects</h1>
        <button
          onClick={() => setShowCreateModal(true)}
          className="btn btn-primary"
        >
          + Create Project
        </button>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {projects.length === 0 ? (
        <div className="text-center py-12 card">
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-4">
            No projects yet
          </p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn btn-primary"
          >
            Create Your First Project
          </button>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map((project) => (
            <div key={project.id} className="card">
              <h3 className="text-xl font-semibold mb-2">{project.name}</h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">
                {project.description || 'No description'}
              </p>
              <div className="text-xs text-gray-500 mb-4">
                Version: {project.version}
              </div>
              <div className="flex flex-col space-y-2">
                <Link
                  to={`/projects/${project.id}/execute`}
                  className="btn btn-primary text-center"
                >
                  Execute Goal
                </Link>
                <Link
                  to={`/projects/${project.id}/hot`}
                  className="btn btn-secondary text-center"
                >
                  Hot Workspace
                </Link>
                <Link
                  to={`/projects/${project.id}/cold`}
                  className="btn btn-secondary text-center"
                >
                  Cold Storage
                </Link>
                <button
                  onClick={() => handleDelete(project.id)}
                  className="btn btn-danger"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-md w-full">
            <h2 className="text-2xl font-bold mb-4">Create New Project</h2>
            <form onSubmit={handleCreate}>
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">
                  Project Name *
                </label>
                <input
                  type="text"
                  name="name"
                  required
                  className="input"
                  placeholder="My Awesome Story"
                />
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">
                  Description
                </label>
                <textarea
                  name="description"
                  rows={3}
                  className="input"
                  placeholder="A brief description of your project"
                />
              </div>
              <div className="mb-6">
                <label className="block text-sm font-medium mb-2">
                  Version
                </label>
                <input
                  type="text"
                  name="version"
                  defaultValue="1.0.0"
                  className="input"
                  placeholder="1.0.0"
                />
              </div>
              <div className="flex space-x-3">
                <button
                  type="submit"
                  disabled={creating}
                  className="btn btn-primary flex-1"
                >
                  {creating ? 'Creating...' : 'Create'}
                </button>
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  disabled={creating}
                  className="btn btn-secondary flex-1"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProjectsPage;
