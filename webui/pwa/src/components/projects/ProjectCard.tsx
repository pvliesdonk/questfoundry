import { Link } from 'react-router-dom';
import { deleteProject } from '../../api/projects';
import { Project } from '../../types/api';

interface Props {
  project: Project;
  onDelete: () => void;
}

export default function ProjectCard({ project, onDelete }: Props) {
  async function handleDelete() {
    if (!confirm('Delete this project? This cannot be undone.')) {
      return;
    }

    try {
      await deleteProject(project.project_id);
      onDelete();
    } catch (error) {
      console.error('Failed to delete project:', error);
      alert('Failed to delete project');
    }
  }

  return (
    <div className="bg-white rounded-lg shadow hover:shadow-lg transition p-6">
      <Link to={`/projects/${project.project_id}`}>
        <h3 className="text-xl font-semibold mb-2 text-primary hover:underline">
          {project.name}
        </h3>
      </Link>
      {project.description && (
        <p className="text-gray-600 mb-4">{project.description}</p>
      )}
      <div className="flex items-center justify-between text-sm text-gray-500">
        <span>v{project.version || '1.0.0'}</span>
        <button
          onClick={handleDelete}
          className="text-red-600 hover:text-red-800"
        >
          Delete
        </button>
      </div>
    </div>
  );
}
