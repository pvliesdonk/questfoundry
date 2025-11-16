import { Artifact } from '../../types/api';
import { StorageBackend } from '../../api/artifacts';

interface Props {
  artifact: Artifact;
  storage: StorageBackend;
  projectId: string;
  onRefresh: () => void;
}

export default function ArtifactCard({ artifact, storage }: Props) {
  const id = artifact.metadata?.id || 'unknown';
  const title = artifact.data?.title || artifact.data?.name || id;
  const description = artifact.data?.description || artifact.data?.summary || '';

  return (
    <div className={`bg-white rounded-lg shadow hover:shadow-lg transition p-4 border-l-4 ${
      storage === 'hot' ? 'border-red-500' : 'border-blue-500'
    }`}>
      <div className="flex items-start justify-between mb-2">
        <h3 className="font-semibold">{title}</h3>
        <span className={`text-xs px-2 py-1 rounded ${
          storage === 'hot'
            ? 'bg-red-100 text-red-700'
            : 'bg-blue-100 text-blue-700'
        }`}>
          {storage}
        </span>
      </div>
      {description && (
        <p className="text-sm text-gray-600 mb-2 line-clamp-2">
          {description}
        </p>
      )}
      <div className="text-xs text-gray-500">
        <span className="font-medium">Type:</span> {artifact.type}
      </div>
      {id !== 'unknown' && (
        <div className="text-xs text-gray-500 mt-1 truncate">
          <span className="font-medium">ID:</span> {id}
        </div>
      )}
    </div>
  );
}
