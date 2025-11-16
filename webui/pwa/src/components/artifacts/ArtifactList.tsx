import { Artifact } from '../../types/api';
import { StorageBackend } from '../../api/artifacts';
import ArtifactCard from './ArtifactCard';

interface Props {
  artifacts: Artifact[];
  storage: StorageBackend;
  projectId: string;
  onRefresh: () => void;
}

export default function ArtifactList({ artifacts, storage, projectId, onRefresh }: Props) {
  if (artifacts.length === 0) {
    return (
      <div className="text-center text-gray-500 py-12">
        <p>No artifacts in {storage} storage</p>
      </div>
    );
  }

  // Group artifacts by type
  const grouped = artifacts.reduce((acc, artifact) => {
    const type = artifact.type;
    if (!acc[type]) {
      acc[type] = [];
    }
    acc[type].push(artifact);
    return acc;
  }, {} as Record<string, Artifact[]>);

  return (
    <div className="space-y-8">
      {Object.entries(grouped).map(([type, items]) => (
        <div key={type}>
          <h2 className="text-xl font-semibold mb-4 capitalize">
            {type.replace(/_/g, ' ')}s ({items.length})
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {items.map((artifact, idx) => (
              <ArtifactCard
                key={artifact.metadata?.id || `${type}-${idx}`}
                artifact={artifact}
                storage={storage}
                projectId={projectId}
                onRefresh={onRefresh}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
