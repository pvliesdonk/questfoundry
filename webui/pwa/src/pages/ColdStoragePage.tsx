import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { listArtifacts } from '../api/artifacts';
import type { Artifact } from '../types/api';
import ArtifactList from '../components/artifacts/ArtifactList';
import { getErrorMessage } from '../utils/errorMessage';

export default function ColdStoragePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (projectId) {
      loadArtifacts();
    }
  }, [projectId]);

  async function loadArtifacts() {
    if (!projectId) return;

    try {
      setLoading(true);
      setError(null);
      const data = await listArtifacts(projectId, 'cold');
      setArtifacts(data);
    } catch (err: unknown) {
      setError(getErrorMessage(err, 'Failed to load cold artifacts'));
    } finally {
      setLoading(false);
    }
  }

  if (!projectId) {
    return (
      <div className="text-center py-12">
        <div className="text-xl text-gray-600">Project not found.</div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="text-xl text-gray-600">Loading cold storage...</div>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-8">
        <Link to="/projects" className="text-blue-600 hover:underline mb-2 inline-block">
          ← Back to Projects
        </Link>
        <h1 className="text-3xl font-bold mb-2">
          Cold Storage ❄️
        </h1>
        <p className="text-gray-600">
          Validated, immutable artifacts ready for export
        </p>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {artifacts.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-lg shadow p-6">
          <p className="text-xl text-gray-600 mb-4">
            No validated artifacts in cold storage
          </p>
          <p className="text-sm text-gray-500">
            Create drafts in hot workspace and run gatecheck to validate them
          </p>
        </div>
      ) : (
        <ArtifactList
          artifacts={artifacts}
          storage="cold"
          projectId={projectId}
          onRefresh={loadArtifacts}
        />
      )}
    </div>
  );
}
