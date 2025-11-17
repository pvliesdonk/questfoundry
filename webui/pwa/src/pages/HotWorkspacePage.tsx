import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { listArtifacts } from '../api/artifacts';
import type { Artifact } from '../types/api';
import ArtifactList from '../components/artifacts/ArtifactList';
import { getErrorMessage } from '../utils/errorMessage';

export default function HotWorkspacePage() {
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
      const data = await listArtifacts(projectId, 'hot');
      setArtifacts(data);
    } catch (err: unknown) {
      setError(getErrorMessage(err, 'Failed to load hot artifacts'));
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
        <div className="text-xl text-gray-600">Loading hot workspace...</div>
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
          Hot Workspace 🔥
        </h1>
        <p className="text-gray-600">
          Working drafts and in-progress artifacts (24h TTL)
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
            No drafts in hot workspace
          </p>
          <Link
            to={`/projects/${projectId}/execute`}
            className="px-4 py-2 bg-primary text-white rounded hover:bg-blue-600"
          >
            Execute Goals to Create Drafts
          </Link>
        </div>
      ) : (
        <ArtifactList
          artifacts={artifacts}
          storage="hot"
          projectId={projectId}
          onRefresh={loadArtifacts}
        />
      )}
    </div>
  );
}
