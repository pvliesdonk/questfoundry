import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { listArtifacts } from '../api/artifacts';
import { ArtifactResponse } from '../types/api';

const ColdStoragePage: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const [artifacts, setArtifacts] = useState<ArtifactResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadArtifacts();
  }, [projectId]);

  const loadArtifacts = async () => {
    if (!projectId) return;
    
    try {
      setLoading(true);
      setError(null);
      const data = await listArtifacts(projectId, 'cold');
      setArtifacts(data);
    } catch (err: any) {
      setError(err.detail || 'Failed to load artifacts');
    } finally {
      setLoading(false);
    }
  };

  // Group artifacts by type
  const groupedArtifacts = artifacts.reduce((acc, artifact) => {
    if (!acc[artifact.type]) {
      acc[artifact.type] = [];
    }
    acc[artifact.type].push(artifact);
    return acc;
  }, {} as Record<string, ArtifactResponse[]>);

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="text-xl text-gray-600 dark:text-gray-400">Loading cold storage...</div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <Link to="/projects" className="text-blue-600 hover:underline mb-2 inline-block">
          ← Back to Projects
        </Link>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Cold Storage
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Validated artifacts (permanent storage)
        </p>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      {artifacts.length === 0 ? (
        <div className="text-center py-12 card">
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-4">
            No validated artifacts in cold storage
          </p>
          <p className="text-sm text-gray-500">
            Create drafts in hot workspace and run gatecheck to validate them
          </p>
        </div>
      ) : (
        <div className="space-y-8">
          {Object.entries(groupedArtifacts).map(([type, items]) => (
            <div key={type}>
              <h2 className="text-2xl font-semibold mb-4 capitalize">
                {type.replace(/_/g, ' ')}
                <span className="text-sm font-normal text-gray-500 ml-2">
                  ({items.length})
                </span>
              </h2>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {items.map((artifact) => (
                  <div key={artifact.id} className="card">
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="font-semibold">{artifact.metadata.id || artifact.id}</h3>
                      <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                        COLD
                      </span>
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      {artifact.metadata.title || artifact.data.title || 'No title'}
                    </div>
                    <div className="text-xs text-gray-500">
                      Updated: {new Date(artifact.updated_at).toLocaleDateString()}
                    </div>
                    <div className="mt-4">
                      <pre className="text-xs bg-gray-100 dark:bg-gray-700 p-2 rounded overflow-x-auto">
                        {JSON.stringify(artifact.data, null, 2).slice(0, 200)}...
                      </pre>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ColdStoragePage;
