import { useState } from 'react';
import { runGatecheck } from '../../api/execution';
import type { GatecheckResult } from '../../types/api';
import { getErrorMessage } from '../../utils/errorMessage';

interface Props {
  projectId: string;
}

export default function GatecheckForm({ projectId }: Props) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GatecheckResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await runGatecheck(projectId);
      setResult(data);
    } catch (err: unknown) {
      setError(getErrorMessage(err, 'Failed to run gatecheck'));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-2xl font-bold mb-4">Quality Gatecheck</h2>
      <p className="text-gray-600 mb-6">
        Validate all artifacts against the 8 quality bars
      </p>

      <button
        type="button"
        onClick={handleRun}
        className="px-6 py-3 bg-secondary text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 font-medium"
        disabled={loading}
      >
        {loading ? 'Running Gatecheck...' : 'Run Gatecheck'}
      </button>

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded text-red-700">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="mt-6">
          <div className={`p-4 rounded-lg ${
            result.passed
              ? 'bg-green-50 border-2 border-green-300'
              : 'bg-red-50 border-2 border-red-300'
          }`}>
            <h3 className="text-lg font-semibold mb-2">
              {result.passed ? '✅ Passed' : '❌ Failed'}
            </h3>
            {result.issues.length > 0 && (
              <div className="mt-4 space-y-2">
                {result.issues.map((issue) => (
                  <div
                    key={`${issue.artifact_id ?? issue.message}-${issue.severity}`}
                    className="p-3 bg-white rounded border"
                  >
                    <div className="flex items-start justify-between">
                      <span className={`text-xs font-medium px-2 py-1 rounded ${
                        issue.severity === 'error'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-yellow-100 text-yellow-700'
                      }`}>
                        {issue.severity}
                      </span>
                      {issue.artifact_id && (
                        <span className="text-xs text-gray-500">
                          {issue.artifact_id}
                        </span>
                      )}
                    </div>
                    <p className="mt-2 text-sm">{issue.message}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
