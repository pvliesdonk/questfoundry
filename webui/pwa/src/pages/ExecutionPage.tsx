import React, { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { executeGoal, runGatecheck } from '../api/execution';

const ExecutionPage: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const [goal, setGoal] = useState('');
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [gatecheckResult, setGatecheckResult] = useState<any>(null);
  const [runningGatecheck, setRunningGatecheck] = useState(false);

  const handleExecute = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!projectId || !goal.trim()) return;

    try {
      setExecuting(true);
      setError(null);
      setResult(null);
      const response = await executeGoal(projectId, goal);
      setResult(response);
    } catch (err: any) {
      setError(err.detail || 'Execution failed');
    } finally {
      setExecuting(false);
    }
  };

  const handleGatecheck = async () => {
    if (!projectId) return;

    try {
      setRunningGatecheck(true);
      setGatecheckResult(null);
      const response = await runGatecheck(projectId);
      setGatecheckResult(response);
    } catch (err: any) {
      alert(err.detail || 'Gatecheck failed');
    } finally {
      setRunningGatecheck(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <Link to="/projects" className="text-blue-600 hover:underline mb-2 inline-block">
          ← Back to Projects
        </Link>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
          Goal Execution
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Use natural language to create story elements
        </p>
      </div>

      <div className="card mb-6">
        <form onSubmit={handleExecute}>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              What would you like to create?
            </label>
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              rows={4}
              className="input"
              placeholder="Example: Create a hook card called 'The Mysterious Letter' that introduces the main mystery of the story"
              disabled={executing}
            />
          </div>
          <div className="flex space-x-3">
            <button
              type="submit"
              disabled={executing || !goal.trim()}
              className="btn btn-primary"
            >
              {executing ? 'Executing...' : 'Execute Goal'}
            </button>
            <button
              type="button"
              onClick={handleGatecheck}
              disabled={runningGatecheck}
              className="btn btn-secondary"
            >
              {runningGatecheck ? 'Running...' : 'Run Gatecheck'}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="card mb-6">
          <h2 className="text-xl font-semibold mb-4">Execution Result</h2>
          <div className="mb-2">
            <span
              className={`text-sm px-2 py-1 rounded ${
                result.status === 'success'
                  ? 'bg-green-100 text-green-800'
                  : 'bg-red-100 text-red-800'
              }`}
            >
              {result.status}
            </span>
          </div>
          <pre className="bg-gray-100 dark:bg-gray-700 p-4 rounded overflow-x-auto text-sm">
            {JSON.stringify(result.result || result, null, 2)}
          </pre>
          {result.status === 'success' && (
            <div className="mt-4">
              <Link
                to={`/projects/${projectId}/hot`}
                className="text-blue-600 hover:underline"
              >
                View in Hot Workspace →
              </Link>
            </div>
          )}
        </div>
      )}

      {gatecheckResult && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Gatecheck Results</h2>
          <div className="mb-4">
            <span
              className={`text-sm px-2 py-1 rounded ${
                gatecheckResult.status === 'success'
                  ? 'bg-green-100 text-green-800'
                  : 'bg-yellow-100 text-yellow-800'
              }`}
            >
              {gatecheckResult.status}
            </span>
          </div>
          {gatecheckResult.issues && gatecheckResult.issues.length > 0 ? (
            <div className="space-y-2">
              {gatecheckResult.issues.map((issue: any, idx: number) => (
                <div
                  key={idx}
                  className={`p-3 rounded border ${
                    issue.severity === 'error'
                      ? 'bg-red-50 border-red-200'
                      : 'bg-yellow-50 border-yellow-200'
                  }`}
                >
                  <div className="font-semibold text-sm capitalize">{issue.severity}</div>
                  <div className="text-sm">{issue.message}</div>
                  {issue.artifact_id && (
                    <div className="text-xs text-gray-500 mt-1">
                      Artifact: {issue.artifact_id}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-green-600">✓ All quality checks passed!</div>
          )}
        </div>
      )}

      <div className="card mt-6 bg-blue-50 dark:bg-blue-900">
        <h3 className="font-semibold mb-2">Tips:</h3>
        <ul className="text-sm space-y-1 text-gray-700 dark:text-gray-300">
          <li>• Be specific about what you want to create</li>
          <li>• Include details like names, descriptions, and relationships</li>
          <li>• Run gatecheck to validate drafts before promoting to cold storage</li>
          <li>• Check Hot Workspace to see created drafts</li>
        </ul>
      </div>
    </div>
  );
};

export default ExecutionPage;
