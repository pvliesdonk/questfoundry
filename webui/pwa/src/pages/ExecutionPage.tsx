import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { executeGoal } from '../api/execution';
import GatecheckForm from '../components/execution/GatecheckForm';

export default function ExecutionPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [goal, setGoal] = useState('');
  const [executing, setExecuting] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleExecute(e: React.FormEvent) {
    e.preventDefault();
    if (!projectId || !goal.trim()) return;

    try {
      setExecuting(true);
      setError(null);
      setResult(null);
      const response = await executeGoal(projectId, goal);
      setResult(response);
      setGoal(''); // Clear goal after successful execution
    } catch (err: any) {
      setError(err.detail || err.message || 'Execution failed');
    } finally {
      setExecuting(false);
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="mb-8">
        <Link to="/projects" className="text-blue-600 hover:underline mb-2 inline-block">
          ← Back to Projects
        </Link>
        <h1 className="text-3xl font-bold mb-2">
          Goal Execution 🎯
        </h1>
        <p className="text-gray-600">
          Use natural language to create story elements
        </p>
      </div>

      {/* Goal Execution Form */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4">Create Story Elements</h2>
        <form onSubmit={handleExecute}>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              What would you like to create?
            </label>
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              rows={4}
              className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-primary"
              placeholder="Example: Create a hook card called 'The Mysterious Letter' that introduces the main mystery of the story"
              disabled={executing}
            />
          </div>
          <button
            type="submit"
            disabled={executing || !goal.trim()}
            className="px-6 py-3 bg-primary text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 font-medium"
          >
            {executing ? 'Executing...' : 'Execute Goal'}
          </button>
        </form>

        {error && (
          <div className="mt-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="mt-6 p-4 bg-green-50 border-2 border-green-300 rounded-lg">
            <h3 className="font-semibold text-lg mb-2">✅ Success!</h3>
            <p className="text-sm text-gray-700 mb-4">
              Your goal has been executed. New artifacts have been created in the hot workspace.
            </p>
            <div className="flex space-x-3">
              <Link
                to={`/projects/${projectId}/hot`}
                className="px-4 py-2 bg-primary text-white rounded hover:bg-blue-600"
              >
                View in Hot Workspace →
              </Link>
            </div>
            <details className="mt-4">
              <summary className="cursor-pointer text-sm text-gray-600 hover:text-gray-800">
                View raw result
              </summary>
              <pre className="mt-2 bg-gray-100 p-4 rounded overflow-x-auto text-xs">
                {JSON.stringify(result, null, 2)}
              </pre>
            </details>
          </div>
        )}
      </div>

      {/* Gatecheck Form */}
      {projectId && <GatecheckForm projectId={projectId} />}

      {/* Tips */}
      <div className="bg-blue-50 rounded-lg shadow p-6">
        <h3 className="font-semibold mb-3">💡 Tips for Goal Execution</h3>
        <ul className="text-sm space-y-2 text-gray-700">
          <li>• <strong>Be specific:</strong> Include names, descriptions, and relationships</li>
          <li>• <strong>One goal at a time:</strong> Break complex tasks into smaller goals</li>
          <li>• <strong>Check drafts:</strong> View created artifacts in Hot Workspace</li>
          <li>• <strong>Validate:</strong> Run gatecheck before promoting to cold storage</li>
          <li>• <strong>Iterate:</strong> Refine drafts by executing refinement goals</li>
        </ul>
      </div>
    </div>
  );
}
