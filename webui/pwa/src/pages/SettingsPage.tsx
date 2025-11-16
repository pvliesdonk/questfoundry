import { useState, useEffect } from 'react';
import { getUserSettings, updateProviderKeys } from '../api/settings';
import { UserSettings, ProviderKeys } from '../types/api';
import { LoadingPage } from '../components/common/LoadingSpinner';

export default function SettingsPage() {
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const [openaiKey, setOpenaiKey] = useState('');
  const [anthropicKey, setAnthropicKey] = useState('');
  const [showOpenaiKey, setShowOpenaiKey] = useState(false);
  const [showAnthropicKey, setShowAnthropicKey] = useState(false);

  useEffect(() => {
    loadSettings();
  }, []);

  async function loadSettings() {
    try {
      setLoading(true);
      setError(null);
      const data = await getUserSettings();
      setSettings(data);
    } catch (err: any) {
      setError(err.detail || 'Failed to load settings');
    } finally {
      setLoading(false);
    }
  }

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();

    const keys: ProviderKeys = {};
    if (openaiKey) keys.openai_api_key = openaiKey;
    if (anthropicKey) keys.anthropic_api_key = anthropicKey;

    if (Object.keys(keys).length === 0) {
      setError('Please enter at least one API key');
      return;
    }

    try {
      setSaving(true);
      setError(null);
      setSuccess(false);
      await updateProviderKeys(keys);
      setSuccess(true);
      setOpenaiKey('');
      setAnthropicKey('');
      await loadSettings();

      setTimeout(() => setSuccess(false), 5000);
    } catch (err: any) {
      setError(err.detail || 'Failed to save keys');
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return <LoadingPage message="Loading settings..." />;
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">Settings</h1>
      <p className="text-gray-600 mb-8">Manage your API keys and preferences</p>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">
          🔑 BYOK (Bring Your Own Keys)
        </h2>
        <p className="text-sm text-gray-600 mb-6">
          Configure your AI provider keys. These keys are encrypted and stored securely.
          You can use your own keys to control costs and data privacy.
        </p>

        {settings && (
          <div className="mb-6 p-4 bg-gray-50 rounded">
            <div className="text-sm space-y-2">
              <div className="flex items-center justify-between">
                <span>OpenAI API Key:</span>
                <span className={settings.provider_keys?.openai_api_key ? 'text-green-600' : 'text-gray-500'}>
                  {settings.provider_keys?.openai_api_key ? '✓ Configured' : '✗ Not configured'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>Anthropic API Key:</span>
                <span className={settings.provider_keys?.anthropic_api_key ? 'text-green-600' : 'text-gray-500'}>
                  {settings.provider_keys?.anthropic_api_key ? '✓ Configured' : '✗ Not configured'}
                </span>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        {success && (
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4">
            ✓ Keys updated successfully and encrypted!
          </div>
        )}

        <form onSubmit={handleSave}>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              OpenAI API Key
            </label>
            <div className="relative">
              <input
                type={showOpenaiKey ? 'text' : 'password'}
                value={openaiKey}
                onChange={(e) => setOpenaiKey(e.target.value)}
                className="w-full px-3 py-2 pr-20 border rounded focus:ring-2 focus:ring-primary"
                placeholder="sk-..."
              />
              <button
                type="button"
                onClick={() => setShowOpenaiKey(!showOpenaiKey)}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-sm text-blue-600 hover:text-blue-800"
              >
                {showOpenaiKey ? '👁️ Hide' : '👁️‍🗨️ Show'}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Get your key from{' '}
              <a
                href="https://platform.openai.com/api-keys"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                platform.openai.com/api-keys
              </a>
            </p>
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium mb-2">
              Anthropic API Key
            </label>
            <div className="relative">
              <input
                type={showAnthropicKey ? 'text' : 'password'}
                value={anthropicKey}
                onChange={(e) => setAnthropicKey(e.target.value)}
                className="w-full px-3 py-2 pr-20 border rounded focus:ring-2 focus:ring-primary"
                placeholder="sk-ant-..."
              />
              <button
                type="button"
                onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-sm text-blue-600 hover:text-blue-800"
              >
                {showAnthropicKey ? '👁️ Hide' : '👁️‍🗨️ Show'}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Get your key from{' '}
              <a
                href="https://console.anthropic.com/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                console.anthropic.com
              </a>
            </p>
          </div>

          <button
            type="submit"
            disabled={saving || (!openaiKey && !anthropicKey)}
            className="w-full px-4 py-3 bg-primary text-white rounded hover:bg-blue-600 disabled:opacity-50 font-medium"
          >
            {saving ? 'Saving & Encrypting...' : 'Save Keys'}
          </button>
        </form>
      </div>

      <div className="bg-yellow-50 border border-yellow-200 rounded-lg shadow p-6">
        <h3 className="font-semibold mb-2">⚠️ Security Notice</h3>
        <ul className="text-sm text-gray-700 space-y-2">
          <li>• Your API keys are encrypted using <strong>Fernet symmetric encryption</strong> before storage</li>
          <li>• Keys are <strong>never logged</strong> or exposed through API responses</li>
          <li>• Only you can access these keys through your authenticated session</li>
          <li>• Treat your keys like passwords - don't share them</li>
        </ul>
      </div>
    </div>
  );
}
