import React, { useState, useEffect } from 'react';
import { getUserSettings, updateProviderKeys } from '../api/settings';
import { UserSettings, ProviderKeys } from '../types/api';

const SettingsPage: React.FC = () => {
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

  const loadSettings = async () => {
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
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const keys: ProviderKeys = {};
    if (openaiKey) keys.openai_api_key = openaiKey;
    if (anthropicKey) keys.anthropic_api_key = anthropicKey;

    if (Object.keys(keys).length === 0) {
      alert('Please enter at least one API key');
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
      
      setTimeout(() => setSuccess(false), 3000);
    } catch (err: any) {
      setError(err.detail || 'Failed to save keys');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center py-12">
        <div className="text-xl text-gray-600 dark:text-gray-400">Loading settings...</div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Settings
      </h1>

      <div className="card mb-6">
        <h2 className="text-xl font-semibold mb-4">
          BYOK (Bring Your Own Keys)
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
          Configure your AI provider keys. These keys are encrypted and stored securely.
          You can use your own keys to control costs and data privacy.
        </p>

        {settings && (
          <div className="mb-6 p-4 bg-gray-50 dark:bg-gray-700 rounded">
            <div className="text-sm">
              <div className="flex items-center justify-between mb-2">
                <span>OpenAI API Key:</span>
                <span className={settings.has_openai_key ? 'text-green-600' : 'text-gray-500'}>
                  {settings.has_openai_key ? '✓ Configured' : '✗ Not configured'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>Anthropic API Key:</span>
                <span className={settings.has_anthropic_key ? 'text-green-600' : 'text-gray-500'}>
                  {settings.has_anthropic_key ? '✓ Configured' : '✗ Not configured'}
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
            Keys updated successfully!
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
                className="input pr-20"
                placeholder="sk-..."
              />
              <button
                type="button"
                onClick={() => setShowOpenaiKey(!showOpenaiKey)}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-sm text-blue-600 hover:text-blue-800"
              >
                {showOpenaiKey ? 'Hide' : 'Show'}
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
                platform.openai.com
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
                className="input pr-20"
                placeholder="sk-ant-..."
              />
              <button
                type="button"
                onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 text-sm text-blue-600 hover:text-blue-800"
              >
                {showAnthropicKey ? 'Hide' : 'Show'}
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
            disabled={saving}
            className="btn btn-primary w-full"
          >
            {saving ? 'Saving...' : 'Save Keys'}
          </button>
        </form>
      </div>

      <div className="card bg-yellow-50 dark:bg-yellow-900 border-yellow-200">
        <h3 className="font-semibold mb-2">⚠️ Security Notice</h3>
        <p className="text-sm text-gray-700 dark:text-gray-300">
          Your API keys are encrypted using Fernet symmetric encryption before storage.
          They are never logged or exposed through API responses. However, anyone with
          access to your account can use these keys to make AI requests on your behalf.
        </p>
      </div>
    </div>
  );
};

export default SettingsPage;
