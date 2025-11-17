// Settings API calls

import { apiFetch } from './client';
import type { UserSettings, ProviderKeys } from '../types/api';

export async function getUserSettings(): Promise<UserSettings> {
  return apiFetch<UserSettings>('/user/settings');
}

export async function updateProviderKeys(keys: ProviderKeys): Promise<void> {
  return apiFetch<void>('/user/settings/keys', {
    method: 'PUT',
    body: JSON.stringify(keys),
  });
}

export async function deleteProviderKeys(): Promise<void> {
  return apiFetch<void>('/user/settings/keys', {
    method: 'DELETE',
  });
}
