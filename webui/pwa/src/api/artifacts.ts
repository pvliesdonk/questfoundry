// Artifact API calls

import { apiFetch } from './client';
import type { Artifact, ArtifactResponse } from '../types/api';

export type StorageBackend = 'hot' | 'cold';

export async function listArtifacts(
  projectId: string,
  storage: StorageBackend = 'cold',
  filters?: Record<string, string>
): Promise<ArtifactResponse[]> {
  const params: Record<string, string> = { storage, ...filters };
  return apiFetch<ArtifactResponse[]>(`/projects/${projectId}/artifacts`, {
    params,
  });
}

export async function getArtifact(
  projectId: string,
  artifactId: string,
  storage: StorageBackend = 'cold'
): Promise<ArtifactResponse> {
  return apiFetch<ArtifactResponse>(
    `/projects/${projectId}/artifacts/${artifactId}`,
    { params: { storage } }
  );
}

export async function createArtifact(
  projectId: string,
  artifact: Artifact,
  storage: StorageBackend = 'cold'
): Promise<ArtifactResponse> {
  return apiFetch<ArtifactResponse>(`/projects/${projectId}/artifacts`, {
    method: 'POST',
    params: { storage },
    body: JSON.stringify(artifact),
  });
}

export async function updateArtifact(
  projectId: string,
  artifactId: string,
  artifact: Artifact,
  storage: StorageBackend = 'cold'
): Promise<ArtifactResponse> {
  return apiFetch<ArtifactResponse>(
    `/projects/${projectId}/artifacts/${artifactId}`,
    {
      method: 'PUT',
      params: { storage },
      body: JSON.stringify(artifact),
    }
  );
}

export async function deleteArtifact(
  projectId: string,
  artifactId: string,
  storage: StorageBackend = 'cold'
): Promise<void> {
  return apiFetch<void>(`/projects/${projectId}/artifacts/${artifactId}`, {
    method: 'DELETE',
    params: { storage },
  });
}
