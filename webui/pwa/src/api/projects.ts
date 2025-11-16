// Project API calls

import { apiFetch } from './client';
import { Project, ProjectResponse } from '../types/api';

export async function listProjects(): Promise<ProjectResponse[]> {
  return apiFetch<ProjectResponse[]>('/projects');
}

export async function getProject(projectId: string): Promise<ProjectResponse> {
  return apiFetch<ProjectResponse>(`/projects/${projectId}`);
}

export async function createProject(project: Project): Promise<ProjectResponse> {
  return apiFetch<ProjectResponse>('/projects', {
    method: 'POST',
    body: JSON.stringify(project),
  });
}

export async function deleteProject(projectId: string): Promise<void> {
  return apiFetch<void>(`/projects/${projectId}`, {
    method: 'DELETE',
  });
}
