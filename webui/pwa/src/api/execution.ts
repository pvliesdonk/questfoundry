// Execution API calls

import { apiFetch } from './client';
import { GoalExecutionRequest, GoalExecutionResponse, GatecheckResponse } from '../types/api';

export async function executeGoal(
  projectId: string,
  goal: string
): Promise<GoalExecutionResponse> {
  const request: GoalExecutionRequest = { goal };
  return apiFetch<GoalExecutionResponse>(`/projects/${projectId}/execute`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function runGatecheck(projectId: string): Promise<GatecheckResponse> {
  return apiFetch<GatecheckResponse>(`/projects/${projectId}/gatecheck`, {
    method: 'POST',
  });
}
