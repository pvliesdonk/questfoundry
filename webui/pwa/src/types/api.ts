// API Types matching the backend models

export interface ProjectInfo {
  name: string;
  description?: string;
  version?: string;
  author?: string;
  metadata?: Record<string, unknown>;
}

export interface Project extends ProjectInfo {
  project_id: string;
  owner_id: string;
  created_at?: string;
}

export interface ProjectResponse extends Project {}

export interface Artifact {
  type: string;
  data: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface ArtifactResponse extends Artifact {
  id: string;
  created_at: string;
  updated_at: string;
}

export interface GoalExecutionRequest {
  goal: string;
  context?: Record<string, unknown>;
}

export interface GoalExecutionResponse {
  status: string;
  result?: unknown;
  error?: string;
}

export type ExecutionRequest = GoalExecutionRequest;
export type ExecutionResult = GoalExecutionResponse;

export interface GatecheckIssue {
  severity: string;
  message: string;
  artifact_id?: string;
}

export interface GatecheckResult {
  status: string;
  passed: boolean;
  issues: GatecheckIssue[];
}

export type GatecheckResponse = GatecheckResult;

export interface UserSettings {
  provider_keys?: Record<string, string>;
}

export interface ProviderKeys {
  openai_api_key?: string;
  anthropic_api_key?: string;
}

export interface APIError {
  detail: string;
  status?: number;
}
