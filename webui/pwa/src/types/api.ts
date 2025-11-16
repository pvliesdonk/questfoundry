// API Types matching the backend models

export interface ProjectInfo {
  name: string;
  description?: string;
  version?: string;
  author?: string;
  metadata?: Record<string, any>;
}

export interface Project extends ProjectInfo {
  project_id: string;
  owner_id: string;
  created_at?: string;
}

export interface Artifact {
  type: string;
  data: Record<string, any>;
  metadata: Record<string, any>;
}

export interface ArtifactResponse extends Artifact {
  id: string;
  created_at: string;
  updated_at: string;
}

export interface ExecutionRequest {
  goal: string;
}

export interface ExecutionResult {
  status: string;
  result?: any;
  error?: string;
}

export interface GatecheckResult {
  passed: boolean;
  issues: Array<{
    severity: string;
    message: string;
    artifact_id?: string;
  }>;
}

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
