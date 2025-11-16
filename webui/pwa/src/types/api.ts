// API Types matching the backend models

export interface Project {
  name: string;
  description?: string;
  version: string;
  created_at?: string;
  updated_at?: string;
}

export interface ProjectResponse extends Project {
  id: string;
  owner: string;
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

export interface GoalExecutionRequest {
  goal: string;
}

export interface GoalExecutionResponse {
  status: string;
  result: any;
  error?: string;
}

export interface GatecheckResponse {
  status: string;
  issues: Array<{
    severity: string;
    message: string;
    artifact_id?: string;
  }>;
}

export interface UserSettings {
  user_id: string;
  has_openai_key: boolean;
  has_anthropic_key: boolean;
}

export interface ProviderKeys {
  openai_api_key?: string;
  anthropic_api_key?: string;
}

export interface APIError {
  detail: string;
  status?: number;
}
