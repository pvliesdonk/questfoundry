# QuestFoundry PWA Implementation Guide (Phase 5)

This guide provides step-by-step instructions for implementing the Progressive Web App frontend for QuestFoundry's multi-tenant WebUI.

## Overview

The PWA is a mobile-first React application that communicates with the FastAPI backend via REST. It provides a user-friendly interface for:

- Managing projects
- Viewing Hot (drafts) and Cold (validated) artifacts
- Executing goals via natural language
- Running gatechecks for quality validation
- Configuring BYOK provider keys

## Tech Stack

- **Framework**: React 18+ with TypeScript
- **Build Tool**: Vite (fast dev server, optimized builds)
- **Routing**: React Router v6
- **HTTP Client**: fetch API with custom wrapper
- **PWA**: vite-plugin-pwa (service worker, manifest)
- **Styling**: TailwindCSS (recommended) or CSS Modules
- **State Management**: React Context + hooks (keep it simple)

## Architecture

```
src/
├── api/                    # API client
│   ├── client.ts          # Fetch wrapper
│   ├── auth.ts            # Authentication helpers
│   ├── projects.ts        # Project API calls
│   ├── artifacts.ts       # Artifact API calls
│   ├── execution.ts       # Execution API calls
│   └── settings.ts        # Settings API calls
├── components/            # Reusable UI components
│   ├── common/           # Buttons, forms, etc.
│   ├── layout/           # App, Layout, Navigation
│   ├── projects/         # Project-related components
│   ├── artifacts/        # Artifact-related components
│   ├── execution/        # Goal execution components
│   └── settings/         # Settings components
├── pages/                # Route pages
│   ├── HomePage.tsx
│   ├── ProjectsPage.tsx
│   ├── ProjectDetailPage.tsx
│   ├── HotWorkspacePage.tsx
│   ├── ColdStoragePage.tsx
│   ├── ExecutionPage.tsx
│   └── SettingsPage.tsx
├── hooks/                # Custom React hooks
│   ├── useProjects.ts
│   ├── useArtifacts.ts
│   ├── useExecution.ts
│   └── useSettings.ts
├── context/              # React Context providers
│   ├── AuthContext.tsx
│   └── AppContext.tsx
├── types/                # TypeScript types
│   └── api.ts
├── utils/                # Utility functions
│   └── format.ts
├── App.tsx               # Root component
├── main.tsx              # Entry point
└── vite-env.d.ts         # Vite types

public/
├── manifest.json         # PWA manifest
└── icons/                # App icons
```

## Phase 5.1: Project Setup

### Step 1: Initialize React Project

```bash
cd webui/pwa

# Install dependencies
npm install

# Add additional dependencies
npm install @tanstack/react-query axios
npm install -D tailwindcss postcss autoprefixer
npm install -D @types/node

# Initialize Tailwind
npx tailwindcss init -p
```

### Step 2: Configure TypeScript

Create `tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### Step 3: Configure Vite

Create `vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      manifest: {
        name: 'QuestFoundry',
        short_name: 'QuestFoundry',
        description: 'Interactive Fiction Authoring Platform',
        theme_color: '#ffffff',
        icons: [
          {
            src: '/icon-192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: '/icon-512.png',
            sizes: '512x512',
            type: 'image/png'
          }
        ]
      }
    })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  }
})
```

### Step 4: Configure Tailwind

Update `tailwind.config.js`:

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#3b82f6',
        secondary: '#8b5cf6',
        hot: '#ef4444',
        cold: '#3b82f6'
      }
    },
  },
  plugins: [],
}
```

### Step 5: Create Environment Variables

Create `.env.development`:

```
VITE_API_URL=http://localhost:8000
```

Create `.env.production`:

```
VITE_API_URL=
```

## Phase 5.2: API Client

### Step 1: Create Base API Client

File: `src/api/client.ts`

```typescript
const API_URL = import.meta.env.VITE_API_URL || '';

export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

export interface FetchOptions extends RequestInit {
  params?: Record<string, string>;
}

export async function apiFetch<T>(
  endpoint: string,
  options: FetchOptions = {}
): Promise<T> {
  const { params, ...fetchOptions } = options;

  // Build URL with query params
  let url = `${API_URL}${endpoint}`;
  if (params) {
    const searchParams = new URLSearchParams(params);
    url += `?${searchParams.toString()}`;
  }

  // Add headers
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...fetchOptions.headers,
  };

  // Make request
  const response = await fetch(url, {
    ...fetchOptions,
    headers,
  });

  // Handle response
  if (!response.ok) {
    let errorData;
    try {
      errorData = await response.json();
    } catch {
      errorData = await response.text();
    }
    throw new APIError(
      errorData?.detail || response.statusText,
      response.status,
      errorData
    );
  }

  // Parse JSON response
  if (response.status === 204) {
    return null as T;
  }

  return response.json();
}
```

### Step 2: Create Type Definitions

File: `src/types/api.ts`

```typescript
export interface ProjectInfo {
  name: string;
  description?: string;
  version?: string;
  author?: string;
  created?: string;
  modified?: string;
  metadata?: Record<string, any>;
}

export interface Project extends ProjectInfo {
  project_id: string;
  owner_id: string;
}

export interface Artifact {
  type: string;
  data: Record<string, any>;
  metadata: Record<string, any>;
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
```

### Step 3: Create Projects API

File: `src/api/projects.ts`

```typescript
import { apiFetch } from './client';
import { Project, ProjectInfo } from '@/types/api';

export async function listProjects(): Promise<Project[]> {
  return apiFetch<Project[]>('/projects');
}

export async function getProject(projectId: string): Promise<Project> {
  return apiFetch<Project>(`/projects/${projectId}`);
}

export async function createProject(info: ProjectInfo): Promise<Project> {
  return apiFetch<Project>('/projects', {
    method: 'POST',
    body: JSON.stringify(info),
  });
}

export async function deleteProject(projectId: string): Promise<void> {
  return apiFetch<void>(`/projects/${projectId}`, {
    method: 'DELETE',
  });
}
```

### Step 4: Create Artifacts API

File: `src/api/artifacts.ts`

```typescript
import { apiFetch } from './client';
import { Artifact } from '@/types/api';

export type StorageBackend = 'hot' | 'cold';

export async function listArtifacts(
  projectId: string,
  storage: StorageBackend = 'cold',
  params?: { artifact_type?: string; [key: string]: string | undefined }
): Promise<Artifact[]> {
  return apiFetch<Artifact[]>(`/projects/${projectId}/artifacts`, {
    params: { storage, ...params },
  });
}

export async function getArtifact(
  projectId: string,
  artifactId: string,
  storage: StorageBackend = 'cold'
): Promise<Artifact> {
  return apiFetch<Artifact>(
    `/projects/${projectId}/artifacts/${artifactId}`,
    { params: { storage } }
  );
}

export async function createArtifact(
  projectId: string,
  artifact: Artifact,
  storage: StorageBackend = 'cold'
): Promise<Artifact> {
  return apiFetch<Artifact>(`/projects/${projectId}/artifacts`, {
    method: 'POST',
    body: JSON.stringify(artifact),
    params: { storage },
  });
}

export async function updateArtifact(
  projectId: string,
  artifactId: string,
  artifact: Artifact,
  storage: StorageBackend = 'cold'
): Promise<Artifact> {
  return apiFetch<Artifact>(
    `/projects/${projectId}/artifacts/${artifactId}`,
    {
      method: 'PUT',
      body: JSON.stringify(artifact),
      params: { storage },
    }
  );
}

export async function deleteArtifact(
  projectId: string,
  artifactId: string,
  storage: StorageBackend = 'cold'
): Promise<void> {
  return apiFetch<void>(
    `/projects/${projectId}/artifacts/${artifactId}`,
    {
      method: 'DELETE',
      params: { storage },
    }
  );
}
```

### Step 5: Create Execution API

File: `src/api/execution.ts`

```typescript
import { apiFetch } from './client';
import { ExecutionRequest, ExecutionResult, GatecheckResult } from '@/types/api';

export async function executeGoal(
  projectId: string,
  request: ExecutionRequest
): Promise<ExecutionResult> {
  return apiFetch<ExecutionResult>(`/projects/${projectId}/execute`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function runGatecheck(projectId: string): Promise<GatecheckResult> {
  return apiFetch<GatecheckResult>(`/projects/${projectId}/gatecheck`, {
    method: 'POST',
  });
}
```

### Step 6: Create Settings API

File: `src/api/settings.ts`

```typescript
import { apiFetch } from './client';
import { UserSettings } from '@/types/api';

export async function getUserSettings(): Promise<UserSettings> {
  return apiFetch<UserSettings>('/user/settings');
}

export async function updateProviderKeys(
  keys: Record<string, string>
): Promise<void> {
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
```

## Phase 5.3: Core Components

### Step 1: Create App Root

File: `src/App.tsx`

```typescript
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';
import HomePage from './pages/HomePage';
import ProjectsPage from './pages/ProjectsPage';
import ProjectDetailPage from './pages/ProjectDetailPage';
import HotWorkspacePage from './pages/HotWorkspacePage';
import ColdStoragePage from './pages/ColdStoragePage';
import ExecutionPage from './pages/ExecutionPage';
import SettingsPage from './pages/SettingsPage';
import ErrorBoundary from './components/common/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<HomePage />} />
            <Route path="projects" element={<ProjectsPage />} />
            <Route path="projects/:projectId" element={<ProjectDetailPage />} />
            <Route path="projects/:projectId/hot" element={<HotWorkspacePage />} />
            <Route path="projects/:projectId/cold" element={<ColdStoragePage />} />
            <Route path="projects/:projectId/execute" element={<ExecutionPage />} />
            <Route path="settings" element={<SettingsPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
```

### Step 2: Create Layout Component

File: `src/components/layout/Layout.tsx`

```typescript
import { Outlet } from 'react-router-dom';
import Navigation from './Navigation';

export default function Layout() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      <main className="container mx-auto px-4 py-8">
        <Outlet />
      </main>
    </div>
  );
}
```

### Step 3: Create Navigation Component

File: `src/components/layout/Navigation.tsx`

```typescript
import { Link } from 'react-router-dom';

export default function Navigation() {
  return (
    <nav className="bg-white shadow">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-8">
            <Link to="/" className="text-xl font-bold text-primary">
              QuestFoundry
            </Link>
            <Link
              to="/projects"
              className="text-gray-700 hover:text-primary transition"
            >
              Projects
            </Link>
          </div>
          <div className="flex items-center space-x-4">
            <Link
              to="/settings"
              className="text-gray-700 hover:text-primary transition"
            >
              Settings
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
```

### Step 4: Create Error Boundary

File: `src/components/common/ErrorBoundary.tsx`

```typescript
import { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
          <div className="max-w-md w-full bg-white shadow-lg rounded-lg p-6">
            <h1 className="text-2xl font-bold text-red-600 mb-4">
              Something went wrong
            </h1>
            <p className="text-gray-700 mb-4">
              {this.state.error?.message || 'An unexpected error occurred'}
            </p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-primary text-white rounded hover:bg-blue-600"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
```

## Phase 5.4: Project Management UI

### Step 1: Projects List Page

File: `src/pages/ProjectsPage.tsx`

```typescript
import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { listProjects, createProject } from '@/api/projects';
import { Project, ProjectInfo } from '@/types/api';
import ProjectCard from '@/components/projects/ProjectCard';
import CreateProjectModal from '@/components/projects/CreateProjectModal';

export default function ProjectsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);

  useEffect(() => {
    loadProjects();
  }, []);

  async function loadProjects() {
    try {
      const data = await listProjects();
      setProjects(data);
    } catch (error) {
      console.error('Failed to load projects:', error);
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate(info: ProjectInfo) {
    try {
      await createProject(info);
      await loadProjects();
      setShowCreate(false);
    } catch (error) {
      console.error('Failed to create project:', error);
      throw error;
    }
  }

  if (loading) {
    return <div className="text-center">Loading projects...</div>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold">My Projects</h1>
        <button
          onClick={() => setShowCreate(true)}
          className="px-4 py-2 bg-primary text-white rounded hover:bg-blue-600"
        >
          New Project
        </button>
      </div>

      {projects.length === 0 ? (
        <div className="text-center text-gray-500 py-12">
          <p className="mb-4">No projects yet</p>
          <button
            onClick={() => setShowCreate(true)}
            className="text-primary hover:underline"
          >
            Create your first project
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {projects.map(project => (
            <ProjectCard
              key={project.project_id}
              project={project}
              onDelete={loadProjects}
            />
          ))}
        </div>
      )}

      {showCreate && (
        <CreateProjectModal
          onClose={() => setShowCreate(false)}
          onCreate={handleCreate}
        />
      )}
    </div>
  );
}
```

### Step 2: Project Card Component

File: `src/components/projects/ProjectCard.tsx`

```typescript
import { Link } from 'react-router-dom';
import { deleteProject } from '@/api/projects';
import { Project } from '@/types/api';

interface Props {
  project: Project;
  onDelete: () => void;
}

export default function ProjectCard({ project, onDelete }: Props) {
  async function handleDelete() {
    if (!confirm('Delete this project? This cannot be undone.')) {
      return;
    }

    try {
      await deleteProject(project.project_id);
      onDelete();
    } catch (error) {
      console.error('Failed to delete project:', error);
      alert('Failed to delete project');
    }
  }

  return (
    <div className="bg-white rounded-lg shadow hover:shadow-lg transition p-6">
      <Link to={`/projects/${project.project_id}`}>
        <h3 className="text-xl font-semibold mb-2 text-primary hover:underline">
          {project.name}
        </h3>
      </Link>
      {project.description && (
        <p className="text-gray-600 mb-4">{project.description}</p>
      )}
      <div className="flex items-center justify-between text-sm text-gray-500">
        <span>v{project.version || '1.0.0'}</span>
        <button
          onClick={handleDelete}
          className="text-red-600 hover:text-red-800"
        >
          Delete
        </button>
      </div>
    </div>
  );
}
```

### Step 3: Create Project Modal

File: `src/components/projects/CreateProjectModal.tsx`

```typescript
import { useState, FormEvent } from 'react';
import { ProjectInfo } from '@/types/api';

interface Props {
  onClose: () => void;
  onCreate: (info: ProjectInfo) => Promise<void>;
}

export default function CreateProjectModal({ onClose, onCreate }: Props) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await onCreate({
        name,
        description,
        version: '1.0.0',
      });
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg max-w-md w-full p-6">
        <h2 className="text-2xl font-bold mb-4">Create New Project</h2>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              Project Name *
            </label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-primary"
              required
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              Description
            </label>
            <textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-primary"
              rows={3}
            />
          </div>
          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700">
              {error}
            </div>
          )}
          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded"
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-primary text-white rounded hover:bg-blue-600 disabled:opacity-50"
              disabled={loading}
            >
              {loading ? 'Creating...' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
```

## Phase 5.5: Hot/Cold SoT UI

### Step 1: Hot Workspace Page

File: `src/pages/HotWorkspacePage.tsx`

```typescript
import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { listArtifacts } from '@/api/artifacts';
import { Artifact } from '@/types/api';
import ArtifactList from '@/components/artifacts/ArtifactList';

export default function HotWorkspacePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (projectId) {
      loadArtifacts();
    }
  }, [projectId]);

  async function loadArtifacts() {
    try {
      const data = await listArtifacts(projectId!, 'hot');
      setArtifacts(data);
    } catch (error) {
      console.error('Failed to load hot artifacts:', error);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return <div className="text-center">Loading hot workspace...</div>;
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">
          Hot Workspace
          <span className="ml-3 text-hot">🔥</span>
        </h1>
        <p className="text-gray-600">
          Working drafts and in-progress artifacts (24h TTL)
        </p>
      </div>

      <ArtifactList
        artifacts={artifacts}
        storage="hot"
        onRefresh={loadArtifacts}
      />
    </div>
  );
}
```

### Step 2: Cold Storage Page

File: `src/pages/ColdStoragePage.tsx`

```typescript
import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { listArtifacts } from '@/api/artifacts';
import { Artifact } from '@/types/api';
import ArtifactList from '@/components/artifacts/ArtifactList';

export default function ColdStoragePage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (projectId) {
      loadArtifacts();
    }
  }, [projectId]);

  async function loadArtifacts() {
    try {
      const data = await listArtifacts(projectId!, 'cold');
      setArtifacts(data);
    } catch (error) {
      console.error('Failed to load cold artifacts:', error);
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return <div className="text-center">Loading cold storage...</div>;
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">
          Cold Storage
          <span className="ml-3 text-cold">❄️</span>
        </h1>
        <p className="text-gray-600">
          Validated, immutable artifacts ready for export
        </p>
      </div>

      <ArtifactList
        artifacts={artifacts}
        storage="cold"
        onRefresh={loadArtifacts}
      />
    </div>
  );
}
```

### Step 3: Artifact List Component

File: `src/components/artifacts/ArtifactList.tsx`

```typescript
import { Artifact } from '@/types/api';
import { StorageBackend } from '@/api/artifacts';
import ArtifactCard from './ArtifactCard';

interface Props {
  artifacts: Artifact[];
  storage: StorageBackend;
  onRefresh: () => void;
}

export default function ArtifactList({ artifacts, storage, onRefresh }: Props) {
  if (artifacts.length === 0) {
    return (
      <div className="text-center text-gray-500 py-12">
        <p>No artifacts in {storage} storage</p>
      </div>
    );
  }

  // Group artifacts by type
  const grouped = artifacts.reduce((acc, artifact) => {
    const type = artifact.type;
    if (!acc[type]) {
      acc[type] = [];
    }
    acc[type].push(artifact);
    return acc;
  }, {} as Record<string, Artifact[]>);

  return (
    <div className="space-y-8">
      {Object.entries(grouped).map(([type, items]) => (
        <div key={type}>
          <h2 className="text-xl font-semibold mb-4 capitalize">
            {type.replace('_', ' ')}s ({items.length})
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {items.map(artifact => (
              <ArtifactCard
                key={artifact.metadata.id}
                artifact={artifact}
                storage={storage}
                onRefresh={onRefresh}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
```

### Step 4: Artifact Card Component

File: `src/components/artifacts/ArtifactCard.tsx`

```typescript
import { Artifact } from '@/types/api';
import { StorageBackend } from '@/api/artifacts';

interface Props {
  artifact: Artifact;
  storage: StorageBackend;
  onRefresh: () => void;
}

export default function ArtifactCard({ artifact, storage }: Props) {
  const id = artifact.metadata.id;
  const title = artifact.data.title || artifact.data.name || id;

  return (
    <div className={`bg-white rounded-lg shadow hover:shadow-lg transition p-4 border-l-4 ${
      storage === 'hot' ? 'border-hot' : 'border-cold'
    }`}>
      <div className="flex items-start justify-between mb-2">
        <h3 className="font-semibold">{title}</h3>
        <span className={`text-xs px-2 py-1 rounded ${
          storage === 'hot' 
            ? 'bg-red-100 text-red-700' 
            : 'bg-blue-100 text-blue-700'
        }`}>
          {storage}
        </span>
      </div>
      <p className="text-sm text-gray-600 mb-2 line-clamp-2">
        {artifact.data.description || 'No description'}
      </p>
      <div className="text-xs text-gray-500">
        <span className="font-medium">ID:</span> {id}
      </div>
    </div>
  );
}
```

## Phase 5.6: Goal Execution UI

### Step 1: Execution Page

File: `src/pages/ExecutionPage.tsx`

```typescript
import { useState, FormEvent } from 'react';
import { useParams } from 'react-router-dom';
import { executeGoal } from '@/api/execution';
import { ExecutionResult } from '@/types/api';

export default function ExecutionPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [goal, setGoal] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExecutionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!goal.trim() || !projectId) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await executeGoal(projectId, { goal });
      setResult(data);
      setGoal(''); // Clear input on success
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Execute Goal</h1>
        <p className="text-gray-600">
          Describe what you want to accomplish in natural language
        </p>
      </div>

      <form onSubmit={handleSubmit} className="mb-8">
        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">
            What would you like to do?
          </label>
          <textarea
            value={goal}
            onChange={e => setGoal(e.target.value)}
            className="w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-primary"
            rows={4}
            placeholder="Example: Create a hook card for a mysterious beginning"
            required
            disabled={loading}
          />
        </div>
        <button
          type="submit"
          className="w-full px-6 py-3 bg-primary text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 font-medium"
          disabled={loading}
        >
          {loading ? 'Executing...' : 'Execute Goal'}
        </button>
      </form>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 mb-4">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className={`p-6 rounded-lg border ${
          result.status === 'success' 
            ? 'bg-green-50 border-green-200' 
            : 'bg-yellow-50 border-yellow-200'
        }`}>
          <h2 className="text-xl font-semibold mb-4">Result</h2>
          <p className="mb-2">
            <strong>Status:</strong> {result.status}
          </p>
          {result.result && (
            <pre className="mt-4 p-4 bg-white rounded border overflow-x-auto text-sm">
              {JSON.stringify(result.result, null, 2)}
            </pre>
          )}
          {result.error && (
            <p className="mt-2 text-red-700">
              <strong>Error:</strong> {result.error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
```

## Phase 5.7: Gatecheck UI

### Step 1: Gatecheck Component

File: `src/components/execution/GatecheckForm.tsx`

```typescript
import { useState } from 'react';
import { runGatecheck } from '@/api/execution';
import { GatecheckResult } from '@/types/api';

interface Props {
  projectId: string;
}

export default function GatecheckForm({ projectId }: Props) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GatecheckResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleRun() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await runGatecheck(projectId);
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-2xl font-bold mb-4">Quality Gatecheck</h2>
      <p className="text-gray-600 mb-6">
        Validate all artifacts against the 8 quality bars
      </p>

      <button
        onClick={handleRun}
        className="px-6 py-3 bg-secondary text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 font-medium"
        disabled={loading}
      >
        {loading ? 'Running Gatecheck...' : 'Run Gatecheck'}
      </button>

      {error && (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded text-red-700">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="mt-6">
          <div className={`p-4 rounded-lg ${
            result.passed 
              ? 'bg-green-50 border-2 border-green-300' 
              : 'bg-red-50 border-2 border-red-300'
          }`}>
            <h3 className="text-lg font-semibold mb-2">
              {result.passed ? '✅ Passed' : '❌ Failed'}
            </h3>
            {result.issues.length > 0 && (
              <div className="mt-4 space-y-2">
                {result.issues.map((issue, idx) => (
                  <div key={idx} className="p-3 bg-white rounded border">
                    <div className="flex items-start justify-between">
                      <span className={`text-xs font-medium px-2 py-1 rounded ${
                        issue.severity === 'error' 
                          ? 'bg-red-100 text-red-700'
                          : 'bg-yellow-100 text-yellow-700'
                      }`}>
                        {issue.severity}
                      </span>
                      {issue.artifact_id && (
                        <span className="text-xs text-gray-500">
                          {issue.artifact_id}
                        </span>
                      )}
                    </div>
                    <p className="mt-2 text-sm">{issue.message}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
```

## Phase 5.8: Settings UI

### Step 1: Settings Page

File: `src/pages/SettingsPage.tsx`

```typescript
import { useState, useEffect, FormEvent } from 'react';
import { getUserSettings, updateProviderKeys } from '@/api/settings';
import { UserSettings } from '@/types/api';

export default function SettingsPage() {
  const [settings, setSettings] = useState<UserSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  // Form state
  const [openaiKey, setOpenaiKey] = useState('');
  const [anthropicKey, setAnthropicKey] = useState('');

  useEffect(() => {
    loadSettings();
  }, []);

  async function loadSettings() {
    try {
      const data = await getUserSettings();
      setSettings(data);
    } catch (err) {
      console.error('Failed to load settings:', err);
    } finally {
      setLoading(false);
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setSuccess(false);
    setSaving(true);

    const keys: Record<string, string> = {};
    if (openaiKey) keys.openai_api_key = openaiKey;
    if (anthropicKey) keys.anthropic_api_key = anthropicKey;

    try {
      await updateProviderKeys(keys);
      setSuccess(true);
      setOpenaiKey('');
      setAnthropicKey('');
      await loadSettings();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return <div className="text-center">Loading settings...</div>;
  }

  const hasOpenAI = settings?.provider_keys?.openai_api_key;
  const hasAnthropic = settings?.provider_keys?.anthropic_api_key;

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Settings</h1>

      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-xl font-semibold mb-4">Provider Keys (BYOK)</h2>
        <p className="text-gray-600 mb-6">
          Your API keys are encrypted and stored securely. They never leave your
          account.
        </p>

        <div className="space-y-4 mb-6">
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
            <span className="font-medium">OpenAI API Key</span>
            <span className={hasOpenAI ? 'text-green-600' : 'text-gray-400'}>
              {hasOpenAI ? '✓ Configured' : 'Not configured'}
            </span>
          </div>
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded">
            <span className="font-medium">Anthropic API Key</span>
            <span className={hasAnthropic ? 'text-green-600' : 'text-gray-400'}>
              {hasAnthropic ? '✓ Configured' : 'Not configured'}
            </span>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              OpenAI API Key
            </label>
            <input
              type="password"
              value={openaiKey}
              onChange={e => setOpenaiKey(e.target.value)}
              className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-primary"
              placeholder="sk-..."
            />
          </div>

          <div className="mb-6">
            <label className="block text-sm font-medium mb-2">
              Anthropic API Key
            </label>
            <input
              type="password"
              value={anthropicKey}
              onChange={e => setAnthropicKey(e.target.value)}
              className="w-full px-3 py-2 border rounded focus:ring-2 focus:ring-primary"
              placeholder="sk-ant-..."
            />
          </div>

          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700">
              {error}
            </div>
          )}

          {success && (
            <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded text-green-700">
              Provider keys updated successfully!
            </div>
          )}

          <button
            type="submit"
            className="w-full px-4 py-2 bg-primary text-white rounded hover:bg-blue-600 disabled:opacity-50"
            disabled={saving || (!openaiKey && !anthropicKey)}
          >
            {saving ? 'Saving...' : 'Update Keys'}
          </button>
        </form>
      </div>
    </div>
  );
}
```

## Phase 5.9: Mobile Optimization

### Responsive Design Checklist

1. **Touch Targets**:
   - All buttons and links should be at least 44x44px
   - Add appropriate padding/margin for finger taps
   - Use larger font sizes on mobile

2. **Navigation**:
   - Consider hamburger menu on mobile
   - Bottom navigation bar for key actions
   - Swipe gestures for common actions

3. **Forms**:
   - Use appropriate input types (tel, email, etc.)
   - Auto-capitalize where appropriate
   - Provide clear validation feedback

4. **Performance**:
   - Lazy load images and components
   - Optimize bundle size with code splitting
   - Use service worker for offline capability

5. **PWA Features**:
   - Add to home screen prompt
   - Offline fallback page
   - Background sync for draft saving

### Example Mobile Navigation

File: `src/components/layout/MobileNav.tsx`

```typescript
import { useState } from 'react';
import { Link } from 'react-router-dom';

export default function MobileNav() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="md:hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 rounded hover:bg-gray-100"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute top-16 left-0 right-0 bg-white shadow-lg border-t">
          <nav className="flex flex-col p-4 space-y-2">
            <Link
              to="/projects"
              className="px-4 py-3 rounded hover:bg-gray-100"
              onClick={() => setIsOpen(false)}
            >
              Projects
            </Link>
            <Link
              to="/settings"
              className="px-4 py-3 rounded hover:bg-gray-100"
              onClick={() => setIsOpen(false)}
            >
              Settings
            </Link>
          </nav>
        </div>
      )}
    </div>
  );
}
```

## Testing Strategy

### Unit Tests

Use Vitest + React Testing Library:

```bash
npm install -D vitest @testing-library/react @testing-library/jest-dom
```

Example test for ProjectCard:

```typescript
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import ProjectCard from '@/components/projects/ProjectCard';

describe('ProjectCard', () => {
  const mockProject = {
    project_id: '123',
    owner_id: 'user1',
    name: 'Test Project',
    description: 'A test project',
    version: '1.0.0',
  };

  it('renders project name', () => {
    render(
      <BrowserRouter>
        <ProjectCard project={mockProject} onDelete={() => {}} />
      </BrowserRouter>
    );
    
    expect(screen.getByText('Test Project')).toBeInTheDocument();
  });
});
```

### E2E Tests

Use Playwright for end-to-end testing:

```bash
npm install -D @playwright/test
```

Example E2E test:

```typescript
import { test, expect } from '@playwright/test';

test('create project workflow', async ({ page }) => {
  await page.goto('/projects');
  await page.click('text=New Project');
  await page.fill('input[placeholder*="name"]', 'My Quest');
  await page.click('button:has-text("Create")');
  
  await expect(page.locator('text=My Quest')).toBeVisible();
});
```

## Deployment

### Build for Production

```bash
cd webui/pwa
npm run build
```

This creates an optimized build in `dist/`.

### Docker Image

The Dockerfile is already configured:

```dockerfile
# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Nginx Configuration

Create `webui/pwa/nginx.conf`:

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Serve static files
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
}
```

## Summary

This guide provides complete implementation details for Phase 5 (PWA). Key deliverables:

1. **Project Setup**: Vite + React + TypeScript + Tailwind
2. **API Client**: Type-safe fetch wrapper with error handling
3. **Core Components**: Layout, Navigation, Error Boundary
4. **Project Management**: List, create, delete projects
5. **Hot/Cold SoT**: Separate views for drafts vs validated artifacts
6. **Goal Execution**: Natural language interface
7. **Gatecheck**: Quality validation UI
8. **Settings**: BYOK provider key management
9. **Mobile Optimization**: Responsive design, PWA features
10. **Testing**: Unit tests + E2E tests
11. **Deployment**: Docker + Nginx configuration

Follow the CHECKLIST.md to track progress on each component.
