// Type-safe API client

import { APIError } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export class APIErrorClass extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'APIError';
    this.status = status;
    this.detail = detail;
  }
}

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean>;
}

export async function apiFetch<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const { params, ...fetchOptions } = options;

  // Build URL with query parameters
  let url = `${API_BASE_URL}${endpoint}`;
  if (params) {
    const queryString = new URLSearchParams(
      Object.entries(params).reduce((acc, [key, value]) => {
        acc[key] = String(value);
        return acc;
      }, {} as Record<string, string>)
    ).toString();
    if (queryString) {
      url += `?${queryString}`;
    }
  }

  // Set default headers
  const headers = new Headers(fetchOptions.headers);
  if (!headers.has('Content-Type') && fetchOptions.body) {
    headers.set('Content-Type', 'application/json');
  }

  // Mock X-Forwarded-User for development
  // In production, this would be set by the OIDC proxy
  if (!headers.has('X-Forwarded-User')) {
    headers.set('X-Forwarded-User', 'dev-user');
  }

  try {
    const response = await fetch(url, {
      ...fetchOptions,
      headers,
    });

    // Handle non-2xx responses
    if (!response.ok) {
      let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const errorBody = await response.json();
        errorDetail = errorBody.detail || errorDetail;
      } catch {
        // Response body is not JSON
      }
      throw new APIErrorClass(response.status, errorDetail);
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return undefined as T;
    }

    // Parse JSON response
    const data = await response.json();
    return data as T;
  } catch (error) {
    if (error instanceof APIErrorClass) {
      throw error;
    }
    // Network or other errors
    throw new APIErrorClass(0, error instanceof Error ? error.message : 'Unknown error');
  }
}
