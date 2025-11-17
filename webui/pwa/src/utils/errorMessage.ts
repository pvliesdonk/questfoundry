import { APIErrorClass } from '../api/client';

export function getErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof APIErrorClass) {
    return error.detail;
  }

  if (error instanceof Error) {
    return error.message;
  }

  if (error && typeof error === 'object' && 'detail' in error) {
    const detail = (error as { detail?: unknown }).detail;
    if (typeof detail === 'string' && detail.trim().length > 0) {
      return detail;
    }
  }

  return fallback;
}
