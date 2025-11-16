"""Authentication middleware for WebUI API

This middleware extracts the user ID from the X-Forwarded-User header
set by an OIDC reverse proxy (e.g., Traefik + Authelia).

The API server does NOT implement authentication itself - it trusts
the header set by the reverse proxy.
"""

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Extract user ID from X-Forwarded-User header.

    This middleware must be used with an OIDC reverse proxy that sets
    the X-Forwarded-User header after successful authentication.

    The user_id is stored in request.state for use by endpoint handlers.

    Raises:
        HTTPException: 401 if X-Forwarded-User header is missing
    """

    async def dispatch(self, request: Request, call_next):
        """Process request and extract user ID"""
        # Skip auth for health check and docs
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Extract user from X-Forwarded-User header
        user_id = request.headers.get("X-Forwarded-User")

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Missing X-Forwarded-User header. API must run behind OIDC proxy.",
            )

        # Store user_id in request state for use by handlers
        request.state.user_id = user_id

        response: Response = await call_next(request)
        return response
