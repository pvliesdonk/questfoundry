"""Middleware package for WebUI API"""

from .auth import AuthMiddleware

__all__ = ["AuthMiddleware"]
