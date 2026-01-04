"""Validation utilities for LLM-generated artifacts.

This module provides:
- ValidationFeedback: Action-first structured feedback for LLM self-correction
- Fuzzy field matching for detecting and correcting field name typos
"""

from questfoundry.validation.feedback import ValidationFeedback

__all__ = ["ValidationFeedback"]
