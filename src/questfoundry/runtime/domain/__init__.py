"""
Domain loading, validation, and schema compilation.

This module handles loading studio definitions from directories,
resolving file references, validating against meta/ schemas,
and compiling FieldDefinitions to JSON Schema.
"""

from questfoundry.runtime.domain.compiler import (
    compile_artifact_type_schema,
    compile_schema,
)
from questfoundry.runtime.domain.loader import (
    LoadError,
    LoadResult,
    load_studio,
)

__all__ = [
    # Loader
    "LoadError",
    "LoadResult",
    "load_studio",
    # Compiler
    "compile_schema",
    "compile_artifact_type_schema",
]
