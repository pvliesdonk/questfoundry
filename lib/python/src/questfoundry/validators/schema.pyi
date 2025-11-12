"""Type stubs for schema validation utilities"""

from pathlib import Path
from typing import Any

def validate_instance(instance: dict[str, Any], schema_name: str) -> bool: ...
def validate_instance_detailed(
    instance: dict[str, Any], schema_name: str
) -> dict[str, Any]: ...
def validate_schema(schema_path: str | Path) -> bool: ...
