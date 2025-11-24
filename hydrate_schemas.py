#!/usr/bin/env python3
"""
Schema Hydration Script for QuestFoundry
Converts stub schemas to complete schemas based on artifact markdown definitions.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class SchemaHydrator:
    """Hydrates stub schemas from markdown artifact definitions."""

    def __init__(self, schemas_dir: Path, artifacts_dir: Path):
        self.schemas_dir = schemas_dir
        self.artifacts_dir = artifacts_dir
        self.hydrated_count = 0
        self.skipped_count = 0
        self.missing_docs_count = 0

    def is_stub(self, schema_data: Dict[str, Any]) -> bool:
        """Check if a schema is a stub that needs hydration."""
        description = schema_data.get("description", "")
        props_count = len(schema_data.get("properties", {}))

        # A schema is a stub if it has "stub" in description OR has 0-1 properties
        return "stub" in description.lower() or props_count <= 1

    def find_markdown_file(self, schema_name: str) -> Optional[Path]:
        """Find corresponding markdown file for a schema."""
        # Try exact match first
        md_file = self.artifacts_dir / f"{schema_name}.md"
        if md_file.exists():
            return md_file

        # Try _ENRICHED suffix
        enriched_file = self.artifacts_dir / f"{schema_name}_ENRICHED.md"
        if enriched_file.exists():
            return enriched_file

        return None

    def parse_markdown_title_description(self, content: str) -> Tuple[str, str]:
        """Extract title and description from markdown."""
        lines = content.split('\n')

        # Find title (first H1)
        title = ""
        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                # Remove subtitle part after —
                if '—' in title:
                    title = title.split('—')[0].strip()
                break

        # Find description from status block
        description = ""
        in_status_block = False
        for i, line in enumerate(lines):
            if line.startswith('> **Use:**'):
                # Extract the description from the Use line
                desc_start = line.find('**Use:**') + 8
                desc_line = line[desc_start:].strip()

                # Continue reading until we hit a blank line or end of block
                description = desc_line
                j = i + 1
                while j < len(lines) and lines[j].startswith('>'):
                    more_text = lines[j].lstrip('> ').strip()
                    if more_text:
                        description += " " + more_text
                    j += 1
                break

        return title, description

    def parse_field_definitions(self, content: str) -> List[Dict[str, Any]]:
        """Parse field definitions from markdown Field Definitions section."""
        fields = []
        lines = content.split('\n')

        # Find "Field Definitions" or "Schema" section
        in_field_section = False
        current_field = None

        for i, line in enumerate(lines):
            # Detect start of field definitions section
            if re.match(r'^##\s+(Field Definitions|Schema)', line, re.IGNORECASE):
                in_field_section = True
                continue

            # Stop at next major section
            if in_field_section and re.match(r'^##\s+', line) and not re.match(r'^###', line):
                break

            if not in_field_section:
                continue

            # Look for field headers: #### `field_name` (type, required/optional)
            field_match = re.match(r'^####\s+`([^`]+)`\s*\(([^)]+)\)', line)
            if field_match:
                field_name = field_match.group(1)
                field_info = field_match.group(2)

                # Parse type and required
                field_type = "string"  # default
                required = "required" in field_info.lower()

                if "string" in field_info.lower():
                    field_type = "string"
                elif "array" in field_info.lower():
                    field_type = "array"
                elif "object" in field_info.lower():
                    field_type = "object"
                elif "boolean" in field_info.lower():
                    field_type = "boolean"
                elif "number" in field_info.lower() or "integer" in field_info.lower():
                    field_type = "number"

                current_field = {
                    "name": field_name,
                    "type": field_type,
                    "required": required,
                    "description": "",
                    "constraints": {}
                }
                fields.append(current_field)

                # Look ahead for description and constraints
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()

                    # Stop at next field or section
                    if next_line.startswith('####') or next_line.startswith('##'):
                        break

                    # Capture description
                    if next_line and not next_line.startswith('**') and not next_line.startswith('-'):
                        if not current_field["description"]:
                            current_field["description"] = next_line
                        else:
                            current_field["description"] += " " + next_line

                    # Capture constraints
                    if next_line.startswith('**Values:**') or next_line.startswith('**Format:**'):
                        # Look for enum values or format patterns
                        k = j + 1
                        while k < len(lines):
                            constraint_line = lines[k].strip()

                            # Stop at next section or field
                            if constraint_line.startswith('**Example:') or constraint_line.startswith('####') or constraint_line.startswith('##'):
                                break

                            # Skip blank lines
                            if not constraint_line:
                                k += 1
                                continue

                            if constraint_line.startswith('-'):
                                # Enum value
                                enum_val = constraint_line.lstrip('- ').strip()
                                if 'enum' not in current_field["constraints"]:
                                    current_field["constraints"]["enum"] = []
                                # Extract just the value part (before —)
                                if '`' in enum_val:
                                    enum_val = re.findall(r'`([^`]+)`', enum_val)[0]
                                    # Remove surrounding quotes if present
                                    enum_val = enum_val.strip('"').strip("'")
                                elif '—' in enum_val:
                                    enum_val = enum_val.split('—')[0].strip().strip('"').strip("'")
                                current_field["constraints"]["enum"].append(enum_val)

                            k += 1

                    j += 1

        return fields

    def generate_schema_property(self, field: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a JSON Schema property definition from a field."""
        prop = {
            "type": field["type"],
            "description": field["description"] or f"{field['name']} field."
        }

        # Add constraints
        if "enum" in field["constraints"] and field["constraints"]["enum"]:
            prop["enum"] = field["constraints"]["enum"]

        # Add format hints for common patterns
        if field["type"] == "string":
            desc_lower = field["description"].lower()
            if "iso 8601" in desc_lower or "timestamp" in desc_lower:
                prop["format"] = "date-time"
                prop["pattern"] = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
            elif "sha-256" in desc_lower or "sha256" in desc_lower:
                prop["pattern"] = r"^[a-f0-9]{64}$"
            elif "date" in field["name"].lower() and "yyyy-mm-dd" in desc_lower:
                prop["format"] = "date"
                prop["pattern"] = r"^\d{4}-\d{2}-\d{2}$"

        return prop

    def create_hydrated_schema(
        self,
        schema_name: str,
        original_schema: Dict[str, Any],
        md_content: str
    ) -> Dict[str, Any]:
        """Create a hydrated schema from markdown content."""
        title, description = self.parse_markdown_title_description(md_content)
        fields = self.parse_field_definitions(md_content)

        # Build properties
        properties = {}
        required = []

        # Add standard boilerplate if not present in fields
        field_names = {f["name"] for f in fields}

        if "manifest_version" not in field_names:
            properties["manifest_version"] = {
                "type": "string",
                "const": "1.0",
                "description": "Schema version."
            }
            required.append("manifest_version")

        if "project" not in field_names:
            properties["project"] = {
                "type": "string",
                "description": "Project title.",
                "minLength": 1,
                "maxLength": 200
            }
            required.append("project")

        if "created" not in field_names:
            properties["created"] = {
                "type": "string",
                "format": "date-time",
                "description": "ISO 8601 timestamp of creation.",
                "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
            }
            required.append("created")

        if "last_updated" not in field_names:
            properties["last_updated"] = {
                "type": "string",
                "format": "date-time",
                "description": "ISO 8601 timestamp of last update.",
                "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
            }
            required.append("last_updated")

        # Add parsed fields
        for field in fields:
            properties[field["name"]] = self.generate_schema_property(field)
            if field["required"]:
                required.append(field["name"])

        # Build complete schema
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://questfoundry.liesdonk.nl/schemas/{schema_name}.schema.json",
            "title": title or original_schema.get("title", schema_name.replace("_", " ").title()),
            "description": f"Generated from 02-dictionary/artifacts/{schema_name}.md. {description}" if description else original_schema.get("description", ""),
            "type": "object",
            "properties": properties,
            "required": sorted(required),
            "additionalProperties": False
        }

        return schema

    def hydrate_schema(self, schema_file: Path) -> bool:
        """Hydrate a single schema file."""
        schema_name = schema_file.stem.replace(".schema", "")

        # Read current schema
        with open(schema_file, 'r') as f:
            original_schema = json.load(f)

        # Check if it's a stub
        if not self.is_stub(original_schema):
            print(f"  SKIP: {schema_file.name} (already complete)")
            self.skipped_count += 1
            return False

        # Find markdown documentation
        md_file = self.find_markdown_file(schema_name)
        if not md_file:
            print(f"  SKIP: {schema_file.name} (no markdown documentation found)")
            self.missing_docs_count += 1
            return False

        print(f"  HYDRATE: {schema_file.name} <- {md_file.name}")

        # Read markdown content
        with open(md_file, 'r') as f:
            md_content = f.read()

        # Generate hydrated schema
        hydrated_schema = self.create_hydrated_schema(schema_name, original_schema, md_content)

        # Write back to file
        with open(schema_file, 'w') as f:
            json.dump(hydrated_schema, f, indent=2)
            f.write('\n')  # Add trailing newline

        self.hydrated_count += 1
        return True

    def run(self) -> Dict[str, Any]:
        """Run the hydration process on all schema files."""
        print(f"Schema Hydration Process")
        print(f"=" * 60)
        print(f"Schemas directory: {self.schemas_dir}")
        print(f"Artifacts directory: {self.artifacts_dir}")
        print()

        # Get all schema files
        schema_files = sorted(self.schemas_dir.glob("*.schema.json"))
        print(f"Found {len(schema_files)} schema files")
        print()

        # Process each schema
        for schema_file in schema_files:
            self.hydrate_schema(schema_file)

        # Print summary
        print()
        print(f"=" * 60)
        print(f"Summary:")
        print(f"  Hydrated: {self.hydrated_count}")
        print(f"  Skipped (already complete): {self.skipped_count}")
        print(f"  Skipped (no documentation): {self.missing_docs_count}")
        print(f"  Total processed: {len(schema_files)}")

        return {
            "hydrated": self.hydrated_count,
            "skipped_complete": self.skipped_count,
            "skipped_no_docs": self.missing_docs_count,
            "total": len(schema_files)
        }


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent
    schemas_dir = repo_root / "spec" / "03-schemas"
    artifacts_dir = repo_root / "spec" / "02-dictionary" / "artifacts"

    hydrator = SchemaHydrator(schemas_dir, artifacts_dir)
    hydrator.run()


if __name__ == "__main__":
    main()
