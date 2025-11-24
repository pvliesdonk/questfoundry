#!/usr/bin/env python3
"""
Schema Hydration Script v5 - Canon vs. Derived
Enforces integrity by distinguishing between Canon (L0-L2) and Derived (L5) sources.
"""

import json
import re
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field


# Constants
SCHEMA_DRAFT_URL = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_BASE_URL = "https://questfoundry.liesdonk.nl/schemas"
ISO8601_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
SHA256_PATTERN = r"^[a-f0-9]{64}$"
TU_PATTERN = r"^TU-\d{4}-\d{2}-\d{2}-[A-Z]{2,4}\d{2}$"


@dataclass
class ArtifactSource:
    """Information about where an artifact is referenced."""
    name: str
    source_type: str  # 'L0_system', 'L1_role', 'L2_definition', 'L5_runtime'
    source_path: str
    canon: bool  # True if from L0-L2, False if L5 only


@dataclass
class HydrationReport:
    """Tracks hydration results for reporting."""
    ghosts_deleted: List[str] = field(default_factory=list)
    hydrated_from_canon: List[str] = field(default_factory=list)
    inferred_from_canon: List[str] = field(default_factory=list)
    suspect_artifacts: List[str] = field(default_factory=list)
    skipped_complete: List[str] = field(default_factory=list)


class SchemaHydratorV5:
    """Hydrates schemas based on Canon vs. Derived hierarchy."""

    def __init__(self, repo_root: Path, dry_run: bool = True):
        self.repo_root = repo_root
        self.schemas_dir = repo_root / "spec" / "03-schemas"
        self.l0_dir = repo_root / "spec" / "00-north-star"
        self.l1_dir = repo_root / "spec" / "01-roles"
        self.l2_dir = repo_root / "spec" / "02-dictionary" / "artifacts"
        self.l5_dir = repo_root / "spec" / "05-definitions"
        self.report = HydrationReport()
        self.artifact_index: Dict[str, ArtifactSource] = {}
        self.dry_run = dry_run

    def build_artifact_index(self) -> None:
        """Build comprehensive index of artifacts from all layers."""
        print("Building Artifact Index...")
        print("=" * 70)

        # Layer 2: Defined Artifacts (Canon - highest priority)
        self._scan_l2_artifacts()

        # Layer 0: System Objects (Canon)
        self._scan_l0_system_objects()

        # Layer 1: Referenced Artifacts (Canon)
        self._scan_l1_role_references()

        # Layer 5: Runtime Artifacts (Derived)
        self._scan_l5_runtime_artifacts()

        print(f"\nTotal artifacts indexed: {len(self.artifact_index)}")
        canon_count = sum(1 for a in self.artifact_index.values() if a.canon)
        derived_count = len(self.artifact_index) - canon_count
        print(f"  Canon (L0-L2): {canon_count}")
        print(f"  Derived (L5 only): {derived_count}")
        print()

    def _scan_l2_artifacts(self) -> None:
        """Scan Layer 2 artifact definitions."""
        count = 0
        for md_file in self.l2_dir.glob("*.md"):
            if md_file.name == "README.md":
                continue
            artifact_name = md_file.stem.replace("_ENRICHED", "")
            self.artifact_index[artifact_name] = ArtifactSource(
                name=artifact_name,
                source_type="L2_definition",
                source_path=str(md_file.relative_to(self.repo_root)),
                canon=True
            )
            count += 1
        print(f"  L2 (Artifact Definitions): {count} artifacts")

    def _scan_l0_system_objects(self) -> None:
        """Scan Layer 0 for system objects (Cold SoT format)."""
        count = 0
        cold_sot_file = self.l0_dir / "COLD_SOT_FORMAT.md"
        if cold_sot_file.exists():
            with open(cold_sot_file, 'r') as f:
                content = f.read()

            # Extract system objects mentioned (manifest, book, art_manifest, etc.)
            system_objects = [
                "cold_manifest", "cold_book", "cold_art_manifest",
                "cold_fonts", "cold_build_lock"
            ]

            for obj in system_objects:
                if obj not in self.artifact_index:
                    self.artifact_index[obj] = ArtifactSource(
                        name=obj,
                        source_type="L0_system",
                        source_path=str(cold_sot_file.relative_to(self.repo_root)),
                        canon=True
                    )
                    count += 1

        print(f"  L0 (System Objects): {count} artifacts")

    def _scan_l1_role_references(self) -> None:
        """Scan Layer 1 for artifact references in roles."""
        count = 0
        # Scan charters and briefs for artifact mentions
        for md_file in self.l1_dir.rglob("*.md"):
            if md_file.name in ["README.md", "CHANGELOG.md"]:
                continue

            with open(md_file, 'r') as f:
                content = f.read()

            # Look for common artifact patterns in role documentation
            # This is a simplified scan - could be enhanced
            artifact_patterns = [
                r'`(\w+)\.json`',
                r'`(\w+)\.yaml`',
                r'\*\*(\w+)\*\*.*artifact',
            ]

            for pattern in artifact_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    artifact_name = match.lower().replace(".json", "").replace(".yaml", "")
                    if artifact_name and artifact_name not in self.artifact_index:
                        self.artifact_index[artifact_name] = ArtifactSource(
                            name=artifact_name,
                            source_type="L1_role",
                            source_path=str(md_file.relative_to(self.repo_root)),
                            canon=True
                        )
                        count += 1

        print(f"  L1 (Role References): {count} artifacts")

    def _scan_l5_runtime_artifacts(self) -> None:
        """Scan Layer 5 runtime definitions for inputs/outputs."""
        count = 0
        for yaml_file in self.l5_dir.rglob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)

                if not data:
                    continue

                # Extract inputs and outputs
                artifacts = self._extract_artifacts_from_yaml(data)

                for artifact_name in artifacts:
                    if artifact_name not in self.artifact_index:
                        # Only in L5 - suspect!
                        self.artifact_index[artifact_name] = ArtifactSource(
                            name=artifact_name,
                            source_type="L5_runtime",
                            source_path=str(yaml_file.relative_to(self.repo_root)),
                            canon=False
                        )
                        count += 1
                    # If already in index from L0-L2, don't override

            except Exception as e:
                print(f"  Warning: Could not parse {yaml_file}: {e}")
                continue

        print(f"  L5 (Runtime Only): {count} artifacts")

    def _extract_artifacts_from_yaml(self, data: Dict[str, Any]) -> Set[str]:
        """Extract artifact names from YAML data."""
        artifacts = set()

        def scan_dict(d):
            if isinstance(d, dict):
                # Look for inputs/outputs sections
                if 'inputs' in d:
                    inputs = d['inputs']
                    if isinstance(inputs, dict):
                        for section in ['hot', 'cold']:
                            if section in inputs:
                                items = inputs[section]
                                if isinstance(items, list):
                                    for item in items:
                                        if isinstance(item, str):
                                            # Extract artifact name
                                            name = item.split()[0].replace('_', ' ').replace(' ', '_')
                                            artifacts.add(name)
                                        elif isinstance(item, dict) and 'name' in item:
                                            artifacts.add(item['name'])

                if 'outputs' in d:
                    outputs = d['outputs']
                    if isinstance(outputs, dict):
                        for section in ['hot', 'cold']:
                            if section in outputs:
                                items = outputs[section]
                                if isinstance(items, list):
                                    for item in items:
                                        if isinstance(item, str):
                                            name = item.split()[0].replace('_', ' ').replace(' ', '_')
                                            artifacts.add(name)
                                        elif isinstance(item, dict) and 'name' in item:
                                            artifacts.add(item['name'])

                # Recurse
                for v in d.values():
                    scan_dict(v)
            elif isinstance(d, list):
                for item in d:
                    scan_dict(item)

        scan_dict(data)
        return artifacts

    def classify_schema(self, schema_file: Path) -> str:
        """Classify a schema file into one of 4 categories."""
        schema_name = schema_file.stem.replace(".schema", "")

        # Check if it's in our artifact index
        if schema_name not in self.artifact_index:
            return "GHOST"  # No reference anywhere

        source = self.artifact_index[schema_name]

        # Read the schema to check if it's already complete
        try:
            with open(schema_file, 'r') as f:
                schema_data = json.load(f)

            desc = schema_data.get("description", "")
            props_count = len(schema_data.get("properties", {}))
            is_stub = "stub" in desc.lower() or props_count == 0

            if not is_stub:
                return "COMPLETE"  # Already hydrated

        except Exception as e:
            print(f"  Warning: Could not read {schema_file}: {e}")
            return "ERROR"

        # It's a stub - classify by source
        if source.source_type == "L2_definition":
            return "CANON_L2"  # Has L2 definition - can hydrate

        if source.canon and source.source_type in ["L0_system", "L1_role"]:
            return "CANON_MISSING_L2"  # Referenced in canon but no L2 def - infer

        if source.source_type == "L5_runtime" and not source.canon:
            return "SUSPECT_L5"  # Only in L5 - quarantine

        return "UNKNOWN"

    def process_schema(self, schema_file: Path, classification: str) -> None:
        """Process a schema based on its classification."""
        schema_name = schema_file.stem.replace(".schema", "")

        if classification == "GHOST":
            # Delete the file (or mark for deletion in dry-run)
            if not self.dry_run:
                schema_file.unlink()
            self.report.ghosts_deleted.append(schema_name)
            marker = "[DRY-RUN] " if self.dry_run else ""
            print(f"  ❌ {marker}DELETE: {schema_name} (Ghost - no reference)")

        elif classification == "COMPLETE":
            self.report.skipped_complete.append(schema_name)
            print(f"  ✓ SKIP: {schema_name} (already complete)")

        elif classification == "CANON_L2":
            # Hydrate from L2
            source = self.artifact_index[schema_name]
            md_file = self.repo_root / source.source_path

            if md_file.exists():
                if not self.dry_run:
                    self._hydrate_from_markdown(schema_file, md_file)
                self.report.hydrated_from_canon.append(f"{schema_name} ← {source.source_path}")
                marker = "[DRY-RUN] " if self.dry_run else ""
                print(f"  💧 {marker}HYDRATE: {schema_name} ← {md_file.name}")
            else:
                print(f"  ⚠️  WARNING: {schema_name} - L2 reference but file missing")

        elif classification == "CANON_MISSING_L2":
            # Infer from L0/L1
            if not self.dry_run:
                self._infer_from_canon(schema_file, schema_name)
            source = self.artifact_index[schema_name]
            self.report.inferred_from_canon.append(f"{schema_name} (from {source.source_type})")
            marker = "[DRY-RUN] " if self.dry_run else ""
            print(f"  🧠 {marker}INFER: {schema_name} (Canon L0/L1, missing L2)")

        elif classification == "SUSPECT_L5":
            # Quarantine
            if not self.dry_run:
                self._quarantine_suspect(schema_file, schema_name)
            source = self.artifact_index[schema_name]
            self.report.suspect_artifacts.append(f"{schema_name} (L5 only: {source.source_path})")
            marker = "[DRY-RUN] " if self.dry_run else ""
            print(f"  ⚠️  {marker}QUARANTINE: {schema_name} (L5 only - suspect)")

    def _hydrate_from_markdown(self, schema_file: Path, md_file: Path) -> None:
        """Hydrate schema from L2 markdown definition."""
        # Import the original hydrator logic
        from hydrate_schemas import SchemaHydrator

        hydrator = SchemaHydrator(self.schemas_dir, self.l2_dir)

        with open(schema_file, 'r') as f:
            original_schema = json.load(f)

        with open(md_file, 'r') as f:
            md_content = f.read()

        schema_name = schema_file.stem.replace(".schema", "")
        hydrated_schema = hydrator.create_hydrated_schema(schema_name, original_schema, md_content)

        with open(schema_file, 'w') as f:
            json.dump(hydrated_schema, f, indent=2)
            f.write('\n')

    def _infer_from_canon(self, schema_file: Path, schema_name: str) -> None:
        """Infer minimal schema from Canon L0/L1 reference."""
        source = self.artifact_index[schema_name]

        schema = {
            "$schema": SCHEMA_DRAFT_URL,
            "$id": f"{SCHEMA_BASE_URL}/{schema_name}.schema.json",
            "$comment": "INFERRED_FROM_CANON: Missing Layer 2 definition. Validated by Layer 1 usage.",
            "title": schema_name.replace("_", " ").title(),
            "description": f"Inferred from {source.source_type} reference at {source.source_path}. Awaiting Layer 2 definition.",
            "type": "object",
            "properties": {
                "manifest_version": {
                    "type": "string",
                    "const": "1.0",
                    "description": "Schema version."
                },
                "project": {
                    "type": "string",
                    "description": "Project title.",
                    "minLength": 1
                },
                "tu": {
                    "type": "string",
                    "pattern": TU_PATTERN,
                    "description": "Task unit identifier."
                }
            },
            "required": ["manifest_version", "project", "tu"],
            "additionalProperties": False
        }

        with open(schema_file, 'w') as f:
            json.dump(schema, f, indent=2)
            f.write('\n')

    def _quarantine_suspect(self, schema_file: Path, schema_name: str) -> None:
        """Quarantine suspect artifact with warning."""
        source = self.artifact_index[schema_name]

        schema = {
            "$schema": SCHEMA_DRAFT_URL,
            "$id": f"{SCHEMA_BASE_URL}/{schema_name}.schema.json",
            "$comment": "SUSPECT_ARTIFACT: Found in Layer 5 (Derived) but unknown to Canon (L0-L2). This may be a 'Zombie' artifact. Audit required.",
            "title": schema_name.replace("_", " ").title(),
            "description": f"SUSPECT: Referenced only in L5 runtime ({source.source_path}). No canon basis in L0-L2. May be hallucination or legacy drift.",
            "type": "object",
            "properties": {
                "manifest_version": {
                    "type": "string",
                    "const": "1.0",
                    "description": "Schema version."
                },
                "project": {
                    "type": "string",
                    "description": "Project title.",
                    "minLength": 1
                }
            },
            "required": ["manifest_version", "project"],
            "additionalProperties": False
        }

        with open(schema_file, 'w') as f:
            json.dump(schema, f, indent=2)
            f.write('\n')

    def generate_report(self) -> str:
        """Generate markdown hydration report."""
        report_lines = [
            "# Schema Hydration Report (v5 - Canon vs. Derived)",
            "",
            f"**Generated:** {Path.cwd()}",
            "",
            "## 👻 Ghosts Deleted (No Trace)",
            ""
        ]

        if self.report.ghosts_deleted:
            for item in self.report.ghosts_deleted:
                report_lines.append(f"* `{item}.schema.json` — No reference in L0-L5")
        else:
            report_lines.append("* None")

        report_lines.extend([
            "",
            "## 💧 Hydrated from Canon (Layer 2 Source)",
            ""
        ])

        if self.report.hydrated_from_canon:
            for item in self.report.hydrated_from_canon:
                report_lines.append(f"* `{item}`")
        else:
            report_lines.append("* None")

        report_lines.extend([
            "",
            "## 🧠 Inferred from Canon (Layer 0/1 Source - Gap Fill)",
            ""
        ])

        if self.report.inferred_from_canon:
            for item in self.report.inferred_from_canon:
                report_lines.append(f"* `{item}`")
        else:
            report_lines.append("* None")

        report_lines.extend([
            "",
            "## ⚠️ Suspect Artifacts (Layer 5 Only - Potential Errors)",
            ""
        ])

        if self.report.suspect_artifacts:
            for item in self.report.suspect_artifacts:
                report_lines.append(f"* `{item}` — *Found in runtime definitions but missing from Spec*")
        else:
            report_lines.append("* None")

        report_lines.extend([
            "",
            "## ✓ Skipped (Already Complete)",
            "",
            f"* {len(self.report.skipped_complete)} schemas already hydrated"
        ])

        return "\n".join(report_lines)

    def run(self) -> None:
        """Run the v5 hydration process."""
        print("\n" + "=" * 70)
        print("SCHEMA HYDRATION v5 - Canon vs. Derived")
        if self.dry_run:
            print("*** DRY-RUN MODE - No files will be modified ***")
        print("=" * 70)
        print()

        # Step 1: Build artifact index
        self.build_artifact_index()

        # Step 2: Process all schemas
        print("Processing Schemas...")
        print("=" * 70)

        schema_files = sorted(self.schemas_dir.glob("*.schema.json"))
        print(f"Found {len(schema_files)} schema files\n")

        for schema_file in schema_files:
            classification = self.classify_schema(schema_file)
            self.process_schema(schema_file, classification)

        # Step 3: Generate and save report
        print("\n" + "=" * 70)
        print("Generating Report...")
        print("=" * 70)

        report_content = self.generate_report()
        report_file = self.repo_root / "HYDRATION_REPORT.md"

        if not self.dry_run:
            with open(report_file, 'w') as f:
                f.write(report_content)
            print(f"\nReport saved to: {report_file}")
        else:
            print(f"\n[DRY-RUN] Report would be saved to: {report_file}")

        print("\n" + report_content)


def main():
    """Main entry point."""
    import sys

    repo_root = Path(__file__).parent

    # Check for --apply flag
    dry_run = "--apply" not in sys.argv

    if dry_run:
        print("\n⚠️  Running in DRY-RUN mode. Use --apply to actually modify files.\n")

    hydrator = SchemaHydratorV5(repo_root, dry_run=dry_run)
    hydrator.run()

    if dry_run:
        print("\n⚠️  This was a DRY-RUN. No files were modified.")
        print("    Run with --apply to execute changes.\n")


if __name__ == "__main__":
    main()
