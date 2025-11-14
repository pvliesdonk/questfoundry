"""Core spec compiler orchestrator."""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from questfoundry.compiler.types import BehaviorPrimitive, CompilationError

if TYPE_CHECKING:
    from questfoundry.compiler.assemblers import (
        ReferenceResolver,
        StandalonePromptAssembler,
    )
    from questfoundry.compiler.manifest_builder import ManifestBuilder
    from questfoundry.compiler.validators import ReferenceValidator


# Import these after types to avoid circular dependency
# Import at module level for proper PEP 8 compliance
def _import_helpers() -> tuple[
    type["ReferenceResolver"],
    type["StandalonePromptAssembler"],
    type["ManifestBuilder"],
    type["ReferenceValidator"],
]:
    """Lazy import helpers to break circular dependency."""
    from questfoundry.compiler.assemblers import (
        ReferenceResolver,
        StandalonePromptAssembler,
    )
    from questfoundry.compiler.manifest_builder import ManifestBuilder
    from questfoundry.compiler.validators import ReferenceValidator

    return (
        ReferenceResolver,
        StandalonePromptAssembler,
        ManifestBuilder,
        ReferenceValidator,
    )


class SpecCompiler:
    """Main spec compiler orchestrator."""

    def __init__(self, spec_root: Path):
        """Initialize compiler with spec root directory.

        Args:
            spec_root: Root directory of spec/ (containing 05-behavior/)
        """
        self.spec_root = Path(spec_root)
        self.behavior_dir = self.spec_root / "05-behavior"
        self.primitives: dict[str, BehaviorPrimitive] = {}
        self.reference_pattern = re.compile(r"@(\w+):([a-z_0-9]+)(?:#([a-z_0-9-]+))?")
        self.frontmatter_pattern = re.compile(
            r"\A---\s*\n(.*?)\n---\s*\n(.*)",
            re.DOTALL,
        )

    def load_all_primitives(self) -> None:
        """Load all behavior primitives from disk."""
        # Load expertises
        self._load_markdown_primitives(
            self.behavior_dir / "expertises", "expertise", self._extract_expertise_refs
        )

        # Load procedures
        self._load_markdown_primitives(
            self.behavior_dir / "procedures", "procedure", self._extract_procedure_refs
        )

        # Load snippets
        self._load_markdown_primitives(
            self.behavior_dir / "snippets", "snippet", self._extract_snippet_refs
        )

        # Load playbooks
        self._load_yaml_primitives(
            self.behavior_dir / "playbooks", "playbook", self._extract_playbook_refs
        )

        # Load adapters
        self._load_yaml_primitives(
            self.behavior_dir / "adapters", "adapter", self._extract_adapter_refs
        )

    def _load_markdown_primitives(
        self,
        directory: Path,
        prim_type: str,
        ref_extractor: Callable[[dict[str, Any], str], dict[str, list[str]]],
    ) -> None:
        """Load markdown files with YAML frontmatter.

        Args:
            directory: Directory containing primitives
            prim_type: Type of primitive ('expertise', 'procedure', 'snippet')
            ref_extractor: Function to extract references from metadata and content
        """
        if not directory.exists():
            return

        for md_file in directory.glob("*.md"):
            try:
                content = md_file.read_text()

                match = self.frontmatter_pattern.match(content)
                if match:
                    frontmatter_str, markdown_content = match.groups()
                    frontmatter = yaml.safe_load(frontmatter_str) or {}
                    markdown_content = markdown_content.strip()
                else:
                    frontmatter = {}
                    markdown_content = content.strip()

                # Extract ID from frontmatter or filename
                prim_id = frontmatter.get(f"{prim_type}_id", md_file.stem)

                # Extract references
                references = ref_extractor(frontmatter, markdown_content)

                primitive = BehaviorPrimitive(
                    id=prim_id,
                    type=prim_type,
                    content=markdown_content,
                    metadata=frontmatter,
                    references=references,
                    source_path=md_file,
                )

                self.primitives[f"{prim_type}:{prim_id}"] = primitive

            except Exception as e:
                raise CompilationError(
                    f"Error loading {prim_type} from {md_file}: {e}"
                ) from e

    def _load_yaml_primitives(
        self,
        directory: Path,
        prim_type: str,
        ref_extractor: Callable[[dict[str, Any]], dict[str, list[str]]],
    ) -> None:
        """Load pure YAML files (playbooks, adapters).

        Args:
            directory: Directory containing primitives
            prim_type: Type of primitive ('playbook', 'adapter')
            ref_extractor: Function to extract references from YAML data
        """
        if not directory.exists():
            return

        for yaml_file in directory.glob("*.yaml"):
            try:
                data = yaml.safe_load(yaml_file.read_text())

                # Extract ID from data or filename
                prim_id = data.get(
                    f"{prim_type}_id",
                    yaml_file.stem.removesuffix(f".{prim_type}"),
                )

                # Extract references
                references = ref_extractor(data)

                primitive = BehaviorPrimitive(
                    id=prim_id,
                    type=prim_type,
                    content="",  # YAML primitives don't have markdown content
                    metadata=data,
                    references=references,
                    source_path=yaml_file,
                )

                self.primitives[f"{prim_type}:{prim_id}"] = primitive

            except Exception as e:
                raise CompilationError(
                    f"Error loading {prim_type} from {yaml_file}: {e}"
                ) from e

    def _extract_expertise_refs(
        self, frontmatter: dict[str, Any], content: str
    ) -> dict[str, list[str]]:
        """Extract references from expertise files."""
        refs: dict[str, list[str]] = {}

        # Scan content for @references
        for match in self.reference_pattern.finditer(content):
            ref_type, ref_id = match.group(1), match.group(2)
            if ref_type not in refs:
                refs[ref_type] = []
            refs[ref_type].append(ref_id)

        return refs

    def _extract_procedure_refs(
        self, frontmatter: dict[str, Any], content: str
    ) -> dict[str, list[str]]:
        """Extract references from procedure files."""
        refs: dict[str, list[str]] = {}

        # Extract from frontmatter
        if "references_expertises" in frontmatter:
            refs["expertise"] = frontmatter["references_expertises"]
        if "references_schemas" in frontmatter:
            refs["schema"] = frontmatter["references_schemas"]
        if "references_roles" in frontmatter:
            refs["role"] = frontmatter["references_roles"]

        # Scan content for @references
        for match in self.reference_pattern.finditer(content):
            ref_type, ref_id = match.group(1), match.group(2)
            if ref_type not in refs:
                refs[ref_type] = []
            if ref_id not in refs[ref_type]:
                refs[ref_type].append(ref_id)

        return refs

    def _extract_snippet_refs(
        self, frontmatter: dict[str, Any], content: str
    ) -> dict[str, list[str]]:
        """Extract references from snippet files."""
        refs: dict[str, list[str]] = {}

        # Scan content for @references
        for match in self.reference_pattern.finditer(content):
            ref_type, ref_id = match.group(1), match.group(2)
            if ref_type not in refs:
                refs[ref_type] = []
            refs[ref_type].append(ref_id)

        return refs

    def _extract_playbook_refs(self, data: dict[str, Any]) -> dict[str, list[str]]:
        """Extract references from playbook YAML."""
        refs: dict[str, list[str]] = {}

        # Extract procedure references
        if "procedures" in data:
            if "primary" in data["procedures"]:
                refs["procedure"] = self._extract_ref_ids(data["procedures"]["primary"])
            if "supporting" in data["procedures"]:
                supporting = self._extract_ref_ids(data["procedures"]["supporting"])
                refs["procedure"] = refs.get("procedure", []) + supporting

        # Extract adapter references from RACI
        if "raci" in data:
            adapters = []
            for role_type in ["responsible", "accountable", "consulted", "informed"]:
                if role_type in data["raci"]:
                    for item in data["raci"][role_type]:
                        if isinstance(item, dict) and "role" in item:
                            adapter_refs = self._extract_ref_ids([item["role"]])
                            adapters.extend(adapter_refs)
            if adapters:
                refs["adapter"] = adapters

        # Extract schema references
        if "artifacts" in data:
            refs["schema"] = data["artifacts"]

        # Extract snippet references
        if (
            "validation_requirements" in data
            and "reference" in data["validation_requirements"]
        ):
            snippet_ref = data["validation_requirements"]["reference"]
            snippet_ids = self._extract_ref_ids([snippet_ref])
            if snippet_ids:
                refs["snippet"] = snippet_ids

        return refs

    def _extract_adapter_refs(self, data: dict[str, Any]) -> dict[str, list[str]]:
        """Extract references from adapter YAML."""
        refs: dict[str, list[str]] = {}

        # Extract expertise reference
        if "expertise" in data:
            expertise_ids = self._extract_ref_ids([data["expertise"]])
            if expertise_ids:
                refs["expertise"] = expertise_ids

        # Extract procedure references
        if "procedures" in data:
            procedures = []
            if "primary" in data["procedures"]:
                procedures.extend(self._extract_ref_ids(data["procedures"]["primary"]))
            if "supporting" in data["procedures"]:
                procedures.extend(
                    self._extract_ref_ids(data["procedures"]["supporting"])
                )
            if procedures:
                refs["procedure"] = procedures

        # Extract playbook references
        if "loops" in data:
            playbooks = []
            for loop in data["loops"]:
                if isinstance(loop, dict) and "playbook" in loop:
                    playbook_refs = self._extract_ref_ids([loop["playbook"]])
                    playbooks.extend(playbook_refs)
            if playbooks:
                refs["playbook"] = playbooks

        # Extract snippet references
        if "safety_protocols" in data:
            snippets = self._extract_ref_ids(data["safety_protocols"])
            if snippets:
                refs["snippet"] = snippets

        # Extract schema references
        if "artifacts" in data:
            refs["schema"] = data["artifacts"]

        # Extract role references
        if "references" in data and "layer_1" in data["references"]:
            role_refs = self._extract_ref_ids([data["references"]["layer_1"]])
            if role_refs:
                refs["role"] = role_refs

        return refs

    def _extract_ref_ids(self, ref_list: list[str]) -> list[str]:
        """Extract IDs from reference strings like '@type:id'."""
        ids = set()
        for ref in ref_list:
            if isinstance(ref, str):
                match = self.reference_pattern.match(ref)
                if match:
                    ids.add(match.group(2))
        return list(ids)

    def get_primitive(self, ref_type: str, ref_id: str) -> BehaviorPrimitive | None:
        """Get a primitive by type and ID.

        Args:
            ref_type: Type of primitive ('expertise', 'procedure', etc.)
            ref_id: ID of primitive

        Returns:
            BehaviorPrimitive if found, None otherwise
        """
        key = f"{ref_type}:{ref_id}"
        return self.primitives.get(key)

    def compile_playbook(self, playbook_id: str, output_dir: Path) -> dict[str, Any]:
        """Compile a specific playbook.

        Args:
            playbook_id: ID of the playbook to compile
            output_dir: Directory for compiled output

        Returns:
            Compilation result with file paths

        Raises:
            CompilationError: If compilation fails
        """
        # Ensure primitives are loaded
        if not self.primitives:
            self.load_all_primitives()

        # Import helpers
        ReferenceResolver, _, ManifestBuilder, ReferenceValidator = _import_helpers()

        # Validate
        validator = ReferenceValidator(self.primitives, self.spec_root)
        errors = validator.validate_all()
        if errors:
            actual_errors = [e for e in errors if not e.startswith("Warning:")]
            if actual_errors:
                error_msg = "\n".join(actual_errors)
                raise CompilationError(f"Validation failed:\n{error_msg}")

        output_dir = Path(output_dir)
        manifest_dir = output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        # Initialize helpers
        resolver = ReferenceResolver(self.primitives, self.spec_root)
        manifest_builder = ManifestBuilder(self.primitives, resolver)

        # Build manifest
        manifest = manifest_builder.build_playbook_manifest(playbook_id)
        output_path = manifest_dir / f"{playbook_id}.manifest.json"
        output_path.write_text(json.dumps(manifest, indent=2))

        return {
            "playbook_id": playbook_id,
            "manifest_path": str(output_path),
            "compiled_at": manifest["compiled_at"],
        }

    def compile_adapter(self, adapter_id: str, output_dir: Path) -> dict[str, Any]:
        """Compile a specific adapter (role).

        Args:
            adapter_id: ID of the adapter to compile
            output_dir: Directory for compiled output

        Returns:
            Compilation result with file paths

        Raises:
            CompilationError: If compilation fails
        """
        # Ensure primitives are loaded
        if not self.primitives:
            self.load_all_primitives()

        # Import helpers
        (
            ReferenceResolver,
            StandalonePromptAssembler,
            ManifestBuilder,
            ReferenceValidator,
        ) = _import_helpers()

        # Validate
        validator = ReferenceValidator(self.primitives, self.spec_root)
        errors = validator.validate_all()
        if errors:
            actual_errors = [e for e in errors if not e.startswith("Warning:")]
            if actual_errors:
                error_msg = "\n".join(actual_errors)
                raise CompilationError(f"Validation failed:\n{error_msg}")

        output_dir = Path(output_dir)
        manifest_dir = output_dir / "manifests"
        prompt_dir = output_dir / "standalone_prompts"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        prompt_dir.mkdir(parents=True, exist_ok=True)

        # Initialize helpers
        resolver = ReferenceResolver(self.primitives, self.spec_root)
        manifest_builder = ManifestBuilder(self.primitives, resolver)
        prompt_assembler = StandalonePromptAssembler(
            self.primitives, resolver, self.spec_root
        )

        # Build manifest
        manifest = manifest_builder.build_adapter_manifest(adapter_id)
        manifest_path = manifest_dir / f"{adapter_id}.adapter.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Build standalone prompt
        prompt = prompt_assembler.assemble_role_prompt(adapter_id)
        prompt_path = prompt_dir / f"{adapter_id}_full.md"
        prompt_path.write_text(prompt)

        return {
            "adapter_id": adapter_id,
            "manifest_path": str(manifest_path),
            "prompt_path": str(prompt_path),
            "compiled_at": manifest["compiled_at"],
        }

    def compile_all(self, output_dir: Path) -> dict[str, Any]:
        """Run full compilation pipeline.

        Args:
            output_dir: Directory for compiled output (dist/compiled/)

        Returns:
            Compilation report with statistics

        Raises:
            CompilationError: If compilation fails
        """
        output_dir = Path(output_dir)

        # Step 1: Load all primitives
        self.load_all_primitives()

        # Import helpers
        (
            ReferenceResolver,
            StandalonePromptAssembler,
            ManifestBuilder,
            ReferenceValidator,
        ) = _import_helpers()

        # Step 2: Validate references
        validator = ReferenceValidator(self.primitives, self.spec_root)
        errors = validator.validate_all()
        if errors:
            # Separate errors and warnings
            actual_errors = [e for e in errors if not e.startswith("Warning:")]
            warnings = [e for e in errors if e.startswith("Warning:")]

            if actual_errors:
                error_msg = "\n".join(actual_errors)
                raise CompilationError(f"Validation failed:\n{error_msg}")

            # Print warnings but don't fail
            if warnings:
                print("Validation warnings:")
                for warning in warnings:
                    print(f"  {warning}")

        # Create output directories
        manifest_dir = output_dir / "manifests"
        prompt_dir = output_dir / "standalone_prompts"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        prompt_dir.mkdir(parents=True, exist_ok=True)

        # Initialize helpers
        resolver = ReferenceResolver(self.primitives, self.spec_root)
        manifest_builder = ManifestBuilder(self.primitives, resolver)
        prompt_assembler = StandalonePromptAssembler(
            self.primitives, resolver, self.spec_root
        )

        # Compilation stats
        stats: dict[str, Any] = {
            "compiled_at": datetime.now(UTC).isoformat(),
            "primitives_loaded": len(self.primitives),
            "playbook_manifests_generated": 0,
            "adapter_manifests_generated": 0,
            "standalone_prompts_generated": 0,
        }

        # Step 3: Build playbook manifests
        for prim_key, primitive in self.primitives.items():
            if primitive.type == "playbook":
                try:
                    manifest = manifest_builder.build_playbook_manifest(primitive.id)
                    output_path = manifest_dir / f"{primitive.id}.manifest.json"
                    output_path.write_text(json.dumps(manifest, indent=2))
                    stats["playbook_manifests_generated"] += 1
                except Exception as e:
                    raise CompilationError(
                        f"Error building playbook manifest for {primitive.id}: {e}"
                    ) from e

        # Step 4: Build adapter manifests
        for prim_key, primitive in self.primitives.items():
            if primitive.type == "adapter":
                try:
                    manifest = manifest_builder.build_adapter_manifest(primitive.id)
                    output_path = manifest_dir / f"{primitive.id}.adapter.manifest.json"
                    output_path.write_text(json.dumps(manifest, indent=2))
                    stats["adapter_manifests_generated"] += 1
                except Exception as e:
                    raise CompilationError(
                        f"Error building adapter manifest for {primitive.id}: {e}"
                    ) from e

        # Step 5: Assemble standalone prompts
        for prim_key, primitive in self.primitives.items():
            if primitive.type == "adapter":
                try:
                    prompt = prompt_assembler.assemble_role_prompt(primitive.id)
                    output_path = prompt_dir / f"{primitive.id}_full.md"
                    output_path.write_text(prompt)
                    stats["standalone_prompts_generated"] += 1
                except Exception as e:
                    # Log warning but don't fail compilation
                    print(
                        f"Warning: Error assembling standalone prompt "
                        f"for {primitive.id}: {e}"
                    )

        return stats
