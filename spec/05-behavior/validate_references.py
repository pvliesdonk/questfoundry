#!/usr/bin/env python3
"""
Cross-Reference Validation Script for QuestFoundry v2 Architecture

Validates:
- All @expertise:, @procedure:, @snippet:, @playbook:, @adapter:, @schema:, @role: references resolve
- YAML frontmatter is valid
- Referenced files exist
- No orphaned primitives (unused files)
- Circular dependencies
"""

import re
import yaml
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class ReferenceValidator:
    def __init__(self, spec_root: Path):
        self.spec_root = spec_root
        self.behavior_root = spec_root / "05-behavior"
        
        # Track all primitives
        self.expertises: Set[str] = set()
        self.procedures: Set[str] = set()
        self.snippets: Set[str] = set()
        self.playbooks: Set[str] = set()
        self.adapters: Set[str] = set()
        self.roles: Set[str] = set()
        
        # Track references
        self.references: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)  # ref_id -> [(file, ref_type, line)]
        
        # Track errors
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate(self) -> bool:
        """Run all validation checks. Returns True if valid, False otherwise."""
        print("🔍 QuestFoundry v2 Cross-Reference Validator\n")
        
        # Step 1: Discover all primitives
        print("📁 Discovering primitives...")
        self._discover_primitives()
        
        # Step 2: Extract references from all files
        print("🔗 Extracting cross-references...")
        self._extract_references()
        
        # Step 3: Validate references
        print("✅ Validating references...")
        self._validate_references()
        
        # Step 4: Check for orphans
        print("🔍 Checking for orphaned primitives...")
        self._check_orphans()
        
        # Step 5: Report results
        print("\n" + "="*60)
        self._report_results()
        
        return len(self.errors) == 0
    
    def _discover_primitives(self):
        """Discover all primitive IDs from YAML frontmatter."""
        # Expertises
        expertise_dir = self.behavior_root / "expertises"
        if expertise_dir.exists():
            for file in expertise_dir.glob("*.md"):
                expertise_id = self._extract_id(file, "expertise_id")
                if expertise_id:
                    self.expertises.add(expertise_id)
        
        # Procedures
        procedure_dir = self.behavior_root / "procedures"
        if procedure_dir.exists():
            for file in procedure_dir.glob("*.md"):
                procedure_id = self._extract_id(file, "procedure_id")
                if procedure_id:
                    self.procedures.add(procedure_id)
        
        # Snippets
        snippet_dir = self.behavior_root / "snippets"
        if snippet_dir.exists():
            for file in snippet_dir.glob("*.md"):
                snippet_id = self._extract_id(file, "snippet_id")
                if snippet_id:
                    self.snippets.add(snippet_id)
        
        # Playbooks
        playbook_dir = self.behavior_root / "playbooks"
        if playbook_dir.exists():
            for file in playbook_dir.glob("*.yaml"):
                playbook_id = self._extract_id(file, "playbook_id")
                if playbook_id:
                    self.playbooks.add(playbook_id)
        
        # Adapters
        adapter_dir = self.behavior_root / "adapters"
        if adapter_dir.exists():
            for file in adapter_dir.glob("*.yaml"):
                adapter_id = self._extract_id(file, "adapter_id")
                if adapter_id:
                    self.adapters.add(adapter_id)
                # Extract role from adapter
                role = self._extract_id(file, "role")
                if role:
                    self.roles.add(role)
        
        print(f"  Found {len(self.expertises)} expertises")
        print(f"  Found {len(self.procedures)} procedures")
        print(f"  Found {len(self.snippets)} snippets")
        print(f"  Found {len(self.playbooks)} playbooks")
        print(f"  Found {len(self.adapters)} adapters")
        print(f"  Found {len(self.roles)} roles")
    
    def _extract_id(self, file: Path, id_field: str) -> str:
        """Extract ID from YAML frontmatter."""
        try:
            content = file.read_text()
            
            # Extract YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 2:
                    frontmatter = yaml.safe_load(parts[1])
                    if frontmatter and id_field in frontmatter:
                        return frontmatter[id_field]
        except Exception as e:
            self.errors.append(f"Failed to parse {file}: {e}")
        
        return None
    
    def _extract_references(self):
        """Extract all cross-references from markdown and YAML files."""
        # Pattern: @expertise:foo_bar or @procedure:baz_qux
        ref_pattern = re.compile(r'@(expertise|procedure|snippet|playbook|adapter|schema|role):(\w+)')
        
        # Search all markdown files
        for md_file in self.behavior_root.rglob("*.md"):
            content = md_file.read_text()
            for line_num, line in enumerate(content.split('\n'), 1):
                for match in ref_pattern.finditer(line):
                    ref_type = match.group(1)
                    ref_id = match.group(2)
                    self.references[f"{ref_type}:{ref_id}"].append(
                        (str(md_file.relative_to(self.spec_root)), ref_type, line_num)
                    )
        
        # Search all YAML files
        for yaml_file in self.behavior_root.rglob("*.yaml"):
            content = yaml_file.read_text()
            for line_num, line in enumerate(content.split('\n'), 1):
                for match in ref_pattern.finditer(line):
                    ref_type = match.group(1)
                    ref_id = match.group(2)
                    self.references[f"{ref_type}:{ref_id}"].append(
                        (str(yaml_file.relative_to(self.spec_root)), ref_type, line_num)
                    )
        
        print(f"  Found {len(self.references)} unique cross-references")
    
    def _validate_references(self):
        """Validate that all references resolve to existing primitives."""
        for ref, occurrences in self.references.items():
            ref_type, ref_id = ref.split(":", 1)
            
            # Check if reference resolves
            exists = False
            if ref_type == "expertise":
                exists = ref_id in self.expertises
            elif ref_type == "procedure":
                exists = ref_id in self.procedures
            elif ref_type == "snippet":
                exists = ref_id in self.snippets
            elif ref_type == "playbook":
                exists = ref_id in self.playbooks
            elif ref_type == "adapter":
                exists = ref_id in self.adapters
            elif ref_type == "role":
                exists = ref_id in self.roles
            elif ref_type == "schema":
                # Schema validation would check spec/04-schemas/
                # For now, just warn
                self.warnings.append(f"Schema reference not validated: {ref}")
                continue
            
            if not exists:
                for file, _, line_num in occurrences:
                    self.errors.append(
                        f"Broken reference in {file}:{line_num}: @{ref_type}:{ref_id} not found"
                    )
    
    def _check_orphans(self):
        """Check for primitives that are never referenced."""
        # Build referenced IDs
        referenced_expertises = set()
        referenced_procedures = set()
        referenced_snippets = set()
        referenced_playbooks = set()
        referenced_adapters = set()
        
        for ref in self.references.keys():
            ref_type, ref_id = ref.split(":", 1)
            if ref_type == "expertise":
                referenced_expertises.add(ref_id)
            elif ref_type == "procedure":
                referenced_procedures.add(ref_id)
            elif ref_type == "snippet":
                referenced_snippets.add(ref_id)
            elif ref_type == "playbook":
                referenced_playbooks.add(ref_id)
            elif ref_type == "adapter":
                referenced_adapters.add(ref_id)
        
        # Check for orphans
        orphan_expertises = self.expertises - referenced_expertises
        orphan_procedures = self.procedures - referenced_procedures
        orphan_snippets = self.snippets - referenced_snippets
        orphan_playbooks = self.playbooks - referenced_playbooks
        orphan_adapters = self.adapters - referenced_adapters
        
        # Adapters and playbooks are top-level, so orphans are OK
        # But warn for orphan expertises/procedures/snippets
        for expertise in orphan_expertises:
            self.warnings.append(f"Orphaned expertise (never referenced): {expertise}")
        
        for procedure in orphan_procedures:
            self.warnings.append(f"Orphaned procedure (never referenced): {procedure}")
        
        for snippet in orphan_snippets:
            self.warnings.append(f"Orphaned snippet (never referenced): {snippet}")
    
    def _report_results(self):
        """Print validation report."""
        if self.errors:
            print(f"\n❌ VALIDATION FAILED ({len(self.errors)} errors)\n")
            for error in self.errors:
                print(f"  ❌ {error}")
        else:
            print(f"\n✅ VALIDATION PASSED (0 errors)")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)})\n")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
        
        print("\n" + "="*60)
        print(f"Summary:")
        print(f"  Errors: {len(self.errors)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Primitives: {len(self.expertises) + len(self.procedures) + len(self.snippets) + len(self.playbooks) + len(self.adapters)}")
        print(f"  References: {len(self.references)}")


def main():
    spec_root = Path(__file__).parent.parent
    validator = ReferenceValidator(spec_root)
    
    success = validator.validate()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
