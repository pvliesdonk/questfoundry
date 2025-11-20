#!/usr/bin/env python3
"""
Add missing required sections to role YAML files.
"""

import yaml
from pathlib import Path


# Roles that need complete interface/behavior/protocol sections
INCOMPLETE_ROLES = [
    'art_director', 'audio_director', 'audio_producer',
    'export_service', 'illustrator', 'researcher', 'translator'
]


def add_minimal_sections(data: dict, role_id: str) -> bool:
    """Add minimal required sections for schema compliance."""
    changed = False

    # Add minimal interface
    if 'interface' not in data:
        data['interface'] = {
            'inputs': [],
            'outputs': []
        }
        changed = True
        print(f"  + Added minimal interface section")

    # Add minimal behavior with prompt
    if 'behavior' not in data:
        prompt_file = f"../templates/{role_id}_prompt.j2"
        data['behavior'] = {
            'prompt': {
                'template': f"file://{prompt_file}",
                'template_engine': 'jinja2'
            },
            'tools': data.get('tools', [])
        }
        changed = True
        print(f"  + Added minimal behavior section with prompt reference")
    elif 'prompt' not in data['behavior']:
        prompt_file = f"../templates/{role_id}_prompt.j2"
        data['behavior']['prompt'] = {
            'template': f"file://{prompt_file}",
            'template_engine': 'jinja2'
        }
        changed = True
        print(f"  + Added prompt to behavior section")

    # Add minimal protocol
    if 'protocol' not in data:
        data['protocol'] = {
            'intents': {
                'can_send': [],
                'can_receive': []
            }
        }
        changed = True
        print(f"  + Added minimal protocol section")

    return changed


def fix_role_sections(file_path: Path) -> bool:
    """Fix missing sections in role YAML file."""
    role_id = file_path.stem
    print(f"Fixing {file_path.name}...")

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    changed = False

    # Check if this role needs complete sections
    if role_id in INCOMPLETE_ROLES or 'interface' not in data or 'behavior' not in data or 'protocol' not in data:
        if add_minimal_sections(data, role_id):
            changed = True

    # Check if behavior exists but is missing prompt
    if 'behavior' in data and 'prompt' not in data['behavior']:
        prompt_file = f"../templates/{role_id}_prompt.j2"
        data['behavior']['prompt'] = {
            'template': f"file://{prompt_file}",
            'template_engine': 'jinja2'
        }
        changed = True
        print(f"  + Added prompt to existing behavior section")

    if changed:
        # Write back to file
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
        print(f"  ✅ Saved changes to {file_path.name}\n")
        return True
    else:
        print(f"  ℹ No changes needed\n")
        return False


def main():
    roles_dir = Path("spec/05-definitions/roles")
    role_files = sorted(roles_dir.glob("*.yaml"))

    print(f"\n🔧 Adding missing sections to role files...\n")
    print("="*80)

    fixed_count = 0
    for role_file in role_files:
        if fix_role_sections(role_file):
            fixed_count += 1

    print("="*80)
    print(f"\n✅ Fixed {fixed_count} files\n")


if __name__ == "__main__":
    main()
