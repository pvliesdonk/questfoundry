#!/usr/bin/env python3
"""
Automatically fix common validation errors in role YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def fix_role_yaml(file_path: Path) -> bool:
    """Fix common validation errors in a role YAML file."""
    print(f"Fixing {file_path.name}...")

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    changed = False

    # 1. Add missing 'id' field (should match filename without .yaml)
    if 'id' not in data:
        data['id'] = file_path.stem
        changed = True
        print(f"  + Added id: {file_path.stem}")

    # 2. Fix dormancy_policy
    if 'identity' in data and 'dormancy_policy' in data['identity']:
        if data['identity']['dormancy_policy'] == 'default_dormant':
            data['identity']['dormancy_policy'] = 'optional'
            changed = True
            print(f"  ✓ Fixed dormancy_policy: default_dormant -> optional")

    # 3. Fix charter_ref to match expected pattern
    if 'identity' in data and 'charter_ref' not in data['identity']:
        role_name = data.get('identity', {}).get('name', '').lower().replace(' ', '_')
        if role_name:
            data['identity']['charter_ref'] = f"spec/01-roles/charters/{role_name}.md"
            changed = True
            print(f"  + Added charter_ref")

    # 4. Fix wake_conditions - convert dict to string
    if 'identity' in data and 'wake_conditions' in data['identity']:
        wake_conditions = data['identity']['wake_conditions']
        if wake_conditions and isinstance(wake_conditions[0], dict):
            # Convert dict format to string format
            data['identity']['wake_conditions'] = [
                wc.get('description', wc.get('trigger', '')) for wc in wake_conditions
            ]
            changed = True
            print(f"  ✓ Fixed wake_conditions format")

    # 5. Fix fallback_models - convert dict to string
    if 'llm_config' in data and 'fallback_models' in data['llm_config']:
        fallback_models = data['llm_config']['fallback_models']
        if fallback_models and isinstance(fallback_models[0], dict):
            # Convert dict format to model string
            data['llm_config']['fallback_models'] = [
                fm.get('model', '') for fm in fallback_models
            ]
            changed = True
            print(f"  ✓ Fixed fallback_models format")

    # 6. Add missing 'enabled' to tools
    if 'behavior' in data and 'tools' in data['behavior']:
        for tool in data['behavior']['tools']:
            if 'enabled' not in tool:
                tool['enabled'] = True
                changed = True
        if changed:
            print(f"  + Added 'enabled: true' to tools")

    # 7. Add missing 'state_key' to outputs
    if 'interface' in data and 'outputs' in data['interface']:
        for i, output in enumerate(data['interface']['outputs']):
            if 'state_key' not in output:
                # Generate a reasonable default based on artifact_type
                artifact_type = output.get('artifact_type', f'artifact_{i}')
                output['state_key'] = f"artifacts.{artifact_type}"
                changed = True
        if changed:
            print(f"  + Added state_key to outputs")

    # 8. Add missing top-level sections for incomplete roles
    if 'interface' not in data:
        print(f"  ⚠ Warning: Missing 'interface' section (not auto-fixed)")
    if 'behavior' not in data:
        print(f"  ⚠ Warning: Missing 'behavior' section (not auto-fixed)")
    if 'protocol' not in data:
        print(f"  ⚠ Warning: Missing 'protocol' section (not auto-fixed)")

    # 9. Add missing 'prompt' in behavior
    if 'behavior' in data and 'prompt' not in data['behavior']:
        print(f"  ⚠ Warning: Missing 'behavior.prompt' (not auto-fixed)")

    if changed:
        # Write back to file with proper YAML formatting
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

    print(f"\n🔧 Fixing validation errors in {len(role_files)} role files...\n")
    print("="*80)

    fixed_count = 0
    for role_file in role_files:
        if fix_role_yaml(role_file):
            fixed_count += 1

    print("="*80)
    print(f"\n✅ Fixed {fixed_count}/{len(role_files)} files\n")


if __name__ == "__main__":
    main()
