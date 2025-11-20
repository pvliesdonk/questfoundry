#!/usr/bin/env python3
"""
Automatically fix common validation errors in loop YAML files.
"""

import yaml
from pathlib import Path


def fix_loop_yaml(file_path: Path) -> bool:
    """Fix common validation errors in a loop YAML file."""
    print(f"Fixing {file_path.name}...")

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    changed = False

    # 1. Add missing 'id' field
    if 'id' not in data:
        data['id'] = file_path.stem
        changed = True
        print(f"  + Added id: {file_path.stem}")

    # 2. Add missing 'entry_node' in topology
    if 'topology' in data and 'entry_node' not in data['topology']:
        # Try to infer from first node
        nodes = data['topology'].get('nodes', [])
        if nodes:
            first_node = nodes[0]
            entry = first_node.get('node_id') or first_node.get('role') or first_node.get('id', 'unknown')
            data['topology']['entry_node'] = entry
            changed = True
            print(f"  + Added entry_node: {entry}")

    # 3. Fix nodes - ensure 'role_id' field
    if 'topology' in data and 'nodes' in data['topology']:
        for node in data['topology']['nodes']:
            if 'role_id' not in node:
                # Check for 'role' or 'id' field (legacy names)
                if 'role' in node:
                    node['role_id'] = node['role']
                    # del node['role']  # Keep both for compatibility
                    changed = True
                elif 'id' in node and isinstance(node['id'], str) and node['id'] not in data:
                    # If 'id' looks like a role name, use it
                    node['role_id'] = node['id']
                    changed = True

        if changed:
            print(f"  ✓ Fixed node role_id fields")

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
    loops_dir = Path("spec/05-definitions/loops")
    loop_files = sorted(loops_dir.glob("*.yaml"))

    print(f"\n🔧 Fixing validation errors in {len(loop_files)} loop files...\n")
    print("="*80)

    fixed_count = 0
    for loop_file in loop_files:
        if fix_loop_yaml(loop_file):
            fixed_count += 1

    print("="*80)
    print(f"\n✅ Fixed {fixed_count}/{len(loop_files)} files\n")


if __name__ == "__main__":
    main()
