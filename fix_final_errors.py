#!/usr/bin/env python3
"""
Fix final remaining validation errors.
"""

import yaml
from pathlib import Path


def fix_role_final(file_path: Path) -> bool:
    """Fix final role issues."""
    print(f"Fixing {file_path.name}...")

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    changed = False

    # Fix abbreviations: AuD -> AUD, AuP -> AUP
    if 'identity' in data and 'abbreviation' in data['identity']:
        abbrev = data['identity']['abbreviation']
        if abbrev == 'AuD':
            data['identity']['abbreviation'] = 'AUD'
            changed = True
            print(f"  ✓ Fixed abbreviation: AuD -> AUD")
        elif abbrev == 'AuP':
            data['identity']['abbreviation'] = 'AUP'
            changed = True
            print(f"  ✓ Fixed abbreviation: AuP -> AUP")

    if changed:
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
        print(f"  ✅ Saved\n")
    else:
        print(f"  ℹ No changes\n")

    return changed


def fix_loop_final(file_path: Path) -> bool:
    """Fix final loop issues."""
    print(f"Fixing {file_path.name}...")

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    changed = False

    # 1. Add missing tu_lifecycle.required
    if 'traceability' in data and 'tu_lifecycle' in data['traceability']:
        if 'required' not in data['traceability']['tu_lifecycle']:
            data['traceability']['tu_lifecycle']['required'] = True
            changed = True
            print(f"  ✓ Added tu_lifecycle.required")

    # 2. Fix tu_lifecycle enum values
    if 'traceability' in data and 'tu_lifecycle' in data['traceability']:
        tu = data['traceability']['tu_lifecycle']
        if 'starts_in' in tu and tu['starts_in'] == 'proposed':
            tu['starts_in'] = 'hot-proposed'
            changed = True
        if 'must_reach' in tu:
            if isinstance(tu['must_reach'], list) and 'completed' in tu['must_reach']:
                tu['must_reach'] = 'cold-merged'
                changed = True
            elif tu['must_reach'] == 'completed':
                tu['must_reach'] = 'cold-merged'
                changed = True

    # 3. Fix role references - convert objects to strings
    if 'roles' in data:
        for role_type in ['required', 'optional']:
            if role_type in data['roles']:
                roles = data['roles'][role_type]
                if roles and isinstance(roles[0], dict):
                    # Convert dict to string (role name)
                    data['roles'][role_type] = [r.get('role', r.get('role_id', 'unknown')) for r in roles]
                    changed = True
                    print(f"  ✓ Fixed {role_type} roles to strings")

        # Fix wake_dormant
        if 'wake_dormant' in data['roles']:
            wake_dormant = data['roles']['wake_dormant']
            if wake_dormant:
                for wd in wake_dormant:
                    if 'role_id' not in wd and 'role' in wd:
                        wd['role_id'] = wd['role']
                        changed = True
                    if 'wake_condition' not in wd and 'wake_when' in wd:
                        wd['wake_condition'] = wd['wake_when']
                        changed = True

    # 4. Fix artifact references - add state_key
    if 'context' in data and 'required_artifacts' in data['context']:
        for artifact in data['context']['required_artifacts']:
            if 'state_key' not in artifact:
                artifact_type = artifact.get('artifact_type', 'unknown')
                artifact['state_key'] = f"artifacts.{artifact_type}"
                changed = True
        if changed:
            print(f"  ✓ Added state_key to required_artifacts")

    # 5. Fix pre_gate_requirements - should be object
    if 'gates' in data and 'pre_gate_requirements' in data['gates']:
        pre_gate = data['gates']['pre_gate_requirements']
        if isinstance(pre_gate, list):
            data['gates']['pre_gate_requirements'] = {
                'artifacts_complete': pre_gate
            }
            changed = True
            print(f"  ✓ Fixed pre_gate_requirements to object")

    # 6. Fix gate_points - add bars_checked if missing
    if 'gates' in data and 'gate_points' in data['gates']:
        for gp in data['gates']['gate_points']:
            if 'bars_checked' not in gp:
                gp['bars_checked'] = ['Integrity', 'Style', 'Presentation']
                changed = True
        if changed:
            print(f"  ✓ Added bars_checked to gate_points")

    # 7. Fix snapshot_handling - should be object
    if 'traceability' in data and 'snapshot_handling' in data['traceability']:
        snapshot = data['traceability']['snapshot_handling']
        if isinstance(snapshot, str):
            data['traceability']['snapshot_handling'] = {
                'requires_snapshot_ref': True if snapshot == 'read_only' else False,
                'creates_snapshot': False
            }
            changed = True
            print(f"  ✓ Fixed snapshot_handling to object")

    # 8. Fix success_criteria.tu_final_state enum
    if 'success_criteria' in data and 'tu_final_state' in data['success_criteria']:
        if data['success_criteria']['tu_final_state'] == 'completed':
            data['success_criteria']['tu_final_state'] = 'cold-merged'
            changed = True

    # 9. Fix log_level enum
    if 'execution' in data and 'observability' in data['execution']:
        if 'log_level' in data['execution']['observability']:
            log_level = data['execution']['observability']['log_level']
            if log_level == 'info':
                data['execution']['observability']['log_level'] = 'INFO'
                changed = True

    if changed:
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
        print(f"  ✅ Saved\n")
    else:
        print(f"  ℹ No changes\n")

    return changed


def main():
    print("\n🔧 Final validation fixes...\n")
    print("="*80)

    # Fix roles
    print("\nFixing roles...")
    roles_dir = Path("spec/05-definitions/roles")
    for role_file in ['audio_director.yaml', 'audio_producer.yaml']:
        fix_role_final(roles_dir / role_file)

    # Fix loops
    print("\nFixing loops...")
    loops_dir = Path("spec/05-definitions/loops")
    for loop_file in loops_dir.glob("*.yaml"):
        fix_loop_final(loop_file)

    print("="*80)
    print("\n✅ Final fixes complete\n")


if __name__ == "__main__":
    main()
