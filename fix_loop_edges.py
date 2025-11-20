#!/usr/bin/env python3
"""
Fix edge and protocol structures in loop YAML files.
"""

import yaml
from pathlib import Path


def fix_edge_structure(edge: dict) -> dict:
    """Fix edge to match schema requirements."""
    fixed = {}

    # Fix source/target (may be using from/to)
    if 'source' in edge:
        fixed['source'] = edge['source']
    elif 'from' in edge:
        fixed['source'] = edge['from']
    else:
        fixed['source'] = 'UNKNOWN'

    if 'target' in edge:
        fixed['target'] = edge['target']
    elif 'to' in edge:
        fixed['target'] = edge['to']
    else:
        fixed['target'] = 'UNKNOWN'

    # Handle condition - if string, convert to object
    if 'condition' in edge and edge['condition'] is not None:
        if isinstance(edge['condition'], str):
            # Convert string condition to basic python_expression format
            fixed['type'] = 'conditional'
            fixed['condition'] = {
                'evaluator': 'python_expression',
                'expression': edge['condition']
            }
        elif isinstance(edge['condition'], dict):
            fixed['type'] = 'conditional'
            fixed['condition'] = edge['condition']
    else:
        fixed['type'] = 'direct'

    # Copy other fields
    for key in ['description', 'protocol_intent']:
        if key in edge:
            fixed[key] = edge[key]

    return fixed


def fix_exit_condition(exit_cond: dict) -> dict:
    """Fix exit condition to match schema."""
    fixed = {}

    # Infer name if missing
    if 'name' in exit_cond:
        fixed['name'] = exit_cond['name']
    elif 'type' in exit_cond:
        fixed['name'] = exit_cond['type']
    elif 'condition' in exit_cond:
        # Try to infer from condition
        cond_str = str(exit_cond['condition']).lower()
        if 'success' in cond_str or 'complete' in cond_str:
            fixed['name'] = 'success'
        elif 'fail' in cond_str or 'error' in cond_str:
            fixed['name'] = 'failure'
        elif 'defer' in cond_str:
            fixed['name'] = 'deferred'
        else:
            fixed['name'] = 'success'
    else:
        fixed['name'] = 'success'

    # Copy condition and other fields
    for key in ['condition', 'final_state_requirements']:
        if key in exit_cond:
            fixed[key] = exit_cond[key]

    return fixed


def fix_message_sequence(msg: dict, index: int) -> dict:
    """Fix message sequence to match schema."""
    fixed = {}

    # Add step if missing
    if 'step' in msg:
        fixed['step'] = msg['step']
    else:
        fixed['step'] = index + 1

    # Copy required fields, provide defaults if missing
    fixed['sender'] = msg.get('sender', 'UNKNOWN')
    fixed['receiver'] = msg.get('receiver', 'UNKNOWN')
    fixed['intent'] = msg.get('intent', 'unknown')

    # Copy optional fields
    for key in ['payload_type', 'envelope_requirements', 'triggers_edge', 'expected_reply', 'description']:
        if key in msg:
            fixed[key] = msg[key]

    return fixed


def fix_loop_structures(file_path: Path) -> bool:
    """Fix structural issues in loop YAML file."""
    print(f"Fixing {file_path.name}...")

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    changed = False

    # Fix edges
    if 'topology' in data and 'edges' in data['topology']:
        original_edges = data['topology']['edges']
        fixed_edges = []

        for edge in original_edges:
            fixed_edge = fix_edge_structure(edge)
            fixed_edges.append(fixed_edge)

        data['topology']['edges'] = fixed_edges
        changed = True
        print(f"  ✓ Fixed {len(fixed_edges)} edges")

    # Fix exit_conditions
    if 'topology' in data and 'exit_conditions' in data['topology']:
        original_exits = data['topology']['exit_conditions']
        fixed_exits = []

        for exit_cond in original_exits:
            fixed_exit = fix_exit_condition(exit_cond)
            fixed_exits.append(fixed_exit)

        data['topology']['exit_conditions'] = fixed_exits
        changed = True
        print(f"  ✓ Fixed {len(fixed_exits)} exit conditions")

    # Fix message_sequences
    if 'protocol_flow' in data and 'message_sequences' in data['protocol_flow']:
        original_msgs = data['protocol_flow']['message_sequences']
        fixed_msgs = []

        for i, msg in enumerate(original_msgs):
            fixed_msg = fix_message_sequence(msg, i)
            fixed_msgs.append(fixed_msg)

        data['protocol_flow']['message_sequences'] = fixed_msgs
        changed = True
        print(f"  ✓ Fixed {len(fixed_msgs)} message sequences")

    # Fix traceability.produces_cold - should be boolean
    if 'traceability' in data and 'produces_cold' in data['traceability']:
        produces_cold = data['traceability']['produces_cold']
        if not isinstance(produces_cold, bool):
            # If it's a list or string, convert to bool
            data['traceability']['produces_cold'] = bool(produces_cold)
            changed = True
            print(f"  ✓ Fixed produces_cold to boolean")

    # Fix execution.error_handling - should be object
    if 'execution' in data and 'error_handling' in data['execution']:
        error_handling = data['execution']['error_handling']
        if isinstance(error_handling, list):
            # Convert list to object with default fields
            data['execution']['error_handling'] = {
                'on_validation_error': 'retry',
                'on_bar_failure': 'rework',
                'on_protocol_error': 'retry',
                'max_retries': 2,
                'retry_backoff': 'exponential',
                'notes': str(error_handling)  # Preserve original as notes
            }
            changed = True
            print(f"  ✓ Fixed error_handling to object")

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

    print(f"\n🔧 Fixing structural issues in {len(loop_files)} loop files...\n")
    print("="*80)

    fixed_count = 0
    for loop_file in loop_files:
        try:
            if fix_loop_structures(loop_file):
                fixed_count += 1
        except Exception as e:
            print(f"  ❌ Error fixing {loop_file.name}: {e}\n")

    print("="*80)
    print(f"\n✅ Fixed {fixed_count} files\n")


if __name__ == "__main__":
    main()
