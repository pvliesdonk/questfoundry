"""
Script to modernize type hints to Python 3.11+ syntax.

Transformations:
- Optional[X] → X | None
- Union[X, Y] → X | Y
- List[X] → list[X]
- Dict[K, V] → dict[K, V]
- Tuple[X, ...] → tuple[X, ...]
- Set[X] → set[X]
- Remove unused typing imports
"""

import re
import sys
from pathlib import Path
from typing import Set


def modernize_type_hints(content: str) -> tuple[str, bool]:
    """
    Modernize type hints in Python source code.

    Returns:
        Tuple of (modified_content, was_modified)
    """
    original = content

    # Track what typing imports we're using
    using_typing_extensions = "typing_extensions" in content

    # Replace Optional[X] with X | None
    content = re.sub(
        r'Optional\[([^\]]+)\]',
        r'\1 | None',
        content
    )

    # Replace Union[X, Y] with X | Y (handle nested brackets)
    # This is simplified - handles common cases
    content = re.sub(
        r'Union\[([\w\[\], ]+)\]',
        lambda m: ' | '.join(part.strip() for part in m.group(1).split(',')),
        content
    )

    # Replace List[X] with list[X]
    content = re.sub(r'\bList\[', 'list[', content)

    # Replace Dict[K, V] with dict[K, V]
    content = re.sub(r'\bDict\[', 'dict[', content)

    # Replace Tuple[...] with tuple[...]
    content = re.sub(r'\bTuple\[', 'tuple[', content)

    # Replace Set[X] with set[X]
    content = re.sub(r'\bSet\[', 'set[', content)

    # Clean up typing imports
    # Find the current typing import line
    import_pattern = r'from typing import ([^\n]+)'
    match = re.search(import_pattern, content)

    if match:
        imports_str = match.group(1)
        # Parse imports
        imports = [imp.strip() for imp in imports_str.split(',')]

        # Remove old-style imports we've replaced
        deprecated = {'Optional', 'Union', 'List', 'Dict', 'Tuple', 'Set'}
        new_imports = [imp for imp in imports if imp not in deprecated]

        if new_imports:
            # Keep the typing import with remaining items
            new_import_line = f"from typing import {', '.join(new_imports)}"
            content = re.sub(import_pattern, new_import_line, content)
        else:
            # Remove the entire typing import line if nothing left
            content = re.sub(r'from typing import [^\n]+\n', '', content)

    # Check if we modified anything
    was_modified = content != original

    return content, was_modified


def process_file(file_path: Path, dry_run: bool = False) -> bool:
    """
    Process a single Python file.

    Returns:
        True if file was modified
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        new_content, was_modified = modernize_type_hints(content)

        if was_modified:
            if dry_run:
                print(f"Would modify: {file_path}")
            else:
                file_path.write_text(new_content, encoding='utf-8')
                print(f"Modified: {file_path}")
            return True
        else:
            print(f"No changes: {file_path}")
            return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Modernize Python type hints')
    parser.add_argument('paths', nargs='+', type=Path, help='Files or directories to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed')
    args = parser.parse_args()

    files_to_process: Set[Path] = set()

    for path in args.paths:
        if path.is_file():
            if path.suffix == '.py':
                files_to_process.add(path)
        elif path.is_dir():
            files_to_process.update(path.rglob('*.py'))

    modified_count = 0
    for file_path in sorted(files_to_process):
        if process_file(file_path, dry_run=args.dry_run):
            modified_count += 1

    print(f"\n{modified_count}/{len(files_to_process)} files {'would be' if args.dry_run else 'were'} modified")


if __name__ == '__main__':
    main()
