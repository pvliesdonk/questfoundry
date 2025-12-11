#!/bin/bash
# Hook: Block direct edits to generated/ directory
# Exit 0 = allow, Exit 2 = block (message on stdout)

# Get tool input from environment
TOOL_INPUT="$CLAUDE_TOOL_INPUT"

# Check if this is an edit to generated/
if echo "$TOOL_INPUT" | grep -q "src/questfoundry/generated/"; then
    echo "BLOCKED: Cannot edit files in src/questfoundry/generated/ directly."
    echo ""
    echo "These files are AUTO-GENERATED. To make changes:"
    echo "  1. Edit the source in src/questfoundry/domain/"
    echo "  2. Run: qf compile"
    echo "  3. The generated files will be updated automatically"
    echo ""
    echo "See .claude/rules/generated-code.md for details."
    exit 2
fi

# Allow the operation
exit 0
