#!/bin/bash
# Hook: Remind about domain-first principle when editing runtime code
# This is a soft reminder (exit 0), not a blocker

TOOL_INPUT="$CLAUDE_TOOL_INPUT"

# Check if editing runtime code (not generated, not tests)
if echo "$TOOL_INPUT" | grep -q "src/questfoundry/runtime/"; then
    # Only remind, don't block
    echo "REMINDER: When modifying runtime code, verify against domain knowledge first."
    echo "  - Artifact schemas: domain/ontology/artifacts.md"
    echo "  - Role behaviors: domain/roles/*.md"
    echo "  - Workflow flows: domain/loops/*.md"
    echo ""
fi

# Always allow (this is just a reminder)
exit 0
