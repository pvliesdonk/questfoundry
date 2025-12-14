#!/bin/bash
# Hook: Remind about domain-first principle when editing runtime code
# This is a soft reminder (exit 0), not a blocker

TOOL_INPUT="$CLAUDE_TOOL_INPUT"

# Check if editing runtime code (not tests)
if echo "$TOOL_INPUT" | grep -q "src/questfoundry/runtime/"; then
    # Only remind, don't block
    echo "REMINDER: When modifying runtime code, verify against domain-v4 knowledge first."
    echo "  - Artifact schemas: domain-v4/artifacts/*.json"
    echo "  - Agent behaviors: domain-v4/agents/*.json"
    echo "  - Studio config: domain-v4/studio.json"
    echo ""
fi

# Always allow (this is just a reminder)
exit 0
