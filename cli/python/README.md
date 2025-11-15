# QuestFoundry CLI

Command-line interface for the QuestFoundry agentic runtime.

## Installation

```bash
uv pip install -e .
```

## Usage

### Runtime Commands

```bash
# Initialize a new workspace
qf run init

# Send a single message (non-interactive)
qf run send "Create a story about a knight"

# Start interactive chat session
qf run chat

# Run a specific playbook
qf run playbook lore_deepening

# Run with verbose logging
qf run chat --verbose
```

### Configuration Commands

```bash
# Check system health
qf config doctor

# List available playbooks
qf config list-playbooks

# List available roles
qf config list-roles

# Set configuration values
qf config set workspace_path /path/to/workspace
```

## Global Options

```bash
# Set log level
qf --log-level DEBUG run chat

# Enable verbose output
qf -v run chat

# Log to file
qf --log-file app.log run chat
```
