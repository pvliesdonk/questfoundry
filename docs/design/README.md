# Design Documents

Architecture and design documentation for the v4 runtime cleanroom rebuild.

## Documents

| Document | Description | Issue | Status |
|----------|-------------|-------|--------|
| [PHASE0_FOUNDATION.md](PHASE0_FOUNDATION.md) | Domain loader, types, config, project structure | #144 | Complete |
| [PHASE1_SINGLE_AGENT.md](PHASE1_SINGLE_AGENT.md) | LLM providers, sessions, agent runtime | #145 | Complete |
| [PHASE2_TOOL_EXECUTION.md](PHASE2_TOOL_EXECUTION.md) | Tool infrastructure, registry, capability filtering | #146 | Complete |
| [PHASE3_DELEGATION_MESSAGING.md](PHASE3_DELEGATION_MESSAGING.md) | Async messaging, delegation, playbook tracking | #147 | Planned |

## Master Tracking

- **Master Issue**: [#143 - V4 Runtime Cleanroom Rebuild](https://github.com/pvliesdonk/questfoundry/issues/143)

## Phases

| Phase | Focus | Issue | Status |
|-------|-------|-------|--------|
| 0 | Foundation | #144 | Complete |
| 1 | Single Agent Execution | #145 | Complete |
| 2 | Tool Execution | #146 | Complete |
| 3 | Delegation & Messaging | #147 | **In Progress** |
| 4 | Storage & Lifecycle | #148 | Pending |
| 5 | Checkpointing & Resumption | #149 | Pending |
| 6 | Flow Control & Polish | #150 | Pending |

## Key References

- `meta/schemas/core/` — The contract (JSON schemas)
- `meta/docs/README.md` — Design philosophy
- `domain-v4/` — Example studio instance
- `RUNTIME-CLEANROOM-BRIEF.md` — Project brief
