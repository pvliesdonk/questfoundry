# TU Context Template

All messages during active TU should include proper context:

```json
{
  "context": {
    "tu": "TU-YYYY-MM-DD-<ROLE><NN>",
    "loop": "<loop_name>",
    "hot_cold": "hot"  // or "cold" for PN delivery
  }
}
```

**Required context fields:**

- `context.hot_cold` - Always present ("hot" or "cold")
- `context.tu` - TU ID for traceable work (format: TU-YYYY-MM-DD-ROLE-NN)
- `context.loop` - Active loop name (e.g., "lore_deepening", "story_spark")

**For PN delivery:**

- `context.hot_cold = "cold"` (REQUIRED)
- `context.snapshot` - Snapshot ID (REQUIRED)
- `safety.player_safe = true` (REQUIRED)

**For traceability:**

- `correlation_id` - Link response to triggering message
- `refs` - Array of upstream artifact IDs

**Refer to:** `@procedure:tu_lifecycle` for complete TU management.
