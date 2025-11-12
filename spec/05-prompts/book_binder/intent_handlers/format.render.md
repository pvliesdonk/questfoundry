# Intent Handler — format.render

Inputs

- View specification and target format (markdown|html|epub|pdf) with options.

Preconditions

- Cold snapshot assembled; assets present; rendering toolchain available.

Process

1. Render using toolchain per target; apply formatting/stylesheet rules.
2. Verify anchors, crosslinks, accessibility checks (headings, alt text, contrast).
3. Record artifact path in `view_log`.

Outputs

- Rendered artifact reference (out-of-scope) and `ack`; `view_log` updated.
