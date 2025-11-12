# Intent Handler — view.export

Inputs

- Snapshot reference, front matter/UI labels, export targets and options.

Preconditions

- Cold snapshot available; gatecheck passed; PN safety prerequisites satisfied.

Process

1. Validate PN safety prerequisites and gate pass.
2. Assemble view from snapshot; map anchors and crosslinks; verify no dead anchors.
3. Render requested formats (Markdown/HTML/PDF/EPUB); check Presentation and Accessibility.
4. Write `view_log`; send `view.export.result` to PN; notify SR on success/fail.

Outputs

- `view.export.result` envelope (Cold to PN) and `ack` to SR; `view_log` written.

References

- 04-protocol/EXAMPLES/view.export.request.json; 04-protocol/EXAMPLES/view.export.result.json
