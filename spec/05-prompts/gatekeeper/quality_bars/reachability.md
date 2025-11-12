# Quality Bar — Reachability

Checks

- Keystone beats are reachable from at least one viable path without paradox.
- Required keys/conditions are obtainable along at least one path.

Evidence

- Provide one concrete path per keystone (list sections/choices).

Common Failures

- Locked content with no key; circular prerequisites; overlong dependency chains.

Remediation

- Add alternative routes; shorten chains; surface keys earlier; add fail-forward beats.
