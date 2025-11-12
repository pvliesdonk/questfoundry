# Intent Handler — gateway.define

Inputs

- Gateway purpose, state/codeword design, PN enforcement constraints.

Preconditions

- TU open; related sections identified; PN phrasing constraints known.

Process

1. Write single, clear condition (positive form preferred), avoid multi-part checks.
2. Author diegetic PN phrasing that reveals no mechanics.
3. Prove obtainability with at least one path; list where the condition is earned.
4. Define failure path (fallback beat) where applicable; ensure reachability remains sane.
5. Emit `tu.checkpoint` with condition, PN line, obtainability proof, and failure path.

Outputs

- Gateway definition note and PN line guidance; `tu.checkpoint`.

References

- 00-north-star/QUALITY_BARS.md (Gateways, Presentation)
