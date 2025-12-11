"""Code generators for QuestFoundry.

This package contains generators that transform parsed IR into executable Python code.
Generated code is written to the ``generated/`` directory and checked into version control.

Modules
-------
ontology
    Generates Pydantic models from ontology definitions (artifact-type, enum-type).
roles
    Generates role configurations from role definitions (role-meta, role-tools, etc.).
loops
    Generates loop/workflow definitions from loop definitions (loop-meta, graph-node, etc.).

See Also
--------
:mod:`questfoundry.compiler.parser` : MyST directive parsing
:mod:`questfoundry.compiler.models` : Intermediate representation
"""

from questfoundry.compiler.generators._warning import GENERATED_FILE_WARNING
from questfoundry.compiler.generators.loops import generate_loops
from questfoundry.compiler.generators.ontology import generate_models
from questfoundry.compiler.generators.roles import generate_roles

__all__ = ["generate_models", "generate_roles", "generate_loops", "GENERATED_FILE_WARNING"]
