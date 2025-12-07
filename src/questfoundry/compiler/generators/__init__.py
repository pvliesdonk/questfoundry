"""Code generators for QuestFoundry.

This package contains generators that transform parsed IR into executable Python code.
Generated code is written to the ``generated/`` directory and checked into version control.

Modules
-------
ontology
    Generates Pydantic models from ontology definitions (artifact-type, enum-type).

See Also
--------
:mod:`questfoundry.compiler.parser` : MyST directive parsing
:mod:`questfoundry.compiler.models` : Intermediate representation
"""

from questfoundry.compiler.generators.ontology import generate_models

__all__ = ["generate_models"]
