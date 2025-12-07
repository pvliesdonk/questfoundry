"""Compiler module - transforms MyST domain files into generated code.

This package provides the compilation pipeline that reads MyST domain
definitions and generates executable Python code.

Main Entry Points
-----------------
compile_domain
    Compile all domain files to generated code.
compile_ontology
    Compile ontology definitions to Pydantic models.

Example Usage
-------------
>>> from questfoundry.compiler import compile_domain
>>> result = compile_domain()
>>> print(f"Generated {len(result)} files")
"""

from questfoundry.compiler.compile import compile_domain, compile_ontology

__all__ = ["compile_domain", "compile_ontology"]
