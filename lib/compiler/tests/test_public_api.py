"""Regression tests for questfoundry_compiler package exports."""

import importlib


def test_prompt_assemblers_exposed_via_package_root() -> None:
    module = importlib.import_module("questfoundry_compiler")

    assert hasattr(module, "PromptAssembler")
    assert hasattr(module, "ReferenceResolver")
    assert hasattr(module, "StandalonePromptAssembler")


def test_spec_compiler_still_available() -> None:
    module = importlib.import_module("questfoundry_compiler")

    assert hasattr(module, "SpecCompiler")
    assert callable(module.SpecCompiler)
