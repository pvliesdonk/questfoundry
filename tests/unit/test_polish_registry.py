"""Tests for POLISH phase registry."""

from __future__ import annotations

import pytest

from questfoundry.pipeline.registry import PHASE_META_ATTR, PhaseMeta, PhaseRegistry


class TestPhaseRegistry:
    """Tests for the shared PhaseRegistry class."""

    def test_register_phase(self) -> None:
        """Register a phase and retrieve it."""
        registry = PhaseRegistry()
        meta = PhaseMeta(name="test", depends_on=(), is_deterministic=True, priority=0)

        async def dummy(graph, model):
            pass

        registry.register(dummy, meta)
        assert "test" in registry
        assert len(registry) == 1
        assert registry.get_meta("test") == meta
        assert registry.get_function("test") is dummy

    def test_duplicate_registration_raises(self) -> None:
        """Registering the same phase name twice raises ValueError."""
        registry = PhaseRegistry()
        meta = PhaseMeta(name="dup", depends_on=(), is_deterministic=True, priority=0)

        async def fn1(graph, model):
            pass

        async def fn2(graph, model):
            pass

        registry.register(fn1, meta)
        with pytest.raises(ValueError, match="Duplicate phase name"):
            registry.register(fn2, meta)

    def test_validate_missing_dependency(self) -> None:
        """Validate catches missing dependency references."""
        registry = PhaseRegistry()
        meta = PhaseMeta(
            name="phase_b", depends_on=("nonexistent",), is_deterministic=True, priority=0
        )

        async def fn(graph, model):
            pass

        registry.register(fn, meta)
        errors = registry.validate()
        assert len(errors) == 1
        assert "nonexistent" in errors[0]

    def test_validate_cycle_detection(self) -> None:
        """Validate catches dependency cycles."""
        registry = PhaseRegistry()

        async def fn(graph, model):
            pass

        registry.register(
            fn,
            PhaseMeta(name="a", depends_on=("b",), is_deterministic=True, priority=0),
        )
        registry.register(
            fn,
            PhaseMeta(name="b", depends_on=("a",), is_deterministic=True, priority=1),
        )
        errors = registry.validate()
        assert len(errors) == 1
        assert "cycle" in errors[0].lower()

    def test_validate_empty_is_valid(self) -> None:
        """Empty registry validates successfully."""
        registry = PhaseRegistry()
        assert registry.validate() == []

    def test_execution_order_respects_dependencies(self) -> None:
        """execution_order returns dependencies before dependents."""
        registry = PhaseRegistry()

        async def fn(graph, model):
            pass

        registry.register(
            fn,
            PhaseMeta(name="first", depends_on=(), is_deterministic=True, priority=0),
        )
        registry.register(
            fn,
            PhaseMeta(name="second", depends_on=("first",), is_deterministic=True, priority=1),
        )
        registry.register(
            fn,
            PhaseMeta(name="third", depends_on=("second",), is_deterministic=True, priority=2),
        )
        order = registry.execution_order()
        assert order == ["first", "second", "third"]

    def test_execution_order_tiebreaks_on_priority(self) -> None:
        """Independent phases are ordered by priority."""
        registry = PhaseRegistry()

        async def fn(graph, model):
            pass

        registry.register(
            fn,
            PhaseMeta(name="low", depends_on=(), is_deterministic=True, priority=10),
        )
        registry.register(
            fn,
            PhaseMeta(name="high", depends_on=(), is_deterministic=True, priority=1),
        )
        order = registry.execution_order()
        assert order.index("high") < order.index("low")

    def test_phase_names_returns_insertion_order(self) -> None:
        """phase_names returns names in insertion order."""
        registry = PhaseRegistry()

        async def fn(graph, model):
            pass

        for i, name in enumerate(["alpha", "beta", "gamma"]):
            registry.register(
                fn,
                PhaseMeta(name=name, depends_on=(), is_deterministic=True, priority=i),
            )
        assert registry.phase_names == ["alpha", "beta", "gamma"]

    def test_phase_table_format(self) -> None:
        """phase_table returns markdown table."""
        registry = PhaseRegistry()

        async def fn(graph, model):
            pass

        registry.register(
            fn,
            PhaseMeta(name="test_phase", depends_on=(), is_deterministic=False, priority=0),
        )
        table = registry.phase_table()
        assert "test_phase" in table
        assert "llm" in table
        assert "Priority" in table


class TestPolishPhaseDecorator:
    """Tests for the @polish_phase decorator."""

    def test_decorator_registers_phase(self) -> None:
        """@polish_phase registers the function in the POLISH registry."""
        from questfoundry.pipeline.stages.polish.registry import (
            get_polish_registry,
            polish_phase,
        )

        registry = get_polish_registry()
        initial_count = len(registry)

        @polish_phase(name=f"test_phase_{initial_count}", is_deterministic=True)
        async def my_phase(graph, model):
            pass

        assert len(registry) == initial_count + 1
        assert hasattr(my_phase, PHASE_META_ATTR)

    def test_decorator_sets_meta_attribute(self) -> None:
        """@polish_phase sets _phase_meta on the decorated function."""
        from questfoundry.pipeline.stages.polish.registry import (
            get_polish_registry,
            polish_phase,
        )

        registry = get_polish_registry()
        phase_name = f"meta_test_{len(registry)}"

        @polish_phase(name=phase_name, depends_on=["other"], is_deterministic=False, priority=42)
        async def my_phase(graph, model):
            pass

        meta = getattr(my_phase, PHASE_META_ATTR)
        assert meta.name == phase_name
        assert meta.depends_on == ("other",)
        assert meta.is_deterministic is False
        assert meta.priority == 42


class TestGrowRegistryBackwardCompat:
    """Verify GROW registry still works after extraction to shared module."""

    def test_grow_registry_imports(self) -> None:
        """GROW registry classes are importable from both locations."""
        from questfoundry.pipeline.registry import PhaseMeta as SharedPhaseMeta
        from questfoundry.pipeline.registry import PhaseRegistry as SharedRegistry
        from questfoundry.pipeline.stages.grow.registry import PhaseMeta as GrowPhaseMeta
        from questfoundry.pipeline.stages.grow.registry import PhaseRegistry as GrowRegistry

        # They should be the exact same classes
        assert SharedPhaseMeta is GrowPhaseMeta
        assert SharedRegistry is GrowRegistry

    def test_grow_registry_has_phases(self) -> None:
        """GROW registry should have existing phases registered."""
        from questfoundry.pipeline.stages.grow.registry import get_registry

        registry = get_registry()
        assert len(registry) > 0
        assert "validate_dag" in registry

    def test_grow_registry_validates(self) -> None:
        """GROW registry DAG should be valid."""
        from questfoundry.pipeline.stages.grow.registry import get_registry

        registry = get_registry()
        errors = registry.validate()
        assert errors == [], f"GROW registry validation errors: {errors}"
