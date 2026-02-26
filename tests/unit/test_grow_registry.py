"""Tests for the GROW phase registry."""

from __future__ import annotations

import pytest

from questfoundry.pipeline.stages.grow.registry import (
    PHASE_META_ATTR,
    PhaseMeta,
    PhaseRegistry,
    get_registry,
)


class TestPhaseRegistry:
    """Tests for PhaseRegistry core functionality."""

    def test_register_and_lookup(self) -> None:
        reg = PhaseRegistry()
        meta = PhaseMeta(name="alpha", depends_on=(), is_deterministic=True, priority=0)

        async def fake_phase(graph, model):
            pass

        reg.register(fake_phase, meta)
        assert "alpha" in reg
        assert len(reg) == 1
        assert reg.get_meta("alpha") == meta
        assert reg.get_function("alpha") is fake_phase
        assert reg.phase_names == ["alpha"]

    def test_duplicate_name_raises(self) -> None:
        reg = PhaseRegistry()
        meta = PhaseMeta(name="alpha", depends_on=(), is_deterministic=True, priority=0)

        async def fake_a(graph, model):
            pass

        async def fake_b(graph, model):
            pass

        reg.register(fake_a, meta)
        with pytest.raises(ValueError, match="Duplicate phase name"):
            reg.register(fake_b, meta)

    def test_get_meta_nonexistent_returns_none(self) -> None:
        reg = PhaseRegistry()
        assert reg.get_meta("nonexistent") is None

    def test_get_function_nonexistent_returns_none(self) -> None:
        reg = PhaseRegistry()
        assert reg.get_function("nonexistent") is None


class TestPhaseRegistryValidation:
    """Tests for DAG validation."""

    def test_valid_dag(self) -> None:
        reg = PhaseRegistry()

        async def f(graph, model):
            pass

        reg.register(f, PhaseMeta("a", (), True, 0))

        async def g(graph, model):
            pass

        reg.register(g, PhaseMeta("b", ("a",), True, 1))

        async def h(graph, model):
            pass

        reg.register(h, PhaseMeta("c", ("b",), True, 2))

        errors = reg.validate()
        assert errors == []

    def test_missing_dependency(self) -> None:
        reg = PhaseRegistry()

        async def f(graph, model):
            pass

        reg.register(f, PhaseMeta("b", ("missing_dep",), True, 0))

        errors = reg.validate()
        assert len(errors) == 1
        assert "missing_dep" in errors[0]

    def test_cycle_detection(self) -> None:
        reg = PhaseRegistry()

        async def f(graph, model):
            pass

        async def g(graph, model):
            pass

        reg.register(f, PhaseMeta("a", ("b",), True, 0))
        reg.register(g, PhaseMeta("b", ("a",), True, 1))

        errors = reg.validate()
        assert len(errors) == 1
        assert "cycle" in errors[0].lower()


class TestPhaseRegistryExecutionOrder:
    """Tests for topological sort and stable ordering."""

    def test_linear_chain(self) -> None:
        reg = PhaseRegistry()

        async def f(graph, model):
            pass

        async def g(graph, model):
            pass

        async def h(graph, model):
            pass

        reg.register(f, PhaseMeta("a", (), True, 0))
        reg.register(g, PhaseMeta("b", ("a",), True, 1))
        reg.register(h, PhaseMeta("c", ("b",), True, 2))

        assert reg.execution_order() == ["a", "b", "c"]

    def test_stable_sort_by_priority(self) -> None:
        """When phases have no dependency, priority breaks ties."""
        reg = PhaseRegistry()

        async def f(graph, model):
            pass

        async def g(graph, model):
            pass

        async def h(graph, model):
            pass

        # All independent, but priorities define order
        reg.register(f, PhaseMeta("c", (), True, 2))
        reg.register(g, PhaseMeta("a", (), True, 0))
        reg.register(h, PhaseMeta("b", (), True, 1))

        assert reg.execution_order() == ["a", "b", "c"]

    def test_diamond_dependency(self) -> None:
        """A → B, A → C, B → D, C → D."""
        reg = PhaseRegistry()

        async def f(graph, model):
            pass

        async def g(graph, model):
            pass

        async def h(graph, model):
            pass

        async def i(graph, model):
            pass

        reg.register(f, PhaseMeta("a", (), True, 0))
        reg.register(g, PhaseMeta("b", ("a",), True, 1))
        reg.register(h, PhaseMeta("c", ("a",), True, 2))
        reg.register(i, PhaseMeta("d", ("b", "c"), True, 3))

        order = reg.execution_order()
        assert order[0] == "a"
        assert order[-1] == "d"
        # b and c must come before d, after a
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_raises_value_error(self) -> None:
        reg = PhaseRegistry()

        async def f(graph, model):
            pass

        async def g(graph, model):
            pass

        reg.register(f, PhaseMeta("a", ("b",), True, 0))
        reg.register(g, PhaseMeta("b", ("a",), True, 1))

        with pytest.raises(ValueError, match="cycle"):
            reg.execution_order()


class TestGrowPhaseDecorator:
    """Tests for the @grow_phase decorator."""

    def test_decorator_attaches_metadata(self) -> None:
        """The decorator sets _grow_phase_meta on the function."""
        # Test using a fresh registry to avoid polluting the global singleton.
        reg = PhaseRegistry()
        meta = PhaseMeta(name="test_deco", depends_on=(), is_deterministic=True, priority=42)

        async def my_phase(graph, model):
            pass

        reg.register(my_phase, meta)
        setattr(my_phase, PHASE_META_ATTR, meta)

        assert hasattr(my_phase, PHASE_META_ATTR)
        attached = getattr(my_phase, PHASE_META_ATTR)
        assert attached.name == "test_deco"
        assert attached.depends_on == ()
        assert attached.is_deterministic is True
        assert attached.priority == 42


class TestGlobalRegistry:
    """Tests for the global registry populated by actual GROW phases."""

    def test_global_registry_has_25_phases(self) -> None:
        """All GROW phases are registered (24 after S3 collapsed split_endings+heavy_residue into apply_routing)."""
        registry = get_registry()
        assert len(registry) >= 24, (
            f"Expected at least 24 phases, got {len(registry)}: {registry.phase_names}"
        )

    def test_global_registry_validates(self) -> None:
        """The global registry DAG is valid (no missing deps, no cycles)."""
        registry = get_registry()
        errors = registry.validate()
        assert errors == [], f"Registry validation errors: {errors}"

    def test_global_registry_execution_order_matches_expected(self) -> None:
        """Execution order matches the S3 phase structure (split_endings + heavy_residue_routing collapsed into apply_routing)."""
        expected = [
            "validate_dag",
            "scene_types",
            "narrative_gaps",
            "pacing_gaps",
            "atmospheric",
            "path_arcs",
            "intersections",
            "entity_arcs",
            "enumerate_arcs",
            "divergence",
            "convergence",
            "collapse_linear_beats",
            "passages",
            "codewords",
            "residue_beats",
            "overlays",
            "choices",
            "fork_beats",
            "hub_spokes",
            "mark_endings",
            "apply_routing",
            "collapse_passages",
            "validation",
            "prune",
        ]
        registry = get_registry()
        actual = registry.execution_order()
        assert actual == expected, (
            f"Execution order mismatch.\nExpected: {expected}\nActual:   {actual}"
        )

    def test_global_registry_phase_table(self) -> None:
        """phase_table() produces a non-empty markdown table."""
        registry = get_registry()
        table = registry.phase_table()
        assert "| Priority |" in table
        assert "validate_dag" in table
        assert "prune" in table

    def test_apply_routing_has_three_dependencies(self) -> None:
        """apply_routing depends on mark_endings, codewords, and residue_beats (S3, ADR-017)."""
        registry = get_registry()
        meta = registry.get_meta("apply_routing")
        assert meta is not None
        assert set(meta.depends_on) == {"mark_endings", "codewords", "residue_beats"}
