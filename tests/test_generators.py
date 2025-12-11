"""Tests for the code generators.

This module tests the ontology code generator that transforms
parsed IR into Pydantic models.
"""

import tempfile
from pathlib import Path

from questfoundry.compiler.generators.ontology import (
    class_name,
    generate_artifacts_code,
    generate_enums_code,
    generate_init_code,
    generate_models,
    map_type,
    python_safe_name,
)
from questfoundry.compiler.models import (
    ArtifactFieldIR,
    ArtifactTypeIR,
    EnumTypeIR,
    EnumValueIR,
    StoreType,
)

# =============================================================================
# Helper Function Tests
# =============================================================================


class TestMapType:
    """Tests for map_type function."""

    def test_primitives(self) -> None:
        """Primitive types map directly."""
        assert map_type("str", {}) == "str"
        assert map_type("int", {}) == "int"
        assert map_type("float", {}) == "float"
        assert map_type("bool", {}) == "bool"

    def test_string_alias(self) -> None:
        """'string' maps to 'str'."""
        assert map_type("string", {}) == "str"

    def test_generic_containers(self) -> None:
        """Generic containers pass through."""
        assert map_type("list[str]", {}) == "list[str]"
        assert map_type("dict[str, Any]", {}) == "dict[str, Any]"

    def test_enum_reference(self) -> None:
        """Enum types resolve to their name."""
        enums = {"HookType": EnumTypeIR(id="HookType", values=[])}
        assert map_type("HookType", enums) == "HookType"

    def test_unknown_type(self) -> None:
        """Unknown types pass through with comment."""
        result = map_type("UnknownType", {})
        assert "UnknownType" in result


class TestPythonSafeName:
    """Tests for python_safe_name function."""

    def test_valid_name(self) -> None:
        """Valid names pass through unchanged."""
        assert python_safe_name("hook_card") == "hook_card"
        assert python_safe_name("HookType") == "HookType"

    def test_hyphen_to_underscore(self) -> None:
        """Hyphens become underscores."""
        assert python_safe_name("hook-card") == "hook_card"

    def test_special_chars(self) -> None:
        """Special characters become underscores."""
        assert python_safe_name("foo.bar") == "foo_bar"


class TestClassName:
    """Tests for class_name function."""

    def test_snake_case(self) -> None:
        """Snake case converts to PascalCase."""
        assert class_name("hook_card") == "HookCard"
        assert class_name("canon_entry") == "CanonEntry"

    def test_kebab_case(self) -> None:
        """Kebab case converts to PascalCase."""
        assert class_name("hook-card") == "HookCard"

    def test_single_word(self) -> None:
        """Single word capitalizes."""
        assert class_name("hook") == "Hook"


# =============================================================================
# Enum Generator Tests
# =============================================================================


class TestGenerateEnumsCode:
    """Tests for generate_enums_code function."""

    def test_basic_enum(self) -> None:
        """Generate a basic enum with values."""
        enums = {
            "HookType": EnumTypeIR(
                id="HookType",
                description="Hook classification",
                values=[
                    EnumValueIR(name="narrative", description="Story content"),
                    EnumValueIR(name="scene", description="Scene changes"),
                ],
            )
        }

        code = generate_enums_code(enums)

        assert "class HookType(StrEnum):" in code
        assert '"Hook classification"' in code
        assert 'NARRATIVE = "narrative"' in code
        assert 'SCENE = "scene"' in code
        assert '"Story content"' in code

    def test_multiple_enums(self) -> None:
        """Generate multiple enums."""
        enums = {
            "HookType": EnumTypeIR(
                id="HookType",
                values=[EnumValueIR(name="narrative", description="")],
            ),
            "HookStatus": EnumTypeIR(
                id="HookStatus",
                values=[EnumValueIR(name="proposed", description="")],
            ),
        }

        code = generate_enums_code(enums)

        assert "class HookType(StrEnum):" in code
        assert "class HookStatus(StrEnum):" in code

    def test_enum_imports(self) -> None:
        """Generated code has correct imports."""
        code = generate_enums_code({})

        assert "from enum import StrEnum" in code


# =============================================================================
# Artifact Generator Tests
# =============================================================================


class TestGenerateArtifactsCode:
    """Tests for generate_artifacts_code function."""

    def test_basic_artifact(self) -> None:
        """Generate a basic artifact model."""
        artifacts = {
            "hook_card": ArtifactTypeIR(
                id="hook_card",
                name="Hook Card",
                store=StoreType.HOT,
                lifecycle=["proposed", "accepted"],
                fields=[
                    ArtifactFieldIR(
                        artifact="hook_card",
                        name="title",
                        type="str",
                        required=True,
                        description="Hook title",
                    ),
                    ArtifactFieldIR(
                        artifact="hook_card",
                        name="owner",
                        type="str",
                        required=False,
                        description="Owner role",
                    ),
                ],
            )
        }

        code = generate_artifacts_code(artifacts, {})

        assert "class HookCard(BaseModel):" in code
        # Class docstring starts with artifact name
        assert '"""Hook Card.' in code
        # Fields with documentation - now multi-line with title and examples
        assert "title: str = Field(" in code
        assert 'description="Hook title"' in code
        assert "owner: str | None = Field(" in code
        assert 'description="Owner role"' in code

    def test_artifact_with_enum_field(self) -> None:
        """Artifacts with enum fields import the enum."""
        enums = {
            "HookType": EnumTypeIR(
                id="HookType",
                values=[EnumValueIR(name="narrative", description="")],
            )
        }
        artifacts = {
            "hook_card": ArtifactTypeIR(
                id="hook_card",
                name="Hook Card",
                store=StoreType.HOT,
                fields=[
                    ArtifactFieldIR(
                        artifact="hook_card",
                        name="hook_type",
                        type="HookType",
                        required=True,
                    ),
                ],
            )
        }

        code = generate_artifacts_code(artifacts, enums)

        assert "from questfoundry.generated.models.enums import HookType" in code
        assert "hook_type: HookType" in code

    def test_artifact_lifecycle(self) -> None:
        """Lifecycle states are included in the model."""
        artifacts = {
            "hook_card": ArtifactTypeIR(
                id="hook_card",
                name="Hook Card",
                store=StoreType.HOT,
                lifecycle=["draft", "review", "final"],
                fields=[],
            )
        }

        code = generate_artifacts_code(artifacts, {})

        # Lifecycle is now a ClassVar uppercase class constant
        assert 'LIFECYCLE: ClassVar[list[str]] = ["draft", "review", "final"]' in code
        # Docstring includes lifecycle info
        assert "draft → review → final" in code


# =============================================================================
# Integration Tests
# =============================================================================


class TestGenerateModels:
    """Integration tests for generate_models function."""

    def test_generate_to_temp_dir(self) -> None:
        """Generate models to a temporary directory."""
        enums = {
            "HookType": EnumTypeIR(
                id="HookType",
                description="Hook type",
                values=[
                    EnumValueIR(name="narrative", description="Story"),
                    EnumValueIR(name="scene", description="Scene"),
                ],
            ),
        }
        artifacts = {
            "hook_card": ArtifactTypeIR(
                id="hook_card",
                name="Hook Card",
                store=StoreType.HOT,
                lifecycle=["proposed", "accepted"],
                fields=[
                    ArtifactFieldIR(
                        artifact="hook_card",
                        name="title",
                        type="str",
                        required=True,
                    ),
                    ArtifactFieldIR(
                        artifact="hook_card",
                        name="hook_type",
                        type="HookType",
                        required=True,
                    ),
                ],
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_models(enums, artifacts, tmpdir)

            # Check files were created
            assert "enums.py" in result
            assert "artifacts.py" in result
            assert "__init__.py" in result

            # Check enums.py content
            enums_content = Path(result["enums.py"]).read_text()
            assert "class HookType(StrEnum):" in enums_content
            assert 'NARRATIVE = "narrative"' in enums_content

            # Check artifacts.py content
            artifacts_content = Path(result["artifacts.py"]).read_text()
            assert "class HookCard(BaseModel):" in artifacts_content
            assert "from questfoundry.generated.models.enums import HookType" in artifacts_content

            # Check __init__.py content
            init_content = Path(result["__init__.py"]).read_text()
            assert "HookType" in init_content
            assert "HookCard" in init_content


class TestGenerateInitCode:
    """Tests for generate_init_code function."""

    def test_exports_all(self) -> None:
        """Init exports all enums and artifacts."""
        enums = {
            "HookType": EnumTypeIR(id="HookType", values=[]),
            "HookStatus": EnumTypeIR(id="HookStatus", values=[]),
        }
        artifacts = {
            "hook_card": ArtifactTypeIR(
                id="hook_card", name="Hook Card", store=StoreType.HOT, fields=[]
            ),
        }

        code = generate_init_code(enums, artifacts)

        assert '"HookType"' in code
        assert '"HookStatus"' in code
        assert '"HookCard"' in code
        assert "__all__ = [" in code
