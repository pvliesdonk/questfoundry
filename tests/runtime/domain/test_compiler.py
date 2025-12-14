"""
Tests for the schema compiler.

Tests cover:
- Basic field type compilation
- Nested object/array compilation
- Constraints (min/max, length, enum)
- Artifact type compilation with system fields
"""

from questfoundry.runtime.domain.compiler import (
    compile_artifact_type_schema,
    compile_schema,
)
from questfoundry.runtime.models.enums import FieldType
from questfoundry.runtime.models.fields import FieldDefinition


class TestCompileSchema:
    """Tests for compile_schema function."""

    def test_compile_empty_schema(self):
        """compile_schema produces valid schema for empty fields."""
        schema = compile_schema([])

        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert "required" not in schema

    def test_compile_with_title_and_description(self):
        """compile_schema includes title and description."""
        schema = compile_schema(
            fields=[],
            title="Test Schema",
            description="A test schema",
        )

        assert schema["title"] == "Test Schema"
        assert schema["description"] == "A test schema"

    def test_compile_string_field(self):
        """compile_schema handles string fields."""
        fields = [FieldDefinition(name="name", type=FieldType.STRING, required=True)]
        schema = compile_schema(fields)

        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["type"] == "string"
        assert "name" in schema["required"]

    def test_compile_text_field(self):
        """compile_schema handles text fields (long strings)."""
        fields = [FieldDefinition(name="content", type=FieldType.TEXT)]
        schema = compile_schema(fields)

        assert schema["properties"]["content"]["type"] == "string"

    def test_compile_integer_field(self):
        """compile_schema handles integer fields."""
        fields = [FieldDefinition(name="count", type=FieldType.INTEGER)]
        schema = compile_schema(fields)

        assert schema["properties"]["count"]["type"] == "integer"

    def test_compile_number_field(self):
        """compile_schema handles number (float) fields."""
        fields = [FieldDefinition(name="score", type=FieldType.NUMBER)]
        schema = compile_schema(fields)

        assert schema["properties"]["score"]["type"] == "number"

    def test_compile_boolean_field(self):
        """compile_schema handles boolean fields."""
        fields = [FieldDefinition(name="active", type=FieldType.BOOLEAN)]
        schema = compile_schema(fields)

        assert schema["properties"]["active"]["type"] == "boolean"

    def test_compile_date_field(self):
        """compile_schema handles date fields."""
        fields = [FieldDefinition(name="created", type=FieldType.DATE)]
        schema = compile_schema(fields)

        assert schema["properties"]["created"]["type"] == "string"
        assert schema["properties"]["created"]["format"] == "date"

    def test_compile_datetime_field(self):
        """compile_schema handles datetime fields."""
        fields = [FieldDefinition(name="timestamp", type=FieldType.DATETIME)]
        schema = compile_schema(fields)

        assert schema["properties"]["timestamp"]["type"] == "string"
        assert schema["properties"]["timestamp"]["format"] == "date-time"

    def test_compile_uri_field(self):
        """compile_schema handles URI fields."""
        fields = [FieldDefinition(name="link", type=FieldType.URI)]
        schema = compile_schema(fields)

        assert schema["properties"]["link"]["type"] == "string"
        assert schema["properties"]["link"]["format"] == "uri"

    def test_compile_ref_field(self):
        """compile_schema handles reference fields."""
        fields = [
            FieldDefinition(
                name="author_ref",
                type=FieldType.REF,
                ref_target="agent",
            )
        ]
        schema = compile_schema(fields)

        assert schema["properties"]["author_ref"]["type"] == "string"
        assert "Reference to agent" in schema["properties"]["author_ref"]["$comment"]


class TestCompileSchemaConstraints:
    """Tests for constraint compilation."""

    def test_compile_string_with_enum(self):
        """compile_schema handles string enum constraints."""
        fields = [
            FieldDefinition(
                name="status",
                type=FieldType.STRING,
                enum=["draft", "published", "archived"],
            )
        ]
        schema = compile_schema(fields)

        assert schema["properties"]["status"]["enum"] == ["draft", "published", "archived"]

    def test_compile_string_with_format(self):
        """compile_schema handles string format constraints."""
        fields = [
            FieldDefinition(
                name="email",
                type=FieldType.STRING,
                format="email",
            )
        ]
        schema = compile_schema(fields)

        assert schema["properties"]["email"]["format"] == "email"

    def test_compile_string_with_length(self):
        """compile_schema handles string length constraints."""
        fields = [
            FieldDefinition(
                name="code",
                type=FieldType.STRING,
                min_length=3,
                max_length=10,
            )
        ]
        schema = compile_schema(fields)

        assert schema["properties"]["code"]["minLength"] == 3
        assert schema["properties"]["code"]["maxLength"] == 10

    def test_compile_number_with_range(self):
        """compile_schema handles numeric range constraints."""
        fields = [
            FieldDefinition(
                name="percentage",
                type=FieldType.NUMBER,
                min=0.0,
                max=100.0,
            )
        ]
        schema = compile_schema(fields)

        assert schema["properties"]["percentage"]["minimum"] == 0.0
        assert schema["properties"]["percentage"]["maximum"] == 100.0

    def test_compile_field_with_description(self):
        """compile_schema includes field descriptions."""
        fields = [
            FieldDefinition(
                name="title",
                type=FieldType.STRING,
                description="The artifact title",
            )
        ]
        schema = compile_schema(fields)

        assert schema["properties"]["title"]["description"] == "The artifact title"

    def test_compile_field_with_default(self):
        """compile_schema includes field defaults."""
        fields = [
            FieldDefinition(
                name="priority",
                type=FieldType.INTEGER,
                default=1,
            )
        ]
        schema = compile_schema(fields)

        assert schema["properties"]["priority"]["default"] == 1


class TestCompileSchemaArrays:
    """Tests for array field compilation."""

    def test_compile_simple_array(self):
        """compile_schema handles arrays of scalars."""
        fields = [
            FieldDefinition(
                name="tags",
                type=FieldType.ARRAY,
                items_type=FieldType.STRING,
            )
        ]
        schema = compile_schema(fields)

        assert schema["properties"]["tags"]["type"] == "array"
        assert schema["properties"]["tags"]["items"]["type"] == "string"

    def test_compile_array_with_length(self):
        """compile_schema handles array length constraints."""
        fields = [
            FieldDefinition(
                name="choices",
                type=FieldType.ARRAY,
                items_type=FieldType.STRING,
                min_length=1,
                max_length=5,
            )
        ]
        schema = compile_schema(fields)

        assert schema["properties"]["choices"]["minItems"] == 1
        assert schema["properties"]["choices"]["maxItems"] == 5

    def test_compile_array_of_objects(self):
        """compile_schema handles arrays of complex objects."""
        fields = [
            FieldDefinition(
                name="items",
                type=FieldType.ARRAY,
                items=FieldDefinition(
                    name="item",
                    type=FieldType.OBJECT,
                    properties=[
                        FieldDefinition(name="id", type=FieldType.STRING, required=True),
                        FieldDefinition(name="value", type=FieldType.NUMBER),
                    ],
                ),
            )
        ]
        schema = compile_schema(fields)

        items_schema = schema["properties"]["items"]["items"]
        assert items_schema["type"] == "object"
        assert "id" in items_schema["properties"]
        assert "value" in items_schema["properties"]
        assert items_schema["required"] == ["id"]


class TestCompileSchemaObjects:
    """Tests for nested object compilation."""

    def test_compile_nested_object(self):
        """compile_schema handles nested objects."""
        fields = [
            FieldDefinition(
                name="metadata",
                type=FieldType.OBJECT,
                properties=[
                    FieldDefinition(name="author", type=FieldType.STRING),
                    FieldDefinition(name="version", type=FieldType.INTEGER),
                ],
            )
        ]
        schema = compile_schema(fields)

        metadata = schema["properties"]["metadata"]
        assert metadata["type"] == "object"
        assert "author" in metadata["properties"]
        assert "version" in metadata["properties"]

    def test_compile_deeply_nested(self):
        """compile_schema handles deeply nested structures."""
        fields = [
            FieldDefinition(
                name="level1",
                type=FieldType.OBJECT,
                properties=[
                    FieldDefinition(
                        name="level2",
                        type=FieldType.OBJECT,
                        properties=[
                            FieldDefinition(name="level3", type=FieldType.STRING),
                        ],
                    ),
                ],
            )
        ]
        schema = compile_schema(fields)

        level1 = schema["properties"]["level1"]
        level2 = level1["properties"]["level2"]
        level3 = level2["properties"]["level3"]

        assert level3["type"] == "string"


class TestCompileArtifactTypeSchema:
    """Tests for compile_artifact_type_schema function."""

    def test_compile_artifact_type_adds_system_fields(self):
        """compile_artifact_type_schema adds system fields."""
        schema = compile_artifact_type_schema(
            artifact_type_id="section",
            artifact_type_name="Section",
            fields=[
                FieldDefinition(name="title", type=FieldType.STRING, required=True),
            ],
        )

        # Check system fields
        assert "_id" in schema["properties"]
        assert "_type" in schema["properties"]
        assert "_version" in schema["properties"]
        assert "_created_at" in schema["properties"]
        assert "_updated_at" in schema["properties"]
        assert "_created_by" in schema["properties"]
        assert "_lifecycle_state" in schema["properties"]

        # User field also present
        assert "title" in schema["properties"]

    def test_compile_artifact_type_const_type(self):
        """compile_artifact_type_schema sets const for _type."""
        schema = compile_artifact_type_schema(
            artifact_type_id="section",
            artifact_type_name="Section",
            fields=[],
        )

        assert schema["properties"]["_type"]["const"] == "section"

    def test_compile_artifact_type_system_required(self):
        """compile_artifact_type_schema marks system fields as required."""
        schema = compile_artifact_type_schema(
            artifact_type_id="section",
            artifact_type_name="Section",
            fields=[
                FieldDefinition(name="content", type=FieldType.TEXT, required=True),
            ],
        )

        # System fields required
        assert "_id" in schema["required"]
        assert "_type" in schema["required"]
        assert "_version" in schema["required"]

        # User required field also present
        assert "content" in schema["required"]

    def test_compile_artifact_type_with_description(self):
        """compile_artifact_type_schema includes description."""
        schema = compile_artifact_type_schema(
            artifact_type_id="section",
            artifact_type_name="Section",
            fields=[],
            description="A story section",
        )

        assert schema["description"] == "A story section"
