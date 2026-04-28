"""Tests for BRAINSTORM stage implementation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from questfoundry.models import (
    BrainstormDilemmasOutput,
    BrainstormEntitiesOutput,
    BrainstormOutput,
)
from questfoundry.pipeline.stages import BrainstormStage, BrainstormStageError, get_stage


def _two_pass_artifacts(
    entities: list[dict] | None = None,
    dilemmas: list[dict] | None = None,
) -> tuple[BrainstormEntitiesOutput, BrainstormDilemmasOutput]:
    """Build matched (entities, dilemmas) artifacts for the two-pass serialize.

    Tests mock ``serialize_to_artifact`` with ``side_effect`` so the first
    call returns the entities artifact and the second returns the dilemmas
    artifact — mirroring how the BRAINSTORM stage now drives the LLM.
    """
    entities = entities or [
        {
            "entity_id": "character::stub",
            "entity_category": "character",
            "name": "Stub",
            "concept": "c",
        },
        {
            "entity_id": "location::stub_a",
            "entity_category": "location",
            "name": "Stub A",
            "concept": "c",
        },
        {
            "entity_id": "location::stub_b",
            "entity_category": "location",
            "name": "Stub B",
            "concept": "c",
        },
    ]
    dilemmas = dilemmas or [
        {
            "dilemma_id": "dilemma::stub_a_or_b",
            "question": "Stub a or b?",
            "why_it_matters": "stakes",
            "answers": [
                {"answer_id": "yes", "description": "Yes", "is_canonical": True},
                {"answer_id": "no", "description": "No", "is_canonical": False},
            ],
            "central_entity_ids": ["character::stub"],
        }
    ]
    return (
        BrainstormEntitiesOutput(entities=entities),
        BrainstormDilemmasOutput(dilemmas=dilemmas),
    )


def _setup_mock_graph_with_vision(mock_graph: MagicMock, vision_data: dict | None = None) -> None:
    """Configure mock graph to pass DREAM contract validation.

    Args:
        mock_graph: MagicMock instance to configure.
        vision_data: Optional dict to override vision node fields. Defaults to minimal valid vision.
    """
    if vision_data is None:
        vision_data = {
            "genre": "fantasy",
            "tone": ["epic"],
            "themes": ["heroism"],
            "audience": "adult",
            "scope": {"story_size": "short"},
            "human_approved": True,
        }

    # Mock get_node for direct vision lookup
    mock_graph.get_node.return_value = vision_data

    # Mock get_nodes_by_type for validate_dream_output vision contract check
    mock_graph.get_nodes_by_type.return_value = {"vision": vision_data}

    # Mock edges to pass R-1.10 check (no edges on vision)
    mock_graph.get_edges.return_value = []

    # Mock all_node_ids for Output-5 check (only vision node)
    mock_graph.all_node_ids.return_value = ["vision"]


# --- Stage Registration Tests ---


def test_brainstorm_stage_registered() -> None:
    """Brainstorm stage is registered automatically."""
    stage = get_stage("brainstorm")
    assert stage is not None
    assert stage.name == "brainstorm"


def test_brainstorm_stage_name() -> None:
    """BrainstormStage has correct name."""
    stage = BrainstormStage()
    assert stage.name == "brainstorm"


# --- Execute Tests ---


@pytest.mark.asyncio
async def test_execute_requires_project_path() -> None:
    """Execute raises error when project_path is not provided."""
    stage = BrainstormStage()  # No project_path

    mock_model = MagicMock()

    with pytest.raises(BrainstormStageError, match="project_path is required"):
        await stage.execute(model=mock_model, user_prompt="test")


@pytest.mark.asyncio
async def test_execute_requires_vision_in_graph() -> None:
    """Execute raises error when vision is not found in graph."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    mock_graph.get_node.return_value = None  # No vision node

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
    ):
        MockGraph.load.return_value = mock_graph

        with pytest.raises(BrainstormStageError, match="BRAINSTORM requires DREAM"):
            await stage.execute(
                model=mock_model,
                user_prompt="test",
                project_path=Path("/test/project"),
            )


@pytest.mark.asyncio
async def test_execute_calls_all_three_phases() -> None:
    """Execute calls discuss, summarize, and serialize phases."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    _setup_mock_graph_with_vision(mock_graph)

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.brainstorm.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.brainstorm.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.brainstorm.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.brainstorm.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = (
            [HumanMessage(content="hi"), AIMessage(content="hello")],
            2,  # llm_calls
            500,  # tokens
        )
        mock_summarize.return_value = ("Brief summary", 100)
        entities_artifact, dilemmas_artifact = _two_pass_artifacts(
            entities=[
                {
                    "entity_id": "character::hero",
                    "entity_category": "character",
                    "name": "Hero",
                    "concept": "A brave warrior",
                },
                {
                    "entity_id": "location::village",
                    "entity_category": "location",
                    "name": "Village",
                    "concept": "Home",
                },
                {
                    "entity_id": "location::keep",
                    "entity_category": "location",
                    "name": "Keep",
                    "concept": "Stronghold",
                },
            ],
            dilemmas=[
                {
                    "dilemma_id": "dilemma::quest_succeed_or_fail",
                    "question": "Will the hero succeed?",
                    "answers": [
                        {
                            "answer_id": "success",
                            "description": "Hero wins",
                            "is_canonical": True,
                        },
                        {
                            "answer_id": "failure",
                            "description": "Hero fails",
                            "is_canonical": False,
                        },
                    ],
                    "central_entity_ids": ["character::hero"],
                    "why_it_matters": "Core story dilemma",
                }
            ],
        )
        mock_serialize.side_effect = [(entities_artifact, 100, 1), (dilemmas_artifact, 100, 1)]

        artifact, llm_calls, tokens = await stage.execute(
            model=mock_model,
            user_prompt="Let's brainstorm",
            project_path=Path("/test/project"),
        )

        # Verify all phases were called (serialize runs twice — entities + dilemmas).
        mock_discuss.assert_called_once()
        mock_summarize.assert_called_once()
        assert mock_serialize.call_count == 2

        # Verify result
        assert len(artifact["entities"]) == 3
        assert len(artifact["dilemmas"]) == 1
        assert llm_calls == 5  # 2 discuss + 1 summarize + 2 serialize
        assert tokens == 800  # 500 + 100 + 100 + 100


@pytest.mark.asyncio
async def test_execute_emits_phase_progress() -> None:
    """Execute emits phase-level progress callbacks when provided."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    _setup_mock_graph_with_vision(mock_graph)
    on_phase_progress = MagicMock()

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.brainstorm.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.brainstorm.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.brainstorm.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.brainstorm.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = (
            [HumanMessage(content="hi"), AIMessage(content="a"), AIMessage(content="b")],
            2,
            500,
        )
        mock_summarize.return_value = ("Brief summary", 100)
        entities_artifact, dilemmas_artifact = _two_pass_artifacts()
        mock_serialize.side_effect = [(entities_artifact, 100, 1), (dilemmas_artifact, 100, 1)]

        await stage.execute(
            model=mock_model,
            user_prompt="Let's brainstorm",
            project_path=Path("/test/project"),
            on_phase_progress=on_phase_progress,
        )

    assert on_phase_progress.mock_calls == [
        call("discuss", "completed", "2 turns"),
        call("summarize", "completed", None),
        call("serialize entities", "completed", "3 entities"),
        call("serialize dilemmas", "completed", "1 dilemmas"),
    ]


@pytest.mark.asyncio
async def test_execute_passes_vision_context_to_discuss() -> None:
    """Execute passes formatted vision context to discuss phase."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    noir_vision = {
        "genre": "noir",
        "subgenre": "detective",
        "tone": ["dark", "moody"],
        "themes": ["corruption", "justice"],
        "audience": "adult",
        "scope": {"story_size": "short"},
        "human_approved": True,
    }
    _setup_mock_graph_with_vision(mock_graph, noir_vision)

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.brainstorm.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.brainstorm.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.brainstorm.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.brainstorm.get_all_research_tools") as mock_tools,
        patch(
            "questfoundry.pipeline.stages.brainstorm.get_brainstorm_discuss_prompt"
        ) as mock_prompt,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_prompt.return_value = "System prompt with vision"
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        entities_artifact, dilemmas_artifact = _two_pass_artifacts()
        mock_serialize.side_effect = [(entities_artifact, 100, 1), (dilemmas_artifact, 100, 1)]

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # Verify get_brainstorm_discuss_prompt was called with vision context
        mock_prompt.assert_called_once()
        call_kwargs = mock_prompt.call_args.kwargs
        assert "vision_context" in call_kwargs
        # Vision context should include genre info
        assert "noir" in call_kwargs["vision_context"]


@pytest.mark.asyncio
async def test_execute_passes_two_pass_serialize_schemas() -> None:
    """Execute drives two-pass serialize: entities schema, then dilemmas schema."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    _setup_mock_graph_with_vision(mock_graph)

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.brainstorm.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.brainstorm.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.brainstorm.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.brainstorm.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        entities_artifact, dilemmas_artifact = _two_pass_artifacts()
        mock_serialize.side_effect = [(entities_artifact, 100, 1), (dilemmas_artifact, 100, 1)]

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        assert mock_serialize.call_count == 2
        first_call_kwargs, second_call_kwargs = (
            mock_serialize.call_args_list[0].kwargs,
            mock_serialize.call_args_list[1].kwargs,
        )
        assert first_call_kwargs["schema"] is BrainstormEntitiesOutput
        assert second_call_kwargs["schema"] is BrainstormDilemmasOutput


@pytest.mark.asyncio
async def test_execute_uses_brainstorm_summarize_prompt() -> None:
    """Execute uses brainstorm-specific summarize prompt."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    _setup_mock_graph_with_vision(mock_graph)

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.brainstorm.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.brainstorm.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.brainstorm.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.brainstorm.get_all_research_tools") as mock_tools,
        patch(
            "questfoundry.pipeline.stages.brainstorm.get_brainstorm_summarize_prompt"
        ) as mock_prompt,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_prompt.return_value = "Brainstorm summarize prompt"
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        entities_artifact, dilemmas_artifact = _two_pass_artifacts()
        mock_serialize.side_effect = [(entities_artifact, 100, 1), (dilemmas_artifact, 100, 1)]

        await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        # Verify summarize was called with brainstorm prompt
        mock_prompt.assert_called_once()
        assert mock_summarize.call_args.kwargs["system_prompt"] == "Brainstorm summarize prompt"


@pytest.mark.asyncio
async def test_execute_returns_artifact_as_dict() -> None:
    """Execute returns artifact as dictionary, not Pydantic model."""
    stage = BrainstormStage()

    mock_model = MagicMock()
    mock_graph = MagicMock()
    _setup_mock_graph_with_vision(mock_graph)

    with (
        patch("questfoundry.pipeline.stages.brainstorm.Graph") as MockGraph,
        patch("questfoundry.pipeline.stages.brainstorm.run_discuss_phase") as mock_discuss,
        patch("questfoundry.pipeline.stages.brainstorm.summarize_discussion") as mock_summarize,
        patch("questfoundry.pipeline.stages.brainstorm.serialize_to_artifact") as mock_serialize,
        patch("questfoundry.pipeline.stages.brainstorm.get_all_research_tools") as mock_tools,
    ):
        MockGraph.load.return_value = mock_graph
        mock_tools.return_value = []
        mock_discuss.return_value = ([], 1, 100)
        mock_summarize.return_value = ("Brief", 50)
        entities_artifact, dilemmas_artifact = _two_pass_artifacts(
            entities=[
                {
                    "entity_id": "character::kay",
                    "entity_category": "character",
                    "name": "Kay",
                    "concept": "Protagonist",
                },
                {
                    "entity_id": "location::manor",
                    "entity_category": "location",
                    "name": "Manor",
                    "concept": "Estate",
                },
                {
                    "entity_id": "location::archive",
                    "entity_category": "location",
                    "name": "Archive",
                    "concept": "Library",
                },
            ],
            dilemmas=[
                {
                    "dilemma_id": "dilemma::mentor_trust_or_betray",
                    "question": "Can Kay trust the mentor?",
                    "answers": [
                        {"answer_id": "yes", "description": "Trust", "is_canonical": True},
                        {"answer_id": "no", "description": "Betray", "is_canonical": False},
                    ],
                    "central_entity_ids": ["character::kay"],
                    "why_it_matters": "Theme of trust",
                }
            ],
        )
        mock_serialize.side_effect = [(entities_artifact, 60, 1), (dilemmas_artifact, 40, 1)]

        artifact, _, _ = await stage.execute(
            model=mock_model,
            user_prompt="test",
            project_path=Path("/test/project"),
        )

        assert isinstance(artifact, dict)
        assert artifact["entities"][0]["entity_id"] == "character::kay"
        assert artifact["dilemmas"][0]["dilemma_id"] == "dilemma::mentor_trust_or_betray"


# --- Vision Context Formatting Tests ---


def test_format_vision_context_includes_genre() -> None:
    """_format_vision_context includes genre in output."""
    from questfoundry.pipeline.stages.brainstorm import _format_vision_context

    vision = {"genre": "fantasy", "subgenre": "epic"}
    result = _format_vision_context(vision)

    assert "fantasy" in result
    assert "epic" in result


def test_format_vision_context_includes_tone() -> None:
    """_format_vision_context includes tone in output."""
    from questfoundry.pipeline.stages.brainstorm import _format_vision_context

    vision = {"tone": ["dark", "gritty"]}
    result = _format_vision_context(vision)

    assert "dark" in result
    assert "gritty" in result


def test_format_vision_context_includes_themes() -> None:
    """_format_vision_context includes themes in output."""
    from questfoundry.pipeline.stages.brainstorm import _format_vision_context

    vision = {"themes": ["redemption", "sacrifice"]}
    result = _format_vision_context(vision)

    assert "redemption" in result
    assert "sacrifice" in result


def test_format_vision_context_handles_empty() -> None:
    """_format_vision_context handles empty vision node."""
    from questfoundry.pipeline.stages.brainstorm import _format_vision_context

    vision: dict[str, str] = {}
    result = _format_vision_context(vision)

    assert "No creative vision available" in result


# --- Two-Pass Serialize Prompt + Helper Tests (F-6) ---


def test_format_brainstorm_valid_entity_ids_groups_by_category() -> None:
    """Helper renders entity IDs grouped by entity_category with backticks."""
    from questfoundry.agents.prompts import format_brainstorm_valid_entity_ids

    block = format_brainstorm_valid_entity_ids(
        [
            {"entity_id": "character::kay", "entity_category": "character"},
            {"entity_id": "character::mentor", "entity_category": "character"},
            {"entity_id": "location::archive", "entity_category": "location"},
            {"entity_id": "object::dagger", "entity_category": "object"},
            {"entity_id": "faction::guard", "entity_category": "faction"},
        ]
    )

    # Stable category order: character → location → object → faction.
    assert block.index("Characters") < block.index("Locations") < block.index("Objects")
    assert block.index("Objects") < block.index("Factions")

    # Each ID is backticked and listed under its category line.
    for needle in [
        "`character::kay`",
        "`character::mentor`",
        "`location::archive`",
        "`object::dagger`",
        "`faction::guard`",
    ]:
        assert needle in block


def test_format_brainstorm_valid_entity_ids_handles_empty() -> None:
    """Empty input renders an explicit `(none)` marker, not silent truncation."""
    from questfoundry.agents.prompts import format_brainstorm_valid_entity_ids

    assert format_brainstorm_valid_entity_ids([]) == "(none)"


def test_format_brainstorm_valid_entity_ids_skips_malformed_entries() -> None:
    """Entries missing entity_id or entity_category are skipped, not included."""
    from questfoundry.agents.prompts import format_brainstorm_valid_entity_ids

    block = format_brainstorm_valid_entity_ids(
        [
            {"entity_id": "character::kay", "entity_category": "character"},
            {"entity_id": "", "entity_category": "character"},  # empty id
            {"entity_id": "location::archive"},  # missing category
            {"entity_category": "object"},  # missing id
        ]
    )
    assert "`character::kay`" in block
    assert "location::archive" not in block
    # Only one well-formed entity was provided, so only one bullet should appear.
    assert block.count("- **") == 1


def test_format_brainstorm_valid_entity_ids_includes_unknown_categories_alphabetically() -> None:
    """Unrecognised categories sort alphabetically AFTER the known set."""
    from questfoundry.agents.prompts import format_brainstorm_valid_entity_ids

    block = format_brainstorm_valid_entity_ids(
        [
            {"entity_id": "character::kay", "entity_category": "character"},
            {"entity_id": "zzz::late", "entity_category": "zzz"},
            {"entity_id": "aaa::early", "entity_category": "aaa"},
        ]
    )
    # Known category appears first; unknowns sort alphabetically among themselves.
    assert block.index("Characters") < block.index("Aaa")
    assert block.index("Aaa") < block.index("Zzz")


def test_serialize_dilemmas_prompt_injects_valid_entity_ids() -> None:
    """Pass-2 prompt has the formatted valid_entity_ids body in its body."""
    from questfoundry.agents.prompts import get_brainstorm_serialize_dilemmas_prompt

    block = "- **Characters**: `character::kay`\n- **Locations**: `location::archive`"
    prompt = get_brainstorm_serialize_dilemmas_prompt(valid_entity_ids=block)

    assert "### Valid Entity IDs" in prompt
    assert "`character::kay`" in prompt
    assert "`location::archive`" in prompt
    # Pass 2 explicitly forbids inventing entity IDs.
    assert "ONLY use" in prompt and "Valid Entity IDs" in prompt


def test_serialize_entities_prompt_omits_valid_entity_ids_section() -> None:
    """Pass-1 prompt has no Valid Entity IDs slot — entities are the source."""
    from questfoundry.agents.prompts import get_brainstorm_serialize_entities_prompt

    prompt = get_brainstorm_serialize_entities_prompt()

    assert "### Valid Entity IDs" not in prompt
    # Pass 1 explicitly tells the model not to emit dilemmas yet.
    assert "Do NOT emit dilemmas" in prompt


# --- Two-Pass Validator Tests (F-6) ---


def test_entities_only_validator_filters_dilemma_errors() -> None:
    """Pass-1 validator runs entity checks; ignores dilemma-side noise."""
    from questfoundry.pipeline.stages.brainstorm import (
        _validate_brainstorm_entities_only,
    )

    # Fixture deliberately violates two entity-side rules so we can confirm
    # both fire on pass 1. Two locations are present (R-2.4 satisfied), but
    # the duplicate `character::kay` triggers the duplicate-ID rule. The
    # behavior under test is "entity-side errors fire and dilemma-side
    # errors don't" — not which specific entity-side rule fires, so we only
    # assert on the duplicate-ID error below.
    output = {
        "entities": [
            {
                "entity_id": "character::kay",
                "entity_category": "character",
                "name": "Kay",
                "concept": "c",
            },
            {
                "entity_id": "location::archive",
                "entity_category": "location",
                "name": "Archive",
                "concept": "c",
            },
            {
                "entity_id": "location::manor",
                "entity_category": "location",
                "name": "Manor",
                "concept": "c",
            },
            # Duplicate of the first — entity-internal duplicate-ID rule fires.
            {
                "entity_id": "character::kay",
                "entity_category": "character",
                "name": "Kay",
                "concept": "c",
            },
        ],
    }

    errors = _validate_brainstorm_entities_only(output)
    assert all(not e.field_path.startswith("dilemmas") for e in errors)
    # Duplicate entity_id should still surface — entity-internal checks ran.
    assert any("Duplicate entity" in e.issue for e in errors)


def test_dilemmas_validator_merges_pass1_entities_for_cross_check() -> None:
    """Pass-2 validator catches phantom central_entity_ids using pass-1 entities."""
    from questfoundry.pipeline.stages.brainstorm import (
        _make_brainstorm_dilemmas_validator,
    )

    entities_dump = [
        {
            "entity_id": "character::kay",
            "entity_category": "character",
            "name": "Kay",
            "concept": "c",
        },
        {
            "entity_id": "location::archive",
            "entity_category": "location",
            "name": "Archive",
            "concept": "c",
        },
        {
            "entity_id": "location::manor",
            "entity_category": "location",
            "name": "Manor",
            "concept": "c",
        },
    ]
    validator = _make_brainstorm_dilemmas_validator(entities_dump)

    output = {
        "dilemmas": [
            {
                "dilemma_id": "dilemma::trust_or_betray",
                "question": "Trust or betray?",
                "why_it_matters": "stakes",
                "answers": [
                    {"answer_id": "trust", "description": "Trust", "is_canonical": True},
                    {"answer_id": "betray", "description": "Betray", "is_canonical": False},
                ],
                "central_entity_ids": ["character::ghost"],  # phantom
            },
        ],
    }

    errors = validator(output)
    # No entity-side errors leak into dilemma retry feedback.
    assert all(not e.field_path.startswith("entities") for e in errors)
    # Phantom central_entity_id must be flagged so retry receives it.
    assert any(
        e.field_path == "dilemmas.0.central_entity_ids" and "ghost" in e.issue for e in errors
    )


def test_dilemmas_validator_passes_when_central_entity_ids_match_pass1() -> None:
    """Valid pass-2 output (entities exist in pass-1) raises no errors."""
    from questfoundry.pipeline.stages.brainstorm import (
        _make_brainstorm_dilemmas_validator,
    )

    entities_dump = [
        {
            "entity_id": "character::kay",
            "entity_category": "character",
            "name": "Kay",
            "concept": "c",
        },
        {
            "entity_id": "location::archive",
            "entity_category": "location",
            "name": "Archive",
            "concept": "c",
        },
        {
            "entity_id": "location::manor",
            "entity_category": "location",
            "name": "Manor",
            "concept": "c",
        },
    ]
    validator = _make_brainstorm_dilemmas_validator(entities_dump)

    output = {
        "dilemmas": [
            {
                "dilemma_id": "dilemma::trust_or_betray",
                "question": "Trust or betray?",
                "why_it_matters": "stakes",
                "answers": [
                    {"answer_id": "trust", "description": "Trust", "is_canonical": True},
                    {"answer_id": "betray", "description": "Betray", "is_canonical": False},
                ],
                "central_entity_ids": ["character::kay"],
            },
        ],
    }

    assert validator(output) == []


# --- Model Tests ---


def test_brainstorm_output_model_validates() -> None:
    """BrainstormOutput model validates correctly."""
    output = BrainstormOutput(
        entities=[
            {
                "entity_id": "hero",
                "entity_category": "character",
                "name": "Hero",
                "concept": "The protagonist",
            }
        ],
        dilemmas=[
            {
                "dilemma_id": "dilemma::quest",
                "question": "Will the hero complete the quest?",
                "answers": [
                    {
                        "answer_id": "success",
                        "description": "Quest complete",
                        "is_canonical": True,
                    },
                    {
                        "answer_id": "failure",
                        "description": "Quest failed",
                        "is_canonical": False,
                    },
                ],
                "central_entity_ids": ["hero"],
                "why_it_matters": "Central narrative dilemma",
            }
        ],
    )

    assert len(output.entities) == 1
    assert len(output.dilemmas) == 1
    assert output.entities[0].entity_id == "hero"
    assert output.dilemmas[0].dilemma_id == "dilemma::quest"


def test_dilemma_requires_two_answers() -> None:
    """Dilemma model requires exactly two answers."""
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Dilemma

    with pytest.raises(ValidationError, match="List should have at least 2 items"):
        Dilemma(
            dilemma_id="dilemma::test",
            question="Test?",
            answers=[{"answer_id": "one", "description": "Only one", "is_canonical": True}],
            central_entity_ids=["character::test"],
            why_it_matters="Test",
        )


def test_dilemma_requires_one_default_path_answer() -> None:
    """Dilemma model requires exactly one default path answer."""
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Dilemma

    with pytest.raises(ValidationError, match=r"one.*default path"):
        Dilemma(
            dilemma_id="dilemma::test",
            question="Test?",
            answers=[
                {"answer_id": "one", "description": "First", "is_canonical": True},
                {"answer_id": "two", "description": "Second", "is_canonical": True},
            ],
            central_entity_ids=["character::test"],
            why_it_matters="Test",
        )


def test_dilemma_rejects_empty_central_entity_ids() -> None:
    """R-3.6: Dilemma model rejects empty central_entity_ids list (#1524 retry-bypass).

    Was a retry-bypass: Pydantic accepted empty `[]`, in-retry semantic validator
    didn't catch it, graph-mutation post-check rejected with no repair opportunity.
    Now Pydantic refuses at attempt 1 so the existing repair loop fires with the
    valid entity IDs echoed via BrainstormMutationError.to_feedback.
    """
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Dilemma

    with pytest.raises(ValidationError, match="at least 1 item"):
        Dilemma(
            dilemma_id="dilemma::test",
            question="Test?",
            answers=[
                {"answer_id": "one", "description": "First", "is_canonical": True},
                {"answer_id": "two", "description": "Second", "is_canonical": False},
            ],
            central_entity_ids=[],
            why_it_matters="Test",
        )


def test_dilemma_rejects_trailing_or_in_id() -> None:
    """Dilemma model rejects IDs ending with '_or_' (common LLM error)."""
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Dilemma

    _TWO_ANSWERS = [
        {"answer_id": "benevolent", "description": "Kind", "is_canonical": True},
        {"answer_id": "selfish", "description": "Mean", "is_canonical": False},
    ]

    # Trailing _or_ with underscore
    with pytest.raises(ValidationError, match="ends with '_or_'"):
        Dilemma(
            dilemma_id="dilemma::host_benevolent_or_selfish_or_",
            question="Is the host benevolent?",
            answers=_TWO_ANSWERS,
            central_entity_ids=["character::host"],
            why_it_matters="Trust",
        )

    # Trailing _or without underscore
    with pytest.raises(ValidationError, match="ends with '_or_'"):
        Dilemma(
            dilemma_id="dilemma::host_benevolent_or_selfish_or",
            question="Is the host benevolent?",
            answers=_TWO_ANSWERS,
            central_entity_ids=["character::host"],
            why_it_matters="Trust",
        )

    # Valid ID passes
    d = Dilemma(
        dilemma_id="dilemma::host_benevolent_or_selfish",
        question="Is the host benevolent?",
        answers=_TWO_ANSWERS,
        central_entity_ids=["character::host"],
        why_it_matters="Trust",
    )
    assert d.dilemma_id == "dilemma::host_benevolent_or_selfish"


def test_entity_types() -> None:
    """Entity model accepts all valid category types."""
    from questfoundry.models.brainstorm import Entity

    for entity_category in ["character", "location", "object", "faction"]:
        entity = Entity(
            entity_id="test",
            entity_category=entity_category,
            name="Test Entity",
            concept="Test concept",
        )
        assert entity.entity_category == entity_category


def test_dilemma_question_must_end_with_qmark() -> None:
    """R-3.1: dilemma question must end with ?."""
    import pytest
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Dilemma

    with pytest.raises(ValidationError) as exc:
        Dilemma(
            dilemma_id="dilemma::x",
            question="not a question",
            why_it_matters="stakes",
            central_entity_ids=["character::x"],
            answers=[
                {"answer_id": "a", "description": "d", "is_canonical": True},
                {"answer_id": "b", "description": "d", "is_canonical": False},
            ],
        )
    assert "?" in str(exc.value)


def test_entity_name_must_be_non_empty() -> None:
    """R-2.1: entity name is required and non-empty."""
    import pytest
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Entity

    with pytest.raises(ValidationError):
        Entity(entity_id="kay", entity_category="character", concept="c")  # no name

    with pytest.raises(ValidationError):
        Entity(entity_id="kay", entity_category="character", name="", concept="c")

    with pytest.raises(ValidationError):
        Entity(entity_id="kay", entity_category="character", name=None, concept="c")


def test_dilemma_id_must_have_prefix() -> None:
    """R-3.7: dilemma_id must start with 'dilemma::'."""
    import pytest
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import Dilemma

    with pytest.raises(ValidationError) as exc:
        Dilemma(
            dilemma_id="mentor_trust",  # missing prefix
            question="Q?",
            why_it_matters="stakes",
            central_entity_ids=["character::mentor"],
            answers=[
                {"answer_id": "a", "description": "d", "is_canonical": True},
                {"answer_id": "b", "description": "d", "is_canonical": False},
            ],
        )
    assert "dilemma::" in str(exc.value)


def test_brainstorm_output_rejects_unknown_fields() -> None:
    """R-3.8 defensive: BrainstormOutput must not silently accept foreign node data."""
    import pytest
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import BrainstormOutput

    with pytest.raises(ValidationError):
        BrainstormOutput.model_validate(
            {
                "entities": [],
                "dilemmas": [],
                "paths": [{"path_id": "x"}],  # not an allowed field
            }
        )


def test_brainstorm_output_minimum_floor() -> None:
    """R-1.1 floor: must produce ≥1 entity and ≥1 dilemma."""
    import pytest
    from pydantic import ValidationError

    from questfoundry.models.brainstorm import BrainstormOutput

    with pytest.raises(ValidationError):
        BrainstormOutput.model_validate({"entities": [], "dilemmas": []})
