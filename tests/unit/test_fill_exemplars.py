"""Tests for FILL Phase 0b exemplar generation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from questfoundry.graph.fill_context import format_exemplar_passages, format_voice_context
from questfoundry.graph.graph import Graph
from questfoundry.models.fill import FillExemplarOutput
from questfoundry.pipeline.stages.fill import FillStage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_with_voice() -> Graph:
    """Graph with voice node and minimal GROW structure."""
    g = Graph.empty()
    g.create_node(
        "vision",
        {
            "type": "vision",
            "raw_id": "vision",
            "genre": "mystery",
            "tone": ["atmospheric", "tense"],
            "themes": ["trust", "deception"],
        },
    )
    g.create_node(
        "voice::voice",
        {
            "type": "voice",
            "raw_id": "voice",
            "pov": "third_limited",
            "tense": "past",
            "voice_register": "literary",
            "sentence_rhythm": "varied",
            "tone_words": ["atmospheric", "terse"],
            "avoid_words": ["suddenly", "very"],
            "avoid_patterns": ["adverb-heavy tags"],
            "exemplar_passages": [],
        },
    )
    # Minimal arc/passage for grow_summary
    g.create_node("arc::spine", {"type": "arc", "raw_id": "spine", "arc_type": "spine"})
    g.set_last_stage("grow")
    return g


@pytest.fixture
def stage() -> FillStage:
    """FillStage with defaults for direct method testing."""
    return FillStage()


# ---------------------------------------------------------------------------
# TestFormatExemplarPassages
# ---------------------------------------------------------------------------


class TestFormatExemplarPassages:
    def test_no_voice_returns_empty(self) -> None:
        g = Graph.empty()
        assert format_exemplar_passages(g) == ""

    def test_no_exemplars_returns_empty(self, graph_with_voice: Graph) -> None:
        assert format_exemplar_passages(graph_with_voice) == ""

    def test_formats_as_prose(self, graph_with_voice: Graph) -> None:
        graph_with_voice.update_node(
            "voice::voice",
            exemplar_passages=[
                "The corridor narrowed ahead, damp stone glistening.",
                "She turned the page, fingers trembling.",
            ],
        )
        result = format_exemplar_passages(graph_with_voice)
        assert "**Example 1:**" in result
        assert "**Example 2:**" in result
        assert "corridor narrowed" in result
        assert "Do NOT copy" in result

    def test_empty_string_exemplars_ignored(self, graph_with_voice: Graph) -> None:
        graph_with_voice.update_node(
            "voice::voice",
            exemplar_passages=["Good passage here.", "", "  "],
        )
        result = format_exemplar_passages(graph_with_voice)
        assert "**Example 1:**" in result
        # Only one valid exemplar
        assert "**Example 2:**" not in result

    def test_voice_context_excludes_exemplars(self, graph_with_voice: Graph) -> None:
        graph_with_voice.update_node(
            "voice::voice",
            exemplar_passages=["Some exemplar passage here."],
        )
        result = format_voice_context(graph_with_voice)
        assert "exemplar_passages" not in result
        assert "Some exemplar passage" not in result
        # But voice fields should still be present
        assert "pov:" in result
        assert "tense:" in result


# ---------------------------------------------------------------------------
# TestFillExemplarOutput
# ---------------------------------------------------------------------------


class TestFillExemplarOutput:
    def test_valid_two_passages(self) -> None:
        output = FillExemplarOutput(exemplar_passages=["Passage one.", "Passage two."])
        assert len(output.exemplar_passages) == 2

    def test_valid_three_passages(self) -> None:
        output = FillExemplarOutput(exemplar_passages=["One.", "Two.", "Three."])
        assert len(output.exemplar_passages) == 3

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FillExemplarOutput(exemplar_passages=[])

    def test_one_passage_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FillExemplarOutput(exemplar_passages=["Only one."])

    def test_four_passages_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FillExemplarOutput(exemplar_passages=["One.", "Two.", "Three.", "Four."])


# ---------------------------------------------------------------------------
# TestExemplarPhase
# ---------------------------------------------------------------------------


class TestExemplarPhase:
    @pytest.mark.asyncio
    async def test_skipped_without_voice(self, stage: FillStage) -> None:
        g = Graph.empty()
        model = MagicMock()
        result = await stage._phase_0b_exemplar(g, model)
        assert result.status == "skipped"

    @pytest.mark.asyncio
    async def test_corpus_match_skips_llm(self, stage: FillStage, graph_with_voice: Graph) -> None:
        mock_corpus = MagicMock()
        mock_corpus.search_exemplars.return_value = [
            {"sections": [{"content": "First corpus passage."}]},
            {"sections": [{"content": "Second corpus passage."}]},
        ]
        model = MagicMock()

        import sys

        mock_module = MagicMock()
        mock_module.Corpus.return_value = mock_corpus
        sys.modules["ifcraftcorpus"] = mock_module
        try:
            result = await stage._phase_0b_exemplar(graph_with_voice, model)
            assert result.status == "completed"
            assert "corpus" in result.detail
            assert result.llm_calls == 0
            # Exemplars should be stored on voice node
            voice = graph_with_voice.get_node("voice::voice")
            assert voice is not None
            exemplars = voice.get("exemplar_passages", [])
            assert len(exemplars) == 2
        finally:
            del sys.modules["ifcraftcorpus"]

    @pytest.mark.asyncio
    async def test_corpus_no_match_falls_back_to_llm(
        self, stage: FillStage, graph_with_voice: Graph
    ) -> None:
        mock_corpus = MagicMock()
        mock_corpus.search_exemplars.return_value = []  # No matches

        import sys

        mock_module = MagicMock()
        mock_module.Corpus.return_value = mock_corpus
        sys.modules["ifcraftcorpus"] = mock_module

        model = MagicMock()
        llm_output = FillExemplarOutput(exemplar_passages=["LLM passage one.", "LLM passage two."])

        try:
            with patch.object(
                stage,
                "_fill_llm_call",
                new_callable=AsyncMock,
                return_value=(llm_output, 1, 500),
            ):
                result = await stage._phase_0b_exemplar(graph_with_voice, model)
                assert result.status == "completed"
                assert "llm" in result.detail
                assert result.llm_calls == 1
        finally:
            del sys.modules["ifcraftcorpus"]

    @pytest.mark.asyncio
    async def test_corpus_partial_match_falls_back(
        self, stage: FillStage, graph_with_voice: Graph
    ) -> None:
        mock_corpus = MagicMock()
        mock_corpus.search_exemplars.return_value = [
            {"sections": [{"content": "Only one match."}]},
        ]

        import sys

        mock_module = MagicMock()
        mock_module.Corpus.return_value = mock_corpus
        sys.modules["ifcraftcorpus"] = mock_module

        model = MagicMock()
        llm_output = FillExemplarOutput(exemplar_passages=["LLM one.", "LLM two."])

        try:
            with patch.object(
                stage,
                "_fill_llm_call",
                new_callable=AsyncMock,
                return_value=(llm_output, 1, 500),
            ):
                result = await stage._phase_0b_exemplar(graph_with_voice, model)
                assert result.status == "completed"
                assert "llm" in result.detail
        finally:
            del sys.modules["ifcraftcorpus"]

    @pytest.mark.asyncio
    async def test_corpus_import_error_falls_back(
        self, stage: FillStage, graph_with_voice: Graph
    ) -> None:
        """ImportError on ifcraftcorpus falls back to LLM."""
        # Remove ifcraftcorpus from modules so import fails
        import sys

        saved = sys.modules.pop("ifcraftcorpus", None)

        model = MagicMock()
        llm_output = FillExemplarOutput(exemplar_passages=["Fallback one.", "Fallback two."])

        try:
            with patch.object(
                stage,
                "_fill_llm_call",
                new_callable=AsyncMock,
                return_value=(llm_output, 1, 500),
            ):
                # Make the import actually fail
                import builtins

                real_import = builtins.__import__

                def fail_import(name: str, *args: object, **kwargs: object) -> object:
                    if name == "ifcraftcorpus":
                        raise ImportError("No module named 'ifcraftcorpus'")
                    return real_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=fail_import):
                    result = await stage._phase_0b_exemplar(graph_with_voice, model)
                    assert result.status == "completed"
                    assert "llm" in result.detail
        finally:
            if saved is not None:
                sys.modules["ifcraftcorpus"] = saved

    @pytest.mark.asyncio
    async def test_stores_on_voice_node(self, stage: FillStage, graph_with_voice: Graph) -> None:
        """LLM-generated exemplars are stored on the voice node."""
        model = MagicMock()
        llm_output = FillExemplarOutput(
            exemplar_passages=["Stored passage one.", "Stored passage two."]
        )

        import sys

        saved = sys.modules.pop("ifcraftcorpus", None)
        try:
            import builtins

            real_import = builtins.__import__

            def fail_import(name: str, *args: object, **kwargs: object) -> object:
                if name == "ifcraftcorpus":
                    raise ImportError("not available")
                return real_import(name, *args, **kwargs)

            with (
                patch("builtins.__import__", side_effect=fail_import),
                patch.object(
                    stage,
                    "_fill_llm_call",
                    new_callable=AsyncMock,
                    return_value=(llm_output, 1, 500),
                ),
            ):
                await stage._phase_0b_exemplar(graph_with_voice, model)
                voice = graph_with_voice.get_node("voice::voice")
                assert voice is not None
                assert voice["exemplar_passages"] == [
                    "Stored passage one.",
                    "Stored passage two.",
                ]
        finally:
            if saved is not None:
                sys.modules["ifcraftcorpus"] = saved

    @pytest.mark.asyncio
    async def test_uses_creative_model(self, stage: FillStage, graph_with_voice: Graph) -> None:
        """LLM fallback uses creative=True for strong model."""
        model = MagicMock()
        llm_output = FillExemplarOutput(exemplar_passages=["One.", "Two."])

        import sys

        saved = sys.modules.pop("ifcraftcorpus", None)
        try:
            import builtins

            real_import = builtins.__import__

            def fail_import(name: str, *args: object, **kwargs: object) -> object:
                if name == "ifcraftcorpus":
                    raise ImportError("not available")
                return real_import(name, *args, **kwargs)

            with (
                patch("builtins.__import__", side_effect=fail_import),
                patch.object(
                    stage,
                    "_fill_llm_call",
                    new_callable=AsyncMock,
                    return_value=(llm_output, 1, 500),
                ) as mock_llm_call,
            ):
                await stage._phase_0b_exemplar(graph_with_voice, model)
                # Verify creative=True was passed
                mock_llm_call.assert_called_once()
                _, kwargs = mock_llm_call.call_args
                assert kwargs.get("creative") is True
        finally:
            if saved is not None:
                sys.modules["ifcraftcorpus"] = saved

    def test_phase_order(self) -> None:
        stage = FillStage()
        names = [name for _, name in stage._phase_order()]
        voice_idx = names.index("voice")
        exemplar_idx = names.index("exemplar")
        expand_idx = names.index("expand")
        assert exemplar_idx == voice_idx + 1
        assert exemplar_idx < expand_idx


# ---------------------------------------------------------------------------
# TestExemplarValidation
# ---------------------------------------------------------------------------


class TestExemplarValidation:
    def test_pov_check_warns(self, stage: FillStage, caplog: pytest.LogCaptureFixture) -> None:
        """First-person voice with no 'I' pronoun logs warning."""
        voice_data: dict[str, object] = {
            "pov": "first",
            "avoid_words": [],
        }
        stage._language = "en"
        stage._validate_exemplars(voice_data, ["The corridor stretched on."])
        assert any("exemplar_pov_drift" in r.message for r in caplog.records)

    def test_no_pov_warning_when_present(
        self, stage: FillStage, caplog: pytest.LogCaptureFixture
    ) -> None:
        voice_data: dict[str, object] = {
            "pov": "first",
            "avoid_words": [],
        }
        stage._language = "en"
        stage._validate_exemplars(voice_data, ["I walked down the corridor."])
        assert not any("exemplar_pov_drift" in r.message for r in caplog.records)

    def test_avoid_words_warns(self, stage: FillStage, caplog: pytest.LogCaptureFixture) -> None:
        voice_data: dict[str, object] = {
            "pov": "third_limited",
            "avoid_words": ["suddenly"],
        }
        stage._language = "en"
        stage._validate_exemplars(voice_data, ["Suddenly the door opened."])
        assert any("exemplar_avoid_word" in r.message for r in caplog.records)

    def test_non_english_skips_validation(
        self, stage: FillStage, caplog: pytest.LogCaptureFixture
    ) -> None:
        voice_data: dict[str, object] = {
            "pov": "first",
            "avoid_words": ["suddenly"],
        }
        stage._language = "nl"
        stage._validate_exemplars(voice_data, ["Plotseling opende de deur."])
        assert not any("exemplar_pov_drift" in r.message for r in caplog.records)
        assert not any("exemplar_avoid_word" in r.message for r in caplog.records)

    def test_third_person_no_pov_warning(
        self, stage: FillStage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Third-person POV should not trigger first-person pronoun check."""
        voice_data: dict[str, object] = {
            "pov": "third_limited",
            "avoid_words": [],
        }
        stage._language = "en"
        stage._validate_exemplars(voice_data, ["She walked away."])
        assert not any("exemplar_pov_drift" in r.message for r in caplog.records)
