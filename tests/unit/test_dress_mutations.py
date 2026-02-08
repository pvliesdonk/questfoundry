"""Tests for DRESS stage graph mutations."""

from __future__ import annotations

import pytest

from questfoundry.graph.graph import Graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dress_graph() -> Graph:
    """Graph with entities, passages, and codewords for DRESS testing."""
    g = Graph()
    g.create_node(
        "entity::protagonist",
        {
            "type": "entity",
            "raw_id": "protagonist",
            "entity_type": "character",
            "concept": "A young scholar",
        },
    )
    g.create_node(
        "entity::aldric",
        {
            "type": "entity",
            "raw_id": "aldric",
            "entity_type": "character",
            "concept": "A former court advisor",
        },
    )
    g.create_node(
        "entity::bridge",
        {
            "type": "entity",
            "raw_id": "bridge",
            "entity_type": "location",
            "concept": "Ancient stone bridge",
        },
    )
    g.create_node(
        "beat::opening",
        {
            "type": "beat",
            "raw_id": "opening",
            "summary": "Scholar arrives at bridge",
            "scene_type": "establishing",
        },
    )
    g.create_node(
        "passage::opening",
        {
            "type": "passage",
            "raw_id": "opening",
            "from_beat": "beat::opening",
            "prose": "The wind howled across the bridge...",
            "entities": ["entity::protagonist", "entity::bridge"],
        },
    )
    g.create_node(
        "codeword::met_aldric",
        {
            "type": "codeword",
            "raw_id": "met_aldric",
            "trigger": "Player meets aldric at the bridge",
        },
    )
    g.create_node(
        "codeword::found_tome",
        {
            "type": "codeword",
            "raw_id": "found_tome",
            "trigger": "Player discovers the ancient tome",
        },
    )
    return g


# ---------------------------------------------------------------------------
# Graph.remove_edge
# ---------------------------------------------------------------------------


class TestRemoveEdge:
    def test_remove_existing_edge(self) -> None:
        g = Graph()
        g.create_node("a::1", {"type": "a"})
        g.create_node("b::1", {"type": "b"})
        g.add_edge("links", "a::1", "b::1")

        assert g.remove_edge("links", "a::1", "b::1") is True
        assert len(g.get_edges()) == 0

    def test_remove_nonexistent_returns_false(self) -> None:
        assert Graph().remove_edge("links", "a::1", "b::1") is False

    def test_remove_only_matching_edge(self) -> None:
        g = Graph()
        g.create_node("a::1", {"type": "a"})
        g.create_node("b::1", {"type": "b"})
        g.create_node("b::2", {"type": "b"})
        g.add_edge("links", "a::1", "b::1")
        g.add_edge("links", "a::1", "b::2")

        g.remove_edge("links", "a::1", "b::1")
        remaining = g.get_edges()
        assert len(remaining) == 1
        assert remaining[0]["to"] == "b::2"

    def test_type_mismatch_not_removed(self) -> None:
        g = Graph()
        g.create_node("a::1", {"type": "a"})
        g.create_node("b::1", {"type": "b"})
        g.add_edge("links", "a::1", "b::1")

        assert g.remove_edge("other_type", "a::1", "b::1") is False
        assert len(g.get_edges()) == 1


# ---------------------------------------------------------------------------
# Art Direction (Phase 0)
# ---------------------------------------------------------------------------


class TestApplyDressArtDirection:
    def test_creates_nodes_and_edges(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import apply_dress_art_direction

        apply_dress_art_direction(
            dress_graph,
            art_dir={
                "style": "watercolor",
                "medium": "traditional",
                "palette": ["indigo", "rust"],
                "composition_notes": "wide shots",
                "negative_defaults": "photorealistic",
            },
            entity_visuals=[
                {
                    "entity_id": "protagonist",
                    "description": "Young woman, short dark hair",
                    "distinguishing_features": ["jade pendant"],
                    "reference_prompt_fragment": "young woman, jade pendant",
                }
            ],
        )

        ad = dress_graph.get_node("art_direction::main")
        assert ad is not None
        assert ad["style"] == "watercolor"

        ev = dress_graph.get_node("entity_visual::protagonist")
        assert ev is not None
        assert ev["type"] == "entity_visual"

        edges = dress_graph.get_edges(
            from_id="entity_visual::protagonist",
            edge_type="describes_visual",
        )
        assert len(edges) == 1
        assert edges[0]["to"] == "entity::protagonist"

    def test_scoped_entity_id_stripped(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import apply_dress_art_direction

        apply_dress_art_direction(
            dress_graph,
            art_dir={
                "style": "ink",
                "medium": "d",
                "palette": ["b"],
                "composition_notes": "c",
                "negative_defaults": "n",
            },
            entity_visuals=[
                {
                    "entity_id": "entity::aldric",
                    "description": "d",
                    "distinguishing_features": ["f"],
                    "reference_prompt_fragment": "frag",
                }
            ],
        )

        assert dress_graph.get_node("entity_visual::aldric") is not None

    def test_idempotent_rerun(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import apply_dress_art_direction

        kwargs: dict = {
            "art_dir": {
                "style": "ink",
                "medium": "d",
                "palette": ["b"],
                "composition_notes": "c",
                "negative_defaults": "n",
            },
            "entity_visuals": [
                {
                    "entity_id": "protagonist",
                    "description": "d",
                    "distinguishing_features": ["f"],
                    "reference_prompt_fragment": "frag",
                }
            ],
        }
        apply_dress_art_direction(dress_graph, **kwargs)
        apply_dress_art_direction(dress_graph, **kwargs)

        edges = dress_graph.get_edges(
            from_id="entity_visual::protagonist",
            edge_type="describes_visual",
        )
        assert len(edges) == 1

    def test_nonexistent_entity_skips_edge(self) -> None:
        """Unresolvable entity creates visual node but no describes_visual edge."""
        from questfoundry.graph.dress_mutations import apply_dress_art_direction

        g = Graph()
        apply_dress_art_direction(
            g,
            art_dir={
                "style": "ink",
                "medium": "d",
                "palette": ["b"],
                "composition_notes": "c",
                "negative_defaults": "n",
            },
            entity_visuals=[
                {
                    "entity_id": "nonexistent",
                    "description": "d",
                    "distinguishing_features": ["f"],
                    "reference_prompt_fragment": "frag",
                }
            ],
        )

        # Visual node created
        assert g.get_node("entity_visual::nonexistent") is not None
        # No edge because entity doesn't exist in any category
        edges = g.get_edges(from_id="entity_visual::nonexistent", edge_type="describes_visual")
        assert len(edges) == 0


# ---------------------------------------------------------------------------
# Illustration Briefs (Phase 1)
# ---------------------------------------------------------------------------


class TestApplyDressBrief:
    def test_creates_brief_node_and_edge(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import apply_dress_brief

        node_id = apply_dress_brief(
            dress_graph,
            passage_id="passage::opening",
            brief={
                "category": "scene",
                "subject": "Scholar on bridge",
                "composition": "wide",
                "mood": "ominous",
                "caption": "The bridge awaits",
            },
            priority=1,
        )

        assert node_id == "illustration_brief::opening"
        node = dress_graph.get_node(node_id)
        assert node is not None
        assert node["priority"] == 1

        edges = dress_graph.get_edges(from_id=node_id, edge_type="targets")
        assert len(edges) == 1
        assert edges[0]["to"] == "passage::opening"

    def test_idempotent_rerun(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import apply_dress_brief

        kwargs: dict = {
            "passage_id": "opening",
            "priority": 2,
            "brief": {
                "category": "scene",
                "subject": "s",
                "composition": "c",
                "mood": "m",
                "caption": "cap",
            },
        }
        apply_dress_brief(dress_graph, **kwargs)
        apply_dress_brief(dress_graph, **kwargs)

        edges = dress_graph.get_edges(from_id="illustration_brief::opening", edge_type="targets")
        assert len(edges) == 1


# ---------------------------------------------------------------------------
# Codex Entries (Phase 2)
# ---------------------------------------------------------------------------


class TestApplyDressCodex:
    def test_creates_codex_nodes_and_edges(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import apply_dress_codex

        created = apply_dress_codex(
            dress_graph,
            entity_id="entity::aldric",
            entries=[
                {"rank": 1, "content": "A traveling scholar."},
                {"rank": 2, "visible_when": ["met_aldric"], "content": "Former court advisor."},
            ],
        )

        assert created == ["codex::aldric_rank1", "codex::aldric_rank2"]
        node = dress_graph.get_node("codex::aldric_rank1")
        assert node is not None
        assert node["type"] == "codex_entry"

        edges = dress_graph.get_edges(from_id="codex::aldric_rank1", edge_type="HasEntry")
        assert len(edges) == 1
        assert edges[0]["to"] == "entity::aldric"

    def test_missing_rank_raises(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import apply_dress_codex

        with pytest.raises(ValueError, match="missing required 'rank'"):
            apply_dress_codex(dress_graph, entity_id="aldric", entries=[{"content": "c"}])

    def test_idempotent_rerun(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import apply_dress_codex

        kwargs: dict = {"entity_id": "aldric", "entries": [{"rank": 1, "content": "c"}]}
        apply_dress_codex(dress_graph, **kwargs)
        apply_dress_codex(dress_graph, **kwargs)

        edges = dress_graph.get_edges(from_id="codex::aldric_rank1", edge_type="HasEntry")
        assert len(edges) == 1


# ---------------------------------------------------------------------------
# Illustrations (Phase 4)
# ---------------------------------------------------------------------------


class TestApplyDressIllustration:
    def test_creates_illustration_with_edges(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import apply_dress_brief, apply_dress_illustration

        brief_id = apply_dress_brief(
            dress_graph,
            passage_id="opening",
            brief={
                "category": "scene",
                "subject": "s",
                "composition": "c",
                "mood": "m",
                "caption": "cap",
            },
            priority=1,
        )
        illust_id = apply_dress_illustration(
            dress_graph,
            brief_id=brief_id,
            asset_path="assets/abc123.png",
            caption="The bridge",
            category="scene",
        )

        assert illust_id == "illustration::opening"
        node = dress_graph.get_node(illust_id)
        assert node is not None
        assert node["asset"] == "assets/abc123.png"

        depicts = dress_graph.get_edges(from_id=illust_id, edge_type="Depicts")
        assert len(depicts) == 1
        assert depicts[0]["to"] == "passage::opening"

        from_brief = dress_graph.get_edges(from_id=illust_id, edge_type="from_brief")
        assert len(from_brief) == 1
        assert from_brief[0]["to"] == brief_id

    def test_quality_defaults_to_high(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import apply_dress_brief, apply_dress_illustration

        brief_id = apply_dress_brief(
            dress_graph,
            passage_id="opening",
            brief={
                "category": "scene",
                "subject": "s",
                "composition": "c",
                "mood": "m",
                "caption": "c",
            },
            priority=1,
        )
        illust_id = apply_dress_illustration(
            dress_graph,
            brief_id=brief_id,
            asset_path="assets/abc.png",
            caption="cap",
            category="scene",
        )
        node = dress_graph.get_node(illust_id)
        assert node is not None
        assert node["quality"] == "high"

    def test_quality_placeholder(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import apply_dress_brief, apply_dress_illustration

        brief_id = apply_dress_brief(
            dress_graph,
            passage_id="opening",
            brief={
                "category": "scene",
                "subject": "s",
                "composition": "c",
                "mood": "m",
                "caption": "c",
            },
            priority=1,
        )
        illust_id = apply_dress_illustration(
            dress_graph,
            brief_id=brief_id,
            asset_path="assets/abc.png",
            caption="cap",
            category="scene",
            quality="placeholder",
        )
        node = dress_graph.get_node(illust_id)
        assert node is not None
        assert node["quality"] == "placeholder"

    def test_standalone_cover_no_targets_edge(self, dress_graph: Graph) -> None:
        """Cover brief without targets edge creates illustration from brief ID."""
        from questfoundry.graph.dress_mutations import apply_dress_illustration

        dress_graph.create_node(
            "illustration_brief::cover",
            {"type": "illustration_brief", "category": "cover", "subject": "Cover image"},
        )

        illust_id = apply_dress_illustration(
            dress_graph,
            brief_id="illustration_brief::cover",
            asset_path="assets/cover.png",
            caption="",
            category="cover",
        )

        assert illust_id == "illustration::cover"
        node = dress_graph.get_node(illust_id)
        assert node is not None
        assert node["asset"] == "assets/cover.png"
        assert node["category"] == "cover"

        # No Depicts edge for standalone cover
        depicts = dress_graph.get_edges(from_id=illust_id, edge_type="Depicts")
        assert len(depicts) == 0

        # from_brief edge should still exist
        from_brief = dress_graph.get_edges(from_id=illust_id, edge_type="from_brief")
        assert len(from_brief) == 1
        assert from_brief[0]["to"] == "illustration_brief::cover"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateDressCodexEntries:
    def test_valid_entries(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import validate_dress_codex_entries

        errors = validate_dress_codex_entries(
            dress_graph,
            "aldric",
            [
                {"rank": 1, "content": "Base knowledge"},
                {"rank": 2, "visible_when": ["met_aldric"], "content": "Deeper knowledge"},
            ],
        )
        assert errors == []

    def test_empty_entries(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import validate_dress_codex_entries

        errors = validate_dress_codex_entries(dress_graph, "aldric", [])
        assert "no codex entries" in errors[0]

    def test_missing_rank_1(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import validate_dress_codex_entries

        errors = validate_dress_codex_entries(dress_graph, "aldric", [{"rank": 2, "content": "c"}])
        assert any("rank=1" in e for e in errors)

    def test_duplicate_ranks(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import validate_dress_codex_entries

        errors = validate_dress_codex_entries(
            dress_graph, "aldric", [{"rank": 1, "content": "a"}, {"rank": 1, "content": "b"}]
        )
        assert any("duplicate rank=1" in e for e in errors)

    def test_unknown_codeword(self, dress_graph: Graph) -> None:
        from questfoundry.graph.dress_mutations import validate_dress_codex_entries

        errors = validate_dress_codex_entries(
            dress_graph,
            "aldric",
            [
                {"rank": 1, "content": "base"},
                {"rank": 2, "visible_when": ["nonexistent_cw"], "content": "gated"},
            ],
        )
        assert any("unknown codeword" in e for e in errors)
