# tests/unit/test_checkpoint_styles.py
"""Tests for checkpoint style metadata lookup (#1557)."""

from __future__ import annotations

import pytest

from questfoundry.providers.checkpoint_styles import resolve_checkpoint_style


class TestResolveCheckpointStyle:
    """`resolve_checkpoint_style()` returns label/style_hints/incompatible_styles
    for a checkpoint name. Always returns a populated dict (default fallback
    if no specific pattern matches)."""

    @pytest.mark.parametrize(
        ("model", "label_substring"),
        [
            # User's loaded checkpoints at A1111 (2026-04-28)
            ("flux1-dev-bnb-nf4-v2.safetensors", "Flux"),
            ("flux1-dev-bnb-nf4.safetensors", "Flux"),
            ("coloring_book.ckpt", "Coloring Book"),
            ("v1-5-pruned-emaonly.safetensors", "SD 1.5"),
            ("animagine-xl.safetensors", "Animagine"),
            ("Dreamshaper.safetensors", "DreamShaper"),
            ("sd_xl_base_1.0.safetensors", "SDXL Base"),
            ("dreamshaperXL_lightningDPMSDE.safetensors", "DreamShaperXL Lightning"),
            ("dreamshaperXL_v1.safetensors", "DreamShaperXL"),
            ("juggernautXL_ragnarokBy.safetensors", "Juggernaut"),
            ("dreamshaperXL_alpha2Xl10.safetensors", "DreamShaperXL Lightning"),
        ],
    )
    def test_known_checkpoint_resolves_to_specific_label(
        self, model: str, label_substring: str
    ) -> None:
        info = resolve_checkpoint_style(model)
        assert label_substring in info["label"]
        assert info["style_hints"]
        assert info["incompatible_styles"]

    def test_unknown_checkpoint_falls_back_to_default(self) -> None:
        info = resolve_checkpoint_style("totally-made-up-model.safetensors")
        assert "Unknown" in info["label"] or "general-purpose" in info["label"].lower()
        assert info["style_hints"]
        assert info["incompatible_styles"]

    def test_pattern_ordering_dreamshaperxl_lightning_wins_over_generic(self) -> None:
        # The dreamshaper family has three patterns; the most specific must win.
        # dreamshaperxl_lightning|alpha → "DreamShaperXL Lightning / Alpha"
        # dreamshaperxl|dreamshaper.*xl → "DreamShaperXL"
        # dreamshaper                    → "DreamShaper" (SD1.5)
        lightning = resolve_checkpoint_style("dreamshaperXL_lightningDPMSDE.safetensors")
        assert "Lightning" in lightning["label"] or "Alpha" in lightning["label"]

        alpha = resolve_checkpoint_style("dreamshaperXL_alpha2Xl10.safetensors")
        assert "Lightning" in alpha["label"] or "Alpha" in alpha["label"]

        # Standard XL wins over plain SD1.5 fallback.
        standard_xl = resolve_checkpoint_style("dreamshaperXL_v1.safetensors")
        assert "DreamShaperXL" in standard_xl["label"]
        assert "Lightning" not in standard_xl["label"]
        assert "Alpha" not in standard_xl["label"]

        sd15 = resolve_checkpoint_style("Dreamshaper.safetensors")
        assert "Lightning" not in sd15["label"]
        assert "Alpha" not in sd15["label"]
        assert "XL" not in sd15["label"]

    def test_returns_required_keys(self) -> None:
        info = resolve_checkpoint_style("anything.safetensors")
        assert set(info.keys()) >= {
            "label",
            "style_hints",
            "incompatible_styles",
            "good_example",
            "bad_example",
        }

    def test_case_insensitive_matching(self) -> None:
        # Patterns match against lowercased filename.
        upper = resolve_checkpoint_style("FLUX1-DEV.safetensors")
        lower = resolve_checkpoint_style("flux1-dev.safetensors")
        assert upper == lower


class TestPromptFormat:
    """`prompt_format` field on each map entry + `prompt_format_for_checkpoint()`
    helper. Drives the A1111 distiller's clip_tags vs natural_language
    branching for Flux on Forge Neo (#1559)."""

    def test_resolve_checkpoint_style_returns_prompt_format(self) -> None:
        # Every entry must carry a prompt_format key.
        info = resolve_checkpoint_style("anything.safetensors")
        assert "prompt_format" in info
        assert info["prompt_format"] in {"clip_tags", "natural_language"}

    def test_flux_uses_natural_language(self) -> None:
        from questfoundry.providers.checkpoint_styles import (
            prompt_format_for_checkpoint,
        )

        assert (
            prompt_format_for_checkpoint("flux1-dev-bnb-nf4-v2.safetensors") == "natural_language"
        )
        assert prompt_format_for_checkpoint("flux1-dev-bnb-nf4.safetensors") == "natural_language"

    @pytest.mark.parametrize(
        "model",
        [
            "coloring_book.ckpt",
            "v1-5-pruned-emaonly.safetensors",
            "animagine-xl.safetensors",
            "Dreamshaper.safetensors",
            "sd_xl_base_1.0.safetensors",
            "dreamshaperXL_lightningDPMSDE.safetensors",
            "juggernautXL_ragnarokBy.safetensors",
            "dreamshaperXL_alpha2Xl10.safetensors",
            "dreamshaperXL_v1.safetensors",
        ],
    )
    def test_non_flux_checkpoints_use_clip_tags(self, model: str) -> None:
        from questfoundry.providers.checkpoint_styles import (
            prompt_format_for_checkpoint,
        )

        assert prompt_format_for_checkpoint(model) == "clip_tags"

    def test_unknown_checkpoint_uses_clip_tags(self) -> None:
        from questfoundry.providers.checkpoint_styles import (
            prompt_format_for_checkpoint,
        )

        # Default fallback: SD-family is the long tail; safer assumption is CLIP.
        assert prompt_format_for_checkpoint("totally-made-up-model.safetensors") == "clip_tags"

    def test_no_model_uses_clip_tags(self) -> None:
        # When no model is set, preserve the LLM-distill default path.
        from questfoundry.providers.checkpoint_styles import (
            prompt_format_for_checkpoint,
        )

        assert prompt_format_for_checkpoint(None) == "clip_tags"
        assert prompt_format_for_checkpoint("") == "clip_tags"
