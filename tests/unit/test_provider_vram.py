"""Tests for providers.vram — VRAM-aware num_ctx calculation."""

from __future__ import annotations

from typing import Any

import pytest

from questfoundry.providers.vram import (
    DEFAULT_BYTES_PER_WEIGHT,
    MIN_NUM_CTX,
    NUM_CTX_ALIGNMENT,
    QUANT_BYTES_PER_WEIGHT,
    VramTooSmallError,
    calculate_max_context,
    parse_parameter_size,
    parse_quantization,
)


class TestParseQuantization:
    def test_known_quants(self) -> None:
        assert parse_quantization("Q4_K_M") == 0.57
        assert parse_quantization("Q8_0") == 1.00
        assert parse_quantization("F16") == 2.00
        assert parse_quantization("FP16") == 2.00

    def test_case_insensitive(self) -> None:
        assert parse_quantization("q4_k_m") == 0.57
        assert parse_quantization("Q4_K_M ") == 0.57

    def test_unknown_quant_falls_back(self) -> None:
        assert parse_quantization("Q99_K_X") == DEFAULT_BYTES_PER_WEIGHT

    def test_none_falls_back(self) -> None:
        assert parse_quantization(None) == DEFAULT_BYTES_PER_WEIGHT

    def test_table_completeness(self) -> None:
        """All entries in QUANT_BYTES_PER_WEIGHT have positive values."""
        for level, bpw in QUANT_BYTES_PER_WEIGHT.items():
            assert bpw > 0, f"{level} has non-positive bpw {bpw}"


class TestParseParameterSize:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("8.0B", 8.0),
            ("3.8B", 3.8),
            ("70B", 70.0),
            ("1.7B", 1.7),
            ("500M", 0.5),
            ("250M", 0.25),
        ],
    )
    def test_known_formats(self, value: str, expected: float) -> None:
        result = parse_parameter_size(value)
        assert result is not None
        assert abs(result - expected) < 1e-6

    def test_lowercase(self) -> None:
        assert parse_parameter_size("8b") == 8.0

    def test_invalid_returns_none(self) -> None:
        assert parse_parameter_size("not a number") is None
        assert parse_parameter_size("") is None
        assert parse_parameter_size(None) is None


def _llama8b_show_fixture(quant: str = "Q4_K_M") -> dict[str, Any]:
    """Realistic /api/show shape for Llama-3-8B-ish model."""
    return {
        "details": {
            "family": "llama",
            "parameter_size": "8.0B",
            "quantization_level": quant,
        },
        "model_info": {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        },
    }


class TestCalculateMaxContext:
    def test_typical_8b_q4_at_12gb(self) -> None:
        """8B Q4_K_M at 12 GB VRAM yields a reasonable context (≥16K)."""
        ctx = calculate_max_context(12.0, _llama8b_show_fixture())
        assert ctx >= 16_384
        assert ctx % NUM_CTX_ALIGNMENT == 0

    def test_smaller_vram_smaller_ctx(self) -> None:
        """Smaller VRAM budget yields smaller (still aligned) context."""
        ctx_24 = calculate_max_context(24.0, _llama8b_show_fixture())
        ctx_8 = calculate_max_context(8.0, _llama8b_show_fixture())
        assert ctx_24 > ctx_8
        assert ctx_8 % NUM_CTX_ALIGNMENT == 0

    def test_too_small_vram_raises(self) -> None:
        """Weights + overhead alone exceeding budget raises VramTooSmallError."""
        # 8B Q4_K_M weights ~ 4.56 GB; overhead ~ 1.19 GB. 4 GB budget can't fit.
        with pytest.raises(VramTooSmallError) as exc:
            calculate_max_context(4.0, _llama8b_show_fixture())
        assert exc.value.vram_gb == 4.0

    def test_higher_quant_smaller_ctx(self) -> None:
        """Q8_0 leaves less VRAM for KV cache than Q4_K_M at the same budget."""
        ctx_q4 = calculate_max_context(12.0, _llama8b_show_fixture("Q4_K_M"))
        ctx_q8 = calculate_max_context(12.0, _llama8b_show_fixture("Q8_0"))
        assert ctx_q4 > ctx_q8

    def test_architectural_max_caps_calculation(self) -> None:
        """Calculated num_ctx is capped at the model's architectural max."""
        ctx = calculate_max_context(64.0, _llama8b_show_fixture(), architectural_max=8_192)
        assert ctx == 8_192

    def test_aligned_to_1024(self) -> None:
        """Returned num_ctx is always 1024-aligned."""
        for vram in (6.0, 8.0, 12.0, 16.0, 24.0):
            ctx = calculate_max_context(vram, _llama8b_show_fixture())
            assert ctx % NUM_CTX_ALIGNMENT == 0, f"{vram} → {ctx} not aligned"

    def test_at_least_min_num_ctx(self) -> None:
        """Returned num_ctx is at least MIN_NUM_CTX (or VramTooSmallError)."""
        # Find a budget that fits weights+overhead but yields tiny KV room.
        # 8B Q4_K_M: weights 4.56, overhead 1.19, total 5.75. Use 5.85 for ~0.1 GB KV.
        ctx = calculate_max_context(5.85, _llama8b_show_fixture())
        assert ctx >= MIN_NUM_CTX

    def test_invalid_vram_raises(self) -> None:
        with pytest.raises(ValueError, match="vram_gb must be positive"):
            calculate_max_context(0, _llama8b_show_fixture())
        with pytest.raises(ValueError, match="vram_gb must be positive"):
            calculate_max_context(-5, _llama8b_show_fixture())

    def test_missing_arch_field_raises(self) -> None:
        bad = _llama8b_show_fixture()
        del bad["model_info"]["llama.block_count"]
        with pytest.raises(ValueError, match="missing required fields"):
            calculate_max_context(12.0, bad)

    def test_missing_parameter_size_raises(self) -> None:
        bad = _llama8b_show_fixture()
        del bad["details"]["parameter_size"]
        with pytest.raises(ValueError, match="parameter_size"):
            calculate_max_context(12.0, bad)

    def test_gqa_factor_affects_kv_cache(self) -> None:
        """Higher GQA grouping (fewer KV heads) → smaller KV cache → larger context fits."""
        no_gqa = _llama8b_show_fixture()
        no_gqa["model_info"]["llama.attention.head_count_kv"] = 32  # GQA factor 1
        with_gqa = _llama8b_show_fixture()  # GQA factor 4 (32/8)
        ctx_no_gqa = calculate_max_context(12.0, no_gqa)
        ctx_with_gqa = calculate_max_context(12.0, with_gqa)
        # GQA factor 4 means 4x smaller KV cache → much larger ctx
        assert ctx_with_gqa > ctx_no_gqa
