"""VRAM-aware context-length calculation for local Ollama models.

When a local model is run with too large a context, Ollama silently
spills the KV cache to CPU memory and inference becomes 10-100x slower —
the pipeline appears to hang. The user-facing `--max-vram <GB>` flag
asks: "given this VRAM budget, what is the largest ``num_ctx`` that
keeps weights + overhead + KV cache in GPU memory?"

The formula (from https://localllm.in/blog/interactive-vram-calculator):

    VRAM = model_weights + overhead + kv_cache

where:
    model_weights = P * b_w                      # params * bytes/weight
    overhead      = 0.55 + 0.08 * P              # CUDA buffers + scratchpad
    kv_cache      = B * N * 2 * L * (d/g) * b_kv / 1e9

Solving for the max context length N:
    N_max = (vram_gb - model_weights - overhead) * 1e9 / (B * 2 * L * (d/g) * b_kv)

with:
    P     = parameters in billions
    b_w   = bytes per weight (quantization-dependent; see QUANT_BYTES_PER_WEIGHT)
    B     = batch size (1 for QuestFoundry — single request)
    L     = transformer layers (block_count)
    d     = hidden dimension (embedding_length)
    g     = GQA grouping factor (n_head / n_kv_head)
    b_kv  = bytes per KV scalar (typically 2 for FP16 KV cache)

All architectural values come from Ollama's ``/api/show`` endpoint:
``details.parameter_size``, ``details.quantization_level``,
``model_info["*.block_count"]``, ``model_info["*.embedding_length"]``,
``model_info["*.attention.head_count"]``, ``model_info["*.attention.head_count_kv"]``.
"""

from __future__ import annotations

from typing import Any

from questfoundry.observability.logging import get_logger

log = get_logger(__name__)


# Bytes per weight by GGUF quantization format. Practical values that
# include a small overhead allowance — see the interactive VRAM calculator
# referenced in the module docstring. A model packed FP16 takes 2 bytes
# per weight; lower quantizations pack tighter.
QUANT_BYTES_PER_WEIGHT: dict[str, float] = {
    "Q2_K": 0.31,
    "Q3_K_S": 0.39,
    "Q3_K_M": 0.43,
    "Q3_K_L": 0.45,
    "Q4_0": 0.50,
    "Q4_1": 0.53,
    "Q4_K_S": 0.53,
    "Q4_K_M": 0.57,
    "Q5_0": 0.63,
    "Q5_1": 0.65,
    "Q5_K_S": 0.65,
    "Q5_K_M": 0.68,
    "Q6_K": 0.78,
    "Q8_0": 1.00,
    "F16": 2.00,
    "FP16": 2.00,
    "BF16": 2.00,
    "F32": 4.00,
    "FP32": 4.00,
}

# Conservative fallback when quantization is unknown — assume Q4_K_M
# (the most common default). A miscalculation here is forgiving:
# overestimating bytes/weight just gives a smaller num_ctx, never one
# that overshoots VRAM.
DEFAULT_BYTES_PER_WEIGHT = 0.57

# Floor: below 2048 tokens, no QuestFoundry stage produces useful output.
# If max_vram is too small, fail loudly rather than silently choosing 0.
MIN_NUM_CTX = 2048

# Round down to a 1024-aligned value — Ollama tunes for power-of-2-aligned
# contexts and arbitrary sizes don't always help.
NUM_CTX_ALIGNMENT = 1024


class VramTooSmallError(ValueError):
    """Raised when the VRAM budget is below the model's weights + overhead.

    The model can't even load — there's nothing to do but raise. Caller
    surfaces this to the user with the model and budget so they can pick
    a smaller model or a larger budget.
    """

    def __init__(self, model: str, vram_gb: float, weights_gb: float, overhead_gb: float) -> None:
        self.model = model
        self.vram_gb = vram_gb
        self.weights_gb = weights_gb
        self.overhead_gb = overhead_gb
        super().__init__(
            f"Model '{model}' weights ({weights_gb:.2f} GB) plus overhead "
            f"({overhead_gb:.2f} GB) exceed the {vram_gb:.2f} GB VRAM budget "
            f"with no room left for KV cache. Pick a smaller model or raise "
            f"--max-vram."
        )


def parse_quantization(level: str | None) -> float:
    """Map a GGUF quantization level string to bytes per weight.

    Returns the value from ``QUANT_BYTES_PER_WEIGHT`` if known; otherwise
    falls back to ``DEFAULT_BYTES_PER_WEIGHT`` and emits a debug log.
    Match is case-insensitive on the part before any trailing whitespace.
    """
    if not level:
        return DEFAULT_BYTES_PER_WEIGHT
    key = level.strip().upper()
    if key in QUANT_BYTES_PER_WEIGHT:
        return QUANT_BYTES_PER_WEIGHT[key]
    log.debug("vram_unknown_quantization", level=level, fallback=DEFAULT_BYTES_PER_WEIGHT)
    return DEFAULT_BYTES_PER_WEIGHT


def parse_parameter_size(size_str: str | None) -> float | None:
    """Parse Ollama's ``parameter_size`` field (e.g., ``"8.0B"``) to billions.

    Returns the float value or None if the string can't be parsed.
    """
    if not size_str:
        return None
    s = size_str.strip().upper()
    suffix_multipliers = {"B": 1.0, "M": 0.001, "K": 0.000001}
    for suffix, multiplier in suffix_multipliers.items():
        if s.endswith(suffix):
            try:
                return float(s[:-1]) * multiplier
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract_arch_field(model_info: dict[str, Any], suffix: str) -> int | None:
    """Find an architecture field by suffix (e.g., ``.block_count``).

    Ollama prefixes architecture keys with the family name
    (``"qwen2.block_count"``, ``"llama.embedding_length"``). Walking by
    suffix avoids hard-coding the family.
    """
    for key, value in model_info.items():
        if key.endswith(suffix) and isinstance(value, int) and value > 0:
            return value
    return None


def calculate_max_context(
    vram_gb: float,
    show_data: dict[str, Any],
    *,
    batch_size: int = 1,
    bytes_per_kv_scalar: int = 2,
    architectural_max: int | None = None,
) -> int:
    """Compute the largest ``num_ctx`` that fits in ``vram_gb`` GB of VRAM.

    Reads architectural metadata from ``show_data`` (the parsed response
    from Ollama's ``/api/show``), applies the formula in the module
    docstring, and returns a 1024-aligned context length clamped to
    ``[MIN_NUM_CTX, architectural_max]``.

    Raises:
        VramTooSmallError: when weights + overhead alone exceed the budget.
        ValueError: when required architectural fields are missing from
            ``show_data`` (the caller should treat this as "unable to
            compute" and fall back to the standard num_ctx detection).
    """
    if vram_gb <= 0:
        raise ValueError(f"vram_gb must be positive, got {vram_gb}")

    details = show_data.get("details", {}) or {}
    model_info = show_data.get("model_info", {}) or {}

    parameters_b = parse_parameter_size(details.get("parameter_size"))
    if parameters_b is None or parameters_b <= 0:
        raise ValueError("show_data.details.parameter_size missing or unparsable")

    bytes_per_weight = parse_quantization(details.get("quantization_level"))

    block_count = _extract_arch_field(model_info, ".block_count")
    embedding_length = _extract_arch_field(model_info, ".embedding_length")
    head_count = _extract_arch_field(model_info, ".attention.head_count")
    head_count_kv = _extract_arch_field(model_info, ".attention.head_count_kv")

    if not all((block_count, embedding_length, head_count, head_count_kv)):
        missing = [
            name
            for name, val in (
                ("block_count", block_count),
                ("embedding_length", embedding_length),
                ("head_count", head_count),
                ("head_count_kv", head_count_kv),
            )
            if not val
        ]
        raise ValueError(f"show_data.model_info missing required fields: {missing}")

    # mypy: all four are non-None after the all() check above
    assert block_count is not None
    assert embedding_length is not None
    assert head_count is not None
    assert head_count_kv is not None

    weights_gb = parameters_b * bytes_per_weight
    overhead_gb = 0.55 + 0.08 * parameters_b

    available_for_kv_gb = vram_gb - weights_gb - overhead_gb
    if available_for_kv_gb <= 0:
        raise VramTooSmallError(
            model=details.get("family", "<unknown>"),
            vram_gb=vram_gb,
            weights_gb=weights_gb,
            overhead_gb=overhead_gb,
        )

    gqa_factor = head_count / head_count_kv
    kv_per_token_bytes = (
        batch_size * 2 * block_count * (embedding_length / gqa_factor) * bytes_per_kv_scalar
    )
    max_tokens = (available_for_kv_gb * 1e9) / kv_per_token_bytes

    aligned = int(max_tokens // NUM_CTX_ALIGNMENT * NUM_CTX_ALIGNMENT)

    if aligned < MIN_NUM_CTX:
        # Aligned to a value below the floor — caller may want to raise
        # rather than run a degenerate small context. We return MIN_NUM_CTX
        # and let the caller decide; if even that doesn't fit weights+overhead
        # the VramTooSmallError above already fired.
        log.warning(
            "vram_calculated_below_floor",
            calculated=aligned,
            floor=MIN_NUM_CTX,
            vram_gb=vram_gb,
        )
        aligned = MIN_NUM_CTX

    if architectural_max is not None and aligned > architectural_max:
        log.debug(
            "vram_capped_at_architectural_max",
            calculated=aligned,
            architectural_max=architectural_max,
        )
        aligned = architectural_max

    log.info(
        "vram_context_calculated",
        vram_gb=vram_gb,
        parameters_b=parameters_b,
        quantization=details.get("quantization_level"),
        weights_gb=round(weights_gb, 3),
        overhead_gb=round(overhead_gb, 3),
        num_ctx=aligned,
    )
    return aligned
