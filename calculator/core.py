"""Pure memory calculations. No network, UI, or hardware inspection."""
from dataclasses import dataclass, replace
from math import floor, isfinite
from .config import GIB, RESERVES, DEFAULT_OVERHEAD, DEFAULT_QUANT_ALLOWANCE


def positive(name, value, allow_zero=False):
    if not isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'} and finite")


@dataclass(frozen=True)
class Model:
    parameters_b: float = 7.0
    weight_bits: int = 4
    layers: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    context_limit: int | None = None
    architecture_known: bool = False
    quant_allowance: float = DEFAULT_QUANT_ALLOWANCE

    def __post_init__(self):
        for name in ("parameters_b", "layers", "kv_heads", "head_dim"):
            positive(name, getattr(self, name))
        if self.weight_bits not in (4, 8, 16, 32):
            raise ValueError("Unsupported weight precision")
        positive("quantization allowance", self.quant_allowance, True)
        if self.context_limit is not None:
            positive("context limit", self.context_limit)


@dataclass(frozen=True)
class Workload:
    tokens: int = 8192
    sequences: int = 1
    cache_bits: int = 16
    overhead: float = DEFAULT_OVERHEAD
    output_tokens: int = 512
    cache_override_gib: float | None = None

    def __post_init__(self):
        positive("tokens", self.tokens)
        positive("sequences", self.sequences)
        positive("overhead", self.overhead, True)
        positive("output tokens", self.output_tokens, True)
        if self.cache_bits not in (8, 16, 32):
            raise ValueError("Unsupported cache precision")
        if self.cache_override_gib is not None:
            positive("cache budget", self.cache_override_gib)


@dataclass(frozen=True)
class Hardware:
    kind: str
    capacity_bytes: float
    usable_fraction: float
    offload: bool = False
    ram_bytes: float = 0
    ram_fraction: float = 0.75
    bandwidth_gbs: float | None = None
    cpu_weight_multiplier: float = 1.0

    def __post_init__(self):
        if self.kind not in RESERVES:
            raise ValueError("Unknown hardware type")
        positive("capacity", self.capacity_bytes)
        for value in (self.usable_fraction, self.ram_fraction):
            if not 0 < value <= 1:
                raise ValueError("Usable memory fraction must be between 0 and 1")
        positive("RAM", self.ram_bytes, True)
        positive("CPU weight multiplier", self.cpu_weight_multiplier)
        if self.offload and (self.kind != "Discrete GPU" or self.ram_bytes <= 0):
            raise ValueError("Weight offload requires a discrete GPU and system RAM")
        if self.bandwidth_gbs is not None:
            positive("bandwidth", self.bandwidth_gbs)


@dataclass(frozen=True)
class Estimate:
    weights: float
    cache: float
    overhead: float
    total: float
    cache_per_token: float
    confidence: str


@dataclass(frozen=True)
class Comparison:
    verdict: str
    usable: float
    remaining: float
    cpu_weights: float
    gpu_weights: float
    cpu_required: float
    memory_context: int
    recommended_context: int
    prompt_context: int
    speed_ceiling: float | None
    context_exceeds_limit: bool


def estimate(model: Model, workload: Workload) -> Estimate:
    weights = model.parameters_b * 1e9 * model.weight_bits / 8
    if model.weight_bits <= 8:
        weights *= 1 + model.quant_allowance
    per_token = (2 * model.layers * model.kv_heads * model.head_dim
                 * workload.sequences * workload.cache_bits / 8)
    if workload.cache_override_gib is not None:
        per_token = workload.cache_override_gib * GIB / workload.tokens
    cache = per_token * workload.tokens
    overhead = (weights + cache) * workload.overhead
    return Estimate(weights, cache, overhead, weights + cache + overhead, per_token,
                    "Architecture-based estimate" if model.architecture_known else "Low confidence: illustrative architecture")


def compare(model: Model, workload: Workload, hardware: Hardware) -> Comparison:
    result = estimate(model, workload)
    factor = 1 + workload.overhead
    usable = hardware.capacity_bytes * hardware.usable_fraction
    # Use the explicit maximum-offload placement for both fit and context estimates.
    cpu_weights = min(result.weights, hardware.ram_bytes * hardware.ram_fraction / factor / hardware.cpu_weight_multiplier) if hardware.offload and result.total > usable else 0
    resident_weights = result.weights - cpu_weights
    required = (resident_weights + result.cache) * factor
    remaining = usable - required
    verdict = "Does not fit" if remaining < 0 else ("May fit with RAM offload" if cpu_weights else "Fits in memory")
    memory_context = floor(max(0, usable / factor - resident_weights) / result.cache_per_token)
    # Validate the integer boundary through the forward calculation (floating-point rounding).
    def fits(tokens):
        return (resident_weights + result.cache_per_token * tokens) * factor <= usable
    while memory_context > 0 and not fits(memory_context):
        memory_context -= 1
    if fits(memory_context + 1):
        memory_context += 1
    recommended = min(memory_context, model.context_limit) if model.context_limit else memory_context
    ceiling = None
    if hardware.bandwidth_gbs and not cpu_weights and workload.sequences == 1:
        ceiling = hardware.bandwidth_gbs * 1e9 / result.weights
    return Comparison(verdict, usable, remaining, cpu_weights, resident_weights,
                      cpu_weights * factor * hardware.cpu_weight_multiplier, memory_context, recommended,
                      max(0, recommended - workload.output_tokens), ceiling,
                      bool(model.context_limit and workload.tokens > model.context_limit))


def at_context(workload: Workload, tokens: int) -> Workload:
    """Preserve the override's per-token slope when changing context."""
    budget = workload.cache_override_gib
    if budget is not None:
        budget *= tokens / workload.tokens
    return replace(workload, tokens=tokens, cache_override_gib=budget)
