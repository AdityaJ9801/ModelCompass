from dataclasses import replace
import pytest
from calculator.core import Model, Workload, Hardware, estimate, compare, at_context
from calculator.config import GIB, GB


def test_default_components_and_units():
    e = estimate(Model(), Workload())
    assert e.weights == pytest.approx(3.85 * GB)
    assert e.cache == GIB
    assert e.total == pytest.approx((e.weights + e.cache) * 1.15)
    assert GB != GIB


def test_precision_and_grouped_attention():
    base = estimate(Model(), Workload())
    assert estimate(Model(weight_bits=16), Workload()).weights == 14 * GB
    assert estimate(Model(kv_heads=32), Workload()).cache == 4 * base.cache
    assert estimate(Model(), Workload(sequences=2)).cache == 2 * base.cache
    assert estimate(Model(), Workload(cache_bits=8)).cache == base.cache / 2


@pytest.mark.parametrize('capacity', [1, 4, 8, 16, 32, 128])
def test_max_context_forward_boundary(capacity):
    m, w = Model(), Workload()
    h = Hardware('Discrete GPU', capacity * GIB, .9)
    c = compare(m, w, h)
    if c.memory_context:
        assert estimate(m, at_context(w, c.memory_context)).total <= c.usable
    assert estimate(m, at_context(w, c.memory_context + 1)).total > c.usable


def test_model_cap_and_output_reservation():
    c = compare(Model(context_limit=1024), Workload(output_tokens=2048), Hardware('Apple Silicon', 64 * GIB, .67))
    assert c.recommended_context == 1024
    assert c.prompt_context == 0
    assert c.context_exceeds_limit


def test_offload_accounts_for_each_pool():
    m, w = Model(parameters_b=30), Workload()
    h = Hardware('Discrete GPU', 4 * GIB, .9, True, 32 * GIB)
    c = compare(m, w, h)
    assert c.verdict == 'May fit with RAM offload'
    assert c.cpu_required <= h.ram_bytes * h.ram_fraction
    assert c.cpu_weights + c.gpu_weights == estimate(m, w).weights
    assert c.speed_ceiling is None
    assert compare(m, replace(w, tokens=100000), h).verdict == 'Does not fit'


def test_override_scales_and_zero_capacity():
    w = Workload(cache_override_gib=2)
    assert estimate(Model(), at_context(w, 16384)).cache == 4 * GIB
    c = compare(Model(), w, Hardware('CPU / integrated graphics', GIB, .75))
    assert c.memory_context == 0
    assert c.verdict == 'Does not fit'


def test_speed_and_validation():
    assert compare(Model(), Workload(), Hardware('Discrete GPU', 16 * GIB, .9)).speed_ceiling is None
    assert compare(Model(), Workload(), Hardware('Discrete GPU', 16 * GIB, .9, bandwidth_gbs=100)).speed_ceiling == pytest.approx(100 / 3.85)
    with pytest.raises(ValueError):
        Hardware('Apple Silicon', GIB, .67, True, GIB)
    with pytest.raises(ValueError):
        Model(parameters_b=float('nan'))
    with pytest.raises(ValueError):
        Workload(tokens=0)


def test_exact_fit_and_offload_context_boundary():
    m, w = Model(), Workload()
    e = estimate(m, w)
    h = Hardware('Discrete GPU', (e.weights + e.cache) * 1.15, 1)
    assert compare(m, w, h).verdict == 'Fits in memory'
    assert compare(m, w, replace(h, capacity_bytes=h.capacity_bytes - 1)).verdict == 'Does not fit'
    h = Hardware('Discrete GPU', 8 * GIB, .9, True, 4 * GIB)
    c = compare(m, w, h)
    assert (c.gpu_weights + e.cache_per_token * c.memory_context) * 1.15 <= c.usable
    assert (c.gpu_weights + e.cache_per_token * (c.memory_context + 1)) * 1.15 > c.usable
