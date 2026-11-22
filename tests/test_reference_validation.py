"""Independent integer-arithmetic oracle using pinned first-party configurations."""
from pathlib import Path
import json
from fractions import Fraction
from dataclasses import replace
import pytest
from calculator.core import Model, Workload, Hardware, estimate, compare
from calculator.hub import normalize
from calculator.config import GIB

FIXTURES = [json.loads(p.read_text()) for p in sorted((Path(__file__).parent / 'fixtures').glob('*.json'))]


@pytest.mark.parametrize('record', FIXTURES, ids=lambda r:r['repo_id'])
@pytest.mark.parametrize('bits', [4, 8, 16, 32])
@pytest.mark.parametrize('tokens', [1024, 8192, 32768])
def test_source_based_estimates(record, bits, tokens):
    c = record['config']
    metadata = normalize(record['repo_id'], c, record['info'])
    assert metadata.supported
    # Independent source-field calculation; intentionally bypass normalized fields.
    layers = c.get('num_hidden_layers', c.get('n_layer'))
    heads = c.get('num_attention_heads', c.get('n_head'))
    kv = c.get('num_key_value_heads', heads)
    dim = c.get('head_dim') or c.get('hidden_size', c.get('n_embd')) // heads
    cache = 2 * layers * kv * dim * tokens * 2
    weights = Fraction(record['info']['safetensors']['total'] * bits, 8)
    if bits <= 8:
        weights *= Fraction(11, 10)
    expected = (weights + cache) * Fraction(115, 100)
    model = Model(metadata.parameters_b, bits, metadata.layers, metadata.kv_heads, metadata.head_dim, metadata.context_limit, True)
    result = estimate(model, Workload(tokens=tokens))
    assert result.cache == cache
    assert result.weights == pytest.approx(float(weights), rel=1e-12)
    assert result.total == pytest.approx(float(expected), rel=1e-12)
    for kind, capacity, fraction in [('Discrete GPU', 8, .9), ('Apple Silicon', 16, .67), ('CPU / integrated graphics', 32, .75)]:
        h = Hardware(kind, capacity * GIB, fraction)
        comparison = compare(model, Workload(tokens=tokens), h)
        assert (comparison.verdict == 'Fits in memory') == (float(expected) <= h.capacity_bytes * fraction)
        assert comparison.recommended_context <= metadata.context_limit


def test_published_llama2_cache_example():
    # https://huggingface.co/blog/kv-cache-quantization: ~5 GB at 10k tokens.
    e = estimate(Model(layers=32, kv_heads=32, head_dim=128), Workload(tokens=10000))
    assert e.cache == 5_242_880_000
    assert e.cache / 1e9 == pytest.approx(5, rel=.05)


def test_qwen7b_exact_cache_reference():
    record = next(r for r in FIXTURES if r['repo_id'] == 'Qwen/Qwen2.5-7B')
    m = normalize(record['repo_id'], record['config'], record['info'])
    assert (m.layers, m.kv_heads, m.head_dim) == (28, 4, 128)
    assert estimate(Model(m.parameters_b, layers=28, kv_heads=4, head_dim=128), Workload()).cache == 469_762_048
    assert not any('Sliding-window' in w for w in m.warnings)


def test_fp32_offload_consumes_more_ram_and_resident_fit_avoids_offload():
    model, workload = Model(parameters_b=30), Workload()
    h = Hardware('Discrete GPU', 8 * GIB, .9, True, 16 * GIB)
    compressed = compare(model, workload, h)
    fp32 = compare(model, workload, replace(h, cpu_weight_multiplier=4 / .55))
    assert fp32.cpu_weights < compressed.cpu_weights
    assert fp32.cpu_required <= 16 * GIB * .75
    assert fp32.verdict == 'Does not fit'
    fits = compare(Model(), workload, replace(h, capacity_bytes=32 * GIB))
    assert fits.cpu_weights == 0 and fits.verdict == 'Fits in memory'
