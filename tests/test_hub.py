from unittest.mock import patch
import pytest
from calculator.hub import normalize, search_models, load_model, MetadataError

CONFIG = dict(model_type='llama', num_hidden_layers=32, num_attention_heads=32,
              num_key_value_heads=8, hidden_size=4096, max_position_embeddings=8192)


def test_normalization():
    m = normalize('org/model', CONFIG, {'safetensors': {'total': 7_000_000_000}})
    assert m.supported and m.parameters_b == 7 and m.head_dim == 128
    assert m.kv_heads == 8 and m.context_limit == 8192


def test_unknown_and_unsupported():
    assert not normalize('x', {}, {}).supported
    for key in ('num_local_experts', 'vision_config', 'kv_lora_rank'):
        assert not normalize('x', {**CONFIG, key: 8}, {}).supported
    m = normalize('x', {**CONFIG, 'quantization_config': {'bits': 4}}, {'safetensors': {'total': 123}})
    assert m.parameters_b is None
    assert m.warnings


def test_network_failure_is_actionable():
    with patch('calculator.hub.HfApi', side_effect=TimeoutError):
        with pytest.raises(MetadataError, match='manually'):
            search_models()
        with pytest.raises(MetadataError, match='Manual entry'):
            load_model('org/gated')
