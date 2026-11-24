from types import SimpleNamespace
from unittest.mock import patch
import httpx
from calculator.hub import load_model


def missing():
    r = httpx.Response(404, request=httpx.Request('GET', 'https://huggingface.co/config.json'))
    return httpx.HTTPStatusError('missing', request=r.request, response=r)


def record(config=None, gguf=None, base=None):
    return SimpleNamespace(sha='revision', config=config or {}, gguf=gguf,
                           safetensors=None, card_data=SimpleNamespace(to_dict=lambda: {'base_model': base} if base else {}))


def test_gguf_public_metadata_without_config():
    with patch('calculator.hub.HfApi') as api, patch('calculator.hub._read_config', side_effect=missing()):
        api.return_value.model_info.return_value = record(gguf={'architecture':'gemma2','total':9241705984,'context_length':8192})
        m = load_model('bartowski/gemma-2-9b-it-GGUF')
    assert m['supported'] and m['parameters_b'] == 9.241705984
    assert m['context_limit'] == 8192 and m['layers'] is None
    assert 'Public Hub API' in m['provenance']['architecture']


def test_explicit_base_inherits_fields_with_provenance():
    config = dict(model_type='llama', num_hidden_layers=32, num_key_value_heads=8,
                  num_attention_heads=32, hidden_size=4096, max_position_embeddings=8192)
    parent = record()
    parent.safetensors = SimpleNamespace(total=8_000_000_000)
    with patch('calculator.hub.HfApi') as api, patch('calculator.hub._read_config', side_effect=[missing(), config]):
        api.return_value.model_info.side_effect = [record(base='official/base'),parent]
        m = load_model('custom/quant')
    assert m['supported'] and m['parameters_b'] == 8
    assert m['kv_heads'] == 8
    assert m['provenance']['base_model']['repo_id'] == 'official/base'


def test_cycle_is_bounded():
    with patch('calculator.hub.HfApi') as api, patch('calculator.hub._read_config', side_effect=missing()):
        api.return_value.model_info.return_value = record(config={'model_type':'llama'}, base='custom/model')
        m = load_model('custom/model')
        assert api.return_value.model_info.call_count == 1
    assert m['layers'] is None
