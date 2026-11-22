import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path

from calculator.access import clear_access, load_for_session
from calculator.hub import load_model, MetadataError, normalize

CONFIG = dict(model_type="llama", num_hidden_layers=32, num_attention_heads=32,
              num_key_value_heads=8, hidden_size=4096, max_position_embeddings=8192)
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def info():
    return SimpleNamespace(sha="pinned-revision", safetensors=SimpleNamespace(total=7_000_000_000), card_data=None)


@pytest.mark.parametrize("token", [None, "hf_session_test"])
def test_auth_both_requests_and_cross_origin_redirect(token):
    requests = []
    def respond(request):
        requests.append(request)
        if request.url.host == "huggingface.co":
            return httpx.Response(302, headers={"location": "https://cdn.example/config.json"})
        return httpx.Response(200, json=CONFIG)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with patch("calculator.hub.HfApi") as api, patch("calculator.hub.get_session", return_value=client):
            api.return_value.model_info.return_value = info()
            result = load_model("org/model", token)
            assert result["supported"]
            api.assert_called_once_with(endpoint="https://huggingface.co", token=token or False)
    assert "pinned-revision" in str(requests[0].url)
    assert requests[0].headers.get("authorization") == (f"Bearer {token}" if token else None)
    assert "authorization" not in requests[1].headers


@pytest.mark.parametrize("status,reason,attempts", [(401,"access_denied",1),(403,"access_denied",1),(404,"not_found",1),(429,"rate_limited",3),(503,"server_error",3)])
@pytest.mark.parametrize("stage", ["metadata", "configuration"])
def test_http_failures(status, reason, attempts, stage):
    count = []
    def respond(request):
        count.append(request)
        return httpx.Response(status, text="Sensitive upstream details hf_session_test")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with patch("calculator.hub.HfApi") as api, patch("calculator.hub.get_session", return_value=client), patch("calculator.hub.time.sleep"):
            if stage == "metadata":
                response = httpx.Response(status, request=httpx.Request("GET", "https://huggingface.co/api/models/org/model"))
                api.return_value.model_info.side_effect = httpx.HTTPStatusError("hf_session_test", request=response.request, response=response)
            else:
                api.return_value.model_info.return_value = info()
            with pytest.raises(MetadataError) as error:
                load_model("org/model", "hf_session_test")
            assert error.value.reason == reason
            assert "hf_session_test" not in str(error.value)
            assert error.value.__context__ is None
            assert api.return_value.model_info.call_count == attempts


def test_transient_retry_then_success():
    with patch("calculator.hub.HfApi") as api, patch("calculator.hub.time.sleep") as sleep:
        api.return_value.model_info.side_effect = [httpx.ReadTimeout("timeout"), info()]
        with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200, json=CONFIG))) as client:
            with patch("calculator.hub.get_session", return_value=client):
                assert load_model("org/model")["supported"]
        assert api.return_value.model_info.call_count == 2
        sleep.assert_called_once_with(1)


def test_timeout_exhaustion():
    with patch("calculator.hub.HfApi") as api, patch("calculator.hub.time.sleep"):
        api.return_value.model_info.side_effect = httpx.ReadTimeout("timeout")
        with pytest.raises(MetadataError) as error:
            load_model("org/model")
        assert error.value.reason == "timeout"
        assert api.return_value.model_info.call_count == 3


@pytest.mark.parametrize("content", [b"not-json", b"x" * (2 * 1024 * 1024 + 1)])
def test_invalid_or_large_config(content):
    with patch("calculator.hub.HfApi") as api:
        api.return_value.model_info.return_value = info()
        with httpx.Client(transport=httpx.MockTransport(lambda request:httpx.Response(200, content=content))) as client:
            with patch("calculator.hub.get_session", return_value=client), pytest.raises(MetadataError):
                load_model("org/model")
        assert api.return_value.model_info.call_count == 1


def test_cache_routing_and_clear():
    public, private = Mock(return_value={"public": True}), Mock(return_value={"private": True})
    assert load_for_session("org/model", "", public, private) == ({"public": True}, False)
    private.assert_not_called()
    public.reset_mock()
    assert load_for_session("org/model", "hf_test", public, private) == ({"private": True}, True)
    public.assert_not_called()
    state = dict(hf_token="hf_test", metadata_authenticated=True, metadata={"private": True}, params=14, layers=32)
    clear_access(state)
    assert state == {"hf_token": ""}


def test_token_session_isolation_and_export_exclusion():
    at = AppTest.from_file(APP, default_timeout=20).run()
    at.text_input(key="hf_token").set_value("hf_test").run()
    at.session_state["metadata"] = asdict(normalize("restricted/model", CONFIG, {"safetensors": {"total": 7e9}}))
    at.session_state["metadata_authenticated"] = True
    at.run()
    assert not at.exception
    assert len(at.get("download_button")) == 0
    second = AppTest.from_file(APP, default_timeout=20).run()
    assert second.text_input(key="hf_token").value == ""
    assert len(second.get("download_button")) == 1
    at.button(key="clear_token").click().run()
    assert not at.exception
    assert at.text_input(key="hf_token").value == ""
    assert "metadata" not in at.session_state
    assert len(at.get("download_button")) == 1


def test_failed_selection_keeps_previous_model():
    at = AppTest.from_file(APP, default_timeout=20)
    at.session_state["metadata"] = asdict(normalize("previous/model", CONFIG, {}))
    at.session_state["hub_results"] = [{"id": "restricted/model", "parameters_b": 7}]
    at.run()
    with patch("calculator.ui.cached_load", side_effect=MetadataError("Access denied", "access_denied")):
        next(b for b in at.button if b.label == "Use this model").click().run()
    assert not at.exception
    assert at.session_state["metadata"]["repo_id"] == "previous/model"
    assert any("Results still use previous/model" in c.value for c in at.caption)
