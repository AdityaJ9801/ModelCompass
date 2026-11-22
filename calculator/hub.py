"""Session-scoped optional authentication; metadata only, never weights."""
from dataclasses import dataclass, asdict
import json
import time
import httpx
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

HUB_ENDPOINT = "https://huggingface.co"

SUPPORTED = {"llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3_text", "phi3", "phi", "gpt_neox", "gpt2", "opt"}


class MetadataError(Exception):
    def __init__(self, message, reason="unknown", retryable=False):
        super().__init__(message)
        self.reason = reason
        self.retryable = retryable


def classify_error(exc, stage):
    """Return safe messages only: upstream errors can contain request details."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return MetadataError("Access denied. Accept this model's terms and obtain approval on Hugging Face, then use a valid read token from that approved account. Manual entry still works.", "access_denied")
    if status == 404:
        message = "This repository has no accessible config.json at this revision." if stage == "configuration" else "Repository not found or not accessible to this account."
        return MetadataError(message + " Manual entry still works.", "not_found")
    if status == 429:
        return MetadataError("Hugging Face is rate limiting requests. Wait and try again. Manual entry still works.", "rate_limited", True)
    if status is not None and 500 <= status <= 599:
        return MetadataError("Hugging Face is temporarily unavailable. Try again later. Manual entry still works.", "server_error", True)
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return MetadataError("The Hugging Face request timed out. Try again. Manual entry still works.", "timeout", True)
    if isinstance(exc, httpx.TransportError):
        return MetadataError("Could not connect to Hugging Face. Check the connection and try again. Manual entry still works.", "connection", True)
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
        return MetadataError("The model configuration or metadata could not be interpreted. Manual entry still works.", "invalid_metadata")
    return MetadataError("The model metadata could not be processed. Manual entry still works.", "unknown")


@dataclass
class Metadata:
    repo_id: str
    parameters_b: float | None
    layers: int | None
    kv_heads: int | None
    head_dim: int | None
    context_limit: int | None
    supported: bool
    architecture: str
    license: str | None
    quantization: dict | None
    provenance: dict
    warnings: list[str]


def normalize(repo_id, config, info):
    warnings = []
    architecture = config.get("model_type", "unknown")
    unsupported_keys = ("vision_config", "text_config", "num_local_experts", "num_experts", "n_routed_experts", "mamba_d_state", "kv_lora_rank")
    supported = architecture in SUPPORTED and not any(config.get(k) for k in unsupported_keys)
    if not supported:
        warnings.append("This architecture is unsupported or unknown. Automatic estimation is disabled; use manual entry for a supported dense text model.")
    safe = info.get("safetensors") or {}
    params = safe.get("total")
    quant = config.get("quantization_config")
    if quant:
        # Packed quantized tensors need not count original model parameters.
        params = None
        warnings.append("Quantized repository: enter the original parameter count manually; packed tensor counts can be misleading.")
    heads = config.get("num_attention_heads", config.get("n_head"))
    hidden = config.get("hidden_size", config.get("n_embd"))
    dim = config.get("head_dim") or (hidden // heads if hidden and heads and hidden % heads == 0 else None)
    layers = config.get("num_hidden_layers", config.get("n_layer"))
    kv_heads = config.get("num_key_value_heads", heads)
    limit = config.get("max_position_embeddings", config.get("n_positions"))
    if not all((layers, kv_heads, dim)):
        warnings.append("Attention dimensions are incomplete; illustrative defaults will be used for missing fields.")
    if config.get("sliding_window") and config.get("use_sliding_window", True):
        warnings.append("Sliding-window attention detected: full-context cache is a conservative estimate; runtime cache behavior varies.")
    if not params:
        warnings.append("Parameter count is unavailable. Enter it manually before using this model.")
    if config.get("tie_word_embeddings") is not False:
        warnings.append("Tied embeddings may be deduplicated differently by the runtime. Hub tensor totals are a storage-based parameter estimate, not a measured allocation.")
    card = info.get("cardData") or {}
    return Metadata(repo_id, params / 1e9 if params else None, layers, kv_heads, dim, limit,
                    supported, architecture, card.get("license"), quant,
                    {"parameters": "Hub safetensors.total" if params else "manual required",
                     "architecture": f"{repo_id}/config.json", "revision": info.get("sha"),
                     "context": "config maximum; extended RoPE scaling is not inferred"}, warnings)


def search_models(query="", author="", library="", sort="downloads", minimum=0.0, maximum=1000.0, include_unknown=True):
    try:
        api = HfApi(token=False)
        models = api.list_models(search=query or None, author=author or None,
                                 filter=library or None, pipeline_tag="text-generation",
                                 sort=sort, limit=30,
                                 expand=["safetensors", "downloads", "likes", "library_name"])
        rows = []
        for model in models:
            safe = getattr(model, "safetensors", None)
            total = getattr(safe, "total", None) if safe else None
            count = total / 1e9 if total else None
            if (count is None and include_unknown) or (count is not None and minimum <= count <= maximum):
                rows.append({"id": model.id, "parameters_b": count})
        return rows
    except Exception as exc:
        raise MetadataError("Hugging Face search is unavailable. Try again later or enter the model manually.") from exc


def _load_model_once(repo_id, token, stage):
        info = HfApi(endpoint=HUB_ENDPOINT, token=token).model_info(repo_id, timeout=10)
        stage[0] = "configuration"
        # Bounded streamed JSON fetch, pinned to the metadata revision.
        url = hf_hub_url(repo_id, "config.json", revision=info.sha, endpoint=HUB_ENDPOINT)
        # Per-request headers only. Hub's HTTPX client strips Authorization on
        # cross-origin redirects; no login(), disk downloads, or global auth state.
        with get_session().stream("GET", url, headers=build_hf_headers(token=token), timeout=10, follow_redirects=True) as response:
            hf_raise_for_status(response)
            data = bytearray()
            for chunk in response.iter_bytes():
                data.extend(chunk)
                if len(data) > 2 * 1024 * 1024:
                    raise MetadataError("Configuration exceeds the 2 MiB metadata limit. Use manual entry.")
        config = json.loads(data)
        if not isinstance(config, dict):
            raise ValueError("Invalid config")
        safe = getattr(info, "safetensors", None)
        card = getattr(info, "card_data", None)
        raw = {"safetensors": {"total": getattr(safe, "total", None)},
               "cardData": card.to_dict() if card else {}, "sha": info.sha}
        return asdict(normalize(repo_id, config, raw))


def load_model(repo_id, token=None):
    """Load without caching. Callers must never share authenticated results."""
    auth = token.strip() if isinstance(token, str) and token.strip() else False
    failure = None
    for attempt in range(3):
        stage = ["metadata"]
        try:
            return _load_model_once(repo_id, auth, stage)
        except MetadataError as exc:
            failure = exc
            break
        except Exception as exc:
            failure = classify_error(exc, stage[0])
            if not failure.retryable or attempt == 2:
                break
            retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("Retry-After")
            delay = 2**attempt
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    break  # A dated retry instruction is best handled by the user.
            if delay > 10:
                break
            time.sleep(delay)
    # Raise outside the exception handler to avoid retaining sensitive requests
    # in the exposed exception chain.
    raise failure from None
