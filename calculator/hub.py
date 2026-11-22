"""Public Hub metadata only. Never loads model code or weight files."""
from dataclasses import dataclass, asdict
import json
import httpx
from huggingface_hub import HfApi, hf_hub_url

SUPPORTED = {"llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3_text", "phi3", "phi", "gpt_neox", "gpt2", "opt"}


class MetadataError(Exception):
    pass


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


def load_model(repo_id):
    try:
        info = HfApi(token=False).model_info(repo_id, timeout=10)
        # Bounded streamed JSON fetch, pinned to the metadata revision.
        url = hf_hub_url(repo_id, "config.json", revision=info.sha)
        with httpx.stream("GET", url, timeout=10, follow_redirects=True) as response:
            response.raise_for_status()
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
    except MetadataError:
        raise
    except Exception as exc:
        raise MetadataError("Could not read this model's public configuration. It may be gated, missing, or temporarily unavailable. Manual entry still works.") from exc
