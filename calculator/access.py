"""Session state helpers. No credentials or model data are stored globally."""


def forget_authenticated_model(state):
    if state.pop("metadata_authenticated", False):
        for key in ("metadata", "params", "layers", "kv_heads", "head_dim", "limit",
                    "known", "needs_params", "confirm_params"):
            state.pop(key, None)


def clear_access(state):
    state["hf_token"] = ""
    forget_authenticated_model(state)


def load_for_session(repo_id, token, public_loader, authenticated_loader):
    token = token.strip() if token else ""
    if token:
        return authenticated_loader(repo_id, token=token), True
    return public_loader(repo_id), False
