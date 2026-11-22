"""Refresh small, pinned official model configuration fixtures (no weights)."""
from pathlib import Path
import json
from datetime import datetime, timezone
import httpx
from huggingface_hub import HfApi, hf_hub_url

ROOT = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures'
REPOS = ['Qwen/Qwen2.5-7B', 'Qwen/Qwen2.5-0.5B', 'openai-community/gpt2']

def main():
    ROOT.mkdir(exist_ok=True)
    for repo in REPOS:
        info = HfApi(token=False).model_info(repo, timeout=15)
        url = hf_hub_url(repo, 'config.json', revision=info.sha)
        response = httpx.get(url, follow_redirects=True, timeout=15)
        response.raise_for_status()
        safe = info.safetensors
        item = dict(repo_id=repo, revision=info.sha, source=url,
                    captured_at=datetime.now(timezone.utc).isoformat(),
                    config=response.json(), info={'sha': info.sha, 'safetensors': {'total': safe.total if safe else None}})
        (ROOT / (repo.split('/')[-1] + '.json')).write_text(json.dumps(item, indent=2), encoding='utf-8')
        print(repo, info.sha)

if __name__ == '__main__':
    main()
