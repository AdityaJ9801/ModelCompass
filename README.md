<div align="center">

# ModelCompass

### Pick a model. Understand its memory needs. Plan your next step.

A beginner-friendly memory planner for running open-weight language models locally.

**Python · Streamlit · Hugging Face · Independent calculation engine**

</div>

---

## The question behind this project

“Will this model run on my machine?”

It's a question AI developers hear often. A teammate finds an interesting model, sees “7B” in its name, and wants to try it. Someone else is planning a hardware upgrade. Another person already has a powerful setup but still runs out of memory when they send a long document.

The model's parameter count is only the beginning. Weight precision changes how much memory the model occupies. A longer conversation needs more KV-cache memory. The operating system and inference runtime need room too. A model that fits for a short chat may need a different setup for a long document.

**ModelCompass turns those moving parts into an understandable estimate.** It starts with the model, explains its memory requirements, and lets you optionally compare those requirements with hardware specifications you enter yourself.

Built as a small team tool, it aims to make the conversation before a download, experiment, or hardware purchase a little more informed.

## What you can do

- **Start with a model:** enter its parameter count manually or search public Hugging Face models.
- **See where memory goes:** inspect weights, context/KV cache, and runtime overhead.
- **Explore longer context:** compare requirements at 4K, 8K, 16K, and 32K tokens.
- **Compare your hardware:** enter discrete GPU VRAM, Apple unified memory, or CPU/shared RAM.
- **Estimate context capacity:** see the total context that fits and the prompt/history space left after reserving output tokens.
- **Try adjustments:** compare shorter context, lower weight precision, or fewer concurrent sequences.
- **Explore RAM offload:** account for compressed or FP32 CPU weight storage when supported by your runtime.
- **Share the estimate:** download a JSON report containing inputs, results, and assumptions.

The hosted app uses the specifications you provide. It cannot inspect your computer and does not use the hosting server's hardware as your comparison target.

## From a model to a plan

1. Enter the model size, precision, and desired context length.
2. Read its memory breakdown and estimated capacity requirements.
3. Optionally add your hardware to see a fit verdict and context estimate.
4. Adjust the workload and export the result for your team.

**Example:** the official Qwen2.5-7B configuration with 4-bit weights, FP16 cache, and 8,192 total tokens produces an estimate of **about 4.99 GiB**, including default allowances. An entered **8 GiB GPU** with 90% usable memory yields **44,188 total context tokens**, or **43,676** prompt/history tokens after reserving 512 for output.

This is a calculated example, not a measured hardware benchmark. Its inputs and validation are documented in [VALIDATION.md](VALIDATION.md).

## Run locally

Use Python 3.11 or 3.12. From the project directory on Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

On macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Open the local URL printed by Streamlit. After dependencies are installed, manual calculations work offline. Hugging Face search requires internet access.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Connect that repository in Streamlit Community Cloud.
3. Select `app.py` as the entrypoint and choose Python 3.12.
4. Deploy using the included `requirements.txt` and `.streamlit/config.toml`.

Hosting requires no GPU or shared Hugging Face token. Public metadata is cached for one hour; visitor inputs remain in separate Streamlit sessions. The app does not persist hardware specifications or reports. Credentials should never be committed; `.streamlit/secrets.toml` is ignored by Git.

### Restricted models such as Meta Llama

1. Open the model's Hugging Face page, accept its terms, and obtain any required access approval.
2. Create a read token with access to that repository in your own Hugging Face account.
3. Expand **Search Hugging Face**, enter it in the **Hugging Face access token** password field, then select **Use this model**.
4. Use **Clear token** when finished. This also clears authenticated model details and their populated model inputs. Changing the token has the same model-clearing effect.

A token alone does not grant publisher approval. The token is sent to the app server and used for both metadata and configuration requests during your session. Search remains anonymous. Authenticated results bypass the public cache and are never saved to disk. Downloads are disabled while an authenticated model is selected so restricted details cannot enter exported reports. No shared app-owner token is used.

Access denied, missing configuration, timeout, and rate-limit failures have specific messages. Access-denied messages link to the model page. Transient failures receive at most three attempts with bounded backoff; access failures are not retried. A failed selection preserves the current valid model. Manual entry remains available.

## How the estimate works

```text
Weight bytes = parameter count × weight bits / 8
KV-cache bytes = 2 × layers × KV heads × head dimension
                 × context tokens × concurrent sequences × cache bytes per element
Total bytes = (weight bytes + KV-cache bytes) × (1 + runtime overhead)
```

For 4/8-bit weights, a configurable allowance is added to the weight estimate. Weight precision and cache precision are separate settings. Context includes the prompt, conversation history, and generated output.

| Default | Value | Meaning |
| --- | --- | --- |
| Weight precision | 4-bit | Initial planning scenario |
| Cache precision | FP16 | Two bytes per cache element |
| Total context | 8,192 tokens | Includes generated output |
| Concurrent sequences | 1 | One sequence's context budget |
| Quantized-weight allowance | 10% | Heuristic for packing/metadata costs |
| Runtime overhead | 15% | Heuristic for additional memory use |
| Usable discrete GPU memory | 90% | Leaves some VRAM for other use |
| Usable Apple unified memory | 67% | Leaves room for OS and applications |
| Usable CPU/shared memory | 75% | Shared RAM is counted once |

These assumptions are editable. Results use **GiB** (`2^30` bytes); hardware inputs accept GiB or decimal **GB** (`10^9` bytes).

Maximum context comes from remaining memory after weights and overhead. The integer boundary is checked with the forward calculation and capped at a known model context limit. Reserving output tokens gives the remaining prompt/history allowance.

Manual entry initially uses an **illustrative** architecture: 32 layers, 8 KV heads, and head dimension 128. These values are not inferred from parameter count. Verify them in Advanced settings or supply a cache budget. Supported Hugging Face configurations populate architecture details automatically.

## How much should you trust the result?

**Actual memory consumption varies by quantization format, runtime, attention backend, and hardware. These results are planning estimates.**

The latest recorded suite passed **77 automated tests**, including authentication and UI regressions and **108 simulated hardware comparisons** within the source-based matrix. Three official model configurations are preserved with revision hashes. The standard KV-cache formula also matches Hugging Face's published worked example. Gated-access tests use mocked requests; live gated verification requires an approved account.

This validates the tested arithmetic and behavior. Physical inference benchmarks have not yet been run, so the project does not claim a measured prediction-error percentage across real computers.

- Loading and prompt processing may have higher memory peaks than steady inference.
- Quantization formats differ in packing, scales, and unquantized tensors.
- Quantized caches can add metadata and residual buffers beyond the payload estimate.
- Sliding-window implementations may allocate less cache; extended RoPE limits are not inferred.
- Hub tensor totals, especially tied embeddings, can differ from runtime parameter allocations.
- CPU-offload storage depends on runtime; some implementations keep CPU weights in FP32.
- A memory fit does not guarantee runtime compatibility or smooth generation.
- Optional bandwidth output is a theoretical weight-streaming ceiling, not predicted tokens/sec.

Read [the detailed evaluation](VALIDATION.md) for test scenarios, evidence, corrections, and a proposed physical benchmark procedure.

## Modular by design

```text
app.py                  Streamlit entrypoint
calculator/
  core.py               Typed records and pure memory/context calculations
  hub.py                Metadata search and configuration normalization
  ui.py                 Input panels, charts, comparisons, and JSON reports
  config.py             Units and calculation defaults
  access.py             Session-only authentication routing and cleanup
tests/
  fixtures/             Revision-pinned official model metadata
scripts/
  capture_references.py Explicit metadata-fixture refresh utility
```

The calculation engine has no UI or network dependencies and can be reused independently:

```python
from calculator.core import Model, Workload, estimate

model = Model(
    parameters_b=7.615616512,
    layers=28,
    kv_heads=4,
    head_dim=128,
    architecture_known=True,
)
result = estimate(model, Workload(tokens=8192))
print(f"Estimated memory: {result.total / 1024**3:.2f} GiB")
```

The Hub adapter reads metadata and small configuration files only. It never downloads weights or executes repository code. Unsupported architectures are flagged rather than silently evaluated with a dense-model formula.

## Run the tests

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

On macOS/Linux, substitute `.venv/bin/python`. Tests run offline using saved fixtures and mocked network failures. They cover source-based calculations, context boundaries, offload, metadata extraction, UI behavior, and session isolation.

## Current scope

Supports planning for **dense text LLM inference on a single machine**. Training, multi-GPU allocation, MoE, multimodal models, automatic hardware detection, and model execution are outside the current version.

## Primary references

- [Hugging Face: KV-cache memory formula](https://huggingface.co/blog/kv-cache-quantization)
- [Hugging Face: bitsandbytes quantization and offload](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
- [Hugging Face Hub API](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api)
- [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing)
