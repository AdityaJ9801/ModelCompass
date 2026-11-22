# Reference validation and detailed test cases

## Gated-access regression update

The full suite now passes **77 tests**. New mocked checks cover anonymous versus
authenticated requests, authentication on metadata and configuration fetches,
cross-origin redirect credential stripping, 401/403/404/429/503 handling at both
request stages, transient recovery, exhausted retries, malformed/oversized JSON,
shared-cache bypass, token clearing, session isolation, report exclusion, and
retention of the previous model after a failed selection. No real credentials
or gated model downloads were used. The numerical reference evaluation below
remains a formula validation, not a physical inference benchmark.

Validated on 22 November 2026. This validates architecture extraction, arithmetic,
and policy behavior against first-party metadata. It does **not** establish actual
peak RAM/VRAM, loading success, or tokens/sec on physical machines.

## Evidence and reproducibility

Three small snapshots in `tests/fixtures` contain original config fields, Hub
tensor counts, capture timestamps, revision hashes, and pinned source URLs:

| Official publisher/repository | Fields checked | Purpose |
| --- | --- | --- |
| Qwen/Qwen2.5-7B | 28 layers, 4 KV heads, 128 head dimension, 131072 configured context | Grouped-query attention and context cap |
| Qwen/Qwen2.5-0.5B | Published attention configuration and tensor total | Small model and metadata extraction |
| openai-community/gpt2 | Legacy n_layer/n_head/n_embd/n_positions fields | Legacy configuration and MHA fallback |

Run `.venv\Scripts\python -m pytest -q` for offline validation. Refresh snapshots
explicitly with `.venv\Scripts\python scripts/capture_references.py`; this performs
metadata requests only. Refreshing references changes the validation baseline and
should be reviewed. Google Gemma-2-9B returned HTTP 401 without credentials and is
not counted as validated.

Primary references:

- [Published Llama-2 KV-cache example](https://huggingface.co/blog/kv-cache-quantization)
- [Qwen official 7B configuration](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json)
- [Qwen official 0.5B configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json)
- [GPT-2 configuration](https://huggingface.co/openai-community/gpt2/blob/main/config.json)
- [Bitsandbytes offload and memory footprint](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)

## Detailed automated cases

| Case group | Inputs/scenarios | Expected behavior |
| --- | --- | --- |
| Source matrix (36 cases) | Three snapshots × 4/8/16/32-bit weights × 1024/8192/32768 context | Exact cache bytes; weights and total agree with independent rational arithmetic to 1e-12 relative tolerance |
| Hardware submatrix (108 comparisons) | Every source matrix case on 8 GiB GPU, 16 GiB Apple, 32 GiB CPU | Fit matches reference total versus usable memory; recommended context respects model limit |
| Published worked example | Llama-2: 32 layers, 32 KV heads, dimension 128, 10k tokens, FP16 | 5,242,880,000 cache bytes, consistent with published approximate 5 GB |
| Qwen golden value | Official 7B architecture, 8192 tokens | 469,762,048 cache bytes; disabled sliding window produces no active-window warning |
| Context boundaries | 1/4/8/16/32/128 GiB GPUs | Maximum context fits forward calculation; one additional token exceeds capacity |
| Exact fit | Capacity exactly equal to calculated requirement; then one byte less | Fits at equality; fails below equality |
| Capacity semantics | GB versus GiB, Apple/shared memory reserves | Correct byte conversion; no duplicate shared-memory pool |
| Workload scaling | More KV heads, two sequences, FP16 versus INT8 cache | Linear scaling with each factor |
| Model cap and output | Advertised 1024 context, output reservation 2048 | Recommended context capped; prompt capacity clamped to zero |
| Cache override | Double context with a direct cache budget | Budget doubles; per-token slope preserved |
| Offload | Insufficient VRAM, CPU RAM, large cache | Each pool includes overhead; GPU cache must fit; insufficient cache capacity fails |
| FP32 CPU storage | Same hardware with compressed versus FP32 offloaded weights | FP32 permits less offload; RAM footprint stays within its reserve |
| Resident preference | Offload enabled but request fits in VRAM | No unnecessary CPU placement |
| Speed | Missing bandwidth, single sequence, offload | Unknown without suitable inputs; weight-streaming ceiling only |
| Invalid input | NaN parameters, zero tokens, offload on unified memory | Reject invalid inputs |
| Metadata safety | MoE/multimodal/unknown/quantized configs, timeout | Unsupported models blocked; packed counts not trusted; actionable manual fallback |
| UI flow | Manual first; optional hardware; parameter edits | Requirements immediately available; hardware comparison preserves model |
| Sessions/report | Separate Streamlit sessions; JSON download | Inputs isolated; report download exists |

## Findings and adjustments

The standard full-attention FP16 KV formula agrees with the published worked
example and official configuration fields. No universal percentage correction to
that formula is justified by this evidence.

For Qwen2.5-7B at 8192 context, 4-bit weights with the default 10% allowance and
15% runtime overhead produce **5,357,103,799 bytes (~4.99 GiB)**. An entered 8 GiB
GPU at 90% usable capacity yields **44,188 total context tokens**, or **43,676**
prompt/history tokens with 512 reserved for output. These are calculated examples,
not published hardware benchmarks. They use the source tensor total of 7,615,616,512,
not the rounded marketing size of 7 billion.

Corrections made during this audit:

1. Ignore configured sliding_window when use_sliding_window is false.
2. Offer FP32 CPU storage for runtimes such as bitsandbytes INT8 offload; account
   for its extra RAM instead of assuming compressed CPU storage universally.
3. Keep a requested workload resident when it already fits; offload only when needed.
   Maximum context then reflects that chosen placement, not every possible placement.
4. Warn that tied-embedding storage totals can differ from deduplicated runtime
   parameter counts. GPT-2's Hub tensor total is a storage estimate, not proof of
   its runtime allocation.

## Limits and next calibration step

The 10% quantization allowance, 15% overhead, memory reserves, and bandwidth ceiling
remain heuristics. These sources do not establish universal accuracy for them.
Quantized cache may include scales and residual buffers; the simple cache formula
models payload only. Sliding-window implementations may allocate less cache.

Before claiming measured accuracy, benchmark a pinned model artifact and runtime
on named hardware: record idle memory, model-loaded memory, peak prefill memory,
steady decode memory and throughput at 4K/8K/32K, cache precision, concurrency,
offload placement, attention backend, OS and driver. Compare each component against
this estimate, report signed/absolute error, and calibrate per runtime/format rather
than replacing defaults from one machine. No inference benchmarks were run here.
