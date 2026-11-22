from dataclasses import asdict, replace
import json
import altair as alt
import pandas as pd
import streamlit as st
from .config import GIB, GB, RESERVES, CONTEXTS, DEFAULT_OVERHEAD, DEFAULT_QUANT_ALLOWANCE
from .core import Model, Workload, Hardware, estimate, compare, at_context
from .hub import search_models, load_model, MetadataError
from .access import clear_access, forget_authenticated_model, load_for_session
from .theme import render_header, render_footer

cached_search = st.cache_data(ttl=3600, max_entries=128, show_spinner=False)(search_models)
@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def cached_load(repo_id):
    # This cache entrypoint deliberately has no credential parameter.
    return load_model(repo_id)
MEMORY_CAVEAT = "Actual memory consumption varies by quantization format, runtime, attention backend, and hardware. These results are planning estimates."


def memory(value):
    return f"{value / GIB:,.2f} GiB"


def hub_picker():
    with st.expander("Search Hugging Face — optional"):
        st.caption("Public text-generation models only. We read small metadata files, never model weights.")
        st.markdown("**Restricted-model access — optional**")
        st.caption("For gated models such as Llama, accept the publisher's terms and receive access approval on Hugging Face first. A token alone does not grant access. Use a read token from your approved account; it is sent to this app's server for this session only.")
        st.text_input("Hugging Face access token", type="password", key="hf_token",
                      on_change=forget_authenticated_model, args=(st.session_state,))
        st.button("Clear token", key="clear_token", on_click=clear_access, args=(st.session_state,))
        st.caption("Search remains public. Authenticated model details stay in this session and are excluded from shared caches and downloads.")
        with st.form("hub_search"):
            query = st.text_input("Model name or keyword", placeholder="e.g. Llama, Qwen, Gemma")
            a, b, c = st.columns(3)
            author = a.text_input("Author (optional)")
            library = b.selectbox("Library", ["Any", "transformers", "gguf", "mlx"])
            sort = c.selectbox("Sort by", ["downloads", "likes", "lastModified"])
            a, b = st.columns(2)
            minimum = a.number_input("Minimum parameters (billions)", 0.0, 10000.0, 0.0)
            maximum = b.number_input("Maximum parameters (billions)", 0.0, 10000.0, 1000.0)
            unknown = st.checkbox("Include models with unknown parameter count", True)
            submitted = st.form_submit_button("Search models")
        if submitted:
            st.session_state.pop("hub_results", None)
            if minimum > maximum:
                st.error("Minimum parameter count must not exceed maximum.")
            else:
                try:
                    with st.spinner("Finding public models…"):
                        st.session_state.hub_results = cached_search(query, author, "" if library == "Any" else library, sort, minimum, maximum, unknown)
                except MetadataError as exc:
                    st.warning(str(exc))
        rows = st.session_state.get("hub_results")
        if rows:
            mapping = {r["id"]: r["parameters_b"] for r in rows}
            selected = st.selectbox("Search results", list(mapping), format_func=lambda x: f"{x} · {str(round(mapping[x], 2)) + 'B' if mapping[x] else 'size unknown'}")
            st.caption("Filters apply to the first 30 matches. Quantized tensor counts may not represent original parameter counts.")
            if st.button("Use this model"):
                try:
                    with st.spinner("Reading model specifications…"):
                        meta, authenticated = load_for_session(selected, st.session_state.get("hf_token"), cached_load, load_model)
                    st.session_state.metadata = meta
                    st.session_state.metadata_authenticated = authenticated
                    if meta["supported"]:
                        st.session_state.params = float(meta["parameters_b"] or 7.0)
                        st.session_state.layers = int(meta["layers"] or 32)
                        st.session_state.kv_heads = int(meta["kv_heads"] or 8)
                        st.session_state.head_dim = int(meta["head_dim"] or 128)
                        st.session_state.limit = int(meta["context_limit"] or 0)
                        st.session_state.known = all(meta[k] for k in ("layers", "kv_heads", "head_dim"))
                        st.session_state.needs_params = meta["parameters_b"] is None
                        st.session_state.confirm_params = False
                except MetadataError as exc:
                    st.warning(str(exc))
                    if exc.reason == "access_denied":
                        st.link_button("Open model page to request access", f"https://huggingface.co/{selected}")
                    previous = st.session_state.get("metadata")
                    st.caption(f"Results still use {previous['repo_id']}." if previous else "Results still use your current manual inputs.")
        elif rows == []:
            st.info("No matching models in this result batch. Broaden the filters or use manual entry.")


def model_inputs():
    st.subheader("01 / Start with your model", anchor="start-with-your-model")
    st.caption("Start with the model size. Hardware details are optional and come later.")
    hub_picker()
    meta = st.session_state.get("metadata")
    if meta:
        st.info(f"Selected repository: {meta['repo_id']}")
        for warning in meta["warnings"]:
            st.warning(warning)
        st.caption(f"License: {meta['license'] or 'not supplied'} · Architecture: {meta['architecture']}")
        st.caption(f"Repository quantization: {meta['quantization'] or 'not specified'}. Your precision selection below is a planning scenario, not confirmation of an available artifact.")
        if st.button("Clear selected model and use manual entry"):
            forget_authenticated_model(st.session_state)
            st.session_state.pop("metadata", None)
            st.session_state.needs_params = False
            st.rerun()
    defaults = dict(params=7.0, layers=32, kv_heads=8, head_dim=128, limit=0, known=False)
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    a, b, c = st.columns(3)
    params = a.number_input("Model size (billions of parameters)", min_value=0.001, max_value=10000.0, key="params", help="7B means approximately seven billion model parameters.")
    bits = b.selectbox("Weight precision", [4, 8, 16, 32], format_func=lambda x: f"{x}-bit" + (" · smaller memory footprint" if x == 4 else ""))
    tokens = c.number_input("Total context (tokens)", 1, 10000000, 8192, help="Includes your prompt, conversation history, and generated reply. Tokens are pieces of text, not words.")
    confirmation = True
    if st.session_state.get("needs_params"):
        confirmation = st.checkbox("I have entered the original model parameter count", key="confirm_params")
    with st.expander("Advanced settings"):
        st.caption("Only change architecture values when you know the model's configuration.")
        a, b, c = st.columns(3)
        layers = a.number_input("Attention layers", 1, 10000, key="layers")
        heads = b.number_input("KV heads", 1, 10000, key="kv_heads")
        dim = c.number_input("Head dimension", 1, 10000, key="head_dim")
        known = st.checkbox("Architecture values are verified for this model", key="known")
        limit = st.number_input("Model context limit (0 = unknown)", 0, 100000000, key="limit")
        a, b, c = st.columns(3)
        cache_bits = a.selectbox("Cache precision", [16, 8, 32], format_func=lambda x: f"{x}-bit")
        sequences = b.number_input("Concurrent sequences", 1, 1024, 1)
        output = c.number_input("Reserve tokens for generated output", 0, 10000000, 512)
        a, b = st.columns(2)
        overhead = a.number_input("Runtime overhead (%)", 0.0, 100.0, DEFAULT_OVERHEAD * 100)
        allowance = b.number_input("Quantized-weight allowance (%)", 0.0, 100.0, DEFAULT_QUANT_ALLOWANCE * 100)
        override = st.checkbox("Override total KV-cache memory")
        cache_budget = st.number_input("Cache memory for this context and all sequences (GiB)", 0.001, 100000.0, 1.0) if override else None
        if override:
            st.caption("Other context lengths scale this budget linearly; sequence alternatives are omitted because the override covers all sequences.")
    model = Model(params, bits, layers, heads, dim, limit or None, known, allowance / 100)
    workload = Workload(tokens, sequences, cache_bits, overhead / 100, output, cache_budget)
    if meta and not meta["supported"]:
        st.error("Clear the unsupported selection to calculate a supported model manually.")
        st.stop()
    if not confirmation:
        st.info("Confirm the parameter count to display requirements for this repository.")
        st.stop()
    return model, workload


def show_requirements(model, workload):
    result = estimate(model, workload)
    st.subheader("02 / Your memory picture", anchor="your-memory-picture")
    st.caption("Planning estimates for inference, not training. Values below use GiB (1 GiB = 1,073,741,824 bytes).")
    st.caption(MEMORY_CAVEAT)
    if not model.architecture_known:
        st.warning("Low confidence: cache uses illustrative architecture values (default: 32 layers, 8 KV heads, dimension 128). Parameter count alone cannot determine cache size. Verify Advanced settings or supply a cache budget.")
    if model.context_limit and workload.tokens > model.context_limit:
        st.warning("Your requested context exceeds the model's advertised limit. The memory estimate does not make that context supported.")
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Total inference memory", "Model weights", "Context / KV cache", "Runtime overhead"], [result.total, result.weights, result.cache, result.overhead]):
        col.metric(label, memory(value))
    frame = pd.DataFrame({"Component": ["Weights", "KV cache", "Runtime overhead"], "GiB": [result.weights / GIB, result.cache / GIB, result.overhead / GIB], "Memory": ["Estimated memory"] * 3})
    chart = alt.Chart(frame).mark_bar(size=34).encode(x=alt.X("GiB:Q", title="Memory (GiB)"), y=alt.Y("Memory:N", title=None, axis=None), color=alt.Color("Component:N", scale=alt.Scale(domain=["Weights", "KV cache", "Runtime overhead"], range=["#39575E", "#D7BD66", "#CC9586"]), legend=alt.Legend(orient="bottom", title=None)), tooltip=["Component", alt.Tooltip("GiB", format=".2f")]).properties(height=90).configure_view(strokeWidth=0).configure_axis(gridColor="#DED9CB", labelColor="#5E6258", titleColor="#5E6258")
    st.altair_chart(chart, use_container_width=True)
    st.markdown("**Estimated advertised memory needed**")
    for col, (kind, fraction) in zip(st.columns(3), RESERVES.items()):
        col.metric(kind, memory(result.total / fraction))
        col.caption(f"Assumes {fraction:.0%} of total memory is available. Capacity only; runtime support must also match.")
    rows = []
    for count in CONTEXTS:
        scenario = estimate(model, at_context(workload, count))
        rows.append({"Total context": f"{count:,} tokens", "Required memory (GiB)": round(scenario.total / GIB, 2), "Model limit": "Exceeds advertised limit" if model.context_limit and count > model.context_limit else ("Within advertised limit" if model.context_limit else "Unknown")})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    with st.expander("How this estimate works"):
        st.write("Weights = parameter count × precision / 8, plus the quantized-weight allowance for 4/8-bit. KV cache = 2 × layers × KV heads × head dimension × tokens × sequences × cache bytes. Runtime overhead applies to both weights and cache.")
        st.write("GB is decimal (1,000,000,000 bytes); GiB is binary (1,073,741,824 bytes). Weight precision and cache precision are independent. Lower precision needs runtime support and can affect quality.")
        st.write("These estimates do not include a guaranteed peak during loading or prompt processing. Architecture, runtime, attention implementation, and allocator behavior can change actual memory use.")
    return result


def hardware_panel(model, workload):
    st.subheader("03 / Meet your hardware", anchor="meet-your-hardware")
    st.caption("Enter your own specifications. This hosted app cannot inspect your computer and never compares against the hosting server.")
    if not st.checkbox("Add my hardware specifications", key="add_hardware"):
        return None, None
    a, b, c = st.columns(3)
    kind = a.selectbox("Hardware type", list(RESERVES))
    capacity = b.number_input("Total VRAM" if kind == "Discrete GPU" else "Total unified / system memory", 0.1, 100000.0, 16.0)
    unit = c.selectbox("Memory unit", ["GiB", "GB"], help="Use the unit in your hardware specification. GB is decimal; GiB is binary.")
    usable = st.slider("Memory available for inference (%)", 1, 100, round(RESERVES[kind] * 100), key=f"usable_{kind}")
    offload = st.checkbox("Allow model weights to offload to system RAM") if kind == "Discrete GPU" else False
    ram, ram_fraction = 0.0, 0.75
    cpu_multiplier = 1.0
    if offload:
        ram = st.number_input(f"System RAM ({unit})", 0.1, 100000.0, 32.0)
        ram_fraction = st.slider("System RAM available for offload (%)", 1, 100, 75) / 100
        cpu_format = st.selectbox("Offloaded weight storage", ["Same compressed format", "FP32 CPU weights (e.g. bitsandbytes INT8 offload)"])
        if cpu_format.startswith("FP32"):
            cpu_multiplier = 4e9 * model.parameters_b / estimate(model, workload).weights
        st.caption("CPU storage depends on the runtime. Compressed CPU weights are not supported by every offload implementation.")
        st.caption("Maximum-context placement uses as much available RAM for weights as possible and keeps cache on the GPU. Offload needs runtime support and may be substantially slower.")
    bandwidth = None
    if st.checkbox("I know the memory bandwidth"):
        bandwidth = st.number_input("Memory bandwidth (GB/s)", 0.1, 100000.0, 100.0)
    factor = GIB if unit == "GiB" else GB
    hardware = Hardware(kind, capacity * factor, usable / 100, offload, ram * factor, ram_fraction, bandwidth, cpu_multiplier)
    result = compare(model, workload, hardware)
    {"Fits in memory": st.success, "May fit with RAM offload": st.warning, "Does not fit": st.error}[result.verdict](result.verdict)
    a, b, c = st.columns(3)
    a.metric("Memory remaining" if result.remaining >= 0 else "Memory shortfall", memory(abs(result.remaining)))
    b.metric("Estimated maximum total context", f"{result.recommended_context:,} tokens")
    c.metric("Available for prompt + history", f"{result.prompt_context:,} tokens")
    st.caption(f"Memory-based capacity: {result.memory_context:,} tokens. " + (f"Capped at the model limit of {model.context_limit:,}. " if model.context_limit else "Model limit unknown; this is a memory-only estimate. ") + f"Reserving {workload.output_tokens:,} output tokens. This is a capacity estimate, not a runtime allocation guarantee.")
    if result.cpu_weights:
        st.write(f"Weight placement: {memory(result.gpu_weights)} on GPU + {memory(result.cpu_weights)} in RAM. RAM including overhead: {memory(result.cpu_required)}. GPU remaining memory includes its cache and overhead.")
    if workload.output_tokens > result.recommended_context:
        st.warning("The requested output reservation exceeds the estimated context capacity.")
    if result.speed_ceiling is None:
        st.info("Speed unknown. Bandwidth alone cannot predict throughput; offload and concurrent workloads need runtime measurements.")
    else:
        st.info(f"Optimistic weight-streaming ceiling: {result.speed_ceiling:.1f} tokens/sec. This is not predicted or measured speed.")
        if result.speed_ceiling < 8:
            st.warning("Even the optimistic ceiling is below 8 tokens/sec: decoding is likely slow.")
    st.caption("Prompt processing, long-context attention, compute capability, runtime compatibility, and RAM transfers can all limit performance.")
    return hardware, result


def alternatives(model, workload, hardware):
    scenarios = [("Half the context", model, at_context(workload, max(1, workload.tokens // 2)))]
    if model.weight_bits > 4:
        scenarios.append(("4-bit weights", replace(model, weight_bits=4), workload))
    if workload.sequences > 1 and workload.cache_override_gib is None:
        scenarios.append(("One concurrent sequence", model, replace(workload, sequences=1)))
    rows = []
    for label, candidate, work in scenarios:
        row = {"Adjustment": label, "Total memory (GiB)": round(estimate(candidate, work).total / GIB, 2)}
        if hardware:
            row["Hardware result"] = compare(candidate, work, hardware).verdict
        rows.append(row)
    st.markdown("**Ways to reduce memory**")
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render():
    render_header()
    with st.container(key="model_panel"):
        model, workload = model_inputs()
    with st.container(key="results_panel"):
        result = show_requirements(model, workload)
    with st.container(key="hardware_panel"):
        hardware, comparison = hardware_panel(model, workload)
    with st.container(key="alternatives_panel"):
        alternatives(model, workload, hardware)
    if st.session_state.get("metadata_authenticated"):
        st.caption("Downloads are disabled for authenticated model details. Clear the selection to create a manual estimate for export.")
        render_footer()
        return
    report = {"schema_version": 1, "units": "bytes unless named otherwise", "model": asdict(model), "workload": asdict(workload), "estimate": asdict(result), "hardware": asdict(hardware) if hardware else None, "comparison": asdict(comparison) if comparison else None, "metadata": st.session_state.get("metadata"), "assumptions": {"usable_memory_defaults": RESERVES, "cache_override": "Total budget at selected context, scaled linearly" if workload.cache_override_gib else None, "speed": "Weight-streaming ceiling only; not measured throughput", "scope": "Single-machine dense text inference; runtime compatibility is not guaranteed"}}
    report["assumptions"]["memory_variability"] = MEMORY_CAVEAT
    st.download_button("Download estimate · JSON", json.dumps(report, indent=2), "modelcompass-estimate.json", "application/json")
    st.caption("No model weights are downloaded. Your specifications stay in this session; this app does not persist them.")
    render_footer()

