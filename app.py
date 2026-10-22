"""Streamlit demo interface for the LawBot educational prototype."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
DISCLAIMER = "LawBot is for educational use only and is not legal advice."
BASE_MODEL = os.environ.get("LAWBOT_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / "models" / "hf_cache"))
RAG_INDEX_DIR = PROJECT_DIR / "data" / "rag_corpus" / "faiss_index"
RAG_INDEX_READY = (RAG_INDEX_DIR / "index.faiss").exists() and (RAG_INDEX_DIR / "index.pkl").exists()

st.set_page_config(page_title="LawBot", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="Loading language model…")
def load_language_model() -> tuple[Any, Any, threading.Lock]:
    """Load the best local adapter if present, otherwise load the base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    adapter_candidates = [
        Path(os.environ["LAWBOT_ADAPTER_DIR"]) if os.environ.get("LAWBOT_ADAPTER_DIR") else None,
        PROJECT_DIR / "adapters" / "lawbot-grpo",
        PROJECT_DIR / "adapters" / "lawbot-qwen1.5b",
    ]
    adapter_path = next((p for p in adapter_candidates if p and (p / "adapter_config.json").exists()), None)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.eval()
    return model, tokenizer, threading.Lock()


def generate_text(prompt: str, history: list[dict[str, str]] | None = None,
                  system: str | None = None, max_new_tokens: int = 512) -> str:
    """Generate a completion with a short window of conversational context."""
    import torch

    model, tokenizer, generation_lock = load_language_model()
    if system is None:
        complex_request = re.search(
            r"\b(compare|contrast|analy[sz]e|step[- ]by[- ]step|in detail|comprehensive|multiple|pros and cons|explain how|what are the differences)\b",
            prompt, re.I,
        )
        max_new_tokens = min(max_new_tokens, 480 if complex_request else 220)
    messages = [{"role": "system", "content": system or
                 "You are LawBot, a helpful general-purpose assistant with a focus on Indian law. "
                 "Answer ordinary questions as well as legal questions. Start with the direct answer. "
                 "For simple questions, use a few plain-language sentences and stop when the question is answered. "
                 "For complex questions, organize the explanation into clear steps or headings and state any "
                 "important assumptions. For legal topics, explain the "
                 "jurisdiction and date limits, do not invent statute or case citations, and distinguish "
                 "general information from legal advice. If you are unsure, say what is uncertain."}]
    messages.extend(history[-6:] if history else [])
    messages.append({"role": "user", "content": prompt})
    template_options = {"tokenize": False, "add_generation_prompt": True}
    if "Qwen3" in BASE_MODEL:
        template_options["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(messages, **template_options)
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with generation_lock, torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                repetition_penalty=1.04, pad_token_id=tokenizer.eos_token_id)
    completion = output[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


@st.cache_resource(show_spinner=False)
def load_rag_components() -> tuple[Any, Any]:
    from src.rag import load_faiss_index
    return load_faiss_index(PROJECT_DIR)


def rag_answer(query: str, history: list[dict[str, str]] | None, top_k: int) -> dict[str, Any]:
    """Retrieve official text and generate only from passages over a simple floor."""
    try:
        from src.rag import retrieve_with_mmr
        vectorstore, embeddings = load_rag_components()
    except Exception as exc:
        return {"answer": f"I don't have enough evidence in my sources to answer this. Retrieval is unavailable ({type(exc).__name__}).",
                "sources": []}
    candidates = retrieve_with_mmr(query, vectorstore, embeddings, k=top_k, fetch_k=max(20, top_k * 4))
    sources: list[dict[str, Any]] = []
    for item in candidates:
        doc = item["document"]
        meta = doc.metadata
        score = float(item["score"])
        section_numbers = meta.get("section_numbers") or []
        section = meta.get("case_citation") or (
            ("Art. " if meta.get("act") == "Constitution" else "§") + str(meta.get("section_number"))
            if meta.get("section_number") else
            ("Sections " if meta.get("act") != "Constitution" else "Articles ") + ", ".join(map(str, section_numbers))
            if section_numbers else (f"PDF page {meta['page']}" if meta.get("page") else "Summary")
        )
        sources.append({"source": meta.get("doc_title", meta.get("act", "Source")),
                        "section": section, "snippet": doc.page_content[:800], "score": score,
                        "url": meta.get("source_url", ""),
                        "source_type": meta.get("content_type", "bare_act_pdf")})
    usable = [(item, source) for item, source in zip(candidates, sources) if source["score"] >= 0.30]
    if not usable:
        return {"answer": "I don't have enough evidence in my sources to answer this. The retrieved passages did not meet the prototype relevance floor.",
                "sources": sources}
    context = "\n\n".join(
        f"[Source {i + 1}: {source['source']}; {source['section']}]\n"
        f"{item['document'].page_content}"
        for i, (item, source) in enumerate(usable)
    )
    system = (
        "Answer the user's educational question using only the source excerpts below. "
        "If they do not support an answer, say you do not have enough evidence. "
        "Do not add legal claims from memory. Cite supporting source labels such as [Source 1]. "
        "These excerpts are historical IPC/CrPC/Constitution material; flag that limitation when relevant.\n\n"
        f"SOURCE EXCERPTS:\n{context}"
    )
    answer = generate_text(query, history=history, system=system, max_new_tokens=400)
    return {"answer": answer, "sources": sources}


def get_agent(use_rag: bool, tools_enabled: bool, top_k: int):
    from src.agent import LawBotAgent

    rag_handler = (lambda query, history=None: rag_answer(query, history, top_k)) if use_rag else None
    return LawBotAgent(rag_handler=rag_handler, direct_answer=generate_text,
                       project_dir=str(PROJECT_DIR), tools_enabled=tools_enabled,
                       rag_enabled=use_rag)


def render_citations(citations: list[dict[str, Any]]) -> None:
    with st.expander("Retrieved citations", expanded=False):
        if not citations:
            st.caption("No retrieved source citations for this answer.")
            return
        for i, citation in enumerate(citations, start=1):
            title = citation.get("source") or citation.get("title") or "Source"
            section = citation.get("section") or citation.get("citation") or citation.get("term") or ""
            st.markdown(f"**{i}. {title}** — {section}")
            if citation.get("url"):
                st.markdown(f"[Open source]({citation['url']})")
            if citation.get("score") is not None:
                st.caption(f"Similarity score: {citation['score']:.3f} · {citation.get('source_type', '')}")
            if citation.get("snippet"):
                st.caption(citation["snippet"])


def render_trace(trace: list[dict[str, Any]]) -> None:
    with st.expander("Tool trace", expanded=False):
        if not trace:
            st.caption("No tools or retrieval calls were used for this answer.")
        else:
            st.json(trace, expanded=False)


st.markdown("""
<style>
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stSidebar"],
[data-testid="stChatMessage"], [data-testid="stChatInput"] { color:#19263a !important; }
[data-testid="stMain"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * { color:#19263a !important; }
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stTextInput"] input { color:#19263a !important; background-color:#fff !important; }
[data-testid="stChatInput"] textarea::placeholder { opacity:.62; }
[data-testid="stSidebar"] .brand-icon { color:#fff !important; }
[data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 12% 0%,#eef0ff 0,transparent 34%),linear-gradient(180deg,#f7f8fc 0%,#f5f7fb 100%); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1040px; padding-top:2.1rem; padding-bottom:2.5rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#fff 0%,#f5f6fc 100%); border-right:1px solid #e8ebf3; }
.brand { display:flex;align-items:center;gap:12px;margin:0 0 2rem; }
.brand-icon { width:42px;height:42px;border-radius:14px;display:grid;place-items:center;background:linear-gradient(145deg,#7567ef,#5144c5);color:#fff;font-size:21px;box-shadow:0 8px 20px #6558d933; }
.brand-name { color:#19263a;font-size:20px;font-weight:750;letter-spacing:-.5px; }
.brand-sub { color:#78839a;font-size:12px;margin-top:1px; }
.hero { padding:12px 0 14px; }
.eyebrow { display:inline-block;color:#6256d5;background:#eeedff;border:1px solid #e2dfff;border-radius:999px;padding:6px 11px;font-size:11px;font-weight:700;letter-spacing:.08em; }
.hero h1 { color:#17243a;font-size:42px;line-height:1.12;letter-spacing:-1.7px;margin:15px 0 8px; }
.hero p { color:#718097;font-size:16px;margin:0; }
.status-row { display:flex;gap:8px;flex-wrap:wrap;margin-top:18px; }
.status-pill { background:#fff;border:1px solid #e5eaf2;border-radius:999px;padding:7px 11px;color:#5d6980;font-size:12px; }
.welcome-card { background:rgba(255,255,255,.83);border:1px solid #e8ecf4;border-radius:20px;padding:22px 24px;margin:17px 0 10px;box-shadow:0 12px 35px #27324a08; }
.welcome-title { color:#1b2940;font-size:18px;font-weight:700;margin-bottom:5px; }
.welcome-copy { color:#77849a;font-size:14px;line-height:1.5; }
[data-testid="stChatMessage"] { background:rgba(255,255,255,.84);border:1px solid #e8ecf4;border-radius:18px;padding:17px 19px;box-shadow:0 8px 24px #27324a06;margin-bottom:10px; }
[data-testid="stChatInput"] { border:1px solid #dfe5f0;border-radius:18px;box-shadow:0 10px 28px #26344e0b;background:white; }
[data-testid="stChatInput"] textarea { font-size:15px; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#647189; }
div.stButton>button { border-radius:13px;border:1px solid #e2e7f0;background:#fff;color:#344259;min-height:48px;text-align:left;padding:10px 14px;transition:all .15s ease; }
div.stButton>button:hover { border-color:#b8b1ff;color:#5144c5;background:#faf9ff; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="brand"><div class="brand-icon">⚖</div><div><div class="brand-name">LawBot</div><div class="brand-sub">Your clear, practical guide</div></div></div>
    """, unsafe_allow_html=True)
    st.markdown("#### Preferences")
    tools_enabled = st.checkbox("Use glossary and date tools", value=True)
    show_details = st.checkbox("Show technical answer details", value=False)
    if RAG_INDEX_READY:
        use_rag = st.checkbox("Use official source library", value=True)
        top_k = st.slider("Source passages", min_value=3, max_value=10, value=5) if use_rag else 5
    else:
        use_rag, top_k = False, 5
        st.caption("AI answers are generated locally. Verify legal details with current official sources.")
    st.divider()
    st.caption("Educational information only · Not legal advice")
    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("""
<div class="hero"><span class="eyebrow">INDIAN LAW · EVERYDAY QUESTIONS</span>
<h1>Understand the law,<br/>one question at a time.</h1>
<p>Ask a basic question or explore a complex topic. Get a clear explanation in plain language.</p>
<div class="status-row"><span class="status-pill">✦ &nbsp;Local AI assistant</span><span class="status-pill">◉ &nbsp;Available on this device</span></div></div>
""", unsafe_allow_html=True)
st.caption("AI answers can be mistaken. Verify legal details with current official sources.")

if not st.session_state.messages:
    st.markdown("<div class='welcome-card'><div class='welcome-title'>What would you like to understand?</div><div class='welcome-copy'>Try a quick question below, or type your own. I can explain legal ideas and answer general questions.</div></div>", unsafe_allow_html=True)
    suggestions = ["What is law? Explain simply.", "What is the difference between a rule and a law?",
                  "What does consideration mean in contract law?", "Explain Article 21 in simple terms."]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            st.session_state.pending_prompt = suggestion
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="⚖️" if message["role"] == "assistant" else "🙂"):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("citations"):
                render_citations(message.get("citations", []))
            if show_details:
                st.caption(f"Answer type: {message.get('grounding', 'unknown')}")
                render_trace(message.get("tool_trace", []))

prompt = st.chat_input("Ask anything — for example, “What is law?”")
if st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")
if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("Thinking…"):
            try:
                result = get_agent(use_rag, tools_enabled, top_k).run(prompt, history=history)
            except Exception:
                result = {"answer": "I hit a technical problem while preparing that answer. Please try again in a moment.",
                          "route": "error", "grounding": "service_unavailable", "citations": [], "tool_trace": []}
        st.markdown(result["answer"])
        if result.get("citations"):
            render_citations(result.get("citations", []))
        if show_details:
            st.caption(f"Answer type: {result.get('grounding', 'unknown')}")
            render_trace(result.get("tool_trace", []))
    st.session_state.messages.append({"role": "assistant", "content": result["answer"],
                                      "route": result.get("route"), "grounding": result.get("grounding"),
                                      "citations": result.get("citations", []),
                                      "tool_trace": result.get("tool_trace", [])})
