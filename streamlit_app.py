from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False) or load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

import os
import json
from datetime import datetime
import streamlit as st
from graph.langgraph_app import build_graph

try:
    from memory.memory import MemoryManager
except Exception:
    class MemoryManager:
        def __init__(self, k: int = 8):
            self.k = k
            self._history = []

        def add_user(self, msg: str):
            self._history.append(("user", msg))
            self._history = self._history[-2*self.k:]

        def add_ai(self, msg: str):
            self._history.append(("ai", msg))
            self._history = self._history[-2*self.k:]

        def get_formatted(self) -> str:
            lines = []
            for r, c in self._history[-2*self.k:]:
                prefix = "User:" if r == "user" else "Assistant:"
                lines.append(f"{prefix} {c}")
            return "\n".join(lines)

st.set_page_config(page_title="Agentic Travel Assistant", page_icon=" ", layout="wide")
st.title(" Agentic Travel Assistant")
st.caption("Multi-agent, tool-calling (LangGraph + LangChain tools) with sliding memory & RAG.")

with st.sidebar:
    st.header("Configuration")

    default_k = int(os.getenv("MEMORY_K", "8"))
    mem_k = st.number_input("Sliding memory (last K turns)", min_value=2, max_value=20, value=default_k)
    st.session_state.setdefault("mem_k", mem_k)
    st.session_state["mem_k"] = mem_k

    debug = st.checkbox("Debug mode", value=st.session_state.get("debug", False))
    st.session_state["debug"] = debug

    st.divider()
    st.subheader("Quick tests")
    tests = {
        "Flight search": "Find me a round trip from Dubai to Tokyo in August under $1000, Star Alliance",
        "Visa policy": "Do UAE passport holders need a visa for Japan?",
        "Refund policy": "Can I cancel a refundable ticket 48 hours before departure?",
    }
    for label, text in tests.items():
        if st.button(f" {label}"):
            st.session_state.setdefault("pending_prompt", text)
            st.rerun()

    st.divider()
    if st.button("Clear conversation"):
        for k in ("messages", "memory", "graph", "last_state"):
            if k in st.session_state:
                del st.session_state[k]
        st.success("Cleared. Ready for a fresh start.")
        st.rerun()


if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager(k=st.session_state["mem_k"])

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about flights or travel policies."}
    ]

def call_graph(query: str):
    """
    Invoke your LangGraph with the current query + shared memory.
    Your graph nodes set fields like:
      state['response']   -> string to show
      state['current_agent'] -> e.g. 'flight_agent' or 'faq_agent'
      state['results']    -> flight JSON (if any)
      state['rag']        -> policy JSON (if any)
    """
    st.session_state.memory.add_user(query)
    state_in = {
        "query": query,
        "memory": st.session_state.memory,
    }
    out_state = st.session_state.graph.invoke(state_in)
    if out_state.get("response"):
        st.session_state.memory.add_ai(out_state["response"])
    return out_state


def render_debug_panels(state: dict):
    with st.expander("Debug / State"):
        st.json({
            "current_agent": state.get("current_agent"),
            "keys": sorted(list(state.keys())),
        })

    if state.get("rag"):
        with st.expander("RAG Evidence"):
            try:
                st.json(state["rag"])
            except Exception:
                st.write(state["rag"])

    if state.get("results"):
        with st.expander("Flight Results (raw)"):
            st.json(state["results"])


def render_assistant_message(text: str, state: dict):
    with st.chat_message("assistant"):
        st.markdown(text)
        if st.session_state.get("debug"):
            render_debug_panels(state)


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prefill = st.session_state.pop("pending_prompt", None)

user_text = st.chat_input("Type your message…", key="chat_input", disabled=False)
if prefill and not user_text:
    user_text = prefill

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                state = call_graph(user_text)
                st.session_state["last_state"] = state
                reply = state.get("response", "(no response)")
            except Exception as e:
                reply = f"Error: {e}"
                state = {}

        st.markdown(reply)

        if st.session_state.get("debug") and state:
            render_debug_panels(state)

    st.session_state.messages.append({"role": "assistant", "content": reply})
