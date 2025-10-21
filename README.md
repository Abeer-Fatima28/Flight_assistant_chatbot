# Agentic Travel Assistant — Complete README

## Overview

The **Agentic Travel Assistant** is a multi-agent travel chatbot built using **LangGraph**, **LangChain Tools**, and **FAISS RAG**, powered by **OpenAI (GPT-4o-mini)**.  
It supports **tool-calling**, **prompt-based routing**, **state persistence**, and both **terminal** and **Streamlit UI** interfaces.

The system can:
- Answer **flight queries** (e.g., "Find me a round trip from Dubai to Tokyo in August under $1000, Star Alliance").
- Respond to **FAQ and policy questions** using **RAG** (Retrieval-Augmented Generation) from local `.md` documents.
- Maintain conversational **memory** for contextual understanding.
- Run in **debug mode** for transparent insight into system behavior.

---

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/your-repo/agentic-travel-assistant.git
cd agentic-travel-assistant

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Set the OPENAI_API_KEY in the .env file and/or set in the environment for streamlit UI to run.

---

## Running the Assistant

### **Option 1: Run in Terminal (CLI Mode)**

```bash
python main.py
```
You'll see:
```
LLM Agents (Multi-Prompt ReAct + Sliding Memory, GPT-4o-mini) ready. Type 'exit' to quit.
You: Find me a round trip from Dubai to Tokyo in August under $1000, Star Alliance
Bot: Found 1 itinerary matching your criteria.
```

This mode is ideal for backend testing, debugging, or headless environments.

---

### **Option 2: Run via Streamlit UI**

```bash
streamlit run streamlit_app.py
```

Then open:
```
http://localhost:8501
```

#### Streamlit Features
- **Chat-based interface** with message history.
- **Sidebar configuration** for user ID, debug toggle, and quick test prompts.
- **Debug mode** prints raw payloads, timestamps, and state data in real time.
- Supports **predefined test cases** mentioned in the doc (and for the only dataset example we currently have)

---

## Logic & Architecture Overview

The project is built using an **agentic multi-node graph** orchestrated via **LangGraph**. Each agent or node specializes in a domain (flights, FAQs, etc.), and messages flow dynamically based on LLM-inferred intent.

### Core Components
| Component | Description |
|------------|-------------|
| **LangGraph Graph** | Defines nodes (guardrail, primary router, flight agent, FAQ agent) and routing logic. |
| **LangChain Tools** | Provides structured tool interfaces (`@tool`) for flight filtering and RAG search. |
| **FAISS Vector Store** | Stores embeddings from markdown docs (FAQ, visa, refund policy). |
| **Memory Manager** | Sliding window memory maintains conversation context. |
| **OpenAI GPT-4o-mini** | Handles reasoning, intent routing, and natural-language synthesis. |
| **Streamlit UI** | Interactive chat frontend with debug and test scenario modes. |

###  Routing Logic (LangGraph)
1. **Guardrail Node** → Filters unsafe or irrelevant input.  
2. **Primary Router** → Classifies intent (`schedule_search`, `policy_visa`, `policy_refund`, etc.).  
3. **Flight Agent** → Uses `flight_filter` tool with JSON schema.  
4. **FAQ Agent** → Uses `rag_search` tool to ground answers in Markdown docs.  
5. **Response Composer** → Returns user-friendly summaries.

###  Internal States
- **State persistence** allows dynamic context sharing across nodes.
- Every agent reads/writes to a central **state dictionary** containing user query, route, results, and history.
- **Sliding memory** keeps up to *k* messages using LangChain’s `ConversationBufferWindowMemory`.

###  Native Tool-Calling & Jinja Prompting
- Tools are natively registered via LangChain’s `@tool` decorator and exposed to OpenAI’s **function-calling schema**.
- **Prompts are modularized** and version-controlled via **Jinja templates**, enabling fine-tuning and easy prompt evolution.

---

## Creative & Technical Highlights

### What I Covered
- Full **multi-agent orchestration** with **LangGraph**.
- **Tool-based LLM reasoning** — no hardcoded fallbacks or fixed prompts.
- **State persistence & routing logic** for contextual flow.
- **RAG integration** with local markdown documents.
- **Streamlit UI** with *debug mode*, *predefined test cases*, and *state tracking*.
- Support for both **terminal and web interfaces**.

###  What’s Creatively Implemented
- **Dual execution modes:** seamless switching between CLI and Streamlit UI.
- **Debug mode:** exposes raw responses, timestamps, and model calls for transparency.
- **Native tool invocation:** the LLM autonomously decides when to call a tool.
- **Prompt modularity:** via Jinja templates, supporting prompt versioning for experiments.
- **Dynamic RAG rebuild script:** auto-generates FAISS index with chunking and metadata.
- **Case-insensitive flight filtering:** flexible fuzzy matching logic with month parsing and layover handling.

---

##  Building the Vector Store (RAG)

```bash
python scripts/rebuild_rag.py
```
- Creates FAISS index at `data/vectorstore/`
- Embeds `.md` files using SentenceTransformers (`all-MiniLM-L6-v2`).
- Persists metadata and chunk mapping for FAQ retrieval.

---

##  Conclusion

The **Agentic Travel Assistant** demonstrates how to combine **multi-agent orchestration**, **retrieval-augmented reasoning**, and **tool-based LLM workflows** into a cohesive architecture.  
It embodies principles of **stateful AI**, **structured tool invocation**, and **human-like contextual understanding** — while remaining modular and extensible.

---
