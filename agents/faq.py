import logging
from typing import Dict, Any
from agents.base import render, extract_first_json_block
from model_registry.schemas import PolicyAnswer
from tools.tools import openai_tools_for_faq, faq_dispatch
from graph.openai_client import openai_tool_loop

logger = logging.getLogger("agentic_chatbot.faq")

SYSTEM_PROMPT = """You are the FAQ/RAG AGENT.
- You MUST call the tool `rag_search` (exact name) before answering.
- Use ONLY retrieved chunks to compose the answer.
- If the tool returns no results or fails, respond with a JSON whose 'response' says you cannot answer due to missing evidence.
- When done, RETURN ONLY a valid PolicyAnswer JSON (no backticks, no extra text).
"""

USER_TMPL = open('model_registry/prompts/faq_agent.j2', encoding='utf-8').read()

def run_faq(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Entering FAQ Agent (Chat Completions tool-calling).")
    schema = PolicyAnswer.model_json_schema()
    history = state['memory'].get_formatted() if state.get('memory') else ''

    user_prompt = render(
        USER_TMPL,
        conversation_history=history,
        user_input=state['query'],
        schema=schema,
    )

    tools = openai_tools_for_faq()
    dispatch = faq_dispatch()

    resp = openai_tool_loop(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tools=tools,
        dispatch=dispatch,
        max_rounds=3,
        temperature=0.2,
        max_output_tokens=500,
        finalizer_prompt="Return ONLY the final PolicyAnswer JSON now. No backticks, no commentary.",
    )

    text = resp.choices[0].message.content or ""
    try:
        data = extract_first_json_block(text)
    except Exception:
        logger.exception("Failed to parse PolicyAnswer JSON after tool loop.")
        state['response'] = "I couldn't produce the final policy JSON answer. Please try rephrasing."
        state['current_agent'] = 'faq_agent'
        return state

    state['response'] = data.get('response', '')
    state['rag'] = data
    state['current_agent'] = 'faq_agent'
    if state.get('memory'):
        state['memory'].add_ai(state['response'])
    logger.info(f"FAQ Agent answer: {state['response']}")
    return state
