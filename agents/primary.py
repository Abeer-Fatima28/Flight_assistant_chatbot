from typing import Dict, Any
from model_registry.schemas import PrimaryRoute
from agents.base import render, extract_first_json_block
from graph.openai_client import openai_generate
import logging

logger = logging.getLogger("agentic_chatbot.primary")

tmpl = open('model_registry/prompts/primary_router.j2', encoding='utf-8').read()

def run_primary(state: Dict[str, Any]) -> Dict[str, Any]:
    schema = PrimaryRoute.model_json_schema()
    history = state['memory'].get_formatted() if state.get('memory') else ''
    prompt = render(tmpl, schema=schema, query=state['query'], conversation_history=history)

    logger.info("Primary routing started (LLM-only).")
    try:
        out = openai_generate(prompt, max_output_tokens=250, temperature=0.2)
        data = extract_first_json_block(out)
        logger.info(f"Primary LLM intent: {data.get('intent')}")
    except Exception as e:
        # Still LLM-first; if classification fails (rare), we conservatively ask to clarify
        logger.exception("Primary LLM classification failed; routing to clarify.")
        data = {"intent": "clarify_missing_fields", "response": "Let’s clarify a couple of details."}

    pr = PrimaryRoute(
        intent=data.get("intent", "clarify_missing_fields"),
        response=data.get("response", "")
    ).model_dump()

    state['intent'] = pr['intent']
    state['primary'] = pr
    state['response'] = pr.get('response', '')
    if state.get('memory') and state['response']:
        state['memory'].add_ai(state['response'])

    logger.info(f"Primary routing decided: {state['intent']}")
    return state
