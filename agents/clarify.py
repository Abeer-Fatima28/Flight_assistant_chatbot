from typing import Dict, Any
from agents.base import render
from graph.openai_client import openai_generate
import logging

logger = logging.getLogger("agentic_chatbot.clarify")
tmpl = open('model_registry/prompts/clarify_agent.j2', encoding='utf-8').read()

def run_clarify(state: Dict[str, Any]) -> Dict[str, Any]:
    history = state['memory'].get_formatted() if state.get('memory') else ''
    prompt = render(
        tmpl,
        conversation_history=history,
        user_input=state['query'],
    )
    logger.info("Clarify Agent generating contextual follow-up.")
    try:
        question = openai_generate(prompt, max_output_tokens=180, temperature=0.3).strip()
    except Exception as e:
        logger.exception("Clarify Agent failed to generate; providing minimal fallback question.")
        question = "Could you share any missing details so I can proceed?"

    state['response'] = question
    state['current_agent'] = 'clarify_agent'
    if state.get('memory'):
        state['memory'].add_ai(question)
    logger.info(f"Clarify question: {question}")
    return state
