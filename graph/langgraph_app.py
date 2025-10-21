from typing import Dict, Any
from langgraph.graph import StateGraph, END
from graph.guardrail_node import GuardrailNode
from agents.primary import run_primary
from agents.flight import run_flight
from agents.faq import run_faq
from agents.clarify import run_clarify
import logging

logger = logging.getLogger("agentic_chatbot.graph")

def guard_node(state: Dict[str, Any]) -> Dict[str, Any]:
    g = GuardrailNode()(state)
    logger.info(f"Guardrail check complete. blocked={g.get('should_block', False)}")
    if g.get('should_block'):
        state['blocked'] = True
        state['response'] = g['response']
    else:
        state['blocked'] = False
    return state

def primary_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Routing to primary node.")
    return run_primary(state)

def flight_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Routing to flight node.")
    return run_flight(state)

def faq_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Routing to faq node.")
    return run_faq(state)

def clarify_node(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Routing to clarify node.")
    return run_clarify(state)

def guard_cond(state: Dict[str, Any]) -> str:
    return 'blocked' if state.get('blocked') else 'ok'

def intent_router(state: Dict[str, Any]) -> str:
    i = (state.get('intent') or '').lower()
    if i == 'schedule_search': return 'flight'
    if i in ('policy_visa','policy_refund'): return 'faq'
    if i == 'clarify_missing_fields': return 'clarify'
    return 'clarify'  # off_topic or unknown → ask a targeted follow-up

def build_graph():
    g = StateGraph(dict)
    g.add_node('guard', guard_node)
    g.add_node('primary', primary_node)
    g.add_node('flight', flight_node)
    g.add_node('faq', faq_node)
    g.add_node('clarify', clarify_node)

    g.set_entry_point('guard')
    g.add_conditional_edges('guard', guard_cond, {'blocked': END, 'ok': 'primary'})
    g.add_conditional_edges('primary', intent_router, {'flight': 'flight', 'faq': 'faq', 'clarify': 'clarify'})
    g.add_edge('flight', END)
    g.add_edge('faq', END)
    g.add_edge('clarify', END)
    return g.compile()
