import logging
from typing import Dict, Any, List
from agents.base import render, extract_first_json_block
from model_registry.schemas import FlightAnswer
from tools.tools import openai_tools_for_flight, flight_dispatch
from graph.openai_client import openai_tool_loop

logger = logging.getLogger("agentic_chatbot.flight")

SYSTEM_PROMPT = """You are the FLIGHT AGENT.
- If you need data, you MUST call the tool 'flight_filter' (exact name).
- Construct a FlightCriteria JSON string when calling the tool.
- When done, RETURN ONLY a valid FlightAnswer JSON (no backticks, no extra text).
- In the FlightAnswer.summary, include a concise natural-language recap mentioning airline(s), layover(s), price, dates.
"""

USER_TMPL = open('model_registry/prompts/flight_agent.j2', encoding='utf-8').read()


def _format_itinerary(it: Dict[str, Any]) -> str:
    airline = it.get("airline", "Unknown Airline")
    price = it.get("price_usd", 0)
    refundable = it.get("refundable", False)
    segs: List[Dict[str, Any]] = it.get("segments", [])
    hops = []
    for s in segs:
        frm = s.get("from") or s.get("origin") or "?"
        to = s.get("to") or s.get("destination") or "?"
        hops.append(f"{frm}→{to}")
    route = " | ".join(hops) if hops else "(route unavailable)"
    
    dep = segs[0].get("departure_date") if segs else None
    arr = segs[-1].get("arrival_date") if segs else None
    date_str = f"{dep} → {arr}" if dep or arr else "dates N/A"
    
    layovers = []
    if len(segs) > 2:
        for s in segs[:-1]:
            layovers.append(s.get("to") or s.get("destination"))
    elif len(segs) == 2:
        layovers.append(segs[0].get("to") or segs[0].get("destination"))
    layover_str = f" | Layovers: {', '.join([x for x in layovers if x])}" if layovers else ""
    refund_str = "Refundable" if refundable else "Non-refundable"
    return f"{airline} — ${price} — {refund_str} — {date_str} — {route}{layover_str}"


def _format_response(data: Dict[str, Any], max_lines: int = 3) -> str:
    crit = data.get("criteria", {}) or {}
    origin = crit.get("origin") or "?"
    dest = crit.get("destination") or "?"
    month = crit.get("month_hint") or "requested period"
    alliance = crit.get("alliance") or "any"
    price_cap = crit.get("max_price_usd")

    header_bits = [f"{origin} → {dest}", month]
    if alliance and alliance != "any":
        header_bits.append(alliance)
    if price_cap:
        header_bits.append(f"≤ ${price_cap}")
    header = " | ".join([b for b in header_bits if b])

    itins = data.get("itineraries") or []
    if not itins:
        return f"No matching itineraries found for {header}."

    lines = [f"Here are your best {min(len(itins), max_lines)} option(s) for {header}:"]
    for it in itins[:max_lines]:
        lines.append(f"• {_format_itinerary(it)}")

    model_summary = data.get("summary")
    if model_summary and model_summary.strip():
        lines.append("")
        lines.append(model_summary.strip())
    return "\n".join(lines)


def run_flight(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Entering Flight Agent (Chat Completions tool-calling).")
    schema = FlightAnswer.model_json_schema()
    history = state['memory'].get_formatted() if state.get('memory') else ''

    user_prompt = render(
        USER_TMPL,
        conversation_history=history,
        user_input=state['query'],
        schema=schema,
    )

    tools = openai_tools_for_flight()
    dispatch = flight_dispatch()

    resp = openai_tool_loop(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tools=tools,
        dispatch=dispatch,
        max_rounds=4,
        temperature=0.2,
        max_output_tokens=700,
        finalizer_prompt="Return ONLY the final FlightAnswer JSON now. No backticks, no commentary.",
    )

    text = resp.choices[0].message.content or ""
    try:
        data = extract_first_json_block(text)
    except Exception:
        logger.exception("Failed to parse FlightAnswer JSON after tool loop.")
        state['response'] = "I couldn't produce the final flight JSON answer. Please try rephrasing."
        state['current_agent'] = 'flight_agent'
        return state

    pretty = _format_response(data, max_lines=3)

    state['response'] = pretty
    state['results'] = data
    state['current_agent'] = 'flight_agent'
    if state.get('memory'):
        state['memory'].add_ai(state['response'])
    logger.info(f"Flight Agent response composed ({len(data.get('itineraries') or [])} itineraries).")
    return state
