from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, Any, List, Optional
from langchain.tools import tool
import json
from rag_store import search as rag_search_impl
from helpers import load_flights, filter_flights


def _maybe_json(s: str) -> Optional[dict]:
    """Return dict if s is a JSON object string; else None."""
    if not isinstance(s, str):
        return None
    s2 = s.strip()
    if not s2 or not (s2.startswith("{") and s2.endswith("}")):
        return None
    try:
        return json.loads(s2)
    except Exception:
        return None

@tool("rag_search", return_direct=True)
def rag_search(input_text: str) -> str:
    """
    Retrieve top policy chunks for the question from local markdown files.
    INPUT: either plain question text OR a JSON string like {"question": "..."}.
    OUTPUT: JSON list of {title, path, chunk, score, id}.
    """
    q = input_text
    obj = _maybe_json(input_text)
    if isinstance(obj, dict) and "question" in obj:
        q = str(obj["question"])

    hits = rag_search_impl(q)
    return json.dumps(hits, ensure_ascii=False)


@tool("flight_filter", return_direct=True)
def flight_filter(input_text: str) -> str:
    """
    Filter the mock flight dataset.
    INPUT: either a raw FlightCriteria JSON string, or a JSON string like {"criteria_json": "{...}"}.
    OUTPUT: JSON list of itineraries.
    """
    # Accept both shapes:
    #   1) input_text == '{"origin":"Dubai", ...}'   (the criteria dict directly)
    #   2) input_text == '{"criteria_json":"{...}"}' (wrapper with a nested JSON string)
    criteria_dict: Dict[str, Any] = {}

    obj = _maybe_json(input_text)
    if isinstance(obj, dict):
        if "criteria_json" in obj and isinstance(obj["criteria_json"], str):
            try:
                criteria_dict = json.loads(obj["criteria_json"])
            except Exception:
                criteria_dict = {}
        else:
            criteria_dict = obj
    else:
        try:
            criteria_dict = json.loads(input_text)
        except Exception:
            criteria_dict = {}

    matches = filter_flights(load_flights(), criteria_dict)
    return json.dumps(matches, ensure_ascii=False)


def openai_tools_for_flight() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "flight_filter",
            "description": "Filter the mock flight dataset with FlightCriteria and return matching itineraries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria_json": {
                        "type": "string",
                        "description": "FlightCriteria JSON string: {origin, destination, month_hint, alliance, max_price_usd, non_stop_only, refundable_only}"
                    }
                },
                "required": ["criteria_json"]
            }
        }
    ]


def openai_tools_for_faq() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "rag_search",
            "description": "Search local policy/FAQ knowledge and return top chunks for the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's policy question."}
                },
                "required": ["question"]
            }
        }
    ]

def flight_dispatch() -> Dict[str, Any]:
    return {
        "flight_filter": flight_filter.invoke,
    }


def faq_dispatch() -> Dict[str, Any]:
    return {
        "rag_search": rag_search.invoke,
    }
