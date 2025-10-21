import os, json, re
from datetime import datetime
from typing import List, Dict, Any

def load_flights(path: str = None) -> List[Dict[str, Any]]:
    candidates = []
    if path: candidates.append(path)
    candidates += [
        os.path.join("data", "flights.json"),
        os.path.join("data", "flight_listings.json"),
        os.path.join("data", "mock_flights.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                return data
            except Exception:
                pass
    return []


def _month_name_to_num(month: str) -> int | None:
    if not month: return None
    try:
        dt = datetime.strptime(month.strip()[:3].title(), "%b")
        return dt.month
    except Exception:
        return None


def _parse_date_month(date_str: str) -> int | None:
    try:
        return datetime.fromisoformat(date_str).month
    except Exception:
        return None


def _first_nonempty(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _ci_contains(hay: str | None, needle: str | None) -> bool:
    if not hay or not needle: return False
    return needle.lower() in hay.lower()


def _pass_price(item: Dict[str, Any], max_price: Any | None) -> bool:
    if max_price in (None, ""): return True
    try:
        price = _first_nonempty(item, ["price_usd", "price"])
        if price is None:
            return False
        return float(price) <= float(max_price)
    except Exception:
        return False


def _pass_alliance(item: Dict[str, Any], alliance: str | None) -> bool:
    if not alliance: 
        return True

    ali = _first_nonempty(item, ["alliance"])
    if ali and _ci_contains(ali, alliance):
        return True
    return False


def _pass_route(item: Dict[str, Any], origin: str | None, destination: str | None) -> bool:
    src = _first_nonempty(item, ["from", "origin", "source", "from_city"])
    dst = _first_nonempty(item, ["to", "destination", "dest", "to_city"])
    ok_src = True if not origin else _ci_contains(src or "", origin)
    ok_dst = True if not destination else _ci_contains(dst or "", destination)
    return ok_src and ok_dst


def _pass_month(item: Dict[str, Any], month_hint: str | None) -> bool:
    if not month_hint: return True
    m = _month_name_to_num(month_hint)
    if not m: return True  
    d_m = _parse_date_month(_first_nonempty(item, ["departure_date", "depart_date", "outbound_date"]) or "")
    r_m = _parse_date_month(_first_nonempty(item, ["return_date", "inbound_date"]) or "")
    return (d_m == m) or (r_m == m)


def filter_flights(flights: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
    origin = criteria.get("origin")
    destination = criteria.get("destination")
    month_hint = criteria.get("month_hint")
    alliance = criteria.get("alliance")
    max_price = criteria.get("max_price_usd")

    non_stop_only = criteria.get("non_stop_only")
    refundable_only = criteria.get("refundable_only")

    out = []
    for it in flights:
        if not _pass_route(it, origin, destination):            
            continue
        if not _pass_month(it, month_hint):                    
            continue
        if not _pass_price(it, max_price):                      
            continue
        if not _pass_alliance(it, alliance):                    
            continue

        if non_stop_only is True:
            lays = _first_nonempty(it, ["layovers", "stops"]) or []
            if isinstance(lays, list) and len(lays) > 0:
                continue

        if refundable_only is True:
            ref = _first_nonempty(it, ["refundable", "is_refundable"])
            if ref is not True:
                continue

        out.append(it)
    return out
