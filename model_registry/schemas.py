from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class PrimaryRoute(BaseModel):
    intent: str
    response: str = ""
    confidence: float = 0.7

class FlightCriteria(BaseModel):
    origin: Optional[str] = None
    destination: Optional[str] = None
    month_hint: Optional[str] = None
    alliance: Optional[str] = None
    max_price_usd: Optional[float] = None
    non_stop_only: Optional[bool] = None
    refundable_only: Optional[bool] = None
    notes: Optional[str] = ""

class FlightItinerary(BaseModel):
    airline: str
    alliance: Optional[str] = None
    segments: List[Dict[str, Any]] = []
    price_usd: float
    refundable: bool
    match_explanations: List[str] = []

class FlightAnswer(BaseModel):
    intent: str = "schedule_search"
    criteria: FlightCriteria
    itineraries: List[FlightItinerary] = []
    summary: str

class PolicyAnswer(BaseModel):
    intent: str = "policy_answer"
    response: str
    sources: List[Dict[str, str]] = []
    confidence: float = 0.6
