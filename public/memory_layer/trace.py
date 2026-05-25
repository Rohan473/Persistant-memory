"""
Retrieval Tracing - Log all retrieval events for future training/evaluation.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from .config import config

@dataclass
class RetrievalEvent:
    timestamp: str
    query: str
    tool_used: str
    retrieved_ids: List[str]
    retrieved_names: List[str]
    scores: List[float]
    token_budget: int
    tokens_used: int
    compression_ratio: float
    latency_ms: float
    filters: Dict[str, Any]
    user_feedback: Optional[str] = None

class RetrievalLogger:
    """Logger for retrieval events - builds future training data."""

    def __init__(self, log_path: Optional[Path] = None):
        if log_path is None:
            log_path = Path(__file__).parent / "retrieval_log.jsonl"
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: RetrievalEvent) -> None:
        """Append event to log file."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def get_recent(self, n: int = 100) -> List[RetrievalEvent]:
        """Get recent retrieval events."""
        events = []
        if not self.log_path.exists():
            return events
        with open(self.log_path, "r") as f:
            for line in f:
                if line.strip():
                    events.append(RetrievalEvent(**json.loads(line)))
        return events[-n:]

    def get_stats(self) -> Dict:
        """Get aggregate statistics."""
        events = self.get_recent(1000)
        if not events:
            return {"total_events": 0}

        return {
            "total_events": len(events),
            "avg_latency_ms": sum(e.latency_ms for e in events) / len(events),
            "avg_compression": sum(e.compression_ratio for e in events) / len(events),
            "tool_usage": self._count_tool_usage(events),
            "unique_queries": len(set(e.query for e in events)),
        }

    def _count_tool_usage(self, events: List[RetrievalEvent]) -> Dict[str, int]:
        counts = {}
        for e in events:
            counts[e.tool_used] = counts.get(e.tool_used, 0) + 1
        return counts

# Global logger instance
logger = RetrievalLogger()

def trace_retrieval(
    query: str,
    tool_used: str,
    results: List[Dict],
    token_budget: int,
    tokens_used: int,
    filters: Dict = None,
    user_feedback: Optional[str] = None
) -> None:
    """Log a retrieval event."""
    start_time = getattr(trace_retrieval, "_start", time.time())

    event = RetrievalEvent(
        timestamp=datetime.now().isoformat(),
        query=query,
        tool_used=tool_used,
        retrieved_ids=[r.get("id", "") for r in results],
        retrieved_names=[r.get("name", "") for r in results],
        scores=[round(r.get("score", 0), 3) for r in results],
        token_budget=token_budget,
        tokens_used=tokens_used,
        compression_ratio=round(tokens_used / token_budget, 2) if token_budget > 0 else 0,
        latency_ms=round((time.time() - start_time) * 1000, 1),
        filters=filters or {},
        user_feedback=user_feedback
    )

    logger.log(event)

def start_trace() -> float:
    """Mark start of retrieval for latency tracking."""
    return time.time()

def update_latency(start_time: float, event: RetrievalEvent) -> float:
    """Update latency in event after completion."""
    event.latency_ms = round((time.time() - start_time) * 1000, 1)
    return event.latency_ms