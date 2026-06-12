"""Pipeline event types for orchestrator → monitor communication."""

from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class EventType(str, Enum):
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    SCORE_PROPOSED = "score_proposed"
    SCORE_CHALLENGED = "score_challenged"
    SCORE_RESOLVED = "score_resolved"
    ANOMALY_DETECTED = "anomaly_detected"
    STUDENT_COMPLETED = "student_completed"
    BATCH_COMPLETED = "batch_completed"


class PipelineEvent(BaseModel):
    type: EventType
    student: str
    agent: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
