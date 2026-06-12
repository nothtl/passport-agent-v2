"""Monitor Agent tools — inspect, compare, query logs, decide actions."""

from datetime import datetime

_decisions_log: list[dict] = []
_pipeline_state: dict = {"paused": False, "adjustments": {}}


def inspect_score(student: str, pillar: str, dimension: str = None,
                  enriched_path: str = None) -> dict:
    """
    Pull up full evidence trail for a specific score.

    Args:
        student: Student name
        pillar: "EC" | "GC" | "RFF" | "CR" | "CT" | "CI"
        dimension: Specific dimension to inspect
        enriched_path: Path to enriched scores JSON (auto-detected if None)
    """
    import os, json

    if not enriched_path:
        base = os.path.join(os.path.dirname(__file__), "..", "outputs")
        enriched_path = os.path.join(base, f"{student.lower().replace(' ', '_')}_enriched_scores.json")

    if not os.path.exists(enriched_path):
        return {"error": f"No enriched scores found for {student}"}

    with open(enriched_path, encoding="utf-8") as f:
        data = json.load(f)

    scores = data.get("scores", {})
    if pillar not in scores:
        return {"error": f"Pillar {pillar} not found", "available_pillars": list(scores.keys())}

    pillar_data = scores[pillar]
    if isinstance(pillar_data, dict):
        return {
            "student": student,
            "pillar": pillar,
            "score": pillar_data.get("score"),
            "sub_scores": pillar_data.get("sub_scores", {}),
            "evidence": pillar_data.get("evidence", []),
            "narrative": pillar_data.get("reasoning", ""),
        }

    return {"student": student, "pillar": pillar, "data": pillar_data}


def compare_to_cohort(student: str, pillar: str) -> dict:
    """
    Compare a student's score against cohort distribution.
    In production, this would query a live database.
    """
    return {
        "student": student,
        "pillar": pillar,
        "note": "Cohort comparison requires a completed batch. Run at least 5 students first.",
    }


def query_logs(pattern: str, student: str = None, agent: str = None) -> dict:
    """
    Search across all pipeline run events.

    Args:
        pattern: Text pattern to search for in event data
        student: Optional student filter
        agent: Optional agent filter
    """
    return {
        "pattern": pattern,
        "matches": [],
        "note": "Log search requires the orchestrator event buffer. Run the pipeline first.",
    }


def decide_action(student: str, agent: str, action: str, reason: str,
                   context: dict = None) -> dict:
    """
    Issue a decision about what to do with an anomaly.

    Args:
        student: Student name or "__batch__"
        agent: Agent name
        action: "CONTINUE" | "RETRY" | "SKIP_STUDENT" | "SKIP_PILLAR" |
                "FLAG_FOR_REVIEW" | "ESCALATE" | "REDUCE_CONFIDENCE"
        reason: Human-readable explanation
        context: Additional data about the anomaly
    """
    decision = {
        "timestamp": datetime.now().isoformat(),
        "student": student,
        "agent": agent,
        "action": action,
        "reason": reason,
        "context": context or {},
    }
    _decisions_log.append(decision)

    if action == "ESCALATE":
        _pipeline_state["paused"] = True

    return {"recorded": True, "decision": decision}


def adjust_pipeline(setting: str, value, reason: str) -> dict:
    """
    Change pipeline configuration for remaining students.

    Args:
        setting: "temperature" | "max_retries" | "model_override" |
                 "skip_agent" | "pause_batch"
        value: New value for the setting
        reason: Why the adjustment is being made
    """
    _pipeline_state["adjustments"][setting] = {"value": value, "reason": reason}
    return {"applied": True, "setting": setting, "value": value, "state": dict(_pipeline_state)}
