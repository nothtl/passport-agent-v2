"""Validation tools: consistency checking and score validation."""


def check_consistency(field_a: dict, field_b: dict) -> dict:
    """
    Flag contradictions between self-reported fields.

    Args:
        field_a: {"name": "...", "value": "..."}
        field_b: {"name": "...", "value": "..."}

    Returns: {"consistent": bool, "contradiction": str | None}
    """
    name_a = field_a.get("name", "").lower()
    name_b = field_b.get("name", "").lower()
    val_a = str(field_a.get("value", "")).lower()
    val_b = str(field_b.get("value", "")).lower()

    # English proficiency vs written response quality
    english_fields = ["english", "spoken", "comfortable"]
    if any(kw in name_a for kw in english_fields):
        low_english = any(w in val_a for w in ["not comfortable", "not very", "somewhat"])
        if low_english and len(val_b) > 200:
            # Long, detailed response despite low English comfort
            return {
                "consistent": False,
                "contradiction": (
                    f"Student self-rates English as '{val_a}' but wrote "
                    f"a {len(val_b)}-char response suggesting higher proficiency."
                ),
            }

    return {"consistent": True, "contradiction": None}


def validate_score(pillar: str, field_name: str, proposed_score: float,
                    evidence_summary: str = "") -> dict:
    """
    Verify a proposed score falls within plausible bounds.

    Args:
        pillar: "EC" | "GC" | "RFF" | "CR" | "CT" | "CI"
        field_name: The dimension being scored
        proposed_score: The proposed score value
        evidence_summary: Brief summary of evidence found

    Returns: {"valid": bool, "warning": str | None, "suggested_range": [min, max]}
    """
    warnings = []

    # Sanity checks
    if proposed_score < 0 or proposed_score > 100:
        return {
            "valid": False,
            "warning": f"Score {proposed_score} is outside valid range 0-100.",
            "suggested_range": [0, 100],
        }

    # Check if score is maximum but evidence is thin
    if proposed_score >= 95 and len(evidence_summary) < 20:
        warnings.append("Near-perfect score with minimal evidence. Verify before finalizing.")

    # Check if score is exactly 0 but student exists (might mean missing data)
    if proposed_score == 0:
        warnings.append("Score is 0. Confirm this is truly 'no evidence' and not a data loading issue.")

    return {
        "valid": True,
        "warning": "; ".join(warnings) if warnings else None,
        "suggested_range": [
            max(0, proposed_score - 10),
            min(100, proposed_score + 10),
        ],
    }
