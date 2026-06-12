"""Rubric database and reference example lookup."""

# ── GC (Global Citizen) Rubrics ──────────────────────────────────────────────

GC_RUBRICS = {
    "has_volunteer": {
        "dimension_name": "Volunteer Activity Detection",
        "description": "Did the student engage in any volunteering, community service, nonprofit work, food pantry, shelter, faith community service, mutual aid, informal community helping, or unpaid service?",
        "bands": [
            {"score_range": "true", "label": "Evidence found", "criteria": "At least one documented instance of volunteer, community service, or unpaid civic activity in any section of resume or LinkedIn.", "examples": ["Volunteer at Wildlife Conservation Society", "Community outreach program participant", "Church choir member"]},
            {"score_range": "false", "label": "No evidence", "criteria": "No volunteer, community service, or unpaid civic activity found in any document.", "examples": ["Only paid employment listed", "No activities section"]},
        ],
        "evidence_required": "single_signal",
    },
    "volunteer_hours_estimate": {
        "dimension_name": "Volunteer Hours Estimate",
        "description": "Estimate the commitment level of volunteering activity.",
        "bands": [
            {"score_range": "30+", "label": "Extensive", "criteria": "Multiple ongoing volunteer roles spanning years, or a full-time service commitment."},
            {"score_range": "20-30", "label": "Substantial", "criteria": "Regular weekly commitment across multiple months or intensive short-term program."},
            {"score_range": "10-20", "label": "Moderate", "criteria": "Regular volunteering over several months or an intensive short-term project."},
            {"score_range": "1-10", "label": "Limited", "criteria": "One-time or occasional volunteering, or brief community service."},
            {"score_range": "0", "label": "None", "criteria": "No volunteer activity detected."},
        ],
        "evidence_required": "single_signal",
    },
    "community_roles_count": {
        "dimension_name": "Community Roles Count",
        "description": "Count distinct roles that serve the community, social causes, underserved populations, or cultural engagement.",
        "bands": [
            {"score_range": "6-7", "label": "Extensive community engagement", "criteria": "6-7 distinct community-facing roles across multiple organizations and causes."},
            {"score_range": "4-5", "label": "Strong community engagement", "criteria": "4-5 distinct community-facing roles."},
            {"score_range": "2-3", "label": "Moderate community engagement", "criteria": "2-3 distinct community-facing roles."},
            {"score_range": "1", "label": "Limited community engagement", "criteria": "1 community-facing role."},
            {"score_range": "0", "label": "No community engagement", "criteria": "No community-facing roles detected."},
        ],
        "evidence_required": "multiple_signals",
    },
    "network_estimate": {
        "dimension_name": "Professional Network Estimate",
        "description": "Based on internships, professional jobs, and documented professional interactions: how many professionals in their career interest area does this student likely know?",
        "bands": [
            {"score_range": "7-10", "label": "Extensive network", "criteria": "Multiple internships/jobs across different organizations with described professional relationships."},
            {"score_range": "4-6", "label": "Moderate network", "criteria": "1-2 professional roles with described colleague/mentor interactions."},
            {"score_range": "1-3", "label": "Limited network", "criteria": "One professional role or limited documented professional contacts."},
            {"score_range": "0", "label": "No network evidence", "criteria": "No professional roles or network described."},
        ],
        "evidence_required": "single_signal",
    },
    "pre_empathy": {
        "dimension_name": "Pre-Program Empathy",
        "description": "General orientation toward others' feelings, shown through mentoring background, support roles, or family/community service references.",
        "bands": [
            {"score_range": "6-7", "label": "Sustained empathy theme", "criteria": "Central theme across student's entire documented history. Multiple roles demonstrating care for others."},
            {"score_range": "4-5", "label": "Consistent empathy signals", "criteria": "Multiple direct examples from one or more roles/contexts."},
            {"score_range": "2-3", "label": "Some empathy signals", "criteria": "One clear or indirect signal of empathy in documented roles."},
            {"score_range": "1", "label": "No empathy evidence", "criteria": "No evidence of empathy-related activities or roles."},
        ],
        "evidence_required": "single_signal",
    },
    "has_campus_role": {
        "dimension_name": "Campus Role Detection",
        "description": "Has the student held any campus-based role: student leadership, peer tutoring, academic mentoring, student government, campus ambassador, residence advisor, or similar?",
        "bands": [
            {"score_range": "true", "label": "Campus role found", "criteria": "At least one campus-based leadership or support role documented."},
            {"score_range": "false", "label": "No campus role", "criteria": "No campus-based role found."},
        ],
        "evidence_required": "single_signal",
    },
    "has_speakhire": {
        "dimension_name": "SPEAKHIRE Affiliation",
        "description": "Does any LinkedIn company name or job description mention SPEAKHIRE or Speak Hire?",
        "bands": [
            {"score_range": "true", "label": "SPEAKHIRE mentioned", "criteria": "SPEAKHIRE or Speak Hire found in LinkedIn experience."},
            {"score_range": "false", "label": "Not mentioned", "criteria": "No SPEAKHIRE reference found."},
        ],
        "evidence_required": "single_signal",
    },
}

# Generic rubric for dimensions not explicitly defined
_GENERIC_1_7_RUBRIC = {
    "dimension_name": "Generic 1-7 Scale",
    "description": "1 = no evidence at all, 7 = sustained central theme across entire documented history.",
    "bands": [
        {"score_range": "7", "label": "Sustained central theme", "criteria": "Sustained, central theme across the student's entire documented history."},
        {"score_range": "6", "label": "Consistent pattern", "criteria": "Consistent pattern across multiple roles or contexts."},
        {"score_range": "5", "label": "Multiple direct examples", "criteria": "Multiple direct examples from one role or context."},
        {"score_range": "4", "label": "One direct behavioral example", "criteria": "One direct behavioral example described in the docs."},
        {"score_range": "3", "label": "One clear indirect signal", "criteria": "One clear indirect signal."},
        {"score_range": "2", "label": "One vague indirect signal", "criteria": "One vague indirect signal."},
        {"score_range": "1", "label": "No evidence", "criteria": "No evidence at all."},
    ],
    "evidence_required": "single_signal",
}

ALL_GC_DIMENSIONS = [
    "pre_empathy", "pre_humble", "pre_listen", "pre_include",
    "pre_conflict", "pre_lead_auth", "listen", "conflict",
    "include", "reflect", "pre_community_connected",
    "has_volunteer", "volunteer_hours_estimate", "community_roles_count",
    "has_speakhire", "has_campus_role", "network_estimate",
]


def get_rubric(pillar: str, dimension: str) -> dict:
    """
    Return the exact scoring rubric for a dimension.
    ALWAYS call this before scoring — never guess what a score band means.

    Args:
        pillar: "EC" | "GC" | "RFF" | "CR" | "CT" | "CI"
        dimension: e.g., "has_volunteer", "community_roles_count", "pre_empathy"
    """
    if pillar == "GC":
        rubric = GC_RUBRICS.get(dimension)
        if rubric:
            return rubric

    # Fallback: generic rubric
    return dict(_GENERIC_1_7_RUBRIC, dimension_name=dimension)


def lookup_reference_examples(pillar: str, field_name: str,
                               score_range: str = "mid") -> dict:
    """
    Retrieve pre-scored reference examples from a calibrated bank.

    Args:
        pillar: "EC" | "GC" | "RFF" | "CR"
        field_name: The field being scored
        score_range: "low" | "mid" | "high"
    """
    # These would be real calibrated examples in production.
    # For now, return guidance based on the score range.
    examples = {
        "low": [
            {"text": "N/A or blank response", "score": 0.0, "why": "No evidence provided."},
            {"text": "I want to do well", "score": 0.2, "why": "Generic aspiration with no specifics."},
        ],
        "mid": [
            {"text": "I will improve my average to 85", "score": 0.55, "why": "Specific + measurable, partial SMART elements."},
            {"text": "Doctor or nursing", "score": 0.70, "why": "Named career area with some specificity."},
        ],
        "high": [
            {"text": "My goal is to improve my grade by 10% by studying 2 hours daily for the next 3 months", "score": 0.85, "why": "Specific, measurable, time-bound with action plan."},
            {"text": "Video Game Developer — I want to learn C++ and build my first game prototype by December", "score": 0.90, "why": "Highly specific role with clear learning path and timeline."},
        ],
    }

    return {
        "pillar": pillar,
        "field_name": field_name,
        "score_range": score_range,
        "examples": examples.get(score_range, examples["mid"]),
    }
