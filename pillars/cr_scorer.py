"""CR (Career Ready) pillar — mostly deterministic with LLM for C3/C4."""

from .formulas import _to_bool, _parse_hours


def score_cr(fields: dict, llm_call=None) -> dict:
    """
    Score the CR pillar. C1/C2 rule-based, C3/C4 via LLM.

    Args:
        fields: {canonical_name: value_or_None}
        llm_call: optional async callable(prompt_key, **kwargs) -> dict
    """
    g = fields.get

    # ── C1 — Pre-Program Exposure ──────────────────────────────
    ever_vol = _to_bool(g("FY1 - Ever Volunteered?"))
    hours    = g("FY1 - Hours Volunteered")
    had_int  = _to_bool(g("FY1 - Had Internship?"))
    had_job  = _to_bool(g("FY1 - Had/Have Job?"))

    h = _parse_hours(hours)
    if h is None and not ever_vol:  vol_score = 0
    elif h is None and ever_vol:    vol_score = 10
    elif h < 10:                    vol_score = 10
    elif h < 30:                    vol_score = 20
    elif h < 60:                    vol_score = 30
    else:                           vol_score = 40

    c1 = vol_score + (30 if had_int else 0) + (30 if had_job else 0)

    # ── C2 — Foundation Building ───────────────────────────────
    try: attended = float(g("Total Sessions Attended") or 0)
    except (TypeError, ValueError): attended = 0.0
    try: scheduled = float(g("Total Sessions Scheduled") or 0)
    except (TypeError, ValueError): scheduled = 0.0

    sess_score = (min(attended / scheduled, 1.0) * 60) if scheduled > 0 else 0.0
    connected = str(g("Connected Champions") or "")
    champ_count = len([x for x in connected.split(',') if x.strip()])
    champ_score = min(champ_count / 5, 1.0) * 40
    c2 = round(sess_score + champ_score, 1)

    # ── C3 — Skills Developed (LLM) ────────────────────────────
    cpc_text = g("CPC All Session Text") or ""
    c3 = 0
    c3_status = "missing"
    if cpc_text and str(cpc_text).strip() not in ('', 'nan', 'None'):
        if llm_call:
            result = llm_call("c3", cpc_session_text=str(cpc_text))
            c3 = max(0, min(100, int(result.get("c3_score", 30))))
            c3_status = "scored"
        else:
            c3 = 30  # default when LLM unavailable
            c3_status = "unscored"

    # ── C4 — Resume Confirmation ───────────────────────────────
    cpc_resume = g("CPC Resume Text") or ""
    c4 = 0
    c4_status = "missing"
    # Also check if student has a resume file
    if cpc_resume and str(cpc_resume).strip() not in ('', 'nan', 'None'):
        if llm_call:
            result = llm_call("c4", cpc_resume_text=str(cpc_resume))
            built = str(result.get("resume_built", "")).lower() in ("true", "1", "yes")
            c4 = 100 if built else 0
            c4_status = "scored"
        else:
            c4_status = "unscored"

    # ── Final score ────────────────────────────────────────────
    available = [c1, c2]
    if c3_status == "scored": available.append(c3)
    if c4_status == "scored": available.append(c4)
    cr_score = round(sum(available) / len(available), 1)

    return {
        "score": cr_score,
        "sub_scores": {"C1": c1, "C2": c2, "C3": c3, "C4": c4},
    }


# ── LLM prompts (from old cr_scorer.py) ──────────────────────

C3_PROMPT = """You are evaluating a student intern's professional skill development based on observations written by their Career Pathways Champion (a working professional mentor) across multiple mentorship sessions.

NACE CAREER READINESS FRAMEWORK (8 competencies):
1. Career and Self-Development  2. Communication  3. Critical Thinking
4. Equity and Inclusion  5. Leadership  6. Professionalism
7. Teamwork  8. Technology

CHAMPION OBSERVATIONS: {cpc_session_text}

SCORING RUBRIC:
- 80-100: Champion logged rich, specific skill development across 4+ NACE competencies.
- 60-79: Champion logged solid skill development across 2-3 NACE competencies.
- 40-59: Champion logged some skill coverage but limited specificity or depth.
- 20-39: Champion logged minimal skill content.
- 0-19: No meaningful skill content.

Respond in JSON: {{"c3_score": <int 0-100>, "nace_competencies_addressed": [...], "score_rationale": "<one sentence>", "dominant_skills": "<2-3 specific skills>"}}"""

C4_PROMPT = """You are reviewing notes written by a Career Pathways Champion after each mentorship session with a high school intern.

The notes are responses to: "We added the following to the Intern resume:"

CHAMPION'S RESUME NOTES: {cpc_resume_text}

Answer TRUE if the Champion genuinely built or improved the intern's resume (added sections, content, or improvements). Answer FALSE if all entries say "nothing", "N/A", "-", "not yet", etc.

Respond in JSON: {{"resume_built": <true or false>, "confidence": <"high", "medium", or "low">, "reason": "<one sentence>", "key_evidence": "<specific text>"}}"""
