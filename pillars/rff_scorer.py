"""RFF (Reflective & Future-Focused) pillar — hybrid formula + LLM text scoring."""

from .formulas import _norm17, _norm_binary, _norm010, _norm_hope_tags


# ── LLM scoring prompts (from old agent2/tools/rff_scorer.py) ─────

RFF_PROMPTS = {
    "adjectives": """You are evaluating a high school student self-reflection response.
The student was asked: What are three adjectives that describe the person you are and why?

Evaluate SUBSTANCE and SELF-AWARENESS, not grammar or writing style.

STUDENT RESPONSE: {text}

SCORING RUBRIC (0.0 to 1.0):
- 0.0-0.1: Blank, single vague word, or clearly irrelevant
- 0.1-0.3: One or two generic adjectives with no reasoning
- 0.3-0.5: Two or three adjectives, mostly generic, little reasoning
- 0.5-0.7: Three adjectives, at least one specific, some reasoning
- 0.7-0.9: Three meaningful adjectives, most specific, clear reasoning for at least two
- 0.9-1.0: Three specific thoughtful adjectives with career-connected reasoning

Respond in JSON: {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}""",

    "skills": """You are evaluating a high school student response about their skills.
The student was asked: What are three skills you have that will help you in your future career?

Evaluate SUBSTANCE and CAREER-AWARENESS, not grammar or writing style.

STUDENT RESPONSE: {text}

SCORING RUBRIC (0.0 to 1.0):
- 0.0-0.1: Blank, "I do not have any", or non-answer
- 0.1-0.3: Vague non-skills or soft personal traits not linked to any career
- 0.3-0.5: Names one or two real skills but does not connect to any career
- 0.5-0.7: Names two or three real skills, at least one specific and career-relevant
- 0.7-0.9: Names three clear career-relevant skills with some explanation
- 0.9-1.0: Three specific well-articulated skills with strong career connection

Respond in JSON: {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}""",

    "smart_goal": """You are evaluating a high school student SMART goal response.
Evaluate whether this is a genuine structured goal, not grammar or writing quality.

STUDENT RESPONSE: {text}

SCORING RUBRIC (0.0 to 1.0):
- 0.0-0.1: Blank, --, or completely irrelevant
- 0.1-0.3: Vague aspiration with no SMART elements
- 0.3-0.5: Some specificity but missing most SMART elements
- 0.5-0.7: Specific and has one or two SMART elements
- 0.7-0.9: Has three or more SMART elements, clearly career/education connected
- 0.9-1.0: Genuinely SMART with specific, measurable, time-bound, relevant elements

Respond in JSON: {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}""",

    "ideal_job": """You are evaluating a high school student response about their ideal future career.
Evaluate SPECIFICITY and CAREER-AWARENESS, not grammar.

STUDENT RESPONSE: {text}

SCORING RUBRIC (0.0 to 1.0):
- 0.0-0.1: Non-answer: Not yet, Unsure, I don't know, N/A, blank
- 0.1-0.3: Extremely vague: a good job, something that pays well
- 0.3-0.5: Broad career field named but very general
- 0.5-0.7: Named career area with some specificity
- 0.7-0.9: Specific named career role
- 0.9-1.0: Highly specific career role with specialization or clear reasoning

Respond in JSON: {{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}""",
}


def score_rff(fields: dict, llm_score_text=None) -> dict:
    """
    Score the RFF pillar. Formula-scored + LLM-scored text fields.

    Args:
        fields: {canonical_name: value_or_None}
        llm_score_text: optional async callable(text, prompt_key) -> float (0-1)
    """
    g = fields.get

    def _llm(field_name, prompt_key, default=0.5):
        text = g(field_name)
        if text is None or str(text).strip() in ('', 'nan', '--', 'N/A', 'n/a'):
            return 0.0
        if llm_score_text is None:
            return default
        return llm_score_text(str(text), prompt_key)

    # ── LLM-scored text fields ─────────────────────────────────
    n_adj    = _llm("What are three adjectives that describe the person you are and why", "adjectives")
    n_skills = _llm("What are three skills you have that will help you in your future career", "skills")
    n_smart  = _llm("SMART GOAL", "smart_goal")
    n_smart3 = _llm("Remember the SMART Goal you set - next round", "smart_goal")
    n_ideal  = _llm("If you do not have a job, what is your ideal future career job", "ideal_job")

    # ── Formula-scored fields ──────────────────────────────────
    n_hope1     = _norm_hope_tags(g("Hope to Gain"))
    n_hope2     = _norm_hope_tags(g("What do you hope to gain by going through this program"))
    n_pursue    = _norm17(g("Know How To Pursue Careers"))
    n_prepared  = _norm_binary(g("I feel more prepared for my future career"))
    n_ready_col = _norm17(g("I feel ready and prepared for college"))
    n_fy1_ready = _norm17(g("FY1 Feel College Ready and Prepped"))
    n_stronger  = _norm010(g("I feel I am now a stronger candidate for college and careers"))
    n_more_prep = _norm010(g("I feel I am now more prepared for college"))
    n_connect   = _norm_binary(g("FY helped realize doing well connects to my career goals"))
    n_spk_insp  = _norm010(g("The Speaker inspired me to think more about my future career"))
    n_spk_path  = _norm010(g("The Speaker helped me think about my future career pathway"))
    n_spk_model = _norm010(g("The Speaker was a relatable role model"))
    n_top_insp  = _norm010(g("The topic inspired me to think more about my future career"))
    n_top_path  = _norm010(g("The topic helped me think about my future career pathway"))

    # ── Dimensions ─────────────────────────────────────────────
    d1 = n_adj * 0.30 + n_skills * 0.30 + n_hope1 * 0.20 + n_hope2 * 0.20
    d2 = n_smart * 0.70 + n_smart3 * 0.30
    d3 = n_pursue * 0.40 + n_prepared * 0.30 + n_ideal * 0.30
    d4 = (n_ready_col * 0.25 + n_fy1_ready * 0.20 + n_stronger * 0.20 +
          n_more_prep * 0.15 + n_connect * 0.20)

    rff_score = round(
        (d1 * 0.25 + d2 * 0.25 + d3 * 0.25 + d4 * 0.25) * 100,
        1
    )

    return {
        "score": rff_score,
        "sub_scores": {
            "D1_SelfReflection": round(d1, 4),
            "D2_GoalSetting": round(d2, 4),
            "D3_FutureCareer": round(d3, 4),
            "D4_CollegePrep": round(d4, 4),
        },
    }
