"""EC (Effective Communicator) pillar — hybrid formula + LLM text scoring."""

from .formulas import (
    _norm15, _norm17, _norm110, _norm_binary, _to_binary,
    _to_numeric_safe, _english_level, _is_multilingual,
)

def score_ec(fields: dict, llm_score_text=None) -> dict:
    """
    Score the EC pillar. Formula-scored numeric fields + LLM-scored text fields.

    Args:
        fields: {canonical_name: value_or_None}
        llm_score_text: optional async callable(text, rubric_key) -> float (1-5)
    """
    g = fields.get

    # ── Numeric inputs ─────────────────────────────────────────
    eng_int   = _english_level(g("English - Spoken"))
    champ_rat = _norm110(g("Rate your Career Pathways Champion (1-10)"))
    comm_feel = _norm17(g("Community Feel (Quant)"))

    # ── LLM text scores (1-5 scale) ────────────────────────────
    # If llm_score_text is provided, use it. Otherwise use formula defaults.
    def _llm(field_name, rubric_key="written_comm_quality", default=3.0):
        text = g(field_name)
        if text is None or str(text).strip() in ('', 'nan', '--', 'N/A', 'n/a'):
            return 1.0
        if llm_score_text is None:
            return default  # no LLM available, use midpoint
        return llm_score_text(str(text), rubric_key)

    sugg_llm  = _llm("Any suggestions to make the Foundational Year a better experience")
    touch_llm = _llm("Did you find a way to stay in touch")
    cross_llm = _llm("Did you learn something about other careers from other Career Cohorts")
    skill_llm = _llm("What are three skills you have that will help you in your future career")

    written_avg = (sugg_llm + touch_llm + cross_llm) / 3.0
    sugg_depth  = sugg_llm

    # ── V: Verbal Communication (max ~30) ──────────────────────
    V = (
        _norm15(eng_int) * 10 +
        champ_rat        * 10 +
        comm_feel        * 5 +
        _norm15(sugg_depth) * 5
    )

    # ── W: Written Communication (max ~20) ─────────────────────
    W = (
        _norm15(written_avg)        * 15 +
        ((skill_llm - 1.0) / 4.0)  * 5
    )

    # ── I_s: Interpersonal Skills (max ~25) ────────────────────
    conflict_val = g("Deal with conflicts - conflict management")
    listen_val   = g("Listen to others (post)")
    conflict_bonus = 1 if _to_numeric_safe(conflict_val) > 3 else 0
    I_s = (
        _norm17(listen_val)   * 10 +
        _norm17(conflict_val) * 10 +
        conflict_bonus        * 5
    )

    # ── C_s: Cross-Cultural Competence (max ~25) ───────────────
    C_s = (
        _norm17(g("Include others who are different - diversity and inclusion")) * 10 +
        _to_binary(g("After meeting Champions, I better understand people who are different from me")) * 5 +
        _is_multilingual(g("Languages")) * 3 +
        _norm15(eng_int) * 2
    )

    ec_score = round(max(0.0, min(100.0, V + W + I_s + C_s)), 2)

    return {
        "score": ec_score,
        "sub_scores": {
            "Verbal": round(V, 4),
            "Written": round(W, 4),
            "Interpersonal": round(I_s, 4),
            "CrossCultural": round(C_s, 4),
        },
    }
