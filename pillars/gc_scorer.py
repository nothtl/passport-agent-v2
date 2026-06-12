"""GC (Global Citizen) pillar — formula version from old agent2/tools/gc_scorer.py."""

from .formulas import (
    _norm17, _norm_binary, _norm_culture_feel, _norm_career_network,
    _norm_hours_volunteered,
)


def score_gc_formula(fields: dict) -> dict:
    """
    Pure formula GC scorer (no LLM). Uses survey fields only.
    This is the OLD formula-based GC scoring for parity comparison.
    """
    g = fields.get

    # ── D1 Empathy & Humility (weight 0.30) ────────────────────
    d1 = (
        _norm17(g("Pre Empathy")) * 0.05 +
        _norm17(g("Pre Humble")) * 0.05 +
        _norm17(g("Pre Listen")) * 0.05 +
        _norm17(g("Pre Include Others Who Are Different")) * 0.05 +
        _norm17(g("Pre Deal with Conflicts")) * 0.05 +
        _norm17(g("Pre Lead With Authenticity")) * 0.05 +
        _norm17(g("Listen to others")) * 0.15 +
        _norm17(g("Deal with conflicts with other people conflict management")) * 0.15 +
        _norm17(g("Include others who are different from you diversity and inclusion")) * 0.15 +
        _norm17(g("Reflect if you have been in a similar situation as someone you are trying to help non positional leadership")) * 0.20
    )

    # ── D2 Community Feel (weight 0.20) ─────────────────────────
    d2 = (
        _norm17(g("Pre Community Connected")) * 0.15 +
        _norm17(g("Community Feel (Quant)")) * 0.40 +
        _norm_culture_feel(g("Culture Feel")) * 0.45
    )

    # ── D3 Cultural Competency (weight 0.20) ────────────────────
    d3 = (
        _norm17(g("I understand how my cultural values can shape my career choices")) * 0.30 +
        _norm_binary(g("After meeting Champions and working with other peers in my SPEAKHIRE Internship Rounds I better understand people who are different from me")) * 0.30 +
        _norm_binary(g("Were you introduced to diverse career professionals who you can relate to during this Internship Round")) * 0.40
    )

    # ── D4 Network & Growth (weight 0.20) ───────────────────────
    d4 = (
        _norm_binary(g("Meeting with my Champions during school helped me feel like I belong in school")) * 0.15 +
        _norm_binary(g("This SPEAKHIRE Foundational Year helped me understand the value of building a strong network")) * 0.15 +
        _norm_binary(g("I feel more engaged in school and participate more than before")) * 0.15 +
        _norm_binary(g("I made new friends during the Foundational Year")) * 0.15 +
        _norm_career_network(g("How many individuals do you know who work in the career you are interested in")) * 0.40
    )

    # ── D5 Volunteering (weight 0.10) ───────────────────────────
    d5 = (
        _norm_binary(g("FY1 Ever Volunteered")) * 0.35 +
        _norm_hours_volunteered(g("FY1 Hours Volunteered")) * 0.65
    )

    gc_score = round(
        (d1 * 0.30 + d2 * 0.20 + d3 * 0.20 + d4 * 0.20 + d5 * 0.10) * 100,
        1
    )

    return {
        "score": gc_score,
        "sub_scores": {
            "D1_Empathy": round(d1, 4),
            "D2_Community": round(d2, 4),
            "D3_Cultural": round(d3, 4),
            "D4_Network": round(d4, 4),
            "D5_Volunteering": round(d5, 4),
        },
    }
