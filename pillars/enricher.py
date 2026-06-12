"""LLM-driven enrichment — fills missing survey fields from student documents.

Replicates the logic from old Agent 4's ec_enricher, gc_enricher, rff_enricher, cr_enricher.
Uses DeepSeek to infer missing Likert scores, booleans, and other values from resume+LinkedIn.
"""

import json, re

EC_INTERP_PROMPT = """You are estimating behavioral scores for a student's interpersonal communication skills based on their documented activities.

Three dimensions, each on a 1-7 scale:
1 = no evidence | 2 = one vague indirect signal | 3 = one clear indirect signal
4 = one direct behavioral example | 5 = multiple direct examples from one role
6 = consistent pattern across multiple roles | 7 = sustained central theme

STUDENT EVIDENCE:
LinkedIn: {linkedin_text}
Resume: {resume_text}
CPC mentor notes: {cpc_text}

DIMENSIONS:
1. "listen" — listening, facilitating, mentoring, advising, interpreting/translating, peer tutoring
2. "conflict" — navigating disagreements, managing diverse groups, mediating
3. "include" — reaching across differences, multilingual work, diversity outreach, interpreter work
4. "reflect" — using own experience to help others, peer mentoring, shared-experience coaching

Rules: Score from described behaviors only. If no evidence, return 1. Interpreting/translating IS strong evidence for listen(4+) and include(5+).

Return JSON: {{"listen": <int 1-7>, "conflict": <int 1-7>, "include": <int 1-7>, "reflect": <int 1-7>, "reasoning": "<1 sentence>"}}"""


EC_PRE_PROMPT = """You are estimating PRE-PROGRAM behavioral scores for a student based on documents that describe their activities BEFORE or DURING their early program involvement.

Dimensions on a 1-7 scale:
1 = no evidence | 4 = one direct behavioral example | 7 = sustained central theme

STUDENT EVIDENCE:
LinkedIn: {linkedin_text}
Resume: {resume_text}

DIMENSIONS:
1. "pre_empathy" — mentoring, support roles, family/community service references
2. "pre_humble" — openness to learning, language like "curious", "learning", service-oriented
3. "pre_listen" — peer tutoring, counseling, community facilitation, interpreting
4. "pre_include" — multilingual, cross-cultural community involvement
5. "pre_conflict" — team sports, religious/cultural leadership, peer mediation
6. "pre_lead_auth" — founding or initiating something, self-described values, authentic leadership
7. "pre_community_connected" — civic memberships, religious community, cultural associations

Return JSON: {{"pre_empathy": <int 1-7>, "pre_humble": <int 1-7>, "pre_listen": <int 1-7>, "pre_include": <int 1-7>, "pre_conflict": <int 1-7>, "pre_lead_auth": <int 1-7>, "pre_community_connected": <int 1-7>, "reasoning": "<1 sentence>"}}"""


GC_DETECT_PROMPT = """You are detecting specific signals in a student's documents.

LinkedIn: {linkedin_text}
Resume: {resume_text}

Answer based strictly on what is written:
1. has_volunteer — Any volunteering, community service, nonprofit work, food pantry, shelter, faith community service, mutual aid, unpaid service? Answer with true or false.
2. volunteer_hours_estimate — If has_volunteer is true, estimate: "1-10", "10-20", "20-30", or "30+". If false, return "0".
3. community_roles_count — Count distinct roles serving community/social causes/underserved populations (integer 0-7).
4. has_speakhire — Does any experience mention "SPEAKHIRE" or "Speak Hire"? true or false.
5. has_campus_role — Campus-based role: student leadership, peer tutoring, campus ambassador, residence advisor? true or false.
6. network_estimate — How many professionals in their career interest area do they likely know? Integer 0-10.
7. has_career_discussion — Do docs describe career exploration or professional development? true or false.
8. has_college_discussion — Do docs describe college applications, GPA, academic majors? true or false.

Return JSON: {{"has_volunteer": <bool>, "volunteer_hours_estimate": "<string>", "community_roles_count": <int>, "has_speakhire": <bool>, "has_campus_role": <bool>, "network_estimate": <int>, "has_career_discussion": <bool>, "has_college_discussion": <bool>, "reasoning": "<1 sentence>"}}"""


RFF_SIGNALS_PROMPT = """You are detecting career and college readiness signals in a student's documents.

LinkedIn: {linkedin_text}
Resume: {resume_text}
CPC mentor notes: {cpc_text}

Answer based strictly on what is written:
1. has_goal — Does the student describe a specific career or educational goal? true or false.
2. goal_text — If has_goal is true, extract or summarize the specific goal (max 150 chars). If false, return "".
3. career_specificity_score — How specific is their stated career direction? 1-5 scale:
   1 = no career mentioned | 2 = vague field | 3 = broad area named
   4 = specific role named | 5 = specific role with specialization
4. smart_goal_quality — If they have a SMART goal, rate it 0.0-1.0:
   0.0 = no goal | 0.3 = vague aspiration | 0.5 = some SMART elements
   0.7 = 3+ SMART elements | 0.9 = genuinely SMART
5. has_speakhire_on_linkedin — Does LinkedIn mention SPEAKHIRE? true or false.
6. diverse_professionals — Were they introduced to diverse career professionals? true or false.

Return JSON: {{"has_goal": <bool>, "goal_text": "<string>", "career_specificity_score": <int 1-5>, "smart_goal_quality": <float 0.0-1.0>, "has_speakhire_on_linkedin": <bool>, "diverse_professionals": <bool>, "reasoning": "<1 sentence>"}}"""


LANGUAGE_PROMPT = """You are detecting multilingual and cross-cultural signals.

LinkedIn: {linkedin_text}
Resume: {resume_text}

1. languages — List all languages the student speaks (comma-separated string). If only English, return "English".
2. is_multilingual — Does the student speak 2+ languages? true or false.
3. cross_cultural_signals — Count distinct cross-cultural activities (integer 0-5).
4. english_proficiency — Estimate English level: "very comfortable", "comfortable", "somewhat comfortable", "not very comfortable", or "not comfortable".

Return JSON: {{"languages": "<string>", "is_multilingual": <bool>, "cross_cultural_signals": <int>, "english_proficiency": "<string>", "reasoning": "<1 sentence>"}}"""


def enrich_all(student_name: str, missing_fields: list[str], linkedin_text: str,
               resume_text: str, fields: dict, llm_call) -> tuple[dict, dict]:
    """
    Run all enrichment prompts to fill missing survey fields.

    Args:
        student_name: Student name
        missing_fields: List of missing field names
        linkedin_text: LinkedIn markdown text
        resume_text: Resume extracted text
        fields: Current fields dict (mutated in place)
        llm_call: callable(prompt) -> dict (parsed JSON response)

    Returns:
        (enriched dict, updated fields dict)
    """
    enriched = {}
    cpc_text = str(fields.get("CPC All Session Text") or "")
    cpc_resume = str(fields.get("CPC Resume Text") or "")

    # Helper: wrap LLM call
    def _call(prompt):
        try:
            return llm_call(prompt)
        except Exception:
            return {}

    # ── EC: Interpersonal scores (listen, conflict, include, reflect) ──
    interp_prompt = EC_INTERP_PROMPT.format(
        linkedin_text=linkedin_text[:2000] or "none",
        resume_text=resume_text[:2000] or "none",
        cpc_text=cpc_text[:800] or "none",
    )
    interp = _call(interp_prompt)
    for key in ["listen", "conflict", "include", "reflect"]:
        if key in interp and isinstance(interp[key], (int, float)):
            field_map = {
                "listen": "Listen to others (post)",
                "conflict": "Deal with conflicts - conflict management",
                "include": "Include others who are different - diversity and inclusion",
                "reflect": "Reflect if you have been in a similar situation as someone you are trying to help non positional leadership",
            }
            field_name = field_map.get(key)
            if field_name and field_name in missing_fields:
                val = max(1, min(7, int(interp[key])))
                enriched[field_name] = val
                fields[field_name] = val

    # ── EC: Pre-program scores ──────────────────────────────────
    pre_prompt = EC_PRE_PROMPT.format(
        linkedin_text=linkedin_text[:2000] or "none",
        resume_text=resume_text[:2000] or "none",
    )
    pre = _call(pre_prompt)
    pre_field_map = {
        "pre_empathy": "Pre Empathy",
        "pre_humble": "Pre Humble",
        "pre_listen": "Pre Listen",
        "pre_include": "Pre Include Others Who Are Different",
        "pre_conflict": "Pre Deal with Conflicts",
        "pre_lead_auth": "Pre Lead With Authenticity",
        "pre_community_connected": "Pre Community Connected",
    }
    for key, field_name in pre_field_map.items():
        if key in pre and isinstance(pre[key], (int, float)):
            if field_name in missing_fields:
                val = max(1, min(7, int(pre[key])))
                enriched[field_name] = val
                fields[field_name] = val

    # ── GC: Detection signals ───────────────────────────────────
    detect_prompt = GC_DETECT_PROMPT.format(
        linkedin_text=linkedin_text[:2000] or "none",
        resume_text=resume_text[:2000] or "none",
    )
    detect = _call(detect_prompt)
    gc_field_map = {
        "has_volunteer": "FY1 Ever Volunteered",
        "has_speakhire": "FY1 - Had Internship?",
        "has_campus_role": "I feel more engaged in school and participate more than before",
        "network_estimate": "How many individuals do you know who work in the career you are interested in",
        "has_career_discussion": "Know How To Pursue Careers",
        "has_college_discussion": "I feel ready and prepared for college",
    }
    for key, field_name in gc_field_map.items():
        if key in detect and field_name in missing_fields:
            val = detect[key]
            if isinstance(val, bool):
                val = "True" if val else "False"
            enriched[field_name] = val
            fields[field_name] = val

    # community_roles_count → Community Feel (Quant)
    if "community_roles_count" in detect and "Community Feel (Quant)" in missing_fields:
        n = int(detect["community_roles_count"])
        if n > 0:
            val = min(n + 2, 7)
            enriched["Community Feel (Quant)"] = val
            fields["Community Feel (Quant)"] = val

    # volunteer_hours_estimate
    if "volunteer_hours_estimate" in detect and "FY1 Hours Volunteered" in missing_fields:
        val = str(detect["volunteer_hours_estimate"])
        if val != "0":
            enriched["FY1 Hours Volunteered"] = val
            fields["FY1 Hours Volunteered"] = val

    # ── RFF signals ─────────────────────────────────────────────
    rff_prompt = RFF_SIGNALS_PROMPT.format(
        linkedin_text=linkedin_text[:2000] or "none",
        resume_text=resume_text[:2000] or "none",
        cpc_text=cpc_text[:800] or "none",
    )
    rff = _call(rff_prompt)
    rff_field_map = {
        "diverse_professionals": "Were you introduced to diverse career professionals who you can relate to during this Internship Round",
    }
    for key, field_name in rff_field_map.items():
        if key in rff and field_name in missing_fields:
            val = rff[key]
            if isinstance(val, bool):
                val = "1.0" if val else "0.0"
            enriched[field_name] = val
            fields[field_name] = val

    if "career_specificity_score" in rff and "I feel more prepared for my future career" in missing_fields:
        fields["I feel more prepared for my future career"] = min(float(rff["career_specificity_score"]), 5.0)

    # ── Language signals ────────────────────────────────────────
    lang_prompt = LANGUAGE_PROMPT.format(
        linkedin_text=linkedin_text[:2000] or "none",
        resume_text=resume_text[:2000] or "none",
    )
    lang = _call(lang_prompt)
    if "languages" in lang and "Languages" in missing_fields:
        enriched["Languages"] = str(lang["languages"])
        fields["Languages"] = str(lang["languages"])

    # ── Cultural understanding fields ──────────────────────────
    if "is_multilingual" in lang:
        cultural_fields = [
            "I understand how my cultural values can shape my career choices",
            "After meeting Champions, I better understand people who are different from me",
        ]
        for cf in cultural_fields:
            if cf in missing_fields:
                val = 5 if lang.get("is_multilingual") in (True, "true", 1) else 3
                enriched[cf] = val
                fields[cf] = val

    # ── School engagement from campus_role ─────────────────────
    if "has_campus_role" in detect:
        school_fields = [
            "Meeting with my Champions during school helped me feel like I belong in school",
            "I feel more engaged in school and participate more than before",
            "I made new friends during the Foundational Year",
        ]
        has_role = detect["has_campus_role"] in (True, "true", 1)
        for sf in school_fields:
            if sf in missing_fields:
                val = "True" if has_role else "False"
                enriched[sf] = val
                fields[sf] = val

    # ── Network value from network_estimate ────────────────────
    if "network_estimate" in detect and "This SPEAKHIRE Foundational Year helped me understand the value of building a strong network" in missing_fields:
        net = int(detect.get("network_estimate", 0))
        fields["This SPEAKHIRE Foundational Year helped me understand the value of building a strong network"] = "True" if net >= 3 else "False"

    # ── Career/college strength from RFF ───────────────────────
    if "smart_goal_quality" in rff:
        strength_fields = [
            "I feel I am now a stronger candidate for college and careers",
            "I feel I am now more prepared for college",
        ]
        sq = float(rff.get("smart_goal_quality", 0))
        for sf in strength_fields:
            if sf in missing_fields:
                fields[sf] = round(sq * 10, 1)

    if "has_college_discussion" in detect and "FY helped realize doing well connects to my career goals" in missing_fields:
        fields["FY helped realize doing well connects to my career goals"] = "1.0" if detect.get("has_college_discussion") in (True, "true", 1) else "0.0"

    # ── Speaker/topic inspiration fields (optimistic estimates) ─
    speaker_fields = [
        "The Speaker inspired me to think more about my future career",
        "The Speaker helped me think about my future career pathway",
        "The Speaker was a relatable role model",
        "The topic inspired me to think more about my future career",
        "The topic helped me think about my future career pathway",
    ]
    for sf in speaker_fields:
        if sf in missing_fields:
            fields[sf] = 7.0  # neutral-positive default
            enriched[sf] = 7.0

    # ── CR: Build C3 text from resume if missing ───────────────
    if "CPC All Session Text" in missing_fields and resume_text.strip():
        enriched["CPC All Session Text"] = resume_text[:800]
        fields["CPC All Session Text"] = resume_text[:800]

    if "CPC Resume Text" in missing_fields and resume_text.strip():
        enriched["CPC Resume Text"] = f"Resume content: {len(resume_text)} chars of substantive content"
        fields["CPC Resume Text"] = f"Resume content: {len(resume_text)} chars"

    return enriched, fields
