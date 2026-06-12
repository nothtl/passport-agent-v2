"""CT and CI pillar prompts — holistic document-based scoring."""

CT_PROMPT = """You are evaluating a student's critical thinking ability for a workforce development program.
These are high school/early college students in their first professional experiences.

Score from 0-100 based on analytical depth in their documented work.

WHAT CRITICAL THINKING LOOKS LIKE:
- Teaching/tutoring roles: breaking down material, assessing learner needs
- Translation/interpretation: analyzing meaning across languages
- Coordination/program management: organizing workflows, making decisions
- Health/advocacy/social service roles: navigating systems, making judgment calls
- Academic rigour: AP/honors, research, debate
- Certifications in substantive areas

SCORING RUBRIC:
- 80-100: Multiple documented analytical roles with described reasoning/decision-making
- 60-79: At least one clear analytical role with described approach
- 40-59: Some analytical signals alongside execution roles
- 20-39: Mostly execution roles with minimal described analytical component
- 0-19: No analytical signal apparent

STUDENT EVIDENCE:
LinkedIn: {linkedin_text}
Resume: {resume_text}
GitHub: {github_text}

Return JSON:
{{"ct_score": <int 0-100>, "thinking_arc": "<1 sentence, 25-30 words, professional third person>", "key_evidence": "<comma-separated labels>", "depth_signal": "<deep|developing|surface>"}}"""


CI_PROMPT = """You are evaluating a student's creative innovation ability for a workforce development program.
High school/early college students in their first professional experiences.

Score from 0-100 based on evidence of original thinking, novel projects, and creative initiative.

WHAT CREATIVE INNOVATION LOOKS LIKE:
- Content creation: graphic design, social media, digital media producing original material
- Performing arts: acting, modeling, music, dance with described portfolio/production
- Founding/co-founding: clubs, programs, initiatives, community efforts from scratch
- Event design: designing and running original events
- Advocacy/organizing with described strategy and creative materials
- Self-initiated side projects in any domain
- Any role with described self-directed creative contribution

SCORING RUBRIC:
- 80-100: Multiple documented instances of original creative initiative
- 60-79: At least one clearly self-initiated or original creative act
- 40-59: Some creative elements within otherwise standard roles
- 20-39: Standard participation and execution roles, no creative component
- 0-19: No creative signal apparent

CRITICAL THINKING PROFILE (already scored): {ct_arc}
The innovation_arc MUST reference DIFFERENT evidence than the CT note.

STUDENT EVIDENCE:
LinkedIn: {linkedin_text}
Resume: {resume_text}
GitHub: {github_text}
CT arc (DO NOT reuse this evidence): {ct_arc}
Forbidden terms (DO NOT use in innovation_arc): {forbidden_terms}

Return JSON:
{{"ci_score": <int 0-100>, "innovation_arc": "<1 sentence, 25-30 words, professional third person>", "key_evidence": "<comma-separated labels>", "innovation_signal": "<pioneering|developing|conventional>"}}"""
