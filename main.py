#!/usr/bin/env python
"""
Passport Agent V2 — Full 6-Pillar Pipeline with Tool-Using Agents

Runs ALL pillars: EC, GC, RFF, CR, CT, CI + enriched survey fields.
Compares against old Gemini pipeline.

Usage:
  python main.py --student "Abigail Rodriguez"
  python main.py --student "Abigail Rodriguez" --new-only
  python main.py --batch 5
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_ROOT = os.path.join(HERE, "..", "Passport_Agent_Actual_Test_Final_1",
                         "Passport_Agent_Actual_Test", "Passport_Agent_Actual")
sys.path.insert(0, OLD_ROOT)

from agents.base import run_agent, get_usage, reset_usage, OPENCODE_MODEL, _call_api
from tools.evidence import search_evidence, cite_evidence, count_distinct
from tools.rubric import get_rubric, ALL_GC_DIMENSIONS
from tools.documents import parse_linkedin_markdown

# Pillar scorers
from pillars.ec_scorer import score_ec
from pillars.gc_scorer import score_gc_formula
from pillars.rff_scorer import score_rff, RFF_PROMPTS
from pillars.cr_scorer import score_cr, C3_PROMPT, C4_PROMPT
from pillars.ct_ci_prompts import CT_PROMPT, CI_PROMPT
from pillars.formulas import _to_bool

OUTPUTS_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

PILLAR_LABELS = {
    "EC": "Effective Communicator", "GC": "Global Citizen",
    "RFF": "Reflective & Future-Focused", "CR": "Career Ready",
    "CT": "Critical Thinker", "CI": "Creative Innovator",
}


# ═══════════════════════════════════════════════════════════════
# LLM helper — direct API call for simple scoring prompts
# ═══════════════════════════════════════════════════════════════

def _llm_json(prompt: str, temperature: float = 0.0, max_tokens: int = None) -> dict:
    """Quick single LLM call expecting JSON response. No tool loop."""
    if max_tokens is None:
        # DeepSeek V4 Flash uses reasoning tokens — need headroom for long prompts
        max_tokens = max(4096, int(len(prompt) / 2) + 2000)
    messages = [
        {"role": "system", "content": "Return ONLY valid JSON, no markdown, no preamble."},
        {"role": "user", "content": prompt},
    ]
    resp = _call_api(messages=messages, temperature=temperature, max_tokens=max_tokens)
    if "error" in resp:
        return resp
    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try extracting from markdown
        if "```json" in content:
            try:
                return json.loads(content.split("```json")[1].split("```")[0])
            except: pass
        if "```" in content:
            try:
                return json.loads(content.split("```")[1].split("```")[0])
            except: pass
        # Brace extraction
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except: pass
        return {"error": "JSON parse failed", "raw": content[:200]}


# ═══════════════════════════════════════════════════════════════
# New Pipeline — All 6 Pillars
# ═══════════════════════════════════════════════════════════════

async def run_new_pipeline_full(student_name: str) -> dict:
    """Run all 6 pillars + enriched fields on one student."""
    reset_usage()
    slug = re.sub(r'[^a-z0-9]+', '_', student_name.lower()).strip('_')

    print(f"\n{'='*60}")
    print(f"[NEW] Full pipeline: {student_name}")
    print(f"[NEW] Model: {OPENCODE_MODEL}")
    print(f"{'='*60}")

    start = time.time()

    # ── Load documents ─────────────────────────────────────
    student_dir = None
    old_sd = os.path.join(OLD_ROOT, "student_data")
    linkedin_text = ""
    resume_text = ""

    for folder in os.listdir(old_sd):
        if student_name.lower() in folder.lower():
            student_dir = os.path.join(old_sd, folder)
            break

    if student_dir:
        for f in os.listdir(student_dir):
            fp = os.path.join(student_dir, f)
            if "resume" in f.lower() and f.endswith(".pdf"):
                try:
                    from pdfminer.high_level import extract_text
                    resume_text = extract_text(fp)
                except Exception:
                    pass
            if "linkedin" in f.lower() and f.endswith(".md"):
                with open(fp, encoding="utf-8") as fh:
                    linkedin_text = fh.read()

    # ── Load survey data from old agent1 output ─────────────
    old_a1 = os.path.join(OLD_ROOT, "agent1", "outputs", f"{slug}_raw_data.json")
    fields = {}
    if os.path.exists(old_a1):
        with open(old_a1, encoding="utf-8") as f:
            a1 = json.load(f)
        raw_fields = a1.get("fields", {})
        fields = {k: v.get("value") for k, v in raw_fields.items()
                   if v.get("status") == "found"}
        print(f"[NEW] Loaded {len(fields)} survey fields from Agent 1")
    else:
        print(f"[NEW] Warning: No Agent 1 data found, using empty fields")

    # ── Enrich missing survey fields via LLM from documents ──
    print("[NEW] Enriching missing survey fields via LLM...")
    from pillars.enricher import enrich_all
    old_a2 = os.path.join(OLD_ROOT, "agent2", "outputs", f"{slug}_survey_scores.json")
    missing_fields = []
    if os.path.exists(old_a2):
        with open(old_a2, encoding="utf-8") as f:
            missing_fields = json.load(f).get("missing_fields", [])

    enriched = {}
    if missing_fields:
        enriched, fields = enrich_all(
            student_name=student_name,
            missing_fields=missing_fields,
            linkedin_text=linkedin_text,
            resume_text=resume_text,
            fields=fields,
            llm_call=_llm_json,
        )
    print(f"[NEW] Enriched {len(enriched)} fields via LLM")

    # ── Score all pillars ──────────────────────────────────
    all_scores = {}

    # EC — Hybrid (formula + LLM text scoring)
    print("[NEW] Scoring EC...")
    ec_prompt_fields = {
        "Any suggestions to make the Foundational Year a better experience":
            fields.get("Any suggestions to make the Foundational Year a better experience"),
        "Did you find a way to stay in touch":
            fields.get("Did you find a way to stay in touch"),
        "Did you learn something about other careers from other Career Cohorts":
            fields.get("Did you learn something about other careers from other Career Cohorts"),
        "What are three skills you have that will help you in your future career":
            fields.get("What are three skills you have that will help you in your future career"),
    }

    def ec_llm_scorer(text: str, rubric_key: str) -> float:
        """Score EC text field via LLM, return 1.0-5.0."""
        prompt = """Score this student's written response for clarity, specificity, and professional register on a 1-5 scale.

RUBRIC:
5 - Clear, organized, professional tone; specific and actionable.
4 - Generally clear and professional; minor issues.
3 - Understandable but informal or vague.
2 - Difficult to follow or very brief.
1 - Unintelligible, placeholder, or no communicative content.

STUDENT RESPONSE: {text}

Return JSON: {{"score": <int 1-5>, "justification": "<one sentence>"}}""".replace("{text}", text)

        result = _llm_json(prompt)
        if "error" in result:
            return 3.0
        return max(1.0, min(5.0, float(result.get("score", 3))))

    all_scores["EC"] = score_ec(fields, llm_score_text=ec_llm_scorer)

    # GC — Formula for aggregate score + tool-using agent for detail evidence
    print("[NEW] Scoring GC...")
    gc_formula_result = score_gc_formula(fields)
    all_scores["GC"] = {
        "score": gc_formula_result["score"],
        "sub_scores": gc_formula_result["sub_scores"],
        "source": "survey+docs (formula-based aggregate)",
    }

    # RFF — Hybrid (formula + LLM text scoring)
    print("[NEW] Scoring RFF...")
    def rff_llm_scorer(text: str, prompt_key: str) -> float:
        template = RFF_PROMPTS.get(prompt_key, "Score this: {text}")
        result = _llm_json(template.replace("{text}", text))
        if "error" in result: return 0.5
        return max(0.0, min(1.0, float(result.get("score", 0.5))))

    all_scores["RFF"] = score_rff(fields, llm_score_text=rff_llm_scorer)

    # CR — Deterministic + LLM for C3/C4
    print("[NEW] Scoring CR...")
    def cr_llm_caller(prompt_key: str, **kwargs):
        if prompt_key == "c3":
            return _llm_json(C3_PROMPT.replace("{cpc_session_text}", kwargs.get("cpc_session_text", "")))
        elif prompt_key == "c4":
            return _llm_json(C4_PROMPT.replace("{cpc_resume_text}", kwargs.get("cpc_resume_text", "")))
        return {}

    cr_result = score_cr(fields, llm_call=cr_llm_caller)
    # C4 override: if CPC didn't log resume work but student has a substantive resume
    if cr_result["sub_scores"].get("C4", 0) == 0 and len(resume_text.strip()) > 300:
        c4_check = _llm_json(
            "Does this text represent a real, complete student resume with substantive content? "
            f"RESUME: {resume_text[:1200]}\n\n"
            'Return JSON: {{"resume_built": true}} or {{"resume_built": false}}'
        )
        if c4_check.get("resume_built") in (True, "true", 1):
            cr_result["sub_scores"]["C4"] = 100
            cr_result["score"] = round(sum(cr_result["sub_scores"].values()) / 4, 1)
    all_scores["CR"] = cr_result

    # CT — Holistic LLM with evidence
    print("[NEW] Scoring CT & CI...")
    ct_prompt = CT_PROMPT.replace("{linkedin_text}", linkedin_text[:1500] or "none")\
                          .replace("{resume_text}", resume_text[:1500] or "none")\
                          .replace("{github_text}", "none")
    ct_result = _llm_json(ct_prompt)
    # Retry with longer context if empty
    if ct_result.get("ct_score", 0) == 0 and ct_result.get("thinking_arc", "") == "":
        ct_result = _llm_json(ct_prompt)  # retry once
    all_scores["CT"] = {
        "score": int(ct_result.get("ct_score", 0)),
        "thinking_arc": ct_result.get("thinking_arc", ""),
        "key_evidence": ct_result.get("key_evidence", ""),
        "depth_signal": ct_result.get("depth_signal", "developing"),
    }

    # CI — Holistic LLM with CT arc as forbidden context
    ct_arc = ct_result.get("thinking_arc", "")
    ci_prompt = CI_PROMPT.replace("{linkedin_text}", linkedin_text[:2500] or "none")\
                          .replace("{resume_text}", resume_text[:2500] or "none")\
                          .replace("{github_text}", "none")\
                          .replace("{ct_arc}", ct_arc)\
                          .replace("{forbidden_terms}", "")
    ci_result = _llm_json(ci_prompt)
    all_scores["CI"] = {
        "score": int(ci_result.get("ci_score", 0)),
        "innovation_arc": ci_result.get("innovation_arc", ""),
        "key_evidence": ci_result.get("key_evidence", ""),
        "innovation_signal": ci_result.get("innovation_signal", "developing"),
    }

    # ── Assemble output ───────────────────────────────────
    usage = get_usage()
    elapsed = round(time.time() - start, 1)

    # ── Convert to old format for compatibility ────────────
    a1_data = None
    if os.path.exists(old_a1):
        with open(old_a1, encoding="utf-8") as f:
            a1_data = json.load(f)
    a2_data = {}
    if os.path.exists(old_a2):
        with open(old_a2, encoding="utf-8") as f:
            a2_data = json.load(f)

    output = _make_old_format(
        student_name=student_name,
        all_scores=all_scores,
        enriched=enriched,
        old_a1_data=a1_data,
        old_a2_data=a2_data,
        usage=usage,
        elapsed=elapsed,
    )

    out_path = os.path.join(OUTPUTS_DIR, f"{slug}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    # ── Print score summary ───────────────────────────────
    print(f"\n[NEW] Score summary for {student_name}:")
    for pillar in ["EC", "GC", "RFF", "CR", "CT", "CI"]:
        score_data = all_scores.get(pillar, {})
        s = score_data.get("score", "N/A")
        print(f"  {pillar} ({PILLAR_LABELS.get(pillar, pillar)}): {s}")
    print(f"  Enriched fields: {len(enriched)}")
    print(f"  Total LLM calls: {usage['calls']}  Time: {elapsed}s")
    print(f"  Output: {out_path}")

    return output


# ═══════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════

def load_old_pipeline(student_name: str) -> dict:
    """Load pre-computed old pipeline results."""
    slug = re.sub(r'[^a-z0-9]+', '_', student_name.lower()).strip('_')
    old_path = os.path.join(OLD_ROOT, "agent4", "outputs", f"{slug}_enriched_scores.json")
    if not os.path.exists(old_path):
        return {"error": f"No old output for {student_name}"}

    with open(old_path, encoding="utf-8") as f:
        data = json.load(f)

    scores = {}
    for k, v in data.get("scores", {}).items():
        if isinstance(v, dict):
            scores[k] = {"score": v.get("score"), "sub_scores": v.get("sub_scores", {})}
    return {
        "student_name": student_name,
        "scores": scores,
        "enriched_fields": data.get("enriched_fields", {}),
        "llm_calls": data.get("llm_calls_made", "?"),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Passport Agent V2")
    parser.add_argument("--student", default="Abigail Rodriguez")
    parser.add_argument("--new-only", action="store_true")
    parser.add_argument("--batch", type=int, default=0)
    args = parser.parse_args()

    if args.batch > 0:
        # Batch mode — run N students, skip if already have output
        students = [
            "Abigail Rodriguez", "Devin Rhodie", "Benjamin Medrano",
            "Yamin Titikpina", "Bianka Pena", "Cristal Davidson",
        ][:args.batch]

        for name in students:
            slug = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
            out_path = os.path.join(OUTPUTS_DIR, "students", f"{slug}.json")
            if os.path.exists(out_path):
                print(f"[SKIP] {name} — already processed")
                continue
            await run_new_pipeline_full(name)
        return

    student = args.student
    slug = re.sub(r'[^a-z0-9]+', '_', student.lower()).strip('_')

    # ── Run new pipeline ──────────────────────────────────
    new_result = await run_new_pipeline_full(student)

    if args.new_only:
        return

    # ── Compare with old ───────────────────────────────────
    old_result = load_old_pipeline(student)
    if "error" in old_result:
        print(f"[WARN] {old_result['error']}")
        return

    print(f"\n{'='*70}")
    print(f"COMPARISON: {student}")
    print(f"  Old: Gemini Flash-Lite | New: {OPENCODE_MODEL}")
    print(f"{'='*70}")

    print(f"\n{'Pillar':<8} {'Old':>8} {'New':>8} {'Diff':>8}")
    print(f"{'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for pillar in ["EC", "GC", "RFF", "CR", "CT", "CI"]:
        old_score = old_result["scores"].get(pillar, {}).get("score", "N/A")
        new_score = new_result["scores"].get(pillar, {}).get("score", "N/A")
        try:
            diff = round(float(new_score) - float(old_score), 1)
        except (TypeError, ValueError):
            diff = "N/A"
        print(f"{pillar:<8} {str(old_score):>8} {str(new_score):>8} {str(diff):>8}")

    print(f"\nEnriched fields: Old={len(old_result.get('enriched_fields',{}))}, New={new_result.get('fields_enriched',0)}")
    print(f"LLM calls: Old={old_result.get('llm_calls','?')}, New={new_result['usage']['calls']}")
    print(f"Time: New={new_result['elapsed_seconds']}s")


def _make_old_format(student_name, all_scores, enriched, old_a1_data, old_a2_data, usage, elapsed=0):
    """Convert new pipeline output to match old pipeline format exactly."""
    email = ""
    if old_a1_data:
        email = old_a1_data.get("email", "")

    # Compute per-pillar data_coverage from old_a1 if available
    def _coverage(pillar_key):
        if not old_a1_data:
            return 0.0
        raw_fields = old_a1_data.get("fields", {})
        pillar_fields = {k: v for k, v in raw_fields.items() if v.get("pillar") == pillar_key}
        if not pillar_fields:
            return 0.0
        found = sum(1 for v in pillar_fields.values() if v.get("status") == "found")
        return round(found / len(pillar_fields), 2)

    # Build per-pillar scores in old format
    old_scores = {}
    for pillar in ["EC", "GC", "RFF", "CR", "CT", "CI"]:
        new_data = all_scores.get(pillar, {})
        entry = {
            "score": new_data.get("score", 0),
            "source": "survey+docs",
        }
        # Only formula pillars have sub_scores and data_coverage
        if pillar in ["EC", "GC", "RFF", "CR"]:
            entry["sub_scores"] = new_data.get("sub_scores", {})
            entry["data_coverage"] = _coverage(pillar)
        # Add pillar-specific narrative fields
        if pillar == "CT":
            entry["thinking_arc"] = new_data.get("thinking_arc", "")
            entry["key_evidence"] = new_data.get("key_evidence", "")
            entry["depth_signal"] = new_data.get("depth_signal", "developing")
            entry["reasoning"] = new_data.get("thinking_arc", "")
        elif pillar == "CI":
            entry["innovation_arc"] = new_data.get("innovation_arc", "")
            entry["key_evidence"] = new_data.get("key_evidence", "")
            entry["innovation_signal"] = new_data.get("innovation_signal", "developing")
            entry["reasoning"] = new_data.get("innovation_arc", "")
        else:
            subs = new_data.get("sub_scores", {})
            if subs:
                entry["reasoning"] = f"{student_name} demonstrates competency in {pillar} scoring {entry['score']}."
            else:
                entry["reasoning"] = f"Score derived from {entry['source']}."
        old_scores[pillar] = entry

    # Build enriched_fields in old format
    old_enriched = {}
    for k, v in enriched.items():
        old_enriched[k] = {
            "value": v,
            "source": "docs",
            "was_missing": True,
        }

    missing_fields = old_a2_data.get("missing_fields", []) if old_a2_data else []

    still_missing = [f for f in missing_fields if f not in enriched]

    return {
        "student_name": student_name,
        "email": email,
        "scores": old_scores,
        "enriched_fields": old_enriched,
        "fields_enriched_count": len(enriched),
        "fields_still_missing": still_missing,
        "llm_calls_made": usage["calls"],
        "_new_pipeline_metadata": {
            "model": OPENCODE_MODEL,
            "elapsed_seconds": elapsed,
            "usage": usage,
        },
    }


if __name__ == "__main__":
    asyncio.run(main())
