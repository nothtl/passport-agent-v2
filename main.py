#!/usr/bin/env python
"""
Passport Agent V2 — Tool-Using Pipeline Runner

Compares the OLD approach (single-shot Gemini calls via subprocess) with
the NEW approach (tool-using agent loop via OpenCode Zen API).

Usage:
  python main.py --student "Abigail Rodriguez"    # Run both, compare
  python main.py --student "Abigail Rodriguez" --new-only
  python main.py --compare                          # Compare existing outputs
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Add old pipeline to path
OLD_ROOT = os.path.join(HERE, "..", "Passport_Agent_Actual_Test_Final_1",
                         "Passport_Agent_Actual_Test", "Passport_Agent_Actual")
sys.path.insert(0, OLD_ROOT)

from agents.base import run_agent, get_usage, reset_usage, OPENCODE_MODEL
from tools.evidence import search_evidence, cite_evidence, count_distinct
from tools.rubric import get_rubric, lookup_reference_examples, ALL_GC_DIMENSIONS
from tools.validation import check_consistency, validate_score
from tools.documents import (
    extract_pdf_text, identify_resume_structure, extract_section_text,
    parse_linkedin_markdown, find_github_username, scrape_github_profile,
)
from tools.discovery import (
    explore_zip, fuzzy_find_student, semantic_column_search,
    extract_row_field, resolve_identity,
)

OUTPUTS_DIR = os.path.join(HERE, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Agent System Prompts
# ═══════════════════════════════════════════════════════════════════════════════

AGENT4_SCORER_PROMPT = """You are a rigorous student competency scorer for SPEAKHIRE PathCredits.
Your scores affect real students' professional profiles.

Score a student's GLOBAL CITIZEN (GC) pillar across 17 dimensions.
Work efficiently: call search_evidence() ONCE per dimension (NOT multiple times),
then immediately call propose_score() with your findings.

PROCESS PER DIMENSION:
1. search_evidence(query="<concept keywords>", source="all", student_name="<name>")
   → This returns matching text from all documents with context.
2. get_rubric("GC", "<dimension>") if you need rubric guidance.
3. propose_score(pillar="GC", dimension="<dim>", proposed_score=<number>,
   rubric_band="<band>", evidence_quotes=[<exact quotes from search results>],
   reasoning="<why>")

RULES:
- If search_evidence returns total_count=0, score 0 (no evidence).
- If ambiguous, score LOWER — never inflate.
- cite_evidence() is OPTIONAL — only use if you need to verify a specific claim.
- Use SHORT queries in search_evidence (3-5 words). Avoid long sentences.
- Do NOT re-search the same concept. One search per dimension is enough.
- Score ALL 17 dimensions. Be efficient — goal: 2-3 turns per dimension.

Dimensions: has_volunteer, volunteer_hours_estimate, community_roles_count,
network_estimate, pre_empathy, pre_humble, pre_listen, pre_include,
pre_conflict, pre_lead_auth, listen, conflict, include, reflect,
pre_community_connected, has_speakhire, has_campus_role

When ALL 17 are scored, respond: {"done": true, "scored_count": 17}"""


AGENT4_VERIFIER_PROMPT = """You are an adversarial score verifier. Your job is to challenge
proposed scores by searching for counter-evidence the scorer may have missed.

For each proposed score:
1. Call search_evidence() with ALTERNATIVE queries the scorer might not have used.
2. Call get_rubric() to verify the score fits the cited rubric band.
3. If you find counter-evidence or rubric misapplication, challenge the score.
4. If the score is well-supported, uphold it.

Be skeptical. Assume the scorer may have overcounted or missed context.
Only uphold a score if the evidence clearly supports it."""


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Definitions (OpenAI/DeepSeek format)
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS_SCORER = [
    {
        "type": "function",
        "function": {
            "name": "search_evidence",
            "description": "Full-text search across all student documents for a concept. Call BEFORE making any claim about a student.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query. Use OR for alternatives."},
                    "source": {"type": "string", "enum": ["all", "resume", "linkedin", "github"]},
                    "section": {"type": "string", "description": "Optional restrict to section"},
                    "student_name": {"type": "string", "description": "Student name being processed"},
                },
                "required": ["query", "source", "student_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rubric",
            "description": "Return the exact scoring rubric for a dimension. ALWAYS call before scoring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pillar": {"type": "string", "enum": ["EC", "GC", "RFF", "CR", "CT", "CI"]},
                    "dimension": {"type": "string", "description": "e.g. has_volunteer, community_roles_count"},
                },
                "required": ["pillar", "dimension"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cite_evidence",
            "description": "Return the EXACT quote supporting or refuting a claim. Returns found=false if no evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The claim to verify"},
                    "source": {"type": "string", "enum": ["all", "resume", "linkedin", "github"]},
                    "student_name": {"type": "string", "description": "Student name"},
                },
                "required": ["claim", "source", "student_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_distinct",
            "description": "Count distinct instances of a pattern (roles, companies, activities, organizations).",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "enum": ["resume", "linkedin", "github"]},
                    "pattern": {"type": "string", "enum": ["roles", "companies", "volunteer_activities", "community_orgs", "certifications", "projects"]},
                    "student_name": {"type": "string"},
                },
                "required": ["source", "pattern", "student_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_score",
            "description": "Submit a proposed score for a dimension with mandatory evidence citations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pillar": {"type": "string"},
                    "dimension": {"type": "string"},
                    "proposed_score": {"type": "number", "description": "Numeric score"},
                    "rubric_band": {"type": "string", "description": "Which rubric band this matches"},
                    "evidence_quotes": {"type": "array", "items": {"type": "string"}, "description": "Exact quotes from documents"},
                    "reasoning": {"type": "string", "description": "Why this score based on evidence"},
                },
                "required": ["pillar", "dimension", "proposed_score", "rubric_band", "evidence_quotes", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_score",
            "description": "Verify a proposed score falls within plausible bounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pillar": {"type": "string"},
                    "field_name": {"type": "string"},
                    "proposed_score": {"type": "number"},
                    "evidence_summary": {"type": "string"},
                },
                "required": ["pillar", "field_name", "proposed_score", "evidence_summary"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# propose_score tool — captures proposed scores during agent loop
# ═══════════════════════════════════════════════════════════════════════════════

_proposed_scores: list[dict] = []

def propose_score(pillar: str, dimension: str, proposed_score,
                   rubric_band: str = "", evidence_quotes=None,
                   reasoning: str = "") -> dict:
    """Tool: Submit a proposed score with evidence."""
    _proposed_scores.append({
        "pillar": pillar,
        "dimension": dimension,
        "score": float(proposed_score),
        "rubric_band": rubric_band,
        "evidence": evidence_quotes or [],
        "reasoning": reasoning,
    })
    return {"accepted": True, "dimension": dimension, "score": float(proposed_score)}


# ═══════════════════════════════════════════════════════════════════════════════
# NEW Pipeline: Tool-Using Agent Approach
# ═══════════════════════════════════════════════════════════════════════════════

async def run_new_pipeline(student_name: str) -> dict:
    """Run the new tool-using agent pipeline on one student."""
    global _proposed_scores
    _proposed_scores = []
    reset_usage()

    slug = re.sub(r'[^a-z0-9]+', '_', student_name.lower()).strip('_')
    print(f"\n{'='*60}")
    print(f"[NEW PIPELINE] Processing: {student_name}")
    print(f"[NEW PIPELINE] Model: {OPENCODE_MODEL}")
    print(f"{'='*60}")

    start_time = time.time()

    # ── Step 1: Load documents ──────────────────────────────────
    print("\n[Step 1] Loading student documents...")
    student_dir = None
    old_student_data = os.path.join(OLD_ROOT, "student_data")
    for folder in os.listdir(old_student_data):
        if student_name.lower() in folder.lower():
            student_dir = os.path.join(old_student_data, folder)
            break

    docs_summary = {"resume": False, "linkedin": False, "github": False}
    resume_text = ""
    linkedin_text = ""

    if student_dir:
        for f in os.listdir(student_dir):
            if "resume" in f.lower() and f.endswith(".pdf"):
                from pdfminer.high_level import extract_text
                try:
                    resume_text = extract_text(os.path.join(student_dir, f))
                    docs_summary["resume"] = True
                except Exception:
                    pass
            if "linkedin" in f.lower() and f.endswith(".md"):
                with open(os.path.join(student_dir, f), encoding="utf-8") as fh:
                    linkedin_text = fh.read()
                    docs_summary["linkedin"] = True

    # ── Step 2: Run Scorer Agent ────────────────────────────────
    print("[Step 2] Running Scorer Agent (tool-using LLM)...")

    scorer_message = f"""Score ALL 17 GC dimensions for this student.

Student: {student_name}

Available documents:
- Resume: {'YES' if docs_summary['resume'] else 'NO'} ({len(resume_text)} chars)
- LinkedIn: {'YES' if docs_summary['linkedin'] else 'NO'} ({len(linkedin_text)} chars)

RESUME TEXT:
{resume_text[:4000] if resume_text else '(no resume available)'}

LINKEDIN TEXT:
{linkedin_text[:4000] if linkedin_text else '(no LinkedIn available)'}

For EACH of the 17 GC dimensions, call tools in this order:
1. get_rubric("GC", "<dimension>")
2. search_evidence(query="...", source="all", student_name="{student_name}")
3. cite_evidence(claim="...", source="all", student_name="{student_name}")
4. propose_score(...)

The 17 dimensions: {', '.join(ALL_GC_DIMENSIONS)}"""

    tool_executors = {
        "search_evidence": lambda **kw: search_evidence(**kw, student_name=student_name),
        "get_rubric": get_rubric,
        "cite_evidence": lambda **kw: cite_evidence(**kw, student_name=student_name),
        "count_distinct": lambda **kw: count_distinct(**kw, student_name=student_name),
        "propose_score": propose_score,
        "validate_score": validate_score,
    }

    scorer_result = await run_agent(
        system_prompt=AGENT4_SCORER_PROMPT,
        user_message=scorer_message,
        tools=TOOLS_SCORER,
        tool_executors=tool_executors,
        temperature=0.0,
        max_turns=100,
    )

    usage_after_scorer = get_usage()
    print(f"[Step 2] Scorer completed. Scores proposed: {len(_proposed_scores)}")

    # ── Step 3: Quick verification ──────────────────────────────
    # Verify all proposed scores by re-running search with inverse queries
    print("[Step 3] Running verification checks...")
    verification_notes = []

    for ps in _proposed_scores:
        counter_result = search_evidence(
            query=ps["dimension"].replace("_", " OR "),
            source="all",
            student_name=student_name,
        )
        matches_found = counter_result.get("total_count", 0)
        verification_notes.append({
            "dimension": ps["dimension"],
            "score": ps["score"],
            "matches_found_in_docs": matches_found,
            "verdict": "upheld" if (matches_found > 0 and ps["score"] > 0) or (matches_found == 0 and ps["score"] == 0) else "questionable",
        })

    # ── Step 4: Build output ────────────────────────────────────
    elapsed = round(time.time() - start_time, 1)

    scores_output = {}
    for ps in _proposed_scores:
        scores_output[ps["dimension"]] = {
            "score": ps["score"],
            "rubric_band": ps["rubric_band"],
            "evidence": ps["evidence"],
            "reasoning": ps["reasoning"],
        }

    output = {
        "student_name": student_name,
        "pipeline": "new (tool-using agent)",
        "model": OPENCODE_MODEL,
        "scores": scores_output,
        "proposed_count": len(_proposed_scores),
        "verification": verification_notes,
        "usage": usage_after_scorer,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Write output
    out_path = os.path.join(OUTPUTS_DIR, f"{slug}_new_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"[NEW PIPELINE] Done in {elapsed}s")
    print(f"[NEW PIPELINE] LLM calls: {usage_after_scorer['calls']}")
    print(f"[NEW PIPELINE] Tokens: {usage_after_scorer['tokens']}")
    print(f"[NEW PIPELINE] Scores proposed: {len(_proposed_scores)}")
    print(f"[NEW PIPELINE] Output: {out_path}")

    return output


# ═══════════════════════════════════════════════════════════════════════════════
# OLD Pipeline: Subprocess Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_old_pipeline(student_name: str) -> dict:
    """Run the old single-shot Gemini pipeline on one student."""
    import subprocess
    import pandas as pd

    slug = re.sub(r'[^a-z0-9]+', '_', student_name.lower()).strip('_')
    print(f"\n{'='*60}")
    print(f"[OLD PIPELINE] Processing: {student_name}")
    print(f"{'='*60}")

    start_time = time.time()

    # Read existing output files (they were pre-computed)
    old_outputs = os.path.join(OLD_ROOT, "agent4", "outputs")
    old_path = os.path.join(old_outputs, f"{slug}_enriched_scores.json")

    if os.path.exists(old_path):
        with open(old_path, encoding="utf-8") as f:
            data = json.load(f)

        elapsed = round(time.time() - start_time, 1)
        scores = data.get("scores", {})

        # Extract scores for comparison
        gc_scores = {}
        if "GC" in scores:
            gc_data = scores["GC"]
            if isinstance(gc_data, dict) and "sub_scores" in gc_data:
                gc_scores = gc_data["sub_scores"]

        print(f"[OLD PIPELINE] Loaded existing results in {elapsed}s")
        print(f"[OLD PIPELINE] GC score: {scores.get('GC', {}).get('score', 'N/A') if isinstance(scores.get('GC'), dict) else 'N/A'}")

        return {
            "student_name": student_name,
            "pipeline": "old (single-shot Gemini)",
            "scores": scores,
            "gc_sub_scores": gc_scores,
            "llm_calls_made": data.get("llm_calls_made", "unknown"),
            "elapsed_seconds": elapsed,
        }

    print(f"[OLD PIPELINE] No existing output found at {old_path}")
    return {"student_name": student_name, "pipeline": "old", "error": "No existing output"}


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def compare_results(old: dict, new: dict) -> dict:
    """Compare old and new pipeline results."""
    comparison = {
        "student": old.get("student_name"),
        "old_model": "Gemini Flash-Lite",
        "new_model": new.get("model"),
        "comparisons": [],
    }

    old_scores = old.get("gc_sub_scores", {})
    new_scores = new.get("scores", {})

    all_dims = sorted(set(list(old_scores.keys()) + list(new_scores.keys())))

    for dim in all_dims:
        old_val = old_scores.get(dim)
        new_val = new_scores.get(dim, {}).get("score") if isinstance(new_scores.get(dim), dict) else new_scores.get(dim)

        entry = {
            "dimension": dim,
            "old_score": old_val,
            "new_score": new_val,
        }

        if old_val is not None and new_val is not None:
            try:
                diff = abs(float(old_val) - float(new_val))
                entry["difference"] = round(diff, 2)
                if diff < 0.1:
                    entry["agreement"] = "identical"
                elif diff < 1.0:
                    entry["agreement"] = "close"
                elif diff < 3.0:
                    entry["agreement"] = "divergent"
                else:
                    entry["agreement"] = "strongly_divergent"
            except (TypeError, ValueError):
                entry["difference"] = "N/A (non-numeric)"
        else:
            entry["difference"] = "N/A (missing data)"

        comparison["comparisons"].append(entry)

    # Summary stats
    numeric_diffs = [
        c["difference"] for c in comparison["comparisons"]
        if isinstance(c.get("difference"), (int, float))
    ]

    comparison["summary"] = {
        "total_dimensions": len(all_dims),
        "comparable_dimensions": len(numeric_diffs),
        "avg_difference": round(sum(numeric_diffs) / len(numeric_diffs), 2) if numeric_diffs else 0,
        "identical_count": sum(1 for c in comparison["comparisons"] if c.get("agreement") == "identical"),
        "close_count": sum(1 for c in comparison["comparisons"] if c.get("agreement") == "close"),
        "divergent_count": sum(1 for c in comparison["comparisons"] if c.get("agreement") == "divergent"),
        "strongly_divergent_count": sum(1 for c in comparison["comparisons"] if c.get("agreement") == "strongly_divergent"),
        "old_llm_calls": old.get("llm_calls_made", "unknown"),
        "new_llm_calls": new.get("usage", {}).get("calls", 0),
        "old_elapsed_s": old.get("elapsed_seconds", "unknown"),
        "new_elapsed_s": new.get("elapsed_seconds", 0),
    }

    # Evidence quality comparison
    new_scores_with_evidence = sum(
        1 for v in new_scores.values()
        if isinstance(v, dict) and len(v.get("evidence", [])) > 0
    )
    comparison["evidence_quality"] = {
        "old_evidence_traceable": False,  # Old approach has no evidence tracking
        "new_scores_with_evidence": new_scores_with_evidence,
        "new_total_scores": len(new_scores),
        "new_evidence_rate": f"{new_scores_with_evidence}/{len(new_scores)}" if new_scores else "0/0",
    }

    return comparison


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Passport Agent V2 — Compare old vs new pipeline")
    parser.add_argument("--student", default="Abigail Rodriguez", help="Student name to process")
    parser.add_argument("--new-only", action="store_true", help="Only run new pipeline")
    parser.add_argument("--compare", action="store_true", help="Compare existing outputs without re-running")
    args = parser.parse_args()

    student = args.student
    slug = re.sub(r'[^a-z0-9]+', '_', student.lower()).strip('_')

    if args.compare:
        # Load existing outputs and compare
        old_path = os.path.join(OLD_ROOT, "agent4", "outputs", f"{slug}_enriched_scores.json")
        new_path = os.path.join(OUTPUTS_DIR, f"{slug}_new_scores.json")

        if not os.path.exists(old_path):
            print(f"Old output not found: {old_path}")
            sys.exit(1)
        if not os.path.exists(new_path):
            print(f"New output not found: {new_path}")
            sys.exit(1)

        with open(old_path) as f:
            old_data = json.load(f)
        with open(new_path) as f:
            new_data = json.load(f)

        # Convert old to comparison format
        old_result = {
            "student_name": student,
            "pipeline": "old",
            "scores": old_data.get("scores", {}),
            "gc_sub_scores": old_data.get("scores", {}).get("GC", {}).get("sub_scores", {}),
            "llm_calls_made": old_data.get("llm_calls_made", "unknown"),
        }

        comparison = compare_results(old_result, new_data)
        _print_comparison(comparison)
        return

    # ── Run pipelines ──────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# PASSPORT AGENT — Old vs New Pipeline Comparison")
    print(f"# Student: {student}")
    print(f"# Model: {OPENCODE_MODEL}")
    print(f"{'#'*60}")

    if not args.new_only:
        old_result = run_old_pipeline(student)
    else:
        old_result = {"student_name": student, "scores": {}, "gc_sub_scores": {}, "llm_calls_made": 0}

    new_result = await run_new_pipeline(student)

    # ── Compare ────────────────────────────────────────────────
    comparison = compare_results(old_result, new_result)

    comp_path = os.path.join(OUTPUTS_DIR, f"{slug}_comparison.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)

    _print_comparison(comparison)
    print(f"\nComparison saved to: {comp_path}")


def _print_comparison(c: dict):
    """Pretty-print the comparison."""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {c['student']}")
    print(f"  Old: Gemini Flash-Lite (single-shot)")
    print(f"  New: {c['new_model']} (tool-using agent)")
    print(f"{'='*70}")

    s = c["summary"]
    print(f"\nScore Agreement:")
    print(f"  Identical:      {s['identical_count']}")
    print(f"  Close (<1.0):   {s['close_count']}")
    print(f"  Divergent (<3.0): {s['divergent_count']}")
    print(f"  Strongly divergent: {s['strongly_divergent_count']}")
    print(f"  Avg difference: {s['avg_difference']}")

    print(f"\nPerformance:")
    print(f"  Old LLM calls: {s['old_llm_calls']}")
    print(f"  New LLM calls: {s['new_llm_calls']}")
    print(f"  Old elapsed:   {s['old_elapsed_s']}s")
    print(f"  New elapsed:   {s['new_elapsed_s']}s")

    eq = c["evidence_quality"]
    print(f"\nEvidence Traceability:")
    print(f"  Old: No evidence tracking (score is just a number)")
    print(f"  New: {eq['new_evidence_rate']} scores have cited evidence")

    print(f"\nDetailed Dimension Comparison:")
    print(f"  {'Dimension':<35} {'Old':>8} {'New':>8} {'Diff':>8} {'Agreement'}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*15}")
    for dim in c["comparisons"]:
        name = dim["dimension"][:34]
        old = f"{dim['old_score']}" if dim["old_score"] is not None else "N/A"
        new = f"{dim['new_score']}" if dim["new_score"] is not None else "N/A"
        diff = f"{dim['difference']}" if isinstance(dim.get('difference'), (int, float)) else "N/A"
        agree = dim.get("agreement", "N/A")
        print(f"  {name:<35} {old:>8} {new:>8} {diff:>8} {agree}")


if __name__ == "__main__":
    asyncio.run(main())
