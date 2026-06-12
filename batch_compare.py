"""Batch run: compare old vs new pipeline across multiple students."""
import asyncio
import json
import os
import re
import sys
import time
import traceback

# Add main pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import run_new_pipeline, run_old_pipeline, compare_results
from agents.base import OPENCODE_MODEL, reset_usage

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_OUTPUTS = os.path.join(HERE, "..", "Passport_Agent_Actual_Test_Final_1",
                           "Passport_Agent_Actual_Test", "Passport_Agent_Actual",
                           "agent4", "outputs")
NEW_OUTPUTS = os.path.join(HERE, "outputs")
os.makedirs(NEW_OUTPUTS, exist_ok=True)

# Test students — diverse data profiles
STUDENTS = [
    {
        "name": "Devin Rhodie",
        "profile": "Both LinkedIn + resume, strong profile",
    },
    {
        "name": "Benjamin Medrano",
        "profile": "Resume only, no LinkedIn — tests missing source",
    },
    {
        "name": "Yamin Titikpina",
        "profile": "LinkedIn only, no resume — tests single-source scoring",
    },
    {
        "name": "Bianka Pena",
        "profile": "Near-empty LinkedIn (0.2 KB) + resume — tests sparse data",
    },
    {
        "name": "Cristal Davidson",
        "profile": "Both LinkedIn + resume, moderate profile",
    },
]


def slugify(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


async def run_batch():
    batch_results = []
    batch_start = time.time()

    for i, student in enumerate(STUDENTS):
        name = student["name"]
        slug = slugify(name)
        print(f"\n{'#'*70}")
        print(f"# STUDENT {i+1}/{len(STUDENTS)}: {name}")
        print(f"# Profile: {student['profile']}")
        print(f"{'#'*70}")

        reset_usage()
        student_start = time.time()

        # ── Run new pipeline ──────────────────────────────────
        try:
            new_result = await run_new_pipeline(name)
        except Exception as e:
            print(f"[ERROR] New pipeline failed: {e}")
            traceback.print_exc()
            new_result = {"error": str(e), "scores": {}, "usage": {"calls": 0, "tokens": {}}}

        new_elapsed = round(time.time() - student_start, 1)

        # ── Load old results ──────────────────────────────────
        old_path = os.path.join(OLD_OUTPUTS, f"{slug}_enriched_scores.json")
        old_result = {"error": "No old output found"}
        if os.path.exists(old_path):
            with open(old_path, encoding="utf-8") as f:
                old_data = json.load(f)
            old_gc = old_data.get("scores", {}).get("GC", {})
            old_result = {
                "student_name": name,
                "gc_score": old_gc.get("score", "N/A") if isinstance(old_gc, dict) else "N/A",
                "gc_sub_scores": old_gc.get("sub_scores", {}) if isinstance(old_gc, dict) else {},
                "llm_calls": old_data.get("llm_calls_made", "?"),
                "all_pillars": {k: v.get("score", "?") if isinstance(v, dict) else v
                                for k, v in old_data.get("scores", {}).items()},
            }

        # ── Extract new GC scores ─────────────────────────────
        new_gc = new_result.get("scores", {})
        new_gc_summary = {}
        for dim, data in sorted(new_gc.items()):
            if isinstance(data, dict):
                new_gc_summary[dim] = {
                    "score": data.get("score"),
                    "evidence_count": len(data.get("evidence", [])),
                    "reasoning": (data.get("reasoning", "") or "")[:100],
                }
            else:
                new_gc_summary[dim] = {"score": data}

        # ── Evidence stats ────────────────────────────────────
        scores_with_evidence = sum(
            1 for v in new_gc.values()
            if isinstance(v, dict) and len(v.get("evidence", [])) > 0
        )
        total_evidence_quotes = sum(
            len(v.get("evidence", [])) for v in new_gc.values()
            if isinstance(v, dict)
        )

        # ── Assemble student result ───────────────────────────
        student_result = {
            "student": name,
            "profile": student["profile"],
            "new": {
                "model": new_result.get("model", OPENCODE_MODEL),
                "scores": new_gc_summary,
                "total_scores": len(new_gc),
                "scores_with_evidence": scores_with_evidence,
                "total_evidence_quotes": total_evidence_quotes,
                "llm_calls": new_result.get("usage", {}).get("calls", 0),
                "tokens": new_result.get("usage", {}).get("tokens", {}),
                "elapsed_s": new_elapsed,
                "error": new_result.get("error"),
            },
            "old": {
                "gc_score": old_result.get("gc_score", "N/A"),
                "gc_sub_scores": old_result.get("gc_sub_scores", {}),
                "all_pillars": old_result.get("all_pillars", {}),
                "llm_calls": old_result.get("llm_calls", "?"),
            },
        }
        batch_results.append(student_result)

        # Quick summary per student
        print(f"  New: {len(new_gc)} scores, {scores_with_evidence} with evidence, "
              f"{new_result.get('usage', {}).get('calls', 0)} calls, {new_elapsed}s")
        old_gc_score = old_result.get("gc_score", "N/A")
        print(f"  Old: GC={old_gc_score}, {old_result.get('llm_calls', '?')} calls")

    # ── BATCH SUMMARY ─────────────────────────────────────────
    batch_elapsed = round(time.time() - batch_start, 1)
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {len(STUDENTS)} students in {batch_elapsed}s")
    print(f"{'='*70}")

    # Compute aggregate stats
    total_new_calls = sum(r["new"]["llm_calls"] for r in batch_results)
    total_new_tokens_prompt = sum(r["new"]["tokens"].get("prompt", 0) for r in batch_results)
    total_new_tokens_completion = sum(r["new"]["tokens"].get("completion", 0) for r in batch_results)
    total_new_evidence = sum(r["new"]["total_evidence_quotes"] for r in batch_results)
    total_new_scores = sum(r["new"]["total_scores"] for r in batch_results)
    total_new_with_evidence = sum(r["new"]["scores_with_evidence"] for r in batch_results)

    summary = {
        "batch_size": len(STUDENTS),
        "total_elapsed_s": batch_elapsed,
        "new_model": OPENCODE_MODEL,
        "new_stats": {
            "total_llm_calls": total_new_calls,
            "avg_calls_per_student": round(total_new_calls / len(STUDENTS), 1),
            "total_prompt_tokens": total_new_tokens_prompt,
            "total_completion_tokens": total_new_tokens_completion,
            "total_scores_produced": total_new_scores,
            "total_with_evidence": total_new_with_evidence,
            "evidence_rate": f"{round(total_new_with_evidence/max(total_new_scores,1)*100)}%",
            "total_evidence_quotes": total_new_evidence,
            "cost": "$0.00 (OpenCode free tier)",
        },
        "old_stats": {
            "model": "Gemini Flash-Lite",
            "evidence_tracking": "IMPOSSIBLE — scores are opaque numbers",
            "estimated_cost_per_student": "~$0.003",
        },
        "students": batch_results,
    }

    # Write batch results
    batch_path = os.path.join(NEW_OUTPUTS, "batch_comparison.json")
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # Write human-readable log
    log_path = os.path.join(NEW_OUTPUTS, "batch_comparison_log.md")
    _write_log(summary, log_path)

    print(f"\nBatch results: {batch_path}")
    print(f"Comparison log: {log_path}")

    # Print evidence quality table
    print(f"\n{'Student':<25} {'Scores':>7} {'w/Evidence':>11} {'Quotes':>7} {'Calls':>6} {'Time':>7}")
    print(f"{'-'*25} {'-'*7} {'-'*11} {'-'*7} {'-'*6} {'-'*7}")
    for r in batch_results:
        name = r["student"][:24]
        scores = r["new"]["total_scores"]
        with_ev = r["new"]["scores_with_evidence"]
        quotes = r["new"]["total_evidence_quotes"]
        calls = r["new"]["llm_calls"]
        elapsed = f"{r['new']['elapsed_s']}s"
        print(f"{name:<25} {scores:>7} {with_ev:>9}/{scores} {quotes:>7} {calls:>6} {elapsed:>7}")

    return summary


def _write_log(summary, log_path):
    """Write a human-readable markdown comparison log."""
    lines = []
    lines.append("# Batch Pipeline Comparison Log")
    lines.append(f"\n**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Students:** {summary['batch_size']}")
    lines.append(f"**New Model:** {summary['new_model']}")
    lines.append(f"**Old Model:** {summary['old_stats']['model']}")
    lines.append(f"**Total time:** {summary['total_elapsed_s']}s")

    lines.append("\n## Summary Stats")
    lines.append(f"\n| Metric | Old (Gemini) | New (DeepSeek Free) |")
    lines.append(f"|--------|-------------|---------------------|")
    lines.append(f"| Avg calls/student | {summary['old_stats'].get('llm_calls_avg', '?')} | {summary['new_stats']['avg_calls_per_student']} |")
    lines.append(f"| Evidence traceable | ❌ Impossible | ✅ {summary['new_stats']['evidence_rate']} of scores |")
    lines.append(f"| Total evidence quotes | 0 | {summary['new_stats']['total_evidence_quotes']} |")
    lines.append(f"| Cost per student | {summary['old_stats']['estimated_cost_per_student']} | {summary['new_stats']['cost']} |")

    lines.append("\n## Per-Student Results")

    for i, r in enumerate(summary["students"]):
        name = r["student"]
        profile = r["profile"]
        new = r["new"]
        old = r["old"]

        lines.append(f"\n### {i+1}. {name}")
        lines.append(f"**Profile:** {profile}")
        lines.append(f"**Time:** {new['elapsed_s']}s | **Calls:** {new['llm_calls']} | "
                     f"**Evidence:** {new['scores_with_evidence']}/{new['total_scores']} scores")

        if new.get("error"):
            lines.append(f"\n⚠️ **Error:** {new['error']}")
            continue

        lines.append(f"\n| Dimension | New Score | Evidence | Old GC Sub-Score |")
        lines.append(f"|-----------|-----------|----------|-----------------|")

        for dim, data in sorted(new["scores"].items()):
            score = data.get("score", "?")
            ev_count = data.get("evidence_count", 0)
            ev_mark = f"{ev_count} quotes" if ev_count > 0 else "none"
            old_sub = old.get("gc_sub_scores", {})
            # Old has different dimension names, just show all old subs
            old_val = "N/A"
            lines.append(f"| {dim} | {score} | {ev_mark} | {old_val} |")

        # Show old aggregated scores
        lines.append(f"\n**Old aggregated scores:**")
        for pillar, score in old.get("all_pillars", {}).items():
            lines.append(f"- {pillar}: {score}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(run_batch())
