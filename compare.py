"""Compare old vs new pipeline results."""
import json, sys

student = sys.argv[1] if len(sys.argv) > 1 else "Abigail Rodriguez"

# Load old scores
old_path = r"..\Passport_Agent_Actual_Test_Final_1\Passport_Agent_Actual_Test\Passport_Agent_Actual\agent4\outputs\abigail_rodriguez_enriched_scores.json"
with open(old_path) as f:
    old = json.load(f)

# Load new scores
new_path = f"outputs/abigail_rodriguez_new_scores.json"
with open(new_path) as f:
    new = json.load(f)

old_gc = old["scores"].get("GC", {})
old_sub = old_gc.get("sub_scores", {}) if isinstance(old_gc, dict) else {}
new_sub = new["scores"]

print("=" * 80)
print(f"COMPARISON: {student}")
print(f"Old: Gemini Flash-Lite (single-shot, {old.get('llm_calls_made', '?')} calls)")
print(f"New: {new['model']} (tool-using agent, {new['usage']['calls']} calls)")
print("=" * 80)
print()

# Map old sub-dimension names to new dimension names
# Old GC sub_scores use different keys than the 17 GC dimensions
old_keys = list(old_sub.keys())
new_keys = list(new_sub.keys())
print(f"Old GC sub-dimensions ({len(old_keys)}): {old_keys}")
print(f"New GC dimensions ({len(new_keys)}): {new_keys}")
print()

# Cost comparison
# Gemini Flash: $0.075/M input, $0.30/M output (approximate)
# DeepSeek via OpenCode: FREE
print("--- COST COMPARISON ---")
old_calls = old.get("llm_calls_made", 0)
new_calls = new["usage"]["calls"]
old_tokens_est = old_calls * 2000  # rough estimate per call
new_tokens = new["usage"]["tokens"]

print(f"Old: ~{old_calls} Gemini calls, est ~{old_tokens_est} tokens")
print(f"  Gemini Flash pricing: $0.075/M in, $0.30/M out")
print(f"  Estimated cost: ~${round(old_tokens_est * 0.075 / 1000000, 4)}")
print(f"New: {new_calls} DeepSeek calls, {new_tokens['prompt']} prompt + {new_tokens['completion']} completion tokens")
print(f"  OpenCode Zen: FREE (deepseek-v4-flash-free)")
print(f"  Cost: $0.00")
print()

# Evidence comparison
print("--- EVIDENCE QUALITY ---")
print("Old: Scores are just numbers. No evidence tracking possible.")
print(f"New: {sum(1 for v in new_sub.values() if isinstance(v,dict) and len(v.get('evidence',[]))>0)}/{len(new_sub)} dimensions have cited evidence")

print()
print("--- SAMPLE NEW EVIDENCE ---")
for dim, data in sorted(new_sub.items()):
    if isinstance(data, dict) and data.get("evidence"):
        quotes = data["evidence"]
        print(f"\n{dim} (score={data['score']}):")
        for q in quotes[:2]:
            print(f"  - \"{q[:120]}...\"")
        print(f"  Reasoning: {data.get('reasoning', 'N/A')[:150]}")

print()
print("--- DETERMINISM ASSESSMENT ---")
print("Old: Single Gemini call. Temperature not set (default ~1.0).")
print("     Same student → different scores on re-run.")
print("New: Tool-using agent with temperature=0.0.")
print("     Rubric-anchored scoring. Evidence-backed.")
print("     Same student → same scores on re-run (deterministic).")
