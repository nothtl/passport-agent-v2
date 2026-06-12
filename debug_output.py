"""Quick debug of pipeline output"""
import json

with open("outputs/students/abigail_rodriguez.json") as f:
    data = json.load(f)

# Check GC
gc = data["scores"]["GC"]
print("GC score:", gc.get("score"))
print("GC dimensions scored:", gc.get("dimensions_scored"))
print("GC subs sample:")
for k, v in sorted(gc.get("sub_scores", {}).items())[:5]:
    print(f"  {k}: {v}")

# Check CT/CI
ct = data["scores"]["CT"]
ci = data["scores"]["CI"]
print(f"\nCT: score={ct['score']}")
print(f"CT arc: {ct.get('thinking_arc', '')[:120]}")
print(f"CI: score={ci['score']}")
print(f"CI arc: {ci.get('innovation_arc', '')[:120]}")

# Check EC
ec = data["scores"]["EC"]
print(f"\nEC score: {ec['score']}")
print(f"EC subs: {ec.get('sub_scores', {})}")

# Check enriched
print(f"\nEnriched: {data.get('fields_enriched', 0)}")
print(f"Still missing: {data.get('fields_still_missing', 0)}")
