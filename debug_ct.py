"""Debug CT/CI LLM calls — test with larger max_tokens"""
from agents.base import _call_api
import json, re

# Test with larger max_tokens — DeepSeek V4 uses reasoning tokens
for max_tok in [200, 500, 1000, 2000]:
    result = _call_api(
        messages=[{"role": "user", "content": 'Say hello in 3 words'}],
        temperature=0.0, max_tokens=max_tok
    )
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = result.get("usage", {})
    reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    print(f"max_tokens={max_tok}: content={repr(content[:80])} reasoning_tokens={reasoning} total={usage.get('completion_tokens',0)}")
