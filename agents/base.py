"""
Core tool-use agent loop using requests directly (OpenCode Zen API).

OpenCode uses x-api-key header, not Bearer. The OpenAI SDK forces Bearer,
so we use requests directly for this endpoint.
"""

import json
import os
import time
from typing import Any, Callable
import requests

# ── Load config from .env ─────────────────────────────────────────────────
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

_load_env()

OPENCODE_API_KEY = os.environ.get("OPENCODE_API_KEY", "")
OPENCODE_BASE_URL = os.environ.get("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
OPENCODE_MODEL = os.environ.get("OPENCODE_MODEL", "deepseek-v4-flash-free")

# Track usage
_total_llm_calls = 0
_total_tokens = {"prompt": 0, "completion": 0}


def get_usage():
    return {"calls": _total_llm_calls, "tokens": dict(_total_tokens)}


def reset_usage():
    global _total_llm_calls, _total_tokens
    _total_llm_calls = 0
    _total_tokens = {"prompt": 0, "completion": 0}


def _call_api(messages: list[dict], tools: list[dict] = None,
              temperature: float = 0.0, max_tokens: int = 4096,
              model: str = None) -> dict:
    """Make a single API call to OpenCode Zen. Returns response JSON."""
    global _total_llm_calls, _total_tokens

    url = f"{OPENCODE_BASE_URL}/chat/completions"
    headers = {
        "x-api-key": OPENCODE_API_KEY,
        "Content-Type": "application/json",
    }

    payload = {
        "model": model or OPENCODE_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    delays = [2, 4, 8]
    for attempt in range(3):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                _total_llm_calls += 1
                data = r.json()
                if "usage" in data:
                    _total_tokens["prompt"] += data["usage"].get("prompt_tokens", 0)
                    _total_tokens["completion"] += data["usage"].get("completion_tokens", 0)
                return data
            elif r.status_code == 429:
                time.sleep(delays[min(attempt, len(delays) - 1)])
            else:
                time.sleep(delays[min(attempt, len(delays) - 1)])
        except requests.RequestException:
            time.sleep(delays[min(attempt, len(delays) - 1)])

    return {"error": f"API call failed after 3 retries"}


def _extract_tool_calls(response: dict) -> list[dict]:
    """Extract tool calls from API response."""
    choice = response.get("choices", [{}])[0]
    msg = choice.get("message", {})
    return msg.get("tool_calls", [])


def _extract_content(response: dict) -> str:
    """Extract text content from API response."""
    choice = response.get("choices", [{}])[0]
    msg = choice.get("message", {})
    return msg.get("content", "") or ""


async def run_agent(
    *,
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    tool_executors: dict[str, Callable],
    temperature: float = 0.0,
    max_turns: int = 30,
    model_override: str = None,
) -> dict | str:
    """
    Run a tool-use agent loop.

    The LLM receives a task, calls tools as needed, and returns
    a final message when done.
    """
    model_name = model_override or OPENCODE_MODEL

    if not OPENCODE_API_KEY:
        return {"error": "No API key set. Set OPENCODE_API_KEY in .env"}

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for turn in range(max_turns):
        response = _call_api(
            messages=messages,
            tools=tools if tools else None,
            temperature=temperature,
            model=model_name,
        )

        if "error" in response:
            return {"error": response["error"]}

        tool_calls = _extract_tool_calls(response)
        content = _extract_content(response)

        # ── No tool calls — agent is done ─────────────────
        if not tool_calls:
            # Try to parse as JSON
            try:
                return json.loads(content)
            except (json.JSONDecodeError, TypeError):
                if "```json" in content:
                    try:
                        json_str = content.split("```json")[1].split("```")[0]
                        return json.loads(json_str)
                    except (json.JSONDecodeError, IndexError):
                        pass
                if "```" in content:
                    try:
                        json_str = content.split("```")[1].split("```")[0]
                        return json.loads(json_str)
                    except (json.JSONDecodeError, IndexError):
                        pass
                return content

        # ── Execute tool calls ─────────────────────────────
        # Append assistant message
        assistant_msg = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
        messages.append(assistant_msg)

        # Run each tool
        for tc in tool_calls:
            func_name = tc.get("function", {}).get("name", "")
            try:
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
            except json.JSONDecodeError:
                result = {"error": "Invalid JSON arguments"}
            else:
                executor = tool_executors.get(func_name)
                if executor is None:
                    result = {"error": f"Unknown tool: {func_name}"}
                else:
                    try:
                        result = executor(**args)
                    except Exception as e:
                        result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    return {"error": f"Agent exceeded max_turns ({max_turns})"}


async def run_agent_with_fallback(
    *,
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    tool_executors: dict[str, Callable],
    temperature: float = 0.0,
    max_turns: int = 30,
    model_override: str = None,
) -> dict | str:
    """Run agent with fallback (currently single-model)."""
    return await run_agent(
        system_prompt=system_prompt,
        user_message=user_message,
        tools=tools,
        tool_executors=tool_executors,
        temperature=temperature,
        max_turns=max_turns,
        model_override=model_override,
    )
