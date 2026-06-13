# Pipeline Implementation: Framework, Architecture & Tools

This document details the actual approach — what framework to use, how the code is structured, what tools each agent calls, and how the pieces fit together.

For a comparison of old vs new approaches, see `pipeline_comparison.md`.

---

## 1. Framework: What We Use and Why

### What We Install

```
pip install openai pydantic
```

Everything else is standard library (`asyncio`, `json`, `subprocess`) or existing dependencies (`pdfminer.six`, `jinja2`, `pandas`, `requests`).

### Why Not a Real Agent Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  WHAT WE NEED                          WHAT AGENT FRAMEWORKS PROVIDE        │
│  ────────────                          ──────────────────────────           │
│                                                                             │
│  • Fixed execution DAG                 • Dynamic task decomposition         │
│    (A1 → A2∥A3 → A4 → A5)               (LLM decides WHAT to do)           │
│                                                                             │
│  • Tool-use loops at specific          • Multi-agent coordination           │
│    points in the DAG                     (agents spawning agents)           │
│                                                                             │
│  • Structured data passing             • Conversation history management    │
│    between steps (Pydantic)              between agents                     │
│                                                                             │
│  • LLM calls tools autonomously        • LLM calls tools autonomously       │
│    within a bounded task                 within an open-ended task          │
│                                                                             │
│  Our needs are a SUBSET of what        Frameworks add abstraction for       │
│  frameworks provide. The tool-use       complexity we don't have.           │
│  API IS the framework.                  They'd add cost, not value.         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Framework | What It Adds | Why We Skip It |
|-----------|-------------|----------------|
| **LangChain** | Chains, retrievers, prompt templates, vector stores | Abstraction tax. Our DAG is 5 steps — chains add indirection, not clarity. Debugging tool calls through LangChain's abstraction is painful. |
| **LangGraph** | State-machine agent graphs with cycles | Built for complex agent topologies. Our DAG has no cycles, no conditional branching, no dynamic routing. A state machine for `A1→A2∥A3→A4→A5` is overkill. |
| **CrewAI** | Multi-agent role-playing with inter-agent chat | Agents don't chat — they pass structured JSON. The Scorer doesn't "discuss" with the Verifier in natural language; it submits proposals that get programmatically challenged. |
| **AutoGen** | Microsoft's multi-agent framework | Enterprise complexity. `ConversableAgent`, `GroupChat`, `AssistantAgent` — for a pipeline with 3 tool-using LLM calls and a monitor, this is 10× the abstraction we need. |
| **Smolagents** | HuggingFace lightweight agent wrapper | Closest to what we need, but still adds an `Agent` class, a `Tool` registry, and framework-specific conventions. Our tool-use loop is 30 lines — the wrapper is larger than the code it wraps. |

### What We Actually Build

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                          OUR STACK (3 layers)                                │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 1: Model API                                                    │ │
│  │                                                                       │ │
│  │ openai.OpenAI(base_url="https://api.deepseek.com")                    │ │
│  │                                                                       │ │
│  │ Provides:                                                             │ │
│  │ • client.chat.completions.create(model, messages, tools, temperature) │ │
│  │ • Model returns: text OR tool_calls                                   │ │
│  │ • You execute tools, send results back, model continues               │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 2: Tool-Use Loop (30 lines — we write this once)                │ │
│  │                                                                       │ │
│  │ async def run_agent(system_prompt, user_message, tools, executors):   │ │
│  │     messages = [system, user]                                          │ │
│  │     while True:                                                       │ │
│  │         response = client.chat.completions.create(...)                │ │
│  │         if not response.choices[0].message.tool_calls:                │ │
│  │             return response.choices[0].message.content  # done       │ │
│  │         for tool_call in response...tool_calls:                       │ │
│  │             result = executors[tool_call.name](**tool_call.args)      │ │
│  │             messages.append(tool_result)                              │ │
│  │         # loop continues — model sees result, may call more tools     │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ LAYER 3: Pipeline Orchestrator + Monitor (plain Python)               │ │
│  │                                                                       │ │
│  │ async def run_student(name):                                          │ │
│  │     raw    = await agent1(name, data_zip)                             │ │
│  │     survey, docs = await asyncio.gather(                              │ │
│  │         run_agent("You are a survey scorer...", raw, SCORER_TOOLS),   │ │
│  │         run_agent("You are a doc parser...", raw, PARSER_TOOLS)       │ │
│  │     )                                                                 │ │
│  │     scores = await adversarial_enrichment(survey, docs)               │ │
│  │     html   = await render(scores)                                     │ │
│  │     return html                                                       │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. DeepSeek SDK: What It Actually Is

DeepSeek does not have a custom agent SDK. What exists:

### Option A: OpenAI SDK pointed at DeepSeek (recommended)

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="https://api.deepseek.com"
)

# Tool calling works identically to OpenAI:
response = client.chat.completions.create(
    model="deepseek-v4-pro",          # or deepseek-v4-flash
    messages=[...],
    tools=[...],                      # Same JSON Schema format as OpenAI
    tool_choice="auto",               # "auto", "none", "required", or specific
    temperature=0.0,                  # Deterministic
    max_tokens=4096
)
```

### Option B: `deepseek-sdk` (official, thin wrapper)

```bash
pip install deepseek-sdk
```

```python
from deepseek import Client
client = Client()  # reads DEEPSEEK_API_KEY from env
```

Adds no capability beyond the OpenAI SDK. Exists for branding. Skip it.

### Option C: Anthropic-compatible endpoint (beta)

```
https://api.deepseek.com/anthropic
```

Translates Anthropic tool_use blocks to DeepSeek's internal format. Beta quality. Translation layer adds latency and potential bugs. Only use if you have Anthropic SDK code you don't want to rewrite.

### Verdict: Use Option A

```python
pip install openai
```

One dependency. Battle-tested. Every example, library, and StackOverflow answer about OpenAI tool calling applies directly.

### DeepSeek Key Capabilities

| Capability | Supported? | Notes |
|-----------|:---:|-------|
| Tool/Function calling | ✅ | Up to 128 tools per request |
| Parallel tool calls | ✅ | Model can call multiple tools in one response |
| `tool_choice: "auto"` | ✅ | Model decides whether to call a tool |
| `tool_choice: "required"` | ✅ | Force the model to use a tool |
| `tool_choice: {type, function: {name}}` | ✅ | Force a specific tool |
| JSON mode | ✅ | `response_format={'type': 'json_object'}` |
| Strict schema enforcement | ❌ | No server-side schema guarantee. Use Pydantic client-side. |
| Streaming + tools | ✅ | Stream tokens, receive tool_calls when they arrive |
| Thinking/reasoning | ✅ | V4 Pro supports extended thinking before tool calls |
| Context window | ✅ 1M tokens | Fits all student documents at once |
| Temperature control | ✅ | 0.0 to 2.0 |

### DeepSeek Limitations to Plan For

| Limitation | Mitigation |
|-----------|-----------|
| No server-side schema enforcement | Pydantic validates every tool call result client-side |
| API availability can vary | Multi-model fallback: DeepSeek → Claude → GPT |
| Rate limits may change | Orchestrator retries with exponential backoff |
| Tool calling degrades on long chains (15+ turns) | Our longest chain is ~8 turns per phase |
| Text-only (no vision) | Not needed — our inputs are PDF text, markdown, JSON |

---

## 3. Complete Tool Catalog

### Agent 1 — Discovery Tools

```
explore_zip()
├── Description: List all files and sheets in the survey data zip
├── Parameters: {zip_path: string}
├── Returns: {files: [{name, sheets[], row_count, columns[]}]}
└── Replaces: Hardcoded FILE_PRIORITY list

fuzzy_find_student()
├── Description: Search ALL files/sheets for a student name with fuzzy matching
├── Parameters: {name: string, threshold: float (0.0–1.0)}
├── Returns: {matches: [{filename, sheet, row_index, matched_name, score, email}]}
└── Replaces: Exact name matching in student_finder.py

semantic_column_search()
├── Description: Find columns by meaning, not exact header text
├── Parameters: {file: string, sheet: string, concept: string}
├── Example concepts: "english proficiency", "volunteer hours", "career goal"
├── Returns: {columns: [{name, confidence, sample_values[]}]}
└── Replaces: Hardcoded column_variants in field_registry.json

extract_row_field()
├── Description: Extract a specific field value from a matched row
├── Parameters: {filename: string, sheet: string, row_index: int, column: string}
├── Returns: {value, data_type, is_null}
└── Used by: Agent to pull values after finding the right column

resolve_identity()
├── Description: When multiple rows match a name, disambiguate by email/school/cohort
├── Parameters: {candidates: [{filename, sheet, row_index, fields{}}]}
├── Returns: {resolved_row, confidence, disambiguation_reason}
└── Replaces: Implicit first-match-wins behavior
```

### Agent 2 — Scoring Tools

```
get_scoring_rubric()
├── Description: Return the exact rubric for a text-response scoring dimension
├── Parameters: {pillar: "EC"|"GC"|"RFF"|"CR", field_name: string}
├── Returns: {rubric_text, score_bands: [{range, description, examples[]}], max, min}
└── Use: Called BEFORE scoring — model sees the actual rubric, not a prompt summary

lookup_reference_examples()
├── Description: Retrieve pre-scored reference examples from a calibrated bank
├── Parameters: {pillar: string, field_name: string, score_range: "low"|"mid"|"high"}
├── Returns: {examples: [{text, score, scorer_notes, why_this_score}]}
└── Use: Model compares student response to calibrated examples before scoring

check_consistency()
├── Description: Flag contradictions between self-reported fields
├── Parameters: {field_a: {name, value}, field_b: {name, value}}
├── Returns: {consistent: bool, contradiction: string|null}
└── Example: English="Not comfortable" vs eloquent writing → flags inconsistency

validate_score()
├── Description: Verify a proposed score falls within plausible bounds
├── Parameters: {pillar, field_name, proposed_score: float, evidence_summary: string}
├── Returns: {valid: bool, warning: string|null, suggested_range: [min, max]}
└── Use: Sanity check before finalizing
```

### Agent 3 — Document Navigation Tools

```
extract_pdf_text()
├── Description: Extract raw text from a PDF file with page granularity
├── Parameters: {file_path: string, pages: "all"|[start, end]}
├── Returns: {total_pages, pages: [{page_num, text}]}
└── Replaces: Single pdfminer call with no page awareness

identify_resume_structure()
├── Description: Detect the resume's layout and section headers
├── Parameters: {raw_text: string}
├── Returns: {format: "single_column"|"two_column"|"ats"|"creative",
│             sections: [{header, start_line, end_line, confidence}]}
└── Replaces: Single-shot LLM section extraction

extract_section_text()
├── Description: Pull exact text for a named section using identified boundaries
├── Parameters: {raw_text: string, section_name: string, start_line: int, end_line: int}
├── Returns: {section_text, line_range}
└── Use: Agent calls this AFTER identify_resume_structure for precise extraction

parse_linkedin_markdown()
├── Description: Parse LinkedIn markdown export into structured data
├── Parameters: {file_path: string}
├── Returns: {name, headline, about, experience[], education[], skills{}, projects[], certs[]}
└── Replaces: Regex-only parser. Adds LLM fallback for unusual formats.

find_github_username()
├── Description: Search resume + LinkedIn text for GitHub profile references
├── Parameters: {resume_text: string, linkedin_data: object}
├── Returns: {username: string|null, source: "resume"|"linkedin"|null, confidence: float}
└── Handles: "github.com/x", "GitHub: @x", "gh: x", "github/x"

scrape_github_profile()
├── Description: Fetch GitHub profile, repos, READMEs, languages via REST API
├── Parameters: {username: string}
├── Returns: {profile: {bio, public_repos, followers},
│             repos: [{name, description, languages[], readme, stars, forks}]}
└── Same logic as current github_scraper.py, exposed as a tool
```

### Agent 4 — Enrichment & Verification Tools (highest impact)

```
search_evidence() ⭐
├── Description: Full-text search across ALL student documents for a concept
├── Parameters: {
│     query: string,           // "volunteer OR community service OR nonprofit"
│     source: "all"|"resume"|"linkedin"|"github",
│     section: string|null     // restrict to "experience", "leadership", etc.
│   }
├── Returns: {matches: [{
│     source, section, matched_text, surrounding_context,
│     match_type: "direct"|"semantic"|"inferred"
│   }], total_count}
└── Must call before claiming any fact about a student

get_rubric() ⭐
├── Description: Return the exact scoring rubric for any dimension
├── Parameters: {pillar: "EC"|"GC"|"RFF"|"CR"|"CT"|"CI", dimension: string}
├── Returns: {
│     dimension_name, description,
│     bands: [{score_range, label, criteria, examples[]}],
│     evidence_required: "none"|"single_signal"|"multiple_signals"|"sustained"
│   }
└── Model reads rubric, searches for evidence matching each band

cite_evidence() ⭐
├── Description: Return the EXACT quote supporting a claim, or report not found
├── Parameters: {claim: string, source: string}
├── Returns: {
│     found: bool,
│     exact_quote: string|null,
│     location: {source, section, line_range}|null,
│     strength: "direct"|"indirect"|"none"
│   }
└── Every score must have at least one cited quote

count_distinct()
├── Description: Count distinct instances of a pattern (roles, companies, activities)
├── Parameters: {source: string, pattern: "roles"|"companies"|"volunteer_activities"|
│                "community_orgs"|"certifications"|"projects"}
├── Returns: {count: int, items: [{name, context}]}
└── Replaces: Hardcoded counting like len(distinct_companies)

propose_score() ⭐
├── Description: Submit a proposed score with mandatory evidence citations
├── Parameters: {
│     pillar, dimension, proposed_score: float,
│     rubric_band_matched: string,  // which band and why
│     evidence_quotes: [{exact_quote, source, section, relevance}],
│     reasoning: string
│   }
├── Returns: {accepted: bool, adjusted_score: float|null, feedback: string|null}
└── Scoring agent submits. Verifier reviews.

challenge_score() ⭐
├── Description: Adversarially challenge a proposed score
├── Parameters: {proposed_score, dimension, pillar,
│                evidence_quotes: [{exact_quote, source, relevance}],
│                reasoning: string}
├── Returns: {
│     challenge_upheld: bool,
│     reasons: [string],
│     counter_evidence: [{quote, source}],
│     suggested_correction: float|null
│   }
└── Verifier agent uses this. Only upholds with concrete counter-evidence.

resolve_dispute()
├── Description: When scorer and challenger disagree, resolve with final decision
├── Parameters: {proposal, challenge, rubric}
├── Returns: {final_score, resolution_reasoning, accepted_evidence[]}
└── Arbiter agent uses this.

generate_narrative()
├── Description: Generate 1-sentence passport narrative from verified scores + evidence
├── Parameters: {
│     pillar, student_name,
│     top_evidence: [{quote, source}],
│     score_context: string,
│     forbidden_terms: [string]  // from prior narratives to avoid repetition
│   }
├── Returns: {narrative: string, cites_evidence: bool}
└── Replaces single-shot pillar_reasoner.py prompt
```

### Monitor Agent — Tools

```
inspect_score()
├── Description: Pull up full evidence trail, rubric mapping, and verifier notes
├── Parameters: {student: string, pillar: string, dimension: string}
├── Returns: {score, evidence_quotes[], rubric_band, verifier_notes, dispute_history}

compare_to_cohort()
├── Description: Compare a student's scores against cohort distribution
├── Parameters: {student: string, pillar: string}
├── Returns: {student_score, cohort_mean, cohort_std, percentile, is_outlier}

query_logs()
├── Description: Search across all pipeline run events
├── Parameters: {pattern: string, student: string|null, agent: string|null}
├── Returns: {matches: [{event, timestamp, student, agent, data}]}

decide_action()
├── Description: Issue a decision about what to do with an anomaly
├── Parameters: {student, agent,
│     action: "CONTINUE"|"RETRY"|"SKIP_STUDENT"|"SKIP_PILLAR"|
│              "FLAG_FOR_REVIEW"|"ESCALATE"|"REDUCE_CONFIDENCE",
│     reason: string, context: object}
├── Returns: {recorded: bool, decision_id: string}
└── Records decision. If ESCALATE, pauses the batch.

adjust_pipeline()
├── Description: Change pipeline configuration for remaining students
├── Parameters: {
│     setting: "temperature"|"max_retries"|"model_override"|"skip_agent"|"pause_batch",
│     value, reason: string
│   }
├── Returns: {applied: bool, setting, value}
└── Applies to all subsequent students in the batch
```

---

## 4. Code Structure

```
passport-agent/
│
├── requirements.txt              # openai, pydantic, pdfminer.six, jinja2, pandas
│
├── orchestrator.py               # Fixed DAG runner + event emitter
├── monitor.py                    # Monitor Agent (LLM on-demand)
│
├── tools/                        # Tool implementations (plain Python)
│   ├── __init__.py
│   ├── evidence.py               # search_evidence, cite_evidence, count_distinct
│   ├── rubric.py                 # get_rubric, lookup_reference_examples
│   ├── documents.py              # extract_pdf_text, identify_structure, extract_section,
│   │                               parse_linkedin, find_github_user, scrape_github
│   ├── discovery.py              # explore_zip, fuzzy_find_student, semantic_col_search,
│   │                               extract_row_field, resolve_identity
│   ├── validation.py             # check_consistency, validate_score
│   └── pipeline_control.py       # inspect_score, compare_to_cohort, query_logs,
│                                   decide_action, adjust_pipeline (monitor tools)
│
├── agents/                       # System prompts + tool lists for each step
│   ├── __init__.py
│   ├── base.py                   # run_agent() — the 30-line tool-use loop
│   ├── agent1_discovery.py       # System prompt for Agent 1
│   ├── agent2_scorer.py          # System prompt for Agent 2
│   ├── agent3_parser.py          # System prompt for Agent 3
│   ├── agent4_scorer.py          # System prompt for Scorer phase
│   ├── agent4_verifier.py        # System prompt for Verifier phase
│   ├── agent4_arbiter.py         # System prompt for Arbiter phase
│   └── monitor.py                # System prompt for Monitor Agent
│
├── models/                       # Pydantic data contracts
│   ├── __init__.py
│   ├── raw_data.py               # RawDataOutput (from Agent 1)
│   ├── survey_scores.py          # SurveyScoresOutput (from Agent 2)
│   ├── parsed_docs.py            # ParsedDocsOutput (from Agent 3)
│   ├── enriched_scores.py        # EnrichedScoresOutput (from Agent 4)
│   ├── events.py                 # PipelineEvent types
│   └── decisions.py              # MonitorDecision types
│
├── rubric_db/                    # Structured rubric database
│   ├── ec_rubrics.json
│   ├── gc_rubrics.json
│   ├── rff_rubrics.json
│   ├── cr_rubrics.json
│   ├── ct_rubric.json
│   └── ci_rubric.json
│
├── reference_examples/           # Pre-scored calibration examples
│   ├── ec_examples.json
│   ├── rff_examples.json
│   └── gc_examples.json
│
├── templates/                    # Jinja2 passport template (unchanged)
│   └── passport.html.jinja
│
├── student_data/                 # Student folders (unchanged)
│   └── {Student Name}/
│       ├── {Name} - Resume.pdf
│       └── {Name} - LinkedIn.md
│
├── Full_speakhire_data.zip       # Survey data (unchanged)
│
└── outputs/                      # Pipeline output per student
    └── {slug}/
        ├── raw_data.json
        ├── survey_scores.json
        ├── parsed_docs.json
        ├── enriched_scores.json
        └── passport.html
```

### The Tool-Use Loop (`agents/base.py`)

This is the core engine. Used by every agent. Written once.

```python
import json
from typing import Any, Callable
from openai import OpenAI
from models.events import PipelineEvent, EventType

ToolExecutor = Callable[..., dict]

async def run_agent(
    *,
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    tool_executors: dict[str, ToolExecutor],
    temperature: float = 0.0,
    max_turns: int = 30,
    on_event: Callable[[PipelineEvent], None] | None = None,
) -> dict | str:
    """
    Run a tool-use agent loop.

    The LLM receives a task, calls tools as needed, and returns
    a final message when done. The loop handles tool execution
    and result formatting.

    Returns:
        - Parsed JSON dict if the final message is valid JSON
        - Raw string otherwise
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=4096,
        )

        msg = response.choices[0].message

        # ── No more tool calls — agent is done ─────────────────
        if not msg.tool_calls:
            # Try to parse as JSON, fall back to raw text
            try:
                return json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                return msg.content

        # ── Execute tool calls ─────────────────────────────────
        # Append assistant message with tool_calls
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        # Run each tool, collect results
        for tc in msg.tool_calls:
            func_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                result = {"error": f"Invalid JSON arguments: {tc.function.arguments}"}
            else:
                try:
                    result = tool_executors[func_name](**args)
                except Exception as e:
                    result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

        # Loop continues — model sees tool results, may call more tools

    raise RuntimeError(f"Agent exceeded max_turns ({max_turns}) without completing")
```

### Example Agent Definition (`agents/agent4_scorer.py`)

```python
AGENT4_SCORER_SYSTEM_PROMPT = """You are a rigorous student competency scorer for
SPEAKHIRE PathCredits. Your scores affect real students' professional profiles.

RULES:
1. NEVER score without calling get_rubric() first
2. NEVER claim a fact without calling search_evidence() AND cite_evidence()
3. If cite_evidence() returns found=false, you MUST score at the minimum
4. If evidence is ambiguous, score LOWER — never inflate
5. Every propose_score() MUST include at least one exact_quote from cite_evidence()
6. Score dimensions independently — you may call tools in parallel for speed
7. Work through all dimensions systematically before sending your final response

When you have scored ALL dimensions, respond with a JSON summary:
{"pillar": "GC", "scores_proposed": N, "dimensions_scored": [...]}
"""

AGENT4_SCORER_TOOLS = [
    # Tool schemas in OpenAI/DeepSeek format
    {
        "type": "function",
        "function": {
            "name": "search_evidence",
            "description": "Full-text search across all student documents for a concept or keyword. Call BEFORE making any claim about what a student did.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query. Use OR for alternatives. E.g., 'volunteer OR community service OR nonprofit'"
                    },
                    "source": {
                        "type": "string",
                        "enum": ["all", "resume", "linkedin", "github"]
                    },
                    "section": {
                        "type": "string",
                        "description": "Optional: restrict to a section like 'experience' or 'leadership'"
                    }
                },
                "required": ["query", "source"]
            }
        }
    },
    # ... get_rubric, cite_evidence, propose_score, count_distinct
]

# Wiring it together in the orchestrator:
def run_scorer(raw_data, survey_scores, parsed_docs):
    user_msg = build_scorer_message(raw_data, survey_scores, parsed_docs)
    return run_agent(
        client=deepseek_client,
        model="deepseek-v4-pro",
        system_prompt=AGENT4_SCORER_SYSTEM_PROMPT,
        user_message=user_msg,
        tools=AGENT4_SCORER_TOOLS,
        tool_executors={
            "search_evidence": execute_search_evidence,
            "get_rubric": execute_get_rubric,
            "cite_evidence": execute_cite_evidence,
            "propose_score": execute_propose_score,
            "count_distinct": execute_count_distinct,
        },
        temperature=0.0,
    )
```

---

## 5. Orchestrator Logic (Fixed DAG)

```python
import asyncio
from models.events import PipelineEvent, EventType

class PipelineOrchestrator:
    """Runs the fixed DAG. Emits events to Monitor. Does NOT use an LLM."""

    def __init__(self, monitor: "MonitorAgent | None" = None):
        self.monitor = monitor

    async def emit(self, event: PipelineEvent):
        if self.monitor:
            await self.monitor.on_event(event)

    async def run_student(self, name: str, slug: str, zip_path: str):
        ctx = {"name": name, "slug": slug}

        # ── Agent 1 ────────────────────────────────────────────
        self.emit(PipelineEvent(EventType.AGENT_STARTED, name, "agent1"))
        try:
            raw_data = await run_agent(
                system_prompt=AGENT1_PROMPT,
                user_message=f"Find and extract all fields for: {name}",
                tools=AGENT1_TOOLS,
                tool_executors=AGENT1_EXECUTORS,
            )
            self.emit(PipelineEvent(EventType.AGENT_COMPLETED, name, "agent1"))
        except Exception as e:
            self.emit(PipelineEvent(EventType.AGENT_FAILED, name, "agent1", {"error": str(e)}))
            return None

        # ── Agents 2 & 3: PARALLEL ─────────────────────────────
        self.emit(PipelineEvent(EventType.AGENT_STARTED, name, "agent2"))
        self.emit(PipelineEvent(EventType.AGENT_STARTED, name, "agent3"))

        survey_task = run_agent(AGENT2_PROMPT, ..., AGENT2_TOOLS, AGENT2_EXECUTORS)
        docs_task   = run_agent(AGENT3_PROMPT, ..., AGENT3_TOOLS, AGENT3_EXECUTORS)

        survey_scores, parsed_docs = await asyncio.gather(survey_task, docs_task)

        self.emit(PipelineEvent(EventType.AGENT_COMPLETED, name, "agent2"))
        self.emit(PipelineEvent(EventType.AGENT_COMPLETED, name, "agent3"))

        # ── Quality gate: data coverage ────────────────────────
        coverage = raw_data["field_summary"]["found"] / raw_data["field_summary"]["total"]
        if coverage < 0.3:
            self.emit(PipelineEvent(EventType.ANOMALY_DETECTED, name, "quality_gate",
                                    {"reason": "low_coverage", "coverage": coverage}))

        # ── Agent 4: Adversarial (3 sequential phases) ─────────
        self.emit(PipelineEvent(EventType.AGENT_STARTED, name, "agent4"))

        # Phase 1: Scorer
        proposed = await run_agent(SCORER_PROMPT, ..., SCORER_TOOLS, SCORER_EXECUTORS)
        for p in proposed["scores"]:
            self.emit(PipelineEvent(EventType.SCORE_PROPOSED, name, "scorer", p))

        # Phase 2: Verifier
        challenges = await run_agent(VERIFIER_PROMPT, ..., VERIFIER_TOOLS, VERIFIER_EXECUTORS)
        for c in challenges.get("challenges", []):
            self.emit(PipelineEvent(EventType.SCORE_CHALLENGED, name, "verifier", c))

        # Quality gate: dispute rate
        upheld = [c for c in challenges.get("challenges", []) if c.get("challenge_upheld")]
        if len(upheld) / max(len(proposed.get("scores", [])), 1) > 0.4:
            self.emit(PipelineEvent(EventType.ANOMALY_DETECTED, name, "quality_gate",
                                    {"reason": "high_dispute_rate", "rate": len(upheld)/len(proposed["scores"])}))

        # Phase 3: Arbiter
        resolved = await run_agent(ARBITER_PROMPT, ..., ARBITER_TOOLS, ARBITER_EXECUTORS)
        for r in resolved.get("scores", []):
            self.emit(PipelineEvent(EventType.SCORE_RESOLVED, name, "arbiter", r))

        self.emit(PipelineEvent(EventType.AGENT_COMPLETED, name, "agent4"))

        # ── Agent 5: Renderer ──────────────────────────────────
        self.emit(PipelineEvent(EventType.AGENT_STARTED, name, "agent5"))
        passport_path = render_passport(resolved)
        self.emit(PipelineEvent(EventType.AGENT_COMPLETED, name, "agent5"))

        self.emit(PipelineEvent(EventType.STUDENT_COMPLETED, name))
        return passport_path

    async def run_batch(self, students: list[tuple[str, str]], zip_path: str,
                        max_concurrent: int = 5):
        sem = asyncio.Semaphore(max_concurrent)

        async def run_with_limit(name, slug):
            async with sem:
                return await self.run_student(name, slug, zip_path)

        results = await asyncio.gather(
            *[run_with_limit(n, s) for n, s in students],
            return_exceptions=True
        )

        completed = sum(1 for r in results if r and not isinstance(r, Exception))
        failed    = sum(1 for r in results if r is None or isinstance(r, Exception))

        self.emit(PipelineEvent(EventType.BATCH_COMPLETED, "__batch__",
                                {"total": len(students), "completed": completed, "failed": failed}))
        return results
```

---

## 6. Multi-Model Fallback

```python
import os
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

# Model chain: primary → fallback → last resort
MODEL_CHAIN = [
    {
        "name": "deepseek-v4-pro",
        "client": OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.environ["DEEPSEEK_API_KEY"]
        ),
    },
    {
        "name": "claude-sonnet-4-6",
        "client": OpenAI(
            base_url="https://api.anthropic.com/v1",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        ),
    },
]

async def run_agent_with_fallback(
    system_prompt, user_message, tools, tool_executors,
    temperature=0.0, max_turns=30
):
    """Try each model in chain. First one that works, wins."""

    errors = []
    for model_config in MODEL_CHAIN:
        try:
            return await run_agent(
                client=model_config["client"],
                model=model_config["name"],
                system_prompt=system_prompt,
                user_message=user_message,
                tools=tools,
                tool_executors=tool_executors,
                temperature=temperature,
                max_turns=max_turns,
            )
        except (APIError, APITimeoutError, RateLimitError) as e:
            errors.append(f"{model_config['name']}: {e}")
            continue

    raise RuntimeError(f"All models failed: {'; '.join(errors)}")
```

---

## 7. Determinism Guarantees

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  GUARANTEE                     │  HOW WE ENFORCE IT                          │
├────────────────────────────────┼─────────────────────────────────────────────┤
│  Same input → same score       │  temperature=0.0 on every LLM call          │
│                                │  Rubric-anchored scoring (not opinion)      │
│                                │  Reference example calibration              │
├────────────────────────────────┼─────────────────────────────────────────────┤
│  No hallucinated evidence      │  cite_evidence() returns found=false →      │
│                                │  forced minimum score, no guessing          │
│                                │  Every score has at least one exact quote   │
├────────────────────────────────┼─────────────────────────────────────────────┤
│  No contradictory scores       │  check_consistency() catches:               │
│                                │  • Low English proficiency vs fluent text   │
│                                │  • Zero volunteer hours vs volunteer roles  │
│                                │  • No GitHub vs "software engineer" goal    │
├────────────────────────────────┼─────────────────────────────────────────────┤
│  No single-point LLM error     │  Adversarial verifier catches:              │
│                                │  • Overcounted roles/organizations          │
│                                │  • Rubric band misapplication               │
│                                │  • Missed counter-evidence                  │
├────────────────────────────────┼─────────────────────────────────────────────┤
│  No silent pipeline failure    │  Monitor Agent watches:                     │
│                                │  • Low data coverage → flags student        │
│                                │  • High dispute rate → switches model       │
│                                │  • Repeated API errors → escalates          │
│                                │  • Outlier scores → compares to cohort      │
├────────────────────────────────┼─────────────────────────────────────────────┤
│  No malformed JSON drift       │  Pydantic validates every tool call result  │
│                                │  Invalid output → retry with error feedback │
│                                │  Schema versioned with the code             │
├────────────────────────────────┼─────────────────────────────────────────────┤
│  Full audit trail              │  Every score stores:                        │
│                                │  • exact_quote[] + source + section         │
│                                │  • rubric_band + matched criteria           │
│                                │  • verifier_verdict + arbiter_notes         │
│                                │  • monitor_flags (if any)                   │
└────────────────────────────────┴─────────────────────────────────────────────┘
```
