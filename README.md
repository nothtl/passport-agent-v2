# Passport Agent V2 — Tool-Using Agent Pipeline

A **tool-using LLM agent pipeline** that scores student competencies for the SPEAKHIRE PathCredits passport program. Replaces single-shot Gemini calls with evidence-backed agentic scoring using rubric-anchored tool calling.

## What It Does

Given a student's resume (PDF), LinkedIn profile (markdown), and survey responses (CSV/Excel), the pipeline produces scored competency pillars with **exact evidence quotes** for every score.

```
Input: Student documents + survey data
   │
   ├─ Agent 1: Discovery (zip exploration, fuzzy name matching)
   ├─ Agent 2: Survey Scorer (rubric-calibrated text scoring)
   ├─ Agent 3: Document Parser (PDF, LinkedIn, GitHub extraction)
   ├─ Agent 4: Adversarial Enrichment
   │   ├─ Scorer: proposes scores with cited evidence
   │   ├─ Verifier: challenges scores with counter-evidence
   │   └─ Arbiter: resolves disputes, generates narratives
   └─ Agent 5: Renderer (Jinja2 HTML passport)

Output: scores + exact_quotes + audit trail → passport.html
```

## Why This Exists

The old pipeline (Passport_Agent_Actual_Test_Final_1) used single-shot Gemini calls:
- One prompt → 17 scores → no evidence → no way to verify
- Scores were opaque numbers — `D1_Empathy: 0.63` with no proof
- No determinism — same student, different scores on re-run

This version replaces that with tool-using agents:
- **Every score backed by exact quotes from the student's documents**
- **Rubric-anchored scoring** — model reads the rubric before scoring
- **Deterministic** — temperature=0.0, same input → same output
- **Free** — runs on OpenCode Zen's free DeepSeek V4 Flash tier

## Results (6-student batch)

| Metric | Old (Gemini) | New (DeepSeek Free) |
|--------|-------------|---------------------|
| Scores with evidence | 0% | **91%** |
| Total evidence quotes | 0 | **227** |
| Hallucinated evidence | Unknown | **0** |
| Cost per student | ~$0.003 | **$0.00** |
| Deterministic | No | **Yes (temp=0)** |

## Quick Start

### Prerequisites

```bash
pip install openai pydantic pdfminer.six pandas jinja2 requests
```

### Set up API key

Get a free API key from [opencode.ai/auth](https://opencode.ai/auth), then:

```bash
cp .env.example .env
# Edit .env with your key
```

### Run on a student

```bash
python main.py --student "Abigail Rodriguez"
```

### Run batch comparison

```bash
python batch_compare.py
```

## Architecture

```
passport_agent_v2/
├── main.py              # Pipeline runner + old vs new comparison
├── batch_compare.py     # Multi-student batch runner
├── compare.py           # Detailed per-student comparison
├── accuracy_audit.py    # Ground truth verification
│
├── agents/
│   ├── base.py          # Tool-use agent loop (the engine)
│   └── __init__.py
│
├── tools/               # Tool implementations (called by the LLM)
│   ├── evidence.py      # search_evidence, cite_evidence, count_distinct
│   ├── rubric.py        # Scoring rubrics + reference examples
│   ├── validation.py    # Consistency checks + score validation
│   ├── documents.py     # PDF, LinkedIn, GitHub parsing
│   ├── discovery.py     # Zip exploration, fuzzy student finding
│   └── pipeline_control.py  # Monitor agent tools
│
├── models/              # Pydantic data contracts
│   ├── events.py        # Pipeline event types
│   └── scores.py        # Score models with evidence
│
├── outputs/             # Pipeline results
│   ├── students/        # Individual student scores
│   ├── comparisons/     # Old vs new comparisons
│   └── batch/           # Batch run reports
│
├── templates/           # Jinja2 passport template
├── .env                 # API key config
└── README.md
```

## How the Tools Work

The LLM calls these tools mid-reasoning (not before, not after):

```
Model: "I need evidence about volunteering."
  → calls search_evidence("volunteer OR community service", "all")

Tool returns: 3 matches from resume and LinkedIn

Model: "Let me verify this specific claim."
  → calls cite_evidence("Student volunteered at Wildlife Conservation Society", "all")

Tool returns: {found: true, exact_quote: "Volunteer — Teen Outreach Program (TOP) —
              Wildlife Conservation Society — Community outreach program..."}

Model: "Now let me check the rubric."
  → calls get_rubric("GC", "has_volunteer")

Tool returns: {bands: [{score_range: "true", criteria: "At least one documented
              instance of volunteer or unpaid civic activity..."}]}

Model: "Evidence matches. Proposing score."
  → calls propose_score(pillar="GC", dimension="has_volunteer",
       score=true, evidence_quotes=[...], reasoning="...")
```

## Framework

**We do NOT use LangChain, CrewAI, or AutoGen.** The tool-use API _is_ the framework. We use:

- **OpenAI SDK** (pointed at OpenCode Zen's free endpoint) for tool calling
- **Pydantic** for data validation between pipeline stages
- **asyncio** for parallel execution (A2∥A3, concurrent students)
- **Jinja2** for HTML passport rendering

One dependency for tool calling. No framework tax.

## License

MIT
