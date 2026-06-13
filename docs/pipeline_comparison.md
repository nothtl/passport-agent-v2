# Pipeline Architecture: Old Approach vs New Approach

This document compares the current pipeline with the proposed redesign. For implementation details, see `pipeline_implementation.md`.

---

## OLD APPROACH — Linear Script Orchestration with Single-Shot LLM

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              run_pipeline.py                                         │
│                                                                                     │
│  for each student:                                                                  │
│    subprocess.run("python agent1/agent1_data_loader.py --student NAME --zip ...")   │
│    subprocess.run("python agent2/agent2_survey_scorer.py --input raw_data.json")    │
│    subprocess.run("python agent3/agent3_doc_parser.py --input raw_data.json")       │
│    subprocess.run("python agent4/agent4_enrichment.py --survey ... --docs ...")      │
│    subprocess.run("python agent5/agent5_renderer.py --input enriched_scores.json")   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│    AGENT 1       │       │    AGENT 2       │       │    AGENT 3       │       │    AGENT 4       │       │    AGENT 5       │
│   Data Loader    │       │  Survey Scorer   │       │   Doc Parser     │       │   Enrichment     │       │    Renderer      │
│                  │       │                  │       │                  │       │                  │       │                  │
│ ┌──────────────┐ │       │ ┌──────────────┐ │       │ ┌──────────────┐ │       │ ┌──────────────┐ │       │ ┌──────────────┐ │
│ │Hardcoded     │ │       │ │Formula score │ │       │ │pdfminer      │ │       │ │EC/GC/RFF/CR  │ │       │ │Jinja2        │ │
│ │file priority │ │       │ │(numeric flds)│ │       │ │extract text  │ │       │ │enrichers     │ │       │ │template      │ │
│ └──────────────┘ │       │ └──────────────┘ │       │ └──────────────┘ │       │ └──────────────┘ │       │ └──────────────┘ │
│ ┌──────────────┐ │       │ ┌──────────────┐ │       │ ┌──────────────┐ │       │ ┌──────────────┐ │       │                  │
│ │Exact column  │ │       │ │Single Gemini │ │       │ │Regex         │ │       │ │Single Gemini │ │       │                  │
│ │name match    │ │       │ │call per text │ │       │ │LinkedIn      │ │       │ │call per:     │ │       │                  │
│ └──────────────┘ │       │ │field (1-5)   │ │       │ │parser        │ │       │ │• GC 17 fields│ │       │                  │
│ ┌──────────────┐ │       │ └──────────────┘ │       │ └──────────────┘ │       │ │• CT score    │ │       │                  │
│ │Name prefix   │ │       │ ┌──────────────┐ │       │ ┌──────────────┐ │       │ │• CI score    │ │       │                  │
│ │match only    │ │       │ │No reference  │ │       │ │Single Gemini │ │       │ │• 4 pillar    │ │       │                  │
│ └──────────────┘ │       │ │calibration   │ │       │ │call for      │ │       │ │  reasonings  │ │       │                  │
│                  │       │ └──────────────┘ │       │ │resume secs   │ │       │ └──────────────┘ │       │                  │
│                  │       │                  │       │ └──────────────┘ │       │ ┌──────────────┐ │       │                  │
│                  │       │                  │       │ ┌──────────────┐ │       │ │No evidence   │ │       │                  │
│                  │       │                  │       │ │GitHub API    │ │       │ │verification  │ │       │                  │
│                  │       │                  │       │ │scraper       │ │       │ └──────────────┘ │       │                  │
│                  │       │                  │       │ └──────────────┘ │       │ ┌──────────────┐ │       │                  │
│                  │       │                  │       │                  │       │ │Simple string │ │       │                  │
│                  │       │                  │       │                  │       │ │overlap check │ │       │                  │
│                  │       │                  │       │                  │       │ │CT vs CI      │ │       │                  │
│                  │       │                  │       │                  │       │ └──────────────┘ │       │                  │
│     ▼            │       │     ▼            │       │     ▼            │       │     ▼            │       │     ▼            │
│ raw_data.json    │       │ survey_scores    │       │ parsed_docs     │       │ enriched_scores  │       │ passport.html    │
│                  │       │ .json            │       │ .json           │       │ .json            │       │                  │
└──────────────────┘       └──────────────────┘       └──────────────────┘       └──────────────────┘       └──────────────────┘
```

### Key Weaknesses

| Weakness | Where | Impact |
|----------|-------|--------|
| Hardcoded file/column names | Agent 1 | Breaks on new data formats or renamed columns |
| Name prefix matching only | Agent 1 | Misses nicknames, compound surnames, misspellings |
| LLM scores with no calibration | Agent 2 | Same answer → different scores across runs |
| No cross-field consistency check | Agent 2 | Student says "not comfortable with English" but writes eloquent paragraphs |
| Single-shot resume section extraction | Agent 3 | Misclassifies "Clinical Experience" as "Leadership" |
| Regex-only LinkedIn parser | Agent 3 | Fragile to format changes in LinkedIn exports |
| 17-field inference in ONE prompt | Agent 4 | Model hallucinates counts, invents evidence |
| No evidence verification | Agent 4 | Score is a number with no proof. No audit trail. |
| No adversarial check | Agent 4 | No mechanism to catch scoring errors |
| Malformed JSON → silent 0 or 1 | All | Model returns bad output → pipeline silently accepts it |
| No monitoring | All | Pipeline crashes → no one knows why or which student was affected |

---

## NEW APPROACH — Tool-Using Pipeline with Adversarial Verification

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│                        PIPELINE ORCHESTRATOR (Python code)                             │
│                        ──────────────────────────────────                             │
│                                                                                      │
│  Runs the fixed DAG: A1 → (A2 ║ A3) → A4(Scorer→Verifier→Arbiter) → A5             │
│  Each agent is a tool-use loop: the LLM calls tools until the task is done.          │
│                                                                                      │
│                              │                                                       │
│                              │ PipelineEvent stream                                   │
│                              ▼                                                       │
│                        MONITOR AGENT (LLM — on-demand)                                │
│                        Watches events. Silent when normal.                            │
│                        Acts only on anomalies.                                        │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘


┌───────────────────────────────────────────────────────────────────────────┐
│                        AGENT 1 — DISCOVERY                                  │
│                                                                           │
│  Tools: explore_zip(), fuzzy_find_student(), semantic_col_search(),       │
│         extract_row_field(), resolve_identity()                           │
│                                                                           │
│  Loop:                                                                    │
│    1. explore_zip() → dynamic manifest                                    │
│    2. fuzzy_find_student(name, threshold=0.85) → tolerant name matching   │
│    3. For each field: semantic_col_search(concept) → meaning-based match  │
│    4. resolve_identity() if multiple rows match → email disambiguation    │
│                                                                           │
│  Difference: Dynamic discovery replaces hardcoded lists.                  │
│              Semantic search replaces brittle column name variants.       │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        AGENT 2 — CALIBRATED SCORER                         │
│                                                                           │
│  Tools: get_scoring_rubric(), lookup_reference(), check_consistency(),    │
│         validate_score()                                                  │
│                                                                           │
│  Loop (per text field):                                                   │
│    1. get_rubric(field) → anchored scoring bands                          │
│    2. lookup_reference(field, score_range) → calibrated examples          │
│    3. Compare student response to reference examples                      │
│    4. check_consistency() → detect contradictions (e.g. English vs prose) │
│    5. validate_score(proposed) → sanity check                             │
│                                                                           │
│  Difference: Rubric-anchored + reference-calibrated. Same input → same    │
│              score every run. Cross-field consistency prevents nonsense.  │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     AGENT 3 — DOCUMENT NAVIGATOR                           │
│                                                                           │
│  Tools: extract_pdf_text(), identify_structure(), extract_section(),      │
│         parse_linkedin(), find_github_user(), scrape_github()             │
│                                                                           │
│  Loop:                                                                    │
│    1. extract_pdf_text(pdf) → raw text with page granularity              │
│    2. identify_structure(text) → detect layout + section boundaries       │
│    3. extract_section(text, name, start, end) → precise text extraction   │
│    4. parse_linkedin(md) → structured regex + LLM fallback                │
│    5. find_github_user(resume+linkedin) → multi-format detection          │
│    6. scrape_github(username) if found                                    │
│                                                                           │
│  Difference: Iterative structure detection → precise sections.            │
│              LLM fallback for unusual formats. Multi-format detection.    │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│              AGENT 4 — ADVERSARIAL ENRICHMENT (3 phases)                   │
│                                                                           │
│  Tools: search_evidence(), get_rubric(), cite_evidence(),                  │
│         count_distinct(), propose_score(), challenge_score(),              │
│         resolve_dispute(), generate_narrative()                           │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ PHASE 1: SCORER                                                     │ │
│  │                                                                     │ │
│  │  For each dimension:                                                │ │
│  │    get_rubric() → search_evidence() → cite_evidence() →              │ │
│  │    map to rubric band → propose_score(score, evidence_quotes)        │ │
│  │                                                                     │ │
│  │  Evidence found?  → score mapped to rubric band                     │ │
│  │  Evidence absent? → forced minimum score (1 or 0) — no guessing     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ PHASE 2: VERIFIER (adversarial)                                     │ │
│  │                                                                     │ │
│  │  For each proposed score:                                           │ │
│  │    search_evidence(inverse query) → find what scorer missed         │ │
│  │    get_rubric() → verify band match                                 │ │
│  │    challenge_score(proposal) → uphold or dismiss                    │ │
│  │                                                                     │ │
│  │  Example: Scorer found 5 community roles.                           │ │
│  │           Verifier searches → only 3 documented.                    │ │
│  │           Challenge upheld. Suggested correction: 3.                │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                      │
│                                    ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ PHASE 3: ARBITER                                                    │ │
│  │                                                                     │ │
│  │  For each dispute:                                                  │ │
│  │    resolve_dispute() → weigh both sides, apply rubric, final score  │ │
│  │    generate_narrative() → evidence-backed, forbids repeated claims  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  Difference: Single-shot 17-field inference replaced by: evidence        │
│              search → rubric mapping → scored proposal → adversarial     │
│              challenge → arbitration → evidence-backed narrative.        │
│              Every score traceable to an exact quote.                    │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        AGENT 5 — RENDERER (unchanged)                      │
│                                                                           │
│  Jinja2 template. No LLM. Reads enriched_scores.json.                     │
│  Now optionally displays evidence chips, audit trail, monitor flags.      │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Comparison Table

```
┌────────────────────────────┬──────────────────────────────────┬──────────────────────────────────┐
│          ASPECT            │              OLD                 │              NEW                 │
├────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│                            │                                  │                                  │
│  LLM PROVIDER              │ Google Gemini (Flash-Lite)       │ DeepSeek V4 Pro/Flash            │
│                            │                                  │ (with Claude/GPT fallback)       │
│                            │                                  │                                  │
│  DEPENDENCIES              │ requests, pandas, pdfminer,      │ openai (DeepSeek endpoint),      │
│                            │ jinja2, google-generativeai      │ pydantic, pdfminer, jinja2,      │
│                            │                                  │ pandas, requests                 │
│                            │                                  │                                  │
│  ORCHESTRATION             │ subprocess.run() chain           │ Python orchestrator manages      │
│                            │ Scripts communicate via files    │ fixed DAG + parallelism          │
│                            │ No monitoring                    │ Monitor Agent watches events     │
│                            │                                  │                                  │
│  AGENCY MODEL              │ None — scripts run top-to-bottom │ Tool-use LLM calls at each step: │
│                            │ One LLM call → return → write    │ LLM decides which tool, what     │
│                            │                                  │ args, what order → loops until   │
│                            │                                  │ task complete                    │
│                            │                                  │                                  │
│  STUDENT DISCOVERY         │ Hardcoded file priority list     │ explore_zip() → dynamic manifest │
│  (Agent 1)                 │ Exact column name matching       │ semantic_col_search() → meaning  │
│                            │ Name prefix match only           │ fuzzy_find_student() → tolerant  │
│                            │                                  │ resolve_identity() → disambiguate│
│                            │                                  │                                  │
│  TEXT SCORING              │ Single LLM call per field        │ get_rubric() → anchored bands   │
│  (Agent 2)                 │ No reference calibration         │ lookup_reference() → calibrated  │
│                            │ No cross-field consistency       │ check_consistency() → no contrad.│
│                            │ Score = model opinion            │ validate_score() → sanity check  │
│                            │                                  │                                  │
│  DOCUMENT PARSING          │ Single LLM call for sections     │ identify_structure() → layout    │
│  (Agent 3)                 │ Regex-only LinkedIn parser       │ extract_section() → precise text │
│                            │ Single regex for GitHub user     │ LLM fallback for unusual formats │
│                            │                                  │ Multi-format GitHub detection    │
│                            │                                  │                                  │
│  ENRICHMENT & SCORING      │ ONE prompt → 17 fields           │ search_evidence() → find facts   │
│  (Agent 4)                 │ NO evidence verification         │ cite_evidence() → exact quotes   │
│  ★ BIGGEST CHANGE ★        │ NO traceability                  │ get_rubric() → band matching     │
│                            │ NO adversarial check             │ propose_score() → submit         │
│                            │ CT/CI: single opinion score      │ challenge_score() → verify       │
│                            │ Narratives: single prompt        │ resolve_dispute() → finalize     │
│                            │                                  │ generate_narrative() → cited     │
│                            │                                  │                                  │
│  MONITORING                │ None                             │ Monitor Agent watches events     │
│                            │ Pipeline crashes silently        │ Detects anomalies automatically  │
│                            │                                  │ Adjusts pipeline in real time    │
│                            │                                  │ Flags students for human review  │
│                            │                                  │                                  │
│  ERROR HANDLING            │ Malformed JSON → silent 0/1      │ Tool call fails → agent retries  │
│                            │ API 429 → sleep + retry only     │ cite_evidence found=false →      │
│                            │                                  │ forced minimum score (no guess)  │
│                            │                                  │ Monitor escalates on 3+ failures │
│                            │                                  │ Multi-model fallback chain       │
│                            │                                  │                                  │
│  DETERMINISM               │ temp not set (default ~1.0)      │ temperature=0.0 + seed           │
│                            │ Same prompt → different scores   │ Rubric-anchored → same score     │
│                            │                                  │ every run                        │
│                            │                                  │                                  │
│  EVIDENCE TRACEABILITY     │ Score is a number                │ Score + exact_quote[] + source   │
│                            │ Narrative is unverified          │ + rubric_band + verifier verdict │
│                            │ No way to audit                  │ Full audit trail per dimension   │
│                            │                                  │                                  │
│  PARALLELISM               │ None — strictly serial           │ A2 ║ A3 (independent agents)    │
│                            │                                  │ Parallel tool calls within agent │
│                            │                                  │ Students processed concurrently  │
│                            │                                  │                                  │
│  COST PER STUDENT          │ ~$0.01 (Gemini Flash)            │ ~$0.04 (DeepSeek V4 Pro)         │
│                            │                                  │ ~$1.00 if fallback to Claude     │
│                            │                                  │                                  │
└────────────────────────────┴──────────────────────────────────┴──────────────────────────────────┘
```

---

## The Core Loop Difference

```
OLD:                              NEW:
────                              ───

  python script.py \               # Orchestrator (Python) runs the DAG
    --input data.json \            # Each step launches a tool-use loop:
    --output result.json
                                    response = client.chat.completions.create(
  ┌─ Script runs top-to-bottom       model="deepseek-v4-pro",
  │  One LLM call per function       tools=[search_evidence, get_rubric,
  │  Returns whatever JSON                 cite_evidence, ...],
  │  No self-correction              temperature=0.0
  └─ Writes output file             )
  (sequential, no feedback)          ┌─ Model: "I need evidence.
                                     │   Calling search_evidence()"
                                     │─ Tool returns results
                                     │─ Model: "Found 2 matches.
                                     │   Now calling cite_evidence()"
                                     │─ Tool returns exact quote
                                     │─ Model: "Evidence confirmed.
                                     │   Now calling get_rubric()"
                                     │  (continues until task complete,
                                     │   then hands off to Verifier)

  LINEAR                           ITERATIVE + ADVERSARIAL + MONITORED
  5 subprocess calls               Fixed DAG of tool-use loops
  Zero verification                3-phase adversarial verification
  Zero monitoring                  Monitor watches for anomalies
```

---

## Where the Agency Actually Lives

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  PIPELINE ORCHESTRATOR                       AGENTS (inside each step)       │
│  ─────────────────────                       ─────────────────────────       │
│                                                                             │
│  Fixed DAG:                                Each agent is a tool-use loop:   │
│  A1 → A2∥A3 → A4 → A5                                                   │
│                                              while not done:               │
│  This NEVER changes.                          model decides which tool      │
│  The LLM does NOT choose                      to call next                  │
│  which step runs next.                                                     │
│                                              The LLM's decisions are        │
│  Agency: NONE                                WITHIN a fixed task, not       │
│  (it's a for-loop)                           ABOUT which task to run.       │
│                                                                             │
│                                              Agency: TOOL SELECTION         │
│                                              + STRATEGY (search order,      │
│                                              evidence sufficiency,          │
│                                              challenge targets)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
