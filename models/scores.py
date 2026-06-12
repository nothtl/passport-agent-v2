"""Pydantic data contracts between pipeline stages."""

from pydantic import BaseModel, Field


class ScoreEvidence(BaseModel):
    """A single piece of cited evidence for a score."""
    exact_quote: str
    source: str  # "resume" | "linkedin" | "github"
    section: str = ""
    relevance: str = ""


class PillarScore(BaseModel):
    """A final score for one pillar with full audit trail."""
    pillar: str
    score: float = Field(ge=0, le=100)
    sub_scores: dict[str, float] = Field(default_factory=dict)
    evidence: list[ScoreEvidence] = Field(default_factory=list)
    rubric_band: str = ""
    narrative: str = ""
    verifier_verdict: str = ""  # "upheld" | "challenged" | "corrected"
    arbiter_notes: str = ""
    confidence: str = "high"  # "high" | "medium" | "low"
    monitor_flags: list[str] = Field(default_factory=list)


class RawDataOutput(BaseModel):
    student_name: str
    email: str | None = None
    found_in: list[str] = Field(default_factory=list)
    fields: dict = Field(default_factory=dict)
    field_summary: dict = Field(default_factory=dict)


class SurveyScoresOutput(BaseModel):
    student_name: str
    email: str | None = None
    scores: dict = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    llm_calls_made: int = 0


class ParsedDocsOutput(BaseModel):
    student_name: str
    sources_found: list[str] = Field(default_factory=list)
    resume: dict | None = None
    linkedin: dict | None = None
    github: dict | None = None


class EnrichedScoresOutput(BaseModel):
    student_name: str
    email: str | None = None
    scores: dict[str, PillarScore] = Field(default_factory=dict)
    enriched_fields: dict = Field(default_factory=dict)
    fields_enriched_count: int = 0
    fields_still_missing: list[str] = Field(default_factory=list)
    llm_calls_made: int = 0
