"""Evidence search and citation tools — called by the LLM during scoring."""

import re
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STUDENT_DATA = os.path.join(ROOT, "..", "Passport_Agent_Actual_Test_Final_1",
                             "Passport_Agent_Actual_Test", "Passport_Agent_Actual",
                             "student_data")

# In-memory document cache
_doc_cache: dict = {}


def _load_student_docs(student_name: str) -> dict:
    """Load all documents for a student into memory cache."""
    if student_name in _doc_cache:
        return _doc_cache[student_name]

    docs = {"resume": {}, "linkedin": {}, "github": {}}
    student_dir = None

    # Find student folder
    for folder in os.listdir(STUDENT_DATA):
        if student_name.lower() in folder.lower():
            student_dir = os.path.join(STUDENT_DATA, folder)
            break

    if not student_dir:
        _doc_cache[student_name] = docs
        return docs

    # Load resume
    for f in os.listdir(student_dir):
        if f.endswith(".pdf"):
            try:
                from pdfminer.high_level import extract_text
                text = extract_text(os.path.join(student_dir, f))
                docs["resume"]["raw_text"] = text
                docs["resume"]["filename"] = f
            except Exception:
                pass

    # Load LinkedIn
    for f in os.listdir(student_dir):
        if "linkedin" in f.lower() and f.endswith(".md"):
            try:
                with open(os.path.join(student_dir, f), encoding="utf-8") as fh:
                    docs["linkedin"]["raw_text"] = fh.read()
                    docs["linkedin"]["filename"] = f
            except Exception:
                pass

    # Load GitHub if exists
    for f in os.listdir(student_dir):
        if "github" in f.lower() and f.endswith(".json"):
            import json
            try:
                with open(os.path.join(student_dir, f), encoding="utf-8") as fh:
                    docs["github"] = json.load(fh)
            except Exception:
                pass

    _doc_cache[student_name] = docs
    return docs


def search_evidence(query: str, source: str = "all", section: str = None,
                    student_name: str = "") -> dict:
    """
    Full-text search across student documents for a concept or keyword.
    Call BEFORE making any claim about what a student did or didn't do.

    Args:
        query: Search query. Use OR for alternatives.
               E.g., 'volunteer OR community service OR nonprofit'
        source: Which document to search: "all", "resume", "linkedin", "github"
        section: Optional section name to restrict search
        student_name: Name of the student being processed
    """
    docs = _load_student_docs(student_name)

    # Build search terms
    terms = [t.strip().lower() for t in re.split(r'\bOR\b', query)]
    matches = []

    sources_to_search = []
    if source == "all":
        sources_to_search = ["resume", "linkedin"]
        if docs.get("github", {}).get("repos"):
            sources_to_search.append("github")
    else:
        sources_to_search = [source]

    for src in sources_to_search:
        doc = docs.get(src, {})
        if not doc:
            continue

        text = doc.get("raw_text", "")
        if not text:
            # For github, build text from repos
            if src == "github" and doc.get("repos"):
                for repo in doc["repos"]:
                    text += f"\n{repo.get('name', '')} {repo.get('description', '')} {repo.get('readme', '')}"

        if not text:
            continue

        # Split into sentences/paragraphs
        paragraphs = re.split(r'\n\n+', text)

        for para in paragraphs:
            if not para.strip():
                continue
            para_lower = para.lower()

            # Check if any term matches
            matched_terms = [t for t in terms if t in para_lower]
            if not matched_terms:
                continue

            # If section filter, check if paragraph is under that section
            if section and section.lower() not in para_lower[:200].lower():
                continue

            # Extract surrounding context (~2 sentences)
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for i, sent in enumerate(sentences):
                sent_lower = sent.lower()
                if any(t in sent_lower for t in matched_terms):
                    context_start = max(0, i - 1)
                    context_end = min(len(sentences), i + 2)
                    context = " ".join(sentences[context_start:context_end])

                    matches.append({
                        "source": src,
                        "section": section or "unknown",
                        "matched_text": sent.strip()[:300],
                        "surrounding_context": context.strip()[:500],
                        "match_type": "direct" if all(t in sent_lower for t in matched_terms) else "semantic",
                        "matched_terms": matched_terms,
                    })
                    break

    return {
        "matches": matches[:10],  # Top 10 matches
        "total_count": len(matches),
        "query": query,
        "sources_searched": sources_to_search,
    }


def cite_evidence(claim: str, source: str = "all", student_name: str = "") -> dict:
    """
    Return the EXACT quote from a document supporting or refuting a claim.
    Returns found=false if no evidence exists.

    Args:
        claim: The claim to verify, e.g., 'Student volunteered at a food pantry'
        source: Which source to search
        student_name: Name of the student being processed
    """
    # Use search_evidence to find matching text
    result = search_evidence(query=claim, source=source, student_name=student_name)

    if result["total_count"] == 0:
        return {
            "found": False,
            "exact_quote": None,
            "location": None,
            "strength": "none",
        }

    best = result["matches"][0]
    return {
        "found": True,
        "exact_quote": best["matched_text"],
        "location": {
            "source": best["source"],
            "section": best["section"],
        },
        "strength": best["match_type"],
    }


def count_distinct(source: str, pattern: str, student_name: str = "") -> dict:
    """
    Count distinct instances of a pattern (roles, companies, activities, etc.).

    Args:
        source: "resume", "linkedin", or "github"
        pattern: "roles" | "companies" | "volunteer_activities" |
                 "community_orgs" | "certifications" | "projects"
        student_name: Name of the student
    """
    docs = _load_student_docs(student_name)
    text = ""

    if source == "linkedin":
        text = docs.get("linkedin", {}).get("raw_text", "")
    elif source == "resume":
        text = docs.get("resume", {}).get("raw_text", "")
    elif source == "github":
        for repo in docs.get("github", {}).get("repos", []):
            text += f"\n{repo.get('name', '')} {repo.get('description', '')}"

    if not text:
        return {"count": 0, "items": [], "pattern": pattern}

    items = []

    if pattern == "companies":
        # Extract company names from LinkedIn experience section
        companies = set()
        for line in text.splitlines():
            line = line.strip()
            # Match "**Title** — Company" pattern
            m = re.match(r'\*\*.+?\*\*\s*[—–]\s*(.+?)(?:\s*·|$)', line)
            if m:
                company = m.group(1).strip()
                if company and len(company) > 1:
                    companies.add(company)
        items = [{"name": c, "context": "LinkedIn experience"} for c in companies]

    elif pattern == "volunteer_activities":
        volunteer_keywords = ["volunteer", "community service", "nonprofit",
                              "non-profit", "outreach", "food pantry",
                              "shelter", "mutual aid", "faith community"]
        for line in text.splitlines():
            line_lower = line.lower()
            for kw in volunteer_keywords:
                if kw in line_lower and line.strip() not in [i["name"] for i in items]:
                    items.append({"name": line.strip()[:120], "context": kw})
                    break

    elif pattern == "community_orgs":
        org_keywords = ["wildlife", "conservation", "community", "church",
                        "mosque", "temple", "nonprofit", "foundation",
                        "civic", "youth", "outreach", "public school"]
        for line in text.splitlines():
            line_lower = line.lower()
            for kw in org_keywords:
                if kw in line_lower and line.strip() not in [i["name"] for i in items]:
                    items.append({"name": line.strip()[:120], "context": kw})
                    break

    elif pattern == "roles":
        for line in text.splitlines():
            m = re.match(r'\*\*(.+?)\*\*\s*[—–]', line)
            if m:
                items.append({"name": m.group(1).strip(), "context": "LinkedIn title"})

    elif pattern == "projects":
        for line in text.splitlines():
            m = re.match(r'\*\*(.+?)\*\*\s*[—–]', line)
            if m:
                items.append({"name": m.group(1).strip(), "context": "project"})

    elif pattern == "certifications":
        cert_section = False
        for line in text.splitlines():
            if "certification" in line.lower():
                cert_section = True
                continue
            if cert_section and line.startswith("|") and not line.startswith("|---"):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if parts:
                    items.append({"name": parts[0], "context": "certification"})
            elif cert_section and not line.startswith("|") and line.strip():
                cert_section = False

    return {
        "count": len(items),
        "items": items,
        "pattern": pattern,
        "source": source,
    }
