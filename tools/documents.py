"""Document parsing tools — PDF, LinkedIn markdown, GitHub."""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STUDENT_DATA = os.path.join(ROOT, "..", "Passport_Agent_Actual_Test_Final_1",
                             "Passport_Agent_Actual_Test", "Passport_Agent_Actual",
                             "student_data")


def extract_pdf_text(file_path: str, pages: str = "all") -> dict:
    """Extract raw text from a PDF file. Returns page-level granularity."""
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(file_path)
        lines = text.splitlines()
        return {
            "total_pages": "unknown",
            "pages": [{"page_num": 1, "text": text}],
            "total_chars": len(text),
            "total_lines": len(lines),
        }
    except Exception as e:
        return {"error": str(e), "total_pages": 0, "pages": []}


def identify_resume_structure(raw_text: str) -> dict:
    """Detect resume layout and section headers from raw text."""
    if not raw_text:
        return {"format": "unknown", "sections": []}

    # Heuristic section detection based on common headers
    SECTION_PATTERNS = [
        (r'(?i)^\s*(SUMMARY|PROFILE|OBJECTIVE|ABOUT\s*ME)\s*$', "summary"),
        (r'(?i)^\s*(EXPERIENCE|WORK\s*EXPERIENCE|EMPLOYMENT|PROFESSIONAL\s*EXPERIENCE|CLINICAL\s*EXPERIENCE|RESEARCH\s*EXPERIENCE)\s*$', "experience"),
        (r'(?i)^\s*(EDUCATION|ACADEMIC|ACADEMICS)\s*$', "education"),
        (r'(?i)^\s*(SKILLS|COMPETENCIES|QUALIFICATIONS|TECHNICAL\s*SKILLS)\s*$', "skills"),
        (r'(?i)^\s*(PROJECTS|PROJECT\s*EXPERIENCE)\s*$', "projects"),
        (r'(?i)^\s*(CERTIFICATIONS|LICENSES|CERTIFICATES)\s*$', "certifications"),
        (r'(?i)^\s*(LEADERSHIP|VOLUNTEER|COMMUNITY|EXTRACURRICULAR|ACTIVITIES|CAMPUS\s*ENGAGEMENT|HONORS|AWARDS)\s*$', "leadership"),
    ]

    lines = raw_text.splitlines()
    sections = []
    current_section = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        for pattern, section_name in SECTION_PATTERNS:
            if re.match(pattern, stripped):
                if current_section:
                    current_section["end_line"] = i - 1
                current_section = {
                    "header": stripped,
                    "section_type": section_name,
                    "start_line": i + 1,
                    "end_line": None,
                    "confidence": 0.9,
                }
                sections.append(current_section)
                break

    if current_section:
        current_section["end_line"] = len(lines)

    # Detect format
    fmt = "single_column"
    if len(sections) >= 5:
        fmt = "ats"  # Many sections = likely ATS-optimized

    return {
        "format": fmt,
        "sections": sections,
        "total_lines": len(lines),
    }


def extract_section_text(raw_text: str, section_name: str,
                          start_line: int, end_line: int) -> dict:
    """Pull exact text for a named section using identified boundaries."""
    lines = raw_text.splitlines()
    section_lines = lines[max(0, start_line - 1):min(len(lines), end_line or len(lines))]
    text = "\n".join(section_lines).strip()

    return {
        "section_name": section_name,
        "section_text": text[:3000],
        "line_range": [start_line, end_line or len(lines)],
    }


def parse_linkedin_markdown(file_path: str) -> dict:
    """Parse LinkedIn markdown export into structured data."""
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()

    # Extract name (first H1)
    name = None
    for line in lines:
        m = re.match(r'^#\s+(.+)', line.strip())
        if m:
            raw = m.group(1).strip()
            pm = re.search(r'\(([^)]+)\)\s*$', raw)
            if pm:
                name = raw[:pm.start()].strip()
            else:
                name = raw
            break

    # Extract headline (first bold line after H1)
    headline = None
    found_h1 = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# "):
            found_h1 = True
            continue
        if found_h1 and stripped.startswith("**") and stripped.endswith("**"):
            headline = stripped.strip("*").strip()
            break

    # Extract experience entries
    experience = []
    current_exp = None
    for line in lines:
        stripped = line.strip()
        m = re.match(r'^\*\*(.+?)\*\*\s*[—–]\s*(.+)', stripped)
        if m:
            if current_exp:
                experience.append(current_exp)
            current_exp = {
                "title": m.group(1).strip(),
                "company": m.group(2).strip().split("·")[0].strip(),
                "duration": "",
            }
        elif current_exp and stripped and not stripped.startswith("#"):
            if not current_exp["duration"]:
                current_exp["duration"] = stripped
            else:
                current_exp["description"] = (current_exp.get("description", "") + " " + stripped).strip()

    if current_exp:
        experience.append(current_exp)

    # Extract about/summary
    about = None
    in_about = False
    about_lines = []
    for line in lines:
        if re.match(r'^##\s+(About|Summary)', line.strip()):
            in_about = True
            continue
        if in_about:
            if line.strip().startswith("##"):
                break
            if line.strip() and line.strip() != "---":
                about_lines.append(line.strip())

    if about_lines:
        about = "\n".join(about_lines)

    return {
        "name": name,
        "headline": headline,
        "about": about,
        "experience": experience,
        "raw_text": content,
    }


def find_github_username(resume_text: str = "", linkedin_text: str = "") -> dict:
    """Search resume + LinkedIn text for GitHub profile references."""
    EXCLUDED = {"login", "signup", "about", "features", "pricing", "orgs",
                "marketplace", "explore", "topics", "trending", "collections",
                "events", "sponsors"}

    combined = (resume_text or "") + "\n" + (linkedin_text or "")

    # Multiple detection patterns
    patterns = [
        r'github\.com/([a-zA-Z0-9_-]+)',
        r'GitHub:\s*@?([a-zA-Z0-9_-]+)',
        r'github/([a-zA-Z0-9_-]+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, combined)
        for m in matches:
            if m.lower() not in EXCLUDED:
                return {
                    "username": m,
                    "source": "resume" if m in (resume_text or "") else "linkedin",
                    "confidence": 0.9,
                }

    return {"username": None, "source": None, "confidence": 0.0}


def scrape_github_profile(username: str) -> dict:
    """Fetch GitHub profile data via REST API. Falls back gracefully without token."""
    import requests

    try:
        resp = requests.get(
            f"https://api.github.com/users/{username}",
            headers={"Accept": "application/vnd.github+json"},
            timeout=10,
        )
        if resp.status_code == 200:
            profile = resp.json()
            return {
                "username": username,
                "bio": profile.get("bio"),
                "public_repos": profile.get("public_repos", 0),
                "followers": profile.get("followers", 0),
                "repos": [],
            }
    except Exception:
        pass

    return {
        "username": username,
        "error": "Could not fetch GitHub profile (rate limit or network error)",
        "public_repos": 0,
        "repos": [],
    }
