"""Data discovery tools — zip exploration, student finding, field extraction."""

import os
import re
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZIP_PATH = os.path.join(ROOT, "..", "Passport_Agent_Actual_Test_Final_1",
                         "Passport_Agent_Actual_Test", "Passport_Agent_Actual",
                         "Full_speakhire_data.zip")


def explore_zip(zip_path: str = "") -> dict:
    """List all files and sheets in the survey data zip."""
    path = zip_path or ZIP_PATH
    if not os.path.exists(path):
        return {"error": f"Zip not found: {path}", "files": []}

    zf = zipfile.ZipFile(path, "r")
    files = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        files.append({
            "name": os.path.basename(info.filename),
            "path": info.filename,
            "size_kb": round(info.file_size / 1024, 1),
        })

    zf.close()
    return {"zip_path": path, "total_files": len(files), "files": files}


def fuzzy_find_student(name: str, threshold: float = 0.85, zip_path: str = "") -> dict:
    """
    Search ALL files/sheets for a student name with fuzzy matching.

    Args:
        name: Student name to search for
        threshold: Minimum match confidence (0.0-1.0)
        zip_path: Path to zip file
    """
    path = zip_path or ZIP_PATH
    matches = []

    def norm(s):
        return re.sub(r'\s+', ' ', str(s).lower().replace('\xa0', ' ')).strip()

    target = norm(name)
    target_parts = target.split()

    if not os.path.exists(path):
        return {"error": f"Zip not found: {path}", "matches": []}

    zf = zipfile.ZipFile(path, "r")

    for info in zf.infolist():
        if info.is_dir():
            continue
        fname = os.path.basename(info.filename)

        try:
            if fname.endswith(".csv"):
                import pandas as pd
                import io
                raw = zf.read(info.filename)
                df = pd.read_csv(io.BytesIO(raw))

                for col in df.columns:
                    col_lower = str(col).lower()
                    if any(kw in col_lower for kw in ["name", "full name", "student", "intern"]):
                        for idx, val in df[col].dropna().items():
                            val_norm = norm(str(val))
                            # Exact match
                            if val_norm == target:
                                matches.append({
                                    "filename": fname,
                                    "sheet": "_default",
                                    "row_index": int(idx),
                                    "matched_name": str(val),
                                    "match_score": 1.0,
                                    "match_type": "exact",
                                })
                            # Prefix match (first + last name)
                            elif len(target_parts) >= 2:
                                prefix = target_parts[0] + " " + target_parts[1]
                                if val_norm.startswith(prefix):
                                    matches.append({
                                        "filename": fname,
                                        "sheet": "_default",
                                        "row_index": int(idx),
                                        "matched_name": str(val),
                                        "match_score": 0.9,
                                        "match_type": "prefix",
                                    })
                            # Contains match
                            elif target_parts[0] in val_norm and len(target_parts) >= 2 and target_parts[-1] in val_norm:
                                matches.append({
                                    "filename": fname,
                                    "sheet": "_default",
                                    "row_index": int(idx),
                                    "matched_name": str(val),
                                    "match_score": 0.8,
                                    "match_type": "contains",
                                })

            elif fname.endswith(".xlsx"):
                import pandas as pd
                import io
                raw = zf.read(info.filename)
                xls = pd.ExcelFile(io.BytesIO(raw))

                for sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if any(kw in col_lower for kw in ["name", "full name", "student", "intern"]):
                            for idx, val in df[col].dropna().items():
                                val_norm = norm(str(val))
                                if val_norm == target:
                                    matches.append({
                                        "filename": fname,
                                        "sheet": sheet,
                                        "row_index": int(idx),
                                        "matched_name": str(val),
                                        "match_score": 1.0,
                                        "match_type": "exact",
                                    })
                                elif len(target_parts) >= 2:
                                    prefix = target_parts[0] + " " + target_parts[1]
                                    if val_norm.startswith(prefix):
                                        matches.append({
                                            "filename": fname,
                                            "sheet": sheet,
                                            "row_index": int(idx),
                                            "matched_name": str(val),
                                            "match_score": 0.9,
                                            "match_type": "prefix",
                                        })
        except Exception:
            continue

    zf.close()

    return {
        "search_name": name,
        "threshold": threshold,
        "total_matches": len(matches),
        "matches": sorted(matches, key=lambda m: m["match_score"], reverse=True)[:10],
    }


def semantic_column_search(file: str, sheet: str = "_default",
                            concept: str = "", zip_path: str = "") -> dict:
    """
    Find columns by meaning, not exact header text.

    Args:
        file: Filename in zip
        sheet: Sheet name (for Excel files)
        concept: What to search for, e.g., "english proficiency"
        zip_path: Path to zip
    """
    concept_lower = concept.lower()
    CONCEPT_MAP = {
        "english proficiency": ["english", "spoken", "language", "comfortable"],
        "volunteer hours": ["volunteer", "hours", "community service"],
        "career goal": ["career", "goal", "future", "ideal job", "smart"],
        "community feel": ["community feel", "belonging", "connected"],
        "college ready": ["college", "ready", "prepared", "prep"],
        "internship": ["internship", "intern", "job", "work experience"],
    }

    keywords = CONCEPT_MAP.get(concept_lower, concept_lower.split())

    path = zip_path or ZIP_PATH
    if not os.path.exists(path):
        return {"error": f"Zip not found: {path}", "columns": []}

    zf = zipfile.ZipFile(path, "r")
    columns = []

    try:
        if file.endswith(".csv"):
            import pandas as pd
            import io
            raw = zf.read([i for i in zf.namelist() if file in i][0])
            df = pd.read_csv(io.BytesIO(raw))

            for col in df.columns:
                col_lower = str(col).lower()
                score = sum(1 for kw in keywords if kw in col_lower)
                if score > 0:
                    sample = df[col].dropna().head(3).tolist()
                    columns.append({
                        "column_name": str(col),
                        "confidence": min(1.0, score / len(keywords)),
                        "sample_values": [str(s)[:100] for s in sample],
                    })

        elif file.endswith(".xlsx"):
            import pandas as pd
            import io
            raw = zf.read([i for i in zf.namelist() if file in i][0])
            df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet) if sheet != "_default" else pd.read_excel(io.BytesIO(raw))

            for col in df.columns:
                col_lower = str(col).lower()
                score = sum(1 for kw in keywords if kw in col_lower)
                if score > 0:
                    sample = df[col].dropna().head(3).tolist()
                    columns.append({
                        "column_name": str(col),
                        "confidence": min(1.0, score / len(keywords)),
                        "sample_values": [str(s)[:100] for s in sample],
                    })
    except Exception as e:
        zf.close()
        return {"error": str(e), "columns": []}

    zf.close()
    return {
        "concept": concept,
        "file": file,
        "total_columns_found": len(columns),
        "columns": sorted(columns, key=lambda c: c["confidence"], reverse=True),
    }


def extract_row_field(filename: str, sheet: str, row_index: int,
                       column: str, zip_path: str = "") -> dict:
    """Extract a specific field value from a matched row."""
    path = zip_path or ZIP_PATH
    zf = zipfile.ZipFile(path, "r")

    try:
        target = [i for i in zf.namelist() if filename in i][0]
        raw = zf.read(target)

        import pandas as pd, io
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
        else:
            df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet) if sheet != "_default" else pd.read_excel(io.BytesIO(raw))

        if column not in df.columns:
            zf.close()
            return {"value": None, "data_type": "null", "is_null": True, "error": f"Column '{column}' not found"}

        val = df.iloc[row_index][column]
        zf.close()

        return {
            "value": str(val) if not pd.isna(val) else None,
            "data_type": type(val).__name__,
            "is_null": bool(pd.isna(val)),
        }
    except Exception as e:
        zf.close()
        return {"value": None, "data_type": "null", "is_null": True, "error": str(e)}


def resolve_identity(candidates: list) -> dict:
    """
    When multiple rows match a name, disambiguate by email/school/cohort.

    Args:
        candidates: List of candidate objects from fuzzy_find_student
    """
    if len(candidates) <= 1:
        return {
            "resolved_row": candidates[0] if candidates else None,
            "confidence": 1.0 if candidates else 0.0,
            "disambiguation_reason": "Single match" if candidates else "No matches",
        }

    # Prefer exact matches over prefix/contains
    exact = [c for c in candidates if c.get("match_type") == "exact"]
    if exact:
        return {
            "resolved_row": exact[0],
            "confidence": 1.0,
            "disambiguation_reason": f"Selected exact match from {len(candidates)} candidates",
        }

    # Prefer matches from richer files
    priority_files = ["Interns_", "FULL SpeakHire Database"]
    for pf in priority_files:
        for c in candidates:
            if pf in c.get("filename", ""):
                return {
                    "resolved_row": c,
                    "confidence": 0.8,
                    "disambiguation_reason": f"Selected from priority file: {c['filename']}",
                }

    return {
        "resolved_row": candidates[0],
        "confidence": 0.5,
        "disambiguation_reason": f"Selected first of {len(candidates)} ambiguous matches",
    }
