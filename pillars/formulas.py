"""EC scoring formulas — taken verbatim from old agent2/tools/ec_scorer.py."""

ENGLISH_LEVEL_MAP = {
    'very comfortable': 5, 'comfortable': 4, 'somewhat comfortable': 3,
    'not very comfortable': 2, 'not comfortable': 1,
}

KNOWN_HOPE_TAGS = [
    'gain skills i can use at a future job', 'learn about careers',
    'get an internship', 'learn about colleges', 'make new friends',
    'get coaches and mentors', 'get mentors',
    'mentoring', 'skills', 'careers', 'colleges', 'friends', 'internship',
]


def _norm17(val) -> float:
    """1-7 Likert → 0-1"""
    try: return max(0.0, min(1.0, (float(val) - 1) / 6))
    except (TypeError, ValueError): return 0.0

def _norm15(val) -> float:
    """1-5 scale → 0-1"""
    try: return max(0.0, min(1.0, (float(val) - 1) / 4))
    except (TypeError, ValueError): return 0.0

def _norm110(val) -> float:
    """1-10 scale → 0-1"""
    try: return max(0.0, min(1.0, (float(val) - 1) / 9))
    except (TypeError, ValueError): return 0.0

def _norm010(val) -> float:
    """0-10 scale → 0-1"""
    try: return max(0.0, min(1.0, float(val) / 10))
    except (TypeError, ValueError): return 0.0

def _norm_binary(val) -> float:
    if val is None: return 0.0
    sv = str(val).strip().upper()
    if sv in ('YES', 'TRUE', '1', '1.0'): return 1.0
    if sv in ('NO', 'FALSE', '0', '0.0'): return 0.0
    try: return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError): return 0.0

def _to_binary(val) -> int:
    if val is None: return 0
    sv = str(val).strip().lower()
    if sv in ('1', 'true', 'yes', 'y', '1.0'): return 1
    if sv in ('0', 'false', 'no', 'n', '0.0'): return 0
    try: return 1 if float(val) > 0 else 0
    except (TypeError, ValueError): return 0

def _to_numeric_safe(val) -> float:
    try: return float(val)
    except (TypeError, ValueError): return 0.0

def _english_level(val) -> int:
    if val is None: return 3
    v = str(val).strip().lower()
    return ENGLISH_LEVEL_MAP.get(v, 3)

def _is_multilingual(val) -> int:
    if val is None: return 0
    langs = [x.strip() for x in str(val).split(',') if x.strip()]
    return 1 if len(langs) >= 2 else 0

def _norm_hope_tags(val) -> float:
    import re
    if val is None or str(val).strip() in ('', 'nan'): return 0.0
    text_lower = str(val).lower()
    matched = set()
    for tag in KNOWN_HOPE_TAGS:
        if tag in text_lower:
            if 'mentor' in tag: matched.add('mentoring')
            elif 'skill' in tag: matched.add('skills')
            elif 'career' in tag: matched.add('careers')
            elif 'college' in tag: matched.add('colleges')
            elif 'friend' in tag: matched.add('friends')
            elif 'internship' in tag: matched.add('internship')
            else: matched.add(tag)
    parts = re.split(r'[,\n]', str(val))
    for part in parts:
        p = part.strip().lower()
        if len(p) > 3 and p not in ('nan', 'y2'): matched.add(p[:30])
    return min(1.0, len(matched) / 6)

def _parse_hours(val) -> float | None:
    if val is None: return None
    v = str(val).strip()
    if v.lower() in ('', 'nan', 'none', 'false', '0'): return None
    try: return float(v)
    except ValueError:
        try: return float(v.split('-')[0])
        except (ValueError, IndexError): return None

def _norm_culture_feel(val) -> float:
    culture_map = {
        'i feel equally of both': 1.00,
        'i feel more of the culture of my country of origin': 0.70,
        'i feel more american': 0.60,
        'i am confused about my culture': 0.25,
    }
    if val is None: return 0.5
    v = str(val).strip().lower()
    if v in culture_map: return culture_map[v]
    try: return max(0.0, min(1.0, (float(val) - 1) / 6))
    except (TypeError, ValueError): return 0.5

def _norm_career_network(val) -> float:
    try: return min(float(val) / 10.0, 1.0)
    except (TypeError, ValueError): return 0.0

def _norm_hours_volunteered(val) -> float:
    if val is None: return 0.0
    v = str(val).strip()
    if v in ('', 'nan', 'None', 'False', '0', 'false'): return 0.0
    try: num = float(v)
    except ValueError:
        try: num = float(v.split('-')[0])
        except (ValueError, IndexError): return 0.0
    if num == 0: return 0.0
    if num < 10: return 0.1
    if num < 20: return 0.3
    if num < 30: return 0.5
    if num < 60: return 0.7
    return 1.0

def _to_bool(val) -> bool:
    if val is None: return False
    sv = str(val).strip().lower()
    if sv in ('true', 'yes', '1', '1.0'): return True
    if sv in ('false', 'no', '0', '0.0', ''): return False
    try: return float(val) > 0
    except (TypeError, ValueError): return False
