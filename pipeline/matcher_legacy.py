"""Legacy name-only matcher (method = "legacy"): the original fuzzy matcher
that produced the first 117 training pairs.

Names are normalized, 4e "Elder" is aligned to 5e "Ancient", and fuzzy
matches (difflib ratio) are gated by core-token overlap and age agreement.
A pair is accepted when its name score is >= 0.85, or when the 4e name
starts with a 5e base name ("variant promotion"). No metadata is used.

Returns the same record shape as pipeline/matcher.match so the rest of the
pipeline (cardinality, outputs) is shared; `score` is the name score.
"""

import difflib
import re

# Bases that are too generic to promote via substring matching.
VARIANT_PROMOTE_BLOCKLIST = {"shadow"}

AGE_SYNONYMS = {"elder": "ancient"}  # 4e Elder Dragon -> 5e Ancient Dragon

# Stopwords removed when building the semantic "core" tokens used to decide
# whether a fuzzy candidate is plausible.  This lets "Giant Rat" match
# "Diseased Giant Rat" while preventing "Giant Rat" from matching "Giant Goat".
STOPWORDS = {
    "the", "a", "an", "of", "and",
    "young", "adult", "ancient", "elder", "wyrmling", "juvenile",
    "giant", "swarm", "large", "medium", "small", "tiny", "huge", "gargantuan",
    "dragon", "dracolich",
}

AGE_TOKENS = {"young", "adult", "ancient", "elder", "wyrmling", "juvenile"}

ACCEPT_SCORE = 0.85
FUZZY_MIN = 0.55


def normalize(name):
    """Lowercase, drop parentheticals, drop punctuation/digits, collapse ws."""
    s = name.lower()
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\d+", "", s)
    return " ".join(s.split())


def edition_normalize(name):
    return " ".join(AGE_SYNONYMS.get(t, t) for t in normalize(name).split())


def core_tokens(name):
    return set(edition_normalize(name).split()) - STOPWORDS


def age_canon(name):
    tokens = set(normalize(name).split()) & AGE_TOKENS
    if not tokens:
        return None
    age = sorted(tokens)[0]
    return AGE_SYNONYMS.get(age, age)


def is_variant_promotion(name4, name5, score):
    """Promote obvious 4e variant -> 5e base pairs (e.g., Fire Titan -> Fire Giant)."""
    if score < 0.65:
        return False
    n4, n5 = normalize(name4), normalize(name5)
    if n5 in VARIANT_PROMOTE_BLOCKLIST:
        return False
    return n4.startswith(n5)


def pair_is_plausible(name4, name5, score):
    """Filters obvious false positives like Black Dragon -> Blue Dragon."""
    if score >= 0.90:
        return True
    core4, core5 = core_tokens(name4), core_tokens(name5)
    if not core4 or not core5:
        return score >= 0.80
    if not (core4 & core5):
        return False
    age4, age5 = age_canon(name4), age_canon(name5)
    if age4 and age5 and age4 != age5:
        return False
    return score >= FUZZY_MIN


def match(names4, names5):
    """Return (accepted, review) over 4e/5e record names (full names)."""
    # Duplicate names resolve to the last record, as in the original matcher.
    idx4 = {n: i for i, n in enumerate(names4)}
    idx5 = {n: j for j, n in enumerate(names5)}

    found = {}

    def add(n4, n5, score, tier):
        key = (n4, n5)
        if key not in found:
            found[key] = (score, tier)

    by_norm5 = {}
    for n5 in names5:
        by_norm5.setdefault(normalize(n5), []).append(n5)
    for n4 in names4:
        for n5 in by_norm5.get(normalize(n4), []):
            add(n4, n5, 1.0, "exact")

    by_ednorm5 = {}
    for n5 in names5:
        by_ednorm5.setdefault(edition_normalize(n5), []).append(n5)
    for n4 in names4:
        for n5 in by_ednorm5.get(edition_normalize(n4), []):
            if normalize(n4) == normalize(n5):
                continue
            score = difflib.SequenceMatcher(None, n4, n5).ratio()
            add(n4, n5, max(score, 0.99), "exact")

    for n4 in names4:
        for n5 in names5:
            if normalize(n5) == normalize(n4):
                continue
            score = difflib.SequenceMatcher(None, edition_normalize(n4),
                                            edition_normalize(n5)).ratio()
            if pair_is_plausible(n4, n5, score):
                add(n4, n5, score, "fuzzy")

    accepted, review = [], []
    for (n4, n5), (score, tier) in found.items():
        ok = score >= ACCEPT_SCORE or is_variant_promotion(n4, n5, score)
        rec = {"i4": idx4[n4], "i5": idx5[n5], "tier": tier,
               "match_type": "same" if tier == "exact" else "related",
               "score": round(score, 3), "meta_prob": None, "meta_rank": None,
               "name_sim": round(score, 3),
               "note": "" if ok else f"name score {score:.2f} < {ACCEPT_SCORE}"}
        (accepted if ok else review).append(rec)
    return accepted, review
