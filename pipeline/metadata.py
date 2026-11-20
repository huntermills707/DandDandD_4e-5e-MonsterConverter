"""Edition-neutral, non-numeric monster profiles for cross-edition matching.

Matching deliberately ignores numeric stats (HP, level/CR, defenses, ...):
pairs are later used to fit stat-conversion regressions, and choosing pairs
by stat agreement would bias those fits toward whatever looks linear.

Each profile holds small vocabularies that both editions can be mapped
onto: size, creature-type tags, moral alignment, languages, senses,
movement modes, skills, and words from attack/power names. `profile_text`
renders the same facts as a sentence for a sentence-embedding model. Names
and family headers are kept separately (`name`, `alt_names`, `family`) so a
matcher can be evaluated with or without them.
"""

import re

SIZES = ["tiny", "small", "medium", "large", "huge", "gargantuan"]

# 4e origin/type words -> 5e-style type tags.
TYPE_MAP_4E = {
    "aberrant": "aberration",
    "magical beast": "monstrosity",
    "beast": "beast",
    "humanoid": "humanoid",
    "animate": "construct",
    "fey": "fey",
    "elemental": "elemental",
    "shadow": "shadow",
    "immortal": "immortal",
}
# 4e keywords that name a 5e creature type outright.
KEYWORD_TYPES = {
    "undead": "undead", "dragon": "dragon", "giant": "giant",
    "construct": "construct", "ooze": "ooze", "plant": "plant",
    "devil": "fiend", "demon": "fiend", "angel": "celestial",
    "swarm": "swarm", "shapechanger": "shapechanger",
}

LANG_ALIASES = {
    "elvish": "elven", "dwarvish": "dwarven",
    # 4e collapsed several 5e languages into one.
    "supernal": "celestial", "infernal": "celestial",
    "aquan": "primordial", "auran": "primordial", "ignan": "primordial",
    "terran": "primordial",
}
SENSES = ["darkvision", "blindsight", "tremorsense", "truesight",
          "low-light vision", "all-around vision"]
MOVES = ["fly", "swim", "climb", "burrow", "hover", "teleport", "phasing"]
# Energy/effect damage types shared by both editions (4e has no physical types).
DAMAGE = ["acid", "cold", "fire", "force", "lightning", "necrotic", "poison",
          "psychic", "radiant", "thunder"]
STOP = {"of", "the", "a", "an", "and", "actions", "attack", "strike", "s"}


def _words(text):
    return re.findall(r"[a-z][a-z'’-]+", (text or "").lower())


def split_name(name):
    """('Bearded Devil (Barbazu)') -> ('Bearded Devil', ['Barbazu'])."""
    alts = re.findall(r"\(([^)]*)\)", name or "")
    base = re.sub(r"\s*\([^)]*\)", "", name or "").strip()
    return base, [a.strip() for a in alts if a.strip()]


def moral_axis(alignment):
    a = (alignment or "").lower()
    if not a or "unaligned" in a:
        return "unaligned"
    if a.startswith("any"):
        return "any"
    if "evil" in a:
        return "evil"
    if "good" in a:
        return "good"
    return "neutral"


def languages(text):
    text = (text or "").lower()
    if not text or text.strip() in {"-", "—", "none"}:
        return set()
    out = set()
    for part in re.split(r"[,;]", text):
        part = re.sub(r"\d+\s*ft\.?|\d+", "", part).strip()
        if not part or part.startswith(("understands", "can't", "cannot")):
            continue
        if "telepathy" in part:
            out.add("telepathy")
            continue
        word = part.split()[0] if part.split() else ""
        if word in {"common", "deep"}:
            word = "deep speech" if word == "deep" else "common"
        out.add(LANG_ALIASES.get(word, word))
    return out


def tagged(text, vocab):
    text = (text or "").lower()
    return {v for v in vocab if v in text}


def damage_types(text):
    text = (text or "").lower()
    return {d for d in DAMAGE if re.search(rf"\b{d}\b", text)}


def clean_text(text, own_names=()):
    """Block text with numbers, dice, and the monster's own name words
    removed, so similarity reflects content rather than names or stats."""
    t = (text or "").lower().replace("’", "'")
    t = re.sub(r"\d+d\d+|[+-]?\d+", " ", t)
    for word in own_names:
        t = re.sub(rf"\b{re.escape(word)}s?\b", " ", t)
    t = re.sub(r"[^a-z' ]+", " ", t)
    return " ".join(w for w in t.split() if len(w) > 2)


def skills(text):
    return {re.sub(r"\s*[+-]\d+.*", "", s).strip().lower()
            for s in (text or "").split(",") if s.strip()}


def attack_words(names):
    words = set()
    for n in names:
        words.update(w for w in _words(n) if w not in STOP and len(w) > 2)
    return words


def type_tags_4e(type_line):
    t = (type_line or "").lower()
    keywords = re.findall(r"\(([^)]*)\)", t)
    body = re.sub(r"\([^)]*\)", "", t)
    tags = set()
    if "magical beast" in body:
        tags.add(TYPE_MAP_4E["magical beast"])
        body = body.replace("magical beast", "")
    for word, tag in TYPE_MAP_4E.items():
        if re.search(rf"\b{word}\b", body):
            tags.add(tag)
    for kw in (k.strip() for group in keywords for k in group.split(",")):
        tags.add(KEYWORD_TYPES.get(kw, kw))
    if "immortal" in tags and not tags & {"fiend", "celestial"}:
        tags.add("celestial" if "angel" in t else "fiend")
    return {x for x in tags if x}


def type_tags_5e(type_line):
    t = (type_line or "").lower().split(",")[0]
    keywords = re.findall(r"\(([^)]*)\)", t)
    body = re.sub(r"\([^)]*\)", "", t)
    tags = {w for w in body.split() if w not in SIZES}
    for kw in (k.strip() for group in keywords for k in group.split(",")):
        tags.add(KEYWORD_TYPES.get(kw, kw))
    if "swarm" in t:
        tags.add("swarm")
    return {x for x in tags if x and x not in {"of", "swarm"}} | ({"swarm"} if "swarm" in t else set())


def size_of(type_line):
    words = (type_line or "").lower().split()
    return next((w for w in words if w in SIZES), None)


def profile_4e(rec):
    meta = rec.get("meta", {})
    base, alts = split_name(rec["name"])
    powers = meta.get("powers", [])
    return {
        "edition": "4e",
        "name": base,
        "alt_names": alts,
        "family": meta.get("family"),
        "size": size_of(rec.get("type_line")),
        "types": type_tags_4e(rec.get("type_line")),
        "moral": moral_axis(meta.get("alignment")),
        "languages": languages(meta.get("languages")),
        "senses": tagged(meta.get("senses"), SENSES),
        "moves": tagged(meta.get("speed"), MOVES) - {"phasing"},
        "skills": skills(meta.get("skills")),
        "attacks": attack_words(powers),
        "damage": damage_types(meta.get("text")),
        "text": meta.get("text", ""),
        "lore": meta.get("lore", ""),
    }


def profile_5e(rec):
    type_line = rec.get("type_line", "")
    alignment = type_line.split(",", 1)[1] if "," in type_line else ""
    base, alts = split_name(rec["name"])
    return {
        "edition": "5e",
        "name": base,
        "alt_names": alts,
        "family": None,
        "size": size_of(type_line),
        "types": type_tags_5e(type_line),
        "moral": moral_axis(alignment),
        "languages": languages(rec.get("languages")),
        "senses": tagged(rec.get("senses"), SENSES),
        "moves": tagged(rec.get("speed"), MOVES),
        "skills": skills(rec.get("skills")),
        "attacks": attack_words(a.get("name", "") for a in rec.get("attacks", [])),
        "damage": damage_types(rec.get("meta", {}).get("text")),
        "text": rec.get("meta", {}).get("text", ""),
        "lore": rec.get("lore", ""),
    }


def profile_text(p):
    """Name-free sentence describing the profile, for embedding."""
    def fmt(items):
        return ", ".join(sorted(items)) or "none"
    return (
        f"{p['size'] or 'unknown size'} {fmt(p['types'])} creature. "
        f"Alignment: {p['moral']}. Languages: {fmt(p['languages'])}. "
        f"Senses: {fmt(p['senses'])}. Movement: walk{', ' + fmt(p['moves']) if p['moves'] else ''}. "
        f"Skills: {fmt(p['skills'])}. Attacks: {fmt(p['attacks'])}. "
        f"Damage: {fmt(p['damage'])}."
    )
