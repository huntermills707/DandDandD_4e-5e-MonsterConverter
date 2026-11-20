#!/usr/bin/env python3
"""Parse 5e Monster Manual stat blocks from geometry-extracted text.

The 5e MM PDF is a scan with an OCR text layer; geometry extraction
(parsers/extract_text.py) removes most noise, and remaining OCR quirks
("lnt" for Int, "18 {+4)", "4SO XP", "11uge monstrosity", "12 (+l)") are
normalized or fuzzily matched here.

Name lines lost to page breaks or decorative fonts are recovered from the
book's INDEX OF STAT BLOCKS: each block's printed page (pdf page - 1)
locates the index entries for that page, and unclaimed names are assigned
to unnamed blocks (type-line parenthetical match first, then order).
Garbled names ("B,ULETI:E") are fuzzy-repaired against the same page list.

Usage: python3 -m parsers.extract_text "D&D 5E - Monster Manual.pdf" data/raw/mm5_geo.txt
       python3 -m parsers.dnd5e data/raw/mm5_geo.txt data/raw/mm5_monsters.jsonl
"""
import difflib
import os
import json
import re
import sys

# Fuzzy size word: tolerates OCR garbage before/inside ("11uge", "La~ge")
SIZE_FUZZY = r"(?:tiny|small|medium|large|huge|gargantuan|(?<![a-z])(?:iny|mall|edium|arge|uge|argantuan))"
TYPE_WORD = (
    r"aberration|beast|celestial|construct|dragon|elemental|fey|fiend|"
    r"giant|humanoid|monstrosity|ooze|plant|undead"
)
TYPE5_RE = re.compile(rf"^.*{SIZE_FUZZY}\s+\w*.*?({TYPE_WORD})", re.I)
AC_RE = re.compile(r"Armor Class (\d+)(?: \(([^)]+)\))?")
HP_RE = re.compile(r"Hit Points (\d+) \(([^)]+)\)")
SPEED_RE = re.compile(r"^Speed (.+)$")
STR_HDR_RE = re.compile(r"^STR\b")
# Score line anchor: digits followed by a short (however garbled)
# parenthetical modifier; only the score is used, so the modifier's OCR
# noise doesn't matter. "ll" -> 11, "2. (-4)", "18 (+4i", "11 (;t-0)" all ok.
SCORE_RE = re.compile(r"([0-9l]+)[.,]?\s*[({][^)]{0,7}[)}]?")
SAVE_RE = re.compile(r"^Saving Throws (.+)$")
SKILL_RE = re.compile(r"^Skills (.+)$")
SENSES_RE = re.compile(r"^Senses (.+)$")
LANG_RE = re.compile(r"^Languages? (.+)$")
CR_RE = re.compile(r"Challenge ([0-9/]+) \(([^)]+)\)")
MOD_RE = re.compile(r"(Str|Dex|Con|lnt|Int|Wis|Cha)\s*\+(\d+)")
ATK_RE = re.compile(
    r"([A-Z][\w'’ +\-]*?)\. (?:Melee|Ranged|Melee or Ranged) (?:Weapon|Spell) "
    r"Attack: \+(\d+) to hit.*?\bHit: (\d+) \(([\dd+\s]+)\)"
)
FIELD_PREFIX = re.compile(
    r"^(Armor|Hit|Speed|Saving|Skills|Senses|Languages?|Challenge|Damage|"
    r"Condition|STR|ACTIONS|ACT|REACTIONS|LEGENDARY|LAIR|BONUS|MYTHIC|"
    r"REGIONAL|VARIANT)"
)
INDEX_ENTRY_RE = re.compile(r"([A-Z][A-Za-z'’() \-]*?), (\d{1,3})(?=\s+[A-Z]|\s*$)")
PAGE_MARK_RE = re.compile(r"<<<PAGE (\d+)>>>")

# CR -> XP (5e DMG; note the MM prints bespoke XP for some high-CR dragons)
CR_XP = {
    "0": 10, "1/8": 25, "1/4": 50, "1/2": 100, "1": 200, "2": 450, "3": 700,
    "4": 1100, "5": 1800, "6": 2300, "7": 2900, "8": 3900, "9": 5000,
    "10": 5900, "11": 7200, "12": 8400, "13": 10000, "14": 11500,
    "15": 13000, "16": 15000, "17": 18000, "18": 20000, "19": 22000,
    "20": 25000, "21": 33000, "22": 41000, "23": 50000, "24": 62000,
    "25": 75000, "26": 90000, "27": 105000, "28": 120000, "29": 135000,
    "30": 155000,
}

SMALL_WORDS = {"of", "the", "and", "a", "an", "in", "on", "de"}
ABILITIES = ["str", "dex", "con", "int", "wis", "cha"]


def ocr_clean(line: str) -> str:
    line = line.replace("~", "")  # OCR garbage ("La~ge" -> "La ge" handled fuzzy)
    line = re.sub(r"(\w)\.(\w)", r"\1\2", line)  # "neutr.al" -> "neutral"
    return line


def clean_xp(raw: str) -> int | None:
    raw = re.sub(r"[Oo]", "0", raw)
    raw = re.sub(r"[Ss]", "5", raw)
    raw = re.sub(r"[lI]", "1", raw)
    digits = re.sub(r"[^\d]", "", raw)
    return int(digits) if digits else None


def title_name(caps: str) -> str:
    words = caps.lower().split()
    return " ".join(
        w if n > 0 and w in SMALL_WORDS else w.capitalize()
        for n, w in enumerate(words)
    )


def name_plausible(ln: str) -> bool:
    if not 2 < len(ln) < 50 or FIELD_PREFIX.match(ln):
        return False
    alpha = [c for c in ln if c.isalpha()]
    if len(alpha) < 2:
        return False
    return sum(c.isupper() for c in alpha) / len(alpha) > 0.6


def parse_block(name, type_line, block_lines):
    mon = {}
    if name:
        mon["name"] = title_name(name)
        mon["name_caps"] = name
    mon["type_line"] = type_line
    text = " ".join(block_lines)
    for i, ln in enumerate(block_lines):
        if m := AC_RE.search(ln):
            mon["ac"] = int(m.group(1))
            if m.group(2):
                mon["ac_note"] = m.group(2)
        elif m := HP_RE.search(ln):
            mon["hp"] = int(m.group(1))
            mon["hit_dice"] = re.sub(r"\s+", "", m.group(2))
        elif m := SPEED_RE.match(ln):
            mon["speed"] = m.group(1)
        elif STR_HDR_RE.match(ln):
            # scores can be split across lines or under a wrapped header
            blob = " ".join(block_lines[i + 1 : i + 4]).replace("{", "(").replace("}", ")")
            scores = [s.replace("l", "1") for s in SCORE_RE.findall(blob)]
            if len(scores) >= 6:
                mon["abilities"] = {
                    ab: int(s) for ab, s in zip(ABILITIES, scores[:6])
                }
        elif m := SAVE_RE.match(ln):
            mon["saves"] = {
                k.replace("lnt", "Int").lower(): int(v)
                for k, v in MOD_RE.findall(m.group(1))
            }
        elif m := SKILL_RE.match(ln):
            mon["skills"] = m.group(1)
        elif m := SENSES_RE.match(ln):
            mon["senses"] = m.group(1)
        elif m := LANG_RE.match(ln):
            mon["languages"] = m.group(1)
        elif m := CR_RE.search(ln):
            mon["cr"] = m.group(1)
            if (xp := clean_xp(m.group(2))) is not None:
                mon["xp"] = xp
    attacks = [
        {
            "name": m.group(1).strip(),
            "bonus": int(m.group(2)),
            "damage_avg": int(m.group(3)),
            "damage_dice": re.sub(r"\s+", "", m.group(4)),
        }
        for m in ATK_RE.finditer(text)
    ]
    if attacks:
        mon["attacks"] = attacks
    return mon


def load_pages(txt_path):
    """Split extraction into lines with their printed page (pdf page - 1)."""
    lines, pages = [], []
    pdf_page = 0
    for ln in open(txt_path, encoding="utf-8").read().splitlines():
        if m := PAGE_MARK_RE.fullmatch(ln.strip()):
            pdf_page = int(m.group(1))
            continue
        ln = ocr_clean(ln.strip())
        if not ln or re.fullmatch(r"\d{1,3}", ln):
            continue
        lines.append(ln)
        pages.append(pdf_page - 1)
    return lines, pages


def parse_index_file(path):
    """Read the column-extracted index (parsers/extract_index.py output)
    into {printed_page: [names]}. Entries are parsed from the joined token
    stream, so wrapped names and split page numbers resolve naturally."""
    raw = open(path, encoding="utf-8").read()
    lines = [ln.strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"[A-Z]|\d{1,3}", ln)]
    stream = " ".join(lines)
    # page numbers split across a wrap: "Lamia, 20 1 Lemure" -> "... 201 ..."
    stream = re.sub(r"([.,] \d{2}) (\d)(?=\s+[A-Z(])", r"\1\2", stream)
    page_names = {}
    ENTRY = re.compile(r"([A-Z][A-Za-z'’()/\- ]+?)[.,] (\d{2,3})(?=\s+[A-Z(]|\s*$)")
    for m in ENTRY.finditer(stream):
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        page = int(m.group(2))
        if not name or ". See " in name or name.upper().startswith(("INDEX OF", "USE THIS")):
            continue
        if 10 <= page <= 351:
            page_names.setdefault(page, []).append(name)
    return page_names


def fix_names(monsters, pages, index):
    """Recover missing names and repair garbled ones via the index.

    A recovered name is only assigned when the monster's own block text
    references it ("the lemure ...") — blind positional assignment turned
    out to silently mislabel blocks (an Imp name landed on a Lemure).
    """
    recovered = repaired = 0
    by_page = {}
    for mon in monsters:
        by_page.setdefault(pages[mon["_line"]], []).append(mon)

    def text_mentions(mon, name):
        text = mon.get("_text", "").lower()
        return any(
            re.search(rf"\b{re.escape(w)}\b", text)
            for w in name.lower().split()
            if len(w) > 4
        )

    for page, page_mons in by_page.items():
        candidates = list(index.get(page, []))
        # claim candidates for named blocks (exact/fuzzy)
        for mon in page_mons:
            if "name" not in mon:
                continue
            caps = mon["name"].upper()
            if caps in [c.upper() for c in candidates]:
                candidates = [c for c in candidates if c.upper() != caps]
            else:
                close = difflib.get_close_matches(
                    caps, [c.upper() for c in candidates], n=1, cutoff=0.6
                )
                if close:
                    hit = close[0]
                    candidates = [c for c in candidates if c.upper() != hit]
                    if mon["name"].upper() != hit:
                        mon["name"] = title_name(hit)
                        mon["name_repaired"] = True
                        repaired += 1
                else:
                    mon["ocr_suspect"] = True
        # assign remaining candidates to unnamed blocks, but only when the
        # block's own text mentions the candidate
        for mon in page_mons:
            if "name" in mon or not candidates:
                continue
            mentioned = [c for c in candidates if text_mentions(mon, c)]
            if len(mentioned) == 1:
                pick = mentioned[0]
            elif (
                not mentioned
                and len(candidates) == 1
                and sum(1 for m in page_mons if "name" not in m) == 1
            ):
                pick = candidates[0]  # sole candidate, sole unnamed block
            else:
                continue
            candidates.remove(pick)
            mon["name"] = title_name(pick)
            mon["name_recovered"] = True
            recovered += 1
        for mon in page_mons:
            if "name" not in mon:
                mon["ocr_suspect"] = True
    return recovered, repaired


def main(txt_path, out_path, index_path=None):
    lines, pages = load_pages(txt_path)
    index = parse_index_file(index_path) if index_path else {}

    # Block anchors: Armor Class lines; the type line is within 6 lines above
    blocks = []  # (line_idx, name, type_line)
    for i, ln in enumerate(lines):
        if not AC_RE.search(ln):
            continue
        type_idx = None
        for up in range(1, 7):
            if i - up >= 0 and TYPE5_RE.match(lines[i - up]):
                type_idx = i - up
                break
        if type_idx is None:
            continue
        # name is usually directly above the type line, but OCR junk lines
        # can intervene; scan upward over them
        name, start = None, type_idx
        for up in range(1, 9):
            j = type_idx - up
            if j < 0:
                break
            alpha = [c for c in lines[j] if c.isalpha()]
            if len(alpha) < 2:
                continue  # junk line ("\", "n", page furniture)
            if name_plausible(lines[j]):
                name, start = lines[j], j
            break  # first real text line above the type line decides
        blocks.append((i, start, name, lines[type_idx]))

    monsters = []
    for n, (anchor, start, name, type_line) in enumerate(blocks):
        end = blocks[n + 1][1] if n + 1 < len(blocks) else len(lines)
        mon = parse_block(name, type_line, lines[start + 1 : end])
        mon["_line"] = anchor
        mon["_text"] = " ".join(lines[start + 1 : end])
        monsters.append(mon)

    recovered, repaired = fix_names(monsters, pages, index)

    override_path = "manual/local/overrides_5e.json"
    n_overrides = 0
    if os.path.exists(override_path):
        from parsers.overrides import apply_overrides

        n_overrides = apply_overrides(monsters, override_path)

    with open(out_path, "w", encoding="utf-8") as f:
        for mon in monsters:
            mon.pop("_line", None)
            mon.pop("_text", None)
            f.write(json.dumps(mon, ensure_ascii=False) + "\n")

    core = {"name", "ac", "hp", "abilities", "cr", "xp"}
    complete = [m for m in monsters if core <= m.keys()]
    with_atk = [m for m in complete if m.get("attacks")]
    xp_bad = [
        m for m in complete
        if CR_XP.get(m["cr"]) != m["xp"] and not m["cr"].isdigit()
    ]
    print(f"blocks found: {len(monsters)}")
    print(f"complete (name/ac/hp/abilities/cr/xp): {len(complete)} "
          f"({100 * len(complete) / len(monsters):.0f}%)")
    print(f"...and at least one parsed attack: {len(with_atk)}")
    print(f"names recovered via index: {recovered} | repaired: {repaired} | manual overrides: {n_overrides}")
    print(f"still unnamed: {sum(1 for m in monsters if 'name' not in m)}")
    missing = {}
    for m in monsters:
        for k in core - m.keys():
            missing[k] = missing.get(k, 0) + 1
    print(f"missing-field counts: {missing}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
