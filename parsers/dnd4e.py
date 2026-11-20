#!/usr/bin/env python3
"""Parse 4e Monster Manual stat blocks from pdftotext output.

Usage: pdftotext "Monster Manual 4e.pdf" data/raw/mm4.txt
       python3 -m parsers.dnd4e data/raw/mm4.txt data/raw/mm4_monsters.jsonl

Handles the two big layout hazards of the 4e MM:
- stat blocks split across page breaks (the numeric half is orphaned and
  re-attached via the level/role -> XP table as a join key)
- index/sidebar junk that mimics block headers (filtered by name validation
  and by dropping blocks that never gained any stat content)
"""
import json
import os
import re
import sys
import unicodedata

ROLES = (
    r"(?:Elite |Solo )?(?:Artillery|Brute|Controller|Lurker|Skirmisher|"
    r"Soldier|Minion)(?: \(Leader\))?"
)
LEVEL_RE = re.compile(rf"^Level (\d+) ({ROLES})$")
XP_RE = re.compile(r"(?:^|\s)XP ([\d,]+)$")
HP_RE = re.compile(r"HP ([\d,]+)(?:; Bloodied ([\d,]+))?")
PAREN = r"(?:\s*\([^)]*\))?"  # conditional note, e.g. "(34 against ranged attacks)"
DEF_RE = re.compile(
    rf"AC (\d+){PAREN}[;,]\s*Fortitude (\d+){PAREN}[;,]\s*"
    rf"Reflex (\d+){PAREN}[;,]\s*Will (\d+)"
)
INIT_RE = re.compile(r"Initiative \+(\d+)")
PERC_RE = re.compile(r"Perception \+(\d+)")
ABIL_RE = re.compile(r"\b(Str|Dex|Wis|Con|Int|Cha) (\d+) \([+–-]\d+\)")
CHA_RE = re.compile(r"\bCha \d+ \([+–-]\d+\)")
TYPE_RE = re.compile(r"^(Tiny|Small|Medium|Large|Huge|Gargantuan)\b")
ATK_RE = re.compile(r"\+(\d+) vs\. (AC|Fortitude|Reflex|Will); (.+)")
DICE_RE = re.compile(r"(\d+d\d+(?:\s*\+\s*\d+)?)")

# Lines that end a stat block ("X Tactics" / "X Lore" section headings; a
# trailing "$" so mid-block powers like "Feywild Tactics aura 10" don't match)
STOP_RE = re.compile(r"( Tactics$| Lore$)")

# Standard 4e XP by level (elite x2, solo x5, minion /4)
XP_TABLE = {
    1: 100, 2: 125, 3: 150, 4: 175, 5: 200, 6: 250, 7: 300, 8: 350,
    9: 400, 10: 500, 11: 600, 12: 700, 13: 800, 14: 1000, 15: 1200,
    16: 1400, 17: 1600, 18: 2000, 19: 2400, 20: 2800, 21: 3200, 22: 4150,
    23: 5100, 24: 6050, 25: 7000, 26: 9000, 27: 11000, 28: 13000,
    29: 15000, 30: 19000, 31: 23000, 32: 27000, 33: 31000,
}

CORE_FIELDS = {"hp", "ac", "fort", "ref", "will", "xp"}


def expected_xp(level: int, role: str) -> set[int]:
    base = XP_TABLE.get(level)
    if base is None:
        return set()
    if role.startswith("Elite"):
        return {base * 2}
    if role.startswith("Solo"):
        return {base * 5}
    if "Minion" in role:
        # minion XP rounds to the nearest whole (e.g. L6: 250/4 -> 63)
        return {base // 4, -(-base // 4)}
    return {base}


def name_ok(name: str) -> bool:
    """Filter index lines, sidebars, and sentence fragments that precede a
    'Level N Role' line but are not monster names."""
    if len(name) >= 60 or not name:
        return False
    if name.startswith(("Level", "XP ", "Equipment")):
        return False
    if "." in name or name[0].islower():
        return False
    if name.endswith(" Lore") or name == "Encounters":
        return False
    return True


def num(s: str) -> int:
    return int(s.replace(",", ""))


def normalize(text: str) -> str:
    # PyMuPDF splits the fl ligature from the rest of the word ("Reﬂ ex",
    # "ﬂ y"); rejoin it, then fix ligatures via NFKC (Reﬂex -> Reflex)
    return unicodedata.normalize("NFKC", text.replace("ﬂ ", "ﬂ"))


def parse_fields(mon: dict, lines) -> None:
    """Fill stat fields from block lines into mon (in place).

    Field patterns are searched, not anchored: word-level extraction can
    place several fields on one visual line ("HP 97; Bloodied 48 AC 25; ...").
    """
    attacks = []
    for line in lines:
        if "type_line" not in mon and TYPE_RE.match(line):
            mon["type_line"] = XP_RE.sub("", line).strip()
        if xm := XP_RE.search(line):
            mon.setdefault("xp", num(xm.group(1)))
        if im := INIT_RE.search(line):
            mon["initiative"] = int(im.group(1))
        if pm := PERC_RE.search(line):
            mon["perception"] = int(pm.group(1))
        if hm := HP_RE.search(line):
            mon["hp"] = num(hm.group(1))
            if hm.group(2):
                mon["bloodied"] = num(hm.group(2))
        if dm := DEF_RE.search(line):
            mon["ac"], mon["fort"], mon["ref"], mon["will"] = map(int, dm.groups())
        if ABIL_RE.search(line):
            for am in ABIL_RE.finditer(line):
                mon.setdefault("abilities", {})[am.group(1).lower()] = int(am.group(2))
        if am2 := ATK_RE.search(line):
            atk = {
                "bonus": int(am2.group(1)),
                "defense": am2.group(2),
                "text": am2.group(3),
            }
            if dm2 := DICE_RE.search(am2.group(3)):
                atk["damage_dice"] = re.sub(r"\s+", "", dm2.group(1))
            attacks.append(atk)
    if attacks:
        mon.setdefault("attacks", []).extend(attacks)


def block_end(lines, j):
    """True if line j terminates a block (next block header or lore)."""
    return STOP_RE.search(lines[j]) or (
        j + 1 < len(lines)
        and LEVEL_RE.match(lines[j + 1])
        and name_ok(lines[j])
    )


def scan_body(lines, level_idx):
    """Collect body lines following the 'Level N Role' line at level_idx.
    Returns (body_lines, end_index)."""
    j = level_idx + 1
    body = []
    while j < len(lines):
        ln = lines[j]
        if block_end(lines, j):
            break
        body.append(ln)
        j += 1
        # Ability scores end every 4e block; allow one trailing Equipment line
        if CHA_RE.search(ln):
            if j < len(lines) and lines[j].startswith("Equipment"):
                body.append(lines[j])
                j += 1
            break
    return body, j


def scan_block(lines, start):
    """Collect lines of one block beginning at `start` (name line).
    Returns (block_lines, end_index)."""
    body, j = scan_body(lines, start + 1)
    return [lines[start], lines[start + 1]] + body, j


def parse_block(block):
    m = LEVEL_RE.match(block[1])
    mon = {"name": block[0], "level": int(m.group(1)), "role": m.group(2)}
    parse_fields(mon, block[2:])
    return mon


CAPS_RE = re.compile(r"^[A-Z][A-Z'’ ,()-]{3,}$")
SMALL_WORDS = {"of", "the", "and", "a", "an", "in", "on", "de"}


def recover_name(lines, level_idx):
    """Recover a monster name for a header whose name line was lost to a
    page/column break: use the nearest preceding ALL-CAPS entry header."""
    for k in range(level_idx - 1, max(-1, level_idx - 12), -1):
        ln = lines[k]
        if CAPS_RE.match(ln):
            words = ln.lower().split()
            return " ".join(
                w if n > 0 and w in SMALL_WORDS else w.capitalize()
                for n, w in enumerate(words)
            )
    return None


def collect_fragments(lines, spans):
    """Collect orphan fragments: stat-block bodies (starting at a type line)
    that lost their header to a page break."""
    occupied = set()
    for a, b in spans:
        occupied.update(range(a, b))
    fragments = []  # (start_index, fields_dict)
    i = 0
    while i < len(lines):
        if i not in occupied and TYPE_RE.match(lines[i]):
            frag = [lines[i]]
            j = i + 1
            while j < len(lines) and j not in occupied and not block_end(lines, j):
                frag.append(lines[j])
                if CHA_RE.search(lines[j]):
                    j += 1
                    break
                j += 1
            mon = {}
            parse_fields(mon, frag)
            if "xp" in mon:
                mon["_text"] = " ".join(frag)
                fragments.append((i, mon))
                occupied.update(range(i, j))
                i = j
                continue
        i += 1
    return fragments


def merge_fragments(blocks, fragments):
    """Attach orphan bodies to incomplete blocks via expected XP."""
    merged = 0
    remaining = []
    for frag_idx, frag in fragments:
        best = None
        for idx, (start, end, mon) in enumerate(blocks):
            if CORE_FIELDS <= mon.keys():
                continue
            if frag.get("xp") in expected_xp(mon["level"], mon["role"]):
                dist = abs(frag_idx - end)
                if best is None or dist < best[0]:
                    best = (dist, idx)
        if best is None:
            remaining.append((frag_idx, frag))
            continue
        _, idx = best
        start, end, mon = blocks[idx]
        frag_attacks = frag.pop("attacks", [])
        frag.pop("_text", None)
        for k, v in frag.items():
            if k == "abilities":
                mon.setdefault("abilities", {}).update(
                    {a: b for a, b in v.items() if a not in mon.get("abilities", {})}
                )
            else:
                mon.setdefault(k, v)
        mon["attacks"] = frag_attacks + mon.get("attacks", [])
        merged += 1
    return merged, remaining


APPENDIX_RE = re.compile(rf"^(.*?)\s+(Level \d+ {ROLES}(?: \(L\))?)$")


def parse_appendix(lines):
    """'Monsters by Level' appendix -> [(name, level, role)]. The appendix
    header info can stand in for stat-block headers lost to layout
    scrambling (bodies are joined back via expected XP)."""
    start = next(
        (k for k, ln in enumerate(lines)
         if k > 1000 and "MONSTERS" in ln and "BY" in ln and "LEVEL" in ln),
        None,
    )
    entries = []
    if start is None:
        return entries
    for ln in lines[start:]:
        if m := APPENDIX_RE.match(ln):
            name = m.group(1).strip()
            lm = LEVEL_RE.match(m.group(2).replace("(L)", "(Leader)"))
            if lm and name_ok(name):
                entries.append((name, int(lm.group(1)), lm.group(2)))
    return entries


def main(txt_path, out_path):
    raw = normalize(open(txt_path, encoding="utf-8").read())
    lines = [ln.strip() for ln in raw.splitlines()]
    # Drop empties, page numbers, form-feeds, and page markers
    lines = [ln for ln in lines if ln and not re.fullmatch(r"\d{1,3}|\f|<<<PAGE \d+>>>", ln)]
    appendix = parse_appendix(lines)
    # Content ends at the glossary; the index/appendix only mimics blocks.
    # (Take the last occurrence: running heads read "GLOSSARY GLOSSARY"
    # and the TOC has an early entry.)
    glossary = [k for k, ln in enumerate(lines) if ln.startswith("GLOSSARY")]
    if glossary:
        lines = lines[: glossary[-1]]

    # Drop spaced-letter footers and artist credits ("D R AG O N")
    def is_credit(ln: str) -> bool:
        toks = ln.split()
        return (
            len(toks) >= 3
            and all(re.fullmatch(r"[A-Z]{1,3}(-[A-Z]{1,3})*", t) for t in toks)
        )

    lines = [ln for ln in lines if not is_credit(ln)]
    # Merge wrapped continuations into the previous line: parentheticals,
    # and defense lines whose "Will N" part wrapped to the next line
    merged_lines = []
    for ln in lines:
        if merged_lines and (
            ln.startswith("(")
            or (merged_lines[-1].startswith("AC ") and "Will" not in merged_lines[-1])
        ):
            merged_lines[-1] += " " + ln
        else:
            merged_lines.append(ln)
    lines = merged_lines
    # The stat-box header bar has the name and "Level N Role" on the same
    # visual line; split them into name line + Level line
    HEADER_SPLIT_RE = re.compile(rf"^(.*?)\s+(Level \d+ {ROLES})$")
    split_lines = []
    for ln in lines:
        if m := HEADER_SPLIT_RE.match(ln):
            split_lines.extend([m.group(1), m.group(2)])
        else:
            split_lines.append(ln)
    lines = split_lines

    blocks = []  # (start, end, mon)
    i = 0
    while i < len(lines) - 1:
        if LEVEL_RE.match(lines[i + 1]) and name_ok(lines[i]):
            block, j = scan_block(lines, i)
            blocks.append((i, j, parse_block(block)))
            i = j
        else:
            i += 1

    # Recover headers whose name line was lost to a page/column break:
    # Level lines not covered by any block span get a name from the
    # nearest preceding ALL-CAPS entry header.
    covered = set()
    for a, b, _ in blocks:
        covered.update(range(a, b))
    recovered = []
    for idx, ln in enumerate(lines):
        if idx not in covered and LEVEL_RE.match(ln):
            name = recover_name(lines, idx)
            if not name:
                continue
            body, j = scan_body(lines, idx)
            mon = parse_block([name, ln])
            parse_fields(mon, body)
            mon["name_recovered"] = True
            blocks.append((idx, j, mon))
            recovered.append(name)

    spans = [(a, b) for a, b, _ in blocks]
    fragments = collect_fragments(lines, spans)
    n_merged, orphans = merge_fragments(blocks, fragments)

    # Orphan bodies whose headers were lost entirely: re-head them with
    # appendix entries when XP narrows to candidates and the body's own
    # text mentions exactly one ("the dwarf bolter ...")
    n_reheaded = 0
    for _, frag in orphans:
        hits = [
            (name, level, role)
            for name, level, role in appendix
            if frag.get("xp") in expected_xp(level, role)
        ]
        text = frag.pop("_text", "").lower()
        mentioned = [
            h for h in hits
            if any(
                re.search(rf"\b{re.escape(w)}\b", text)
                for w in re.sub(r"\(.*?\)", "", h[0]).lower().split()
                if len(w) > 4
            )
        ]
        if len(mentioned) == 1:
            hits = mentioned
        if len(hits) != 1:
            continue
        name, level, role = hits[0]
        mon = {"name": name, "level": level, "role": role, "name_recovered": True}
        mon.update(frag)
        blocks.append((-1, -1, mon))
        n_reheaded += 1

    # Drop blocks that never gained any stat content (index/sidebar junk)
    monsters = [m for _, _, m in blocks if len(m) > 3]
    dropped = len(blocks) - len(monsters)

    override_path = "manual/local/overrides_4e.json"
    n_overrides = 0
    if os.path.exists(override_path):
        from parsers.overrides import apply_overrides

        n_overrides = apply_overrides(monsters, override_path)

    with open(out_path, "w", encoding="utf-8") as f:
        for mon in monsters:
            f.write(json.dumps(mon, ensure_ascii=False) + "\n")

    complete = [m for m in monsters if CORE_FIELDS <= m.keys()]
    with_atk = [m for m in complete if m.get("attacks")]
    print(f"block headers found: {len(blocks)} ({dropped} junk dropped, {len(recovered)} name-recovered, {n_overrides} manual overrides)")
    if recovered:
        print(f"  recovered: {recovered}")
    print(f"orphan fragments: {len(fragments)} ({n_merged} merged, {n_reheaded} re-headed via appendix, {len(orphans) - n_reheaded} unattached)")
    print(f"blocks with core numeric fields: {len(complete)}")
    print(f"...and at least one parsed attack: {len(with_atk)}")
    if monsters:
        levels = sorted({m["level"] for m in monsters})
        print(f"level range: {levels[0]}-{levels[-1]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
