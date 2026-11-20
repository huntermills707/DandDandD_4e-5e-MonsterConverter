"""Damage measures read from stat-block text, for both editions.

5e
    damage_avg        best single attack, riders included ("Hit: 6 (1d6 + 3)
                      bludgeoning damage plus 9 (2d8) fire damage" = 15)
    damage_per_round  the Multiattack routine ("three attacks: one with its
                      bite and two with its claws" = bite + 2 claws), or the
                      best single attack for monsters without one. None when
                      the routine names an attack that wasn't parsed (block
                      text cut by a page break), rather than a guess. This is
                      the quantity the 5e DMG sets by CR.

4e
    damage_avg        the basic attack (the lowercase m / r power icon),
                      the at-will per-hit damage the 4e DMG sets by level;
                      falls back to the best at-will standard attack.
    damage_max        the largest damage of any power (recharge breath
                      weapons, encounter powers): the old 4e damage_avg.
    attacks_per_round attacks of the best at-will standard action (a power
                      that "makes two claw attacks" counts 2).

Riders, multiattack counts, and condition-dependent extras ("extra 2d8
against a stunned target") are matched by regex; conditional extras and
spells/save-based damage are left out of every measure.
"""

import re

NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6}

# ---------------------------------------------------------------- 5e

ATTACK_5E = re.compile(
    r"([A-Z][\w'’ \-]*?)(?: \([^)]{0,30}\))?\. (Melee|Ranged|Melee or Ranged) "
    r"(?:Weapon|Spell) Attack: \+\s?(\d+) to hit"
    r"((?:(?!Attack:).)*?)\bHit: (\d+) \(([\dd+\-\s]+)\)([^.]*)"
)
RIDER_5E = re.compile(r"plus (\d+) \([\dd+\-\s]+\)")
GENERIC_ATTACK_WORDS = {"melee", "ranged", "weapon", "spell", "other", "more", "of", "its"}


def ocr_fix_5e(text):
    """Repair the OCR slips that break damage expressions in the 5e scan."""
    text = re.sub(r"(?<=[\s(])[lI](?=d\d)", "1", text)       # ld6   -> 1d6
    text = re.sub(r"(?<=\dd)[lI]", "1", text)                  # 2dl0  -> 2d10
    text = re.sub(r"(?<=\dd1) (?=\d[\s)+])", "", text)         # 1d1 2 -> 1d12
    text = re.sub(r"(?<=\dd\d) (?=\d[\s)+])", "", text)        # 2d1 0 -> 2d10
    text = re.sub(r"Attack:\s*\+\s*", "Attack: +", text)       # Attack:+ 13
    return text


def attacks_5e(text):
    """[{name, bonus, base, damage, melee}] for every attack roll in the text."""
    out = []
    for m in ATTACK_5E.finditer(ocr_fix_5e(text)):
        name = m.group(1).replace("ACTIONS", "").strip().lower()
        base = int(m.group(5))
        riders = sum(int(r.group(1)) for r in RIDER_5E.finditer(m.group(7)))
        out.append({"name": name, "bonus": int(m.group(3)), "base": base,
                    "damage": base + riders, "melee": m.group(2).startswith("Melee")})
    return out


def _stem(word):
    word = word.lower().strip(".,:;- ")
    for suffix in ("es", "s"):
        if word.endswith(suffix) and len(word) > 4:
            return word[: -len(suffix)]
    return word


def _attack_named(attacks, word):
    stem = _stem(word)
    return next((a for a in attacks if any(_stem(t) == stem for t in a["name"].split())), None)


def _routine_damage(alt, attacks):
    """Damage of one Multiattack alternative; None if it can't be resolved,
    0.0 if it names no attacks."""
    total, found = 0.0, False
    # "one with its bite and two with its claws"
    for m in re.finditer(r"\b(one|two|three|four)\s+(?:with|using) (?:its|their) ([a-z]+)", alt):
        a = _attack_named(attacks, m.group(2))
        if a is None:
            return None
        total += NUMBER_WORDS[m.group(1)] * a["damage"]
        found = True
    if found:
        return total
    # "uses Fire Ray twice"
    if m := re.search(r"uses (?:its )?([A-Z][a-z]+)[A-Za-z ]*? (twice|three times)", alt):
        a = _attack_named(attacks, m.group(1))
        if a is None:
            return None
        return (2 if m.group(2) == "twice" else 3) * a["damage"]
    # "as many bite attacks as it has heads": counted as five heads
    if m := re.search(r"as many ([a-z]+) attacks as it has heads", alt):
        a = _attack_named(attacks, m.group(1))
        return None if a is None else 5 * a["damage"]
    # "makes 1d4 Rotting Touch attacks"
    if re.search(r"makes 1d4 ", alt):
        return 2.5 * max(a["damage"] for a in attacks)
    # "three claw attacks and one bite attack", "two melee attacks", "two attacks"
    alt = alt.replace("strikes", "attacks").replace("strike", "attack")
    for m in re.finditer(r"\b(one|two|three|four|five|six|a|an)\s+"
                         r"([a-z]+\s+)?([a-z]+\s+)?attacks?\b", alt):
        words = [w.strip() for w in (m.group(2), m.group(3)) if w]
        named = [w for w in words if w not in GENERIC_ATTACK_WORDS]
        a = next((x for x in (_attack_named(attacks, w) for w in named) if x), None)
        if a is None and named and not words[-1] in GENERIC_ATTACK_WORDS:
            return None  # names an attack we don't have
        if a is None:
            pool = attacks
            if "melee" in words:
                pool = [x for x in attacks if x["melee"]] or attacks
            elif "ranged" in words:
                pool = [x for x in attacks if not x["melee"]] or attacks
            a = max(pool, key=lambda x: x["damage"])
        total += NUMBER_WORDS[m.group(1)] * a["damage"]
        found = True
    return total if found else 0.0


def multiattack_damage(text, attacks):
    """Damage of the Multiattack routine: the best of its alternatives.
    None when there is no Multiattack or it can't be resolved."""
    text = ocr_fix_5e(text)
    i = text.find("Multiattack")
    if i < 0 or not attacks:
        return None
    seg = text[i + len("Multiattack"): i + 400]
    j = min((k for k in (seg.find("Weapon Attack"), seg.find("Spell Attack")) if k >= 0),
            default=len(seg))
    seg = seg[:j]
    results = [_routine_damage(alt, attacks)
               for alt in re.split(r"\bor\b|;|Alternatively", seg)]
    if any(r is None for r in results):
        return None
    best = max(results, default=0.0)
    return best or None


def damage_5e(rec):
    """{'damage_avg', 'attack_bonus', 'damage_per_round'} for a parsed 5e block.

    Blocks without text (manual overrides drop text contaminated by a
    neighboring block) fall back to their hand-corrected attack list, and
    have no damage_per_round."""
    text = (rec.get("meta") or {}).get("text", "")
    attacks = attacks_5e(text) if text else []
    if not attacks:
        attacks = [{"name": a.get("name", "").lower(), "bonus": a.get("bonus"),
                    "damage": a["damage_avg"], "melee": True}
                   for a in rec.get("attacks", []) if a.get("damage_avg") is not None]
    if not attacks:
        return {"damage_avg": None, "attack_bonus": None, "damage_per_round": None}
    best = max(attacks, key=lambda a: a["damage"])
    per_round = None
    if text:
        if "Multiattack" in ocr_fix_5e(text):
            multi = multiattack_damage(text, attacks)
            per_round = None if multi is None else max(multi, best["damage"])
        else:
            per_round = best["damage"]
    return {"damage_avg": float(best["damage"]), "attack_bonus": best["bonus"],
            "damage_per_round": None if per_round is None else float(per_round)}


# ---------------------------------------------------------------- 4e

POWER_4E = re.compile(
    r"(?:^|\s)(?:([mrcaMRCA]) )?([A-Z][A-Za-z'’\- ]{1,40}?) "
    r"\(((?:standard|move|minor|free|immediate|no action)[^)]*)\)"
)
ATTACK_4E = re.compile(
    r"\+(\d+) vs\. (AC|Fortitude|Reflex|Will); [^.]*?"
    r"(?:(\d+)d(\d+)(?:\s*\+\s*(\d+))?|(\d+)) (?:[a-z]+ )?damage"
)
MULTI_4E = re.compile(r"makes (two|three|four) (?:[a-z]+ )?(?:basic )?attacks")
PAIR_4E = re.compile(r"makes (?:a|one) [a-z ]+? attack and (?:a|one) [a-z ]+? attack")


def powers_4e(text):
    """[{mark, name, usage, damage, bonus, count, body}] for each power.
    mark: the power icon letter (lowercase m/r = basic attack)."""
    marks = list(POWER_4E.finditer(text))
    out = []
    for k, m in enumerate(marks):
        body = text[m.end(): marks[k + 1].start() if k + 1 < len(marks) else len(text)]
        hits = []
        for a in ATTACK_4E.finditer(body):
            if a.group(6):
                dmg = float(a.group(6))  # minions: fixed damage
            else:
                dmg = int(a.group(3)) * (int(a.group(4)) + 1) / 2 + int(a.group(5) or 0)
            hits.append((dmg, int(a.group(1))))
        count = 1
        if mm := MULTI_4E.search(body):
            count = NUMBER_WORDS[mm.group(1)]
        elif PAIR_4E.search(body):
            count = 2
        best = max(hits, default=(None, None))
        out.append({"mark": m.group(1) or "", "name": m.group(2).strip(),
                    "usage": m.group(3).lower(), "damage": best[0], "bonus": best[1],
                    "count": count, "body": body})
    return out


def _atwill_standard(p):
    return p["usage"].startswith("standard") and "at-will" in p["usage"]


def damage_4e(rec):
    """{'damage_avg', 'attack_bonus', 'damage_max', 'attacks_per_round'}."""
    pw = powers_4e((rec.get("meta") or {}).get("text", ""))
    dealing = [p for p in pw if p["damage"] is not None]
    basic = [p for p in dealing if p["mark"] in ("m", "r")]
    atwill = [p for p in dealing if _atwill_standard(p)]
    main = max(basic or atwill or dealing, key=lambda p: p["damage"], default=None)
    # per-round attacks: a multi-attack at-will standard power, else 1
    multi = [p["count"] for p in pw if _atwill_standard(p) and p["count"] > 1]
    return {
        "damage_avg": None if main is None else float(main["damage"]),
        "attack_bonus": None if main is None else main["bonus"],
        "damage_max": max((p["damage"] for p in dealing), default=None),
        "attacks_per_round": max(multi, default=1) if dealing else None,
    }
