#!/usr/bin/env python3
"""Apply hand-authored corrections (manual/overrides_*.json) to parsed
stat blocks. Each override matches exactly one block on its fields
(`type_contains` does a substring match on type_line) and patches it;
patched blocks are marked with `manual_fix` so downstream steps can
distinguish curated values from parsed ones."""
import json


def apply_overrides(monsters, path):
    with open(path, encoding="utf-8") as f:
        overrides = json.load(f)["overrides"]
    applied = 0
    for ov in overrides:
        match = ov["match"]
        patch = ov.get("set", {})
        drop = ov.get("drop", False)

        def fits(mon):
            for k, v in match.items():
                if k == "type_contains":
                    if v.lower() not in mon.get("type_line", "").lower():
                        return False
                elif str(mon.get(k)) != str(v):
                    return False
            return True

        hits = [m for m in monsters if fits(m)]
        if len(hits) != 1:
            print(f"  override warning: {match} matched {len(hits)} blocks")
            continue
        hit = hits[0]
        if drop:
            monsters.remove(hit)
        else:
            hit.update(patch)
            hit["manual_fix"] = ov.get("note", True)
        applied += 1
    return applied
