"""How many 4e partners each 5e monster may keep.

Matching is 4e -> 5e, so one 5e monster can collect many 4e partners (5e
"Goblin" <- six 4e goblin variants), letting large 4e families dominate
training. Modes:

    many   keep every accepted partner (no limit)
    one    1-to-1: keep the ~exact partner if there is one (the best-scoring
           one if several), else the best-scoring partner
    elbow  1-to-N: keep every ~exact partner (the anchor), then add the
           remaining partners best-first until the score falls by more than
           `elbow_ratio` x from the previous partner. Flat families keep all
           members; a lone standout keeps only itself.

Scores are the method's pair score (metadata probability for the
"metadata" method, name similarity for "legacy").
"""

from collections import defaultdict

MODES = ("many", "one", "elbow")


def select(pairs, mode="many", elbow_ratio=2.0):
    """pairs: dicts with i5, score, match_type ('same' = ~exact).
    Returns (kept, dropped)."""
    if mode not in MODES:
        raise ValueError(f"cardinality must be one of {MODES}, got {mode!r}")
    if mode == "many":
        return list(pairs), []

    groups = defaultdict(list)
    for p in pairs:
        groups[p["i5"]].append(p)

    kept, dropped = [], []
    for group in groups.values():
        anchors = sorted((p for p in group if p["match_type"] == "same"),
                         key=lambda p: -p["score"])
        others = sorted((p for p in group if p["match_type"] != "same"),
                        key=lambda p: -p["score"])
        ordered = anchors + others
        if mode == "one":
            n_keep = 1
        else:
            n_keep = max(len(anchors), 1)
            while n_keep < len(ordered):
                prev, cur = ordered[n_keep - 1]["score"], ordered[n_keep]["score"]
                if cur * elbow_ratio < prev:
                    break
                n_keep += 1
        kept.extend(ordered[:n_keep])
        dropped.extend(ordered[n_keep:])
    return kept, dropped
