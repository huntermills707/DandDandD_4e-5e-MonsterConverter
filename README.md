# D&D Monster Edition Converter

Converts monster stat blocks between Dungeons & Dragons editions, starting
with **4th edition → 5th edition**.

Stat blocks are parsed from rule book PDFs into structured records, matched
across editions by monster name, and the numeric fields (HP, AC, attack
bonus, damage, level/CR, ...) are mapped with simple, human-readable linear
regression models. A Dash web app lets users convert their own typed-in
stats and shows the fitted equations for offline use.

**Legal note:** source PDFs and parsed data stay local and are never
distributed (see `.gitignore`). The web app only accepts user-typed numeric
input — it hosts no copyrighted text.

## Project structure

```
parsers/            edition-specific stat block parsers
    extract_text.py geometry-aware PDF text extraction (word positions)
    extract_index.py 3-column extraction for the 5e MM stat-block index
    dnd4e.py        4e Monster Manual parser
    dnd5e.py        5e Monster Manual parser (OCR-tolerant)
    overrides.py    applies manual/overrides_*.json corrections
manual/             hand-authored per-block corrections (tracked)
data/
    raw/            extracted PDF text + parsed JSONL (local only, gitignored)
    processed/      matched cross-edition training pairs (local only)
models/             fitted regression models + evaluation reports
app/                Dash web app (converter form + equations page)
tests/              parser and model tests
requirements.txt
```

## Pipeline

```
PDFs -> parsers/ -> JSONL stat blocks -> name matching -> training pairs
                                                       |
                              fitted linear models <---+
                                       |
                            Dash app (convert + explain)
```

## Milestones

1. [x] **4e extraction spike** — `parsers/dnd4e.py` parses 432 complete stat
   blocks (HP, AC, Fort/Ref/Will, XP, abilities, attack bonus + damage dice)
   from the 4e MM.
2. [x] **4e parser hardening** — geometry-aware extraction
   (`parsers/extract_text.py`: word positions -> column-aware lines) fixed
   page/column-break scrambling; **479 blocks, all complete** (9 orphan
   bodies remain unidentified). Headers lost to scrambling are re-headed
   from the 'Monsters by Level' appendix, joined to orphan bodies by the
   level/role -> XP table plus self-mentions in the body text.
3. [x] **5e parser** — `parsers/dnd5e.py` parses the OCR'd 5e MM scan:
   **381 blocks** recovered, then **5 scrambled blocks dropped** after
   cross-checking against the 5e SRD, leaving **376 reliable blocks**.
   Lost names are recovered from the book's stat-block index
   (`parsers/extract_index.py`, 3-column block-clustered extraction) keyed
   by printed page; OCR-garbled names are fuzzy-repaired against the same
   page's index entries. Name/CR/HP mismatches against the SRD were
   repaired by content-matching (e.g., a block named "Ancient Red Dragon"
   with Young Red Dragon stats was renamed).
4. [x] **Matching** — `data/processed/build_pairs.py` normalizes names,
   aligns edition-specific age terms (4e "Elder" ↔ 5e "Ancient"), verifies
   5e stats against the SRD, and promotes obvious variant→base pairs. From
   **479 4e** and **376 5e** reliable records it produced **498 candidate
   pairs**: **117 high-confidence** training pairs and **381 flagged for
   review**.  290 unique 4e monsters and 184 unique 5e monsters are
   represented.
5. [x] **Modeling** — `models/fit_models.py` fits per-field linear
   regressions in both directions (4e→5e and 5e→4e). Models now cover HP,
   AC, attack bonus, damage average, CR/level, all six ability scores, and
   saving throws (5e saves from 4e defenses; 4e defenses from 5e saves/abilities).
   Each model uses multiple source-edition covariates and reports a
   scale-adaptive split-conformal 80% prediction interval alongside point
   predictions (relative residual, lower bound clamped at 0).
6. [x] **Web app** — `app/app.py` is a bidirectional Dash dashboard with a
   direction switch (4e → 5e / 5e → 4e), a **Convert** tab showing predicted
   stats with intervals, and an **Explain** tab listing every fitted equation,
   features, R², MAE, and interval width.

## Design notes

- **Parser-first, per edition.** Each edition/source book gets its own
  parser feeding a common schema, so new editions are additive. Pre-scraped
  datasets were considered and rejected in favor of trusted, full-corpus
  local parsing.
- **Manual overrides.** Blocks that defeat parsing are corrected by hand
  in `manual/overrides_*.json` rather than by more parser special-casing;
  the 5e file now contains 68 corrections plus 5 drops. Patched records
  carry `manual_fix` and are listed in `data/processed/review_*.csv` for
  verification.
- **4e monster math changed mid-edition** (MM3 revised HP/damage scaling).
  Book source is tracked per stat block so models can be fit per-book or
  checked for MM1 outliers.
- **Linear models are a feature, not a limitation**: the explanation page
  can show the actual fitted equation per field with fit quality, which a
  black-box model could not offer.

## Usage

```bash
# geometry-aware extraction (word positions -> columns), then parse
python3 -m parsers.extract_text "Monster Manual 4e.pdf" data/raw/mm4_geo.txt
python3 -m parsers.dnd4e data/raw/mm4_geo.txt data/raw/mm4_monsters.jsonl
```

Parsing yields 479 complete stat blocks from the 4e MM. Records that
absorbed a neighbor's stats through merged PDF text lines are flagged in
`data/processed/review_4e.csv` for manual correction rather than further
parser special-casing.

```bash
# build cross-edition matched pairs (training set + review CSV)
python3 data/processed/build_pairs.py

# fit per-field linear models and save equations
.venv/bin/python models/fit_models.py

# run the Dash converter/explanation app
.venv/bin/python app/app.py
```
