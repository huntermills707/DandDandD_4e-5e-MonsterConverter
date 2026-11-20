# D&D Monster Edition Converter

Converts monster stat blocks between Dungeons & Dragons editions, starting
with **4th edition → 5th edition**.

Stat blocks are parsed from rule book PDFs into structured records, matched
across editions by monster name, and the numeric fields (HP, AC, attack
bonus, damage, level/CR, ...) are mapped with simple, human-readable linear
regression models. A Dash web app lets users convert their own typed-in
stats and shows the fitted equations for offline use.

**Legal note:** source PDFs, extracted text, parsed stat blocks, and
hand-transcribed corrections stay local and are never distributed (see
`.gitignore`). What is published is a **surrogate dataset**
(`data/surrogate/`): the numeric game stats only, with monster names and all
descriptive text removed and replaced by opaque ids. The web app only accepts
user-typed numeric input — it hosts no copyrighted text.

## Project structure

```
parsers/            edition-specific stat block parsers
    extract_text.py geometry-aware PDF text extraction (word positions)
    extract_index.py 3-column extraction for the 5e MM stat-block index
    dnd4e.py        4e Monster Manual parser
    dnd5e.py        5e Monster Manual parser (OCR-tolerant)
    overrides.py    applies manual/local/overrides_*.json corrections
manual/
    overrides_*.example.json  format examples (invented monsters, tracked)
    local/          real hand-transcribed corrections (local only, gitignored)
config/
    matching.toml   matching method, cardinality, thresholds
pipeline/
    build_pairs.py  cross-edition matching -> training pairs + review CSV
    matcher_legacy.py  original name-only matcher (method = "legacy")
    cardinality.py  partners per 5e monster: many / one / elbow
    compare_matching.py  scores every configuration on a shared test set
    metadata.py     edition-neutral non-numeric monster profiles
    match_features.py  pairwise metadata features + sentence embeddings
    matcher.py      name tiers gated by the metadata scorer
    eval_matching.py   name-blind retrieval evaluation of the scorer
    datasets.py     loads real local data or the surrogate (--data flag)
    make_surrogate.py  writes data/surrogate/ from local real data
data/
    surrogate/      masked numeric stats: pairs + per-edition monsters (tracked)
    raw/            extracted PDF text + parsed JSONL (local only, gitignored)
    processed/      matched cross-edition training pairs (local only)
models/             fitted regression models + evaluation reports
app/                Dash web apps (edition converter, difficulty scaler)
tests/              surrogate masking, model reproduction, override tests
requirements.txt
requirements-match.txt  extra deps for matching (CPU torch, sentence-transformers)
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
4. [x] **Matching** — `pipeline/build_pairs.py` pairs 4e and 5e
   monsters using **name evidence gated by non-numeric metadata**; numeric
   stats are never used, so pair selection can't favor the stat
   relationships the regressions later fit.
   - The parsers keep each block's metadata (local only): size/type/keywords,
     alignment, languages, senses, movement, skills, power and attack names,
     damage types, block text, and the 4e family header and lore.
   - **Rule** (`pipeline/matcher.py`): if names are approximately exact
     (**exact** incl. OCR word splits and 4e "Elder" ↔ 5e "Ancient";
     **alias** via parenthetical alternate names; **ocr** near-identical
     spelling) → match as the same creature. Otherwise the **metadata
     model** proposes its top 5 candidates, and the best-ranked one whose
     **name similarity** reaches 0.65 is accepted as a related creature
     ("Horde Ghoul" → "Ghoul", "Fire Titan" → "Fire Giant",
     "Efreet Karadjin" → "Efreeti"). Dragon ages must agree.
   - **Metadata model** (`pipeline/match_features.py`): logistic regression
     over structured agreement, TF-IDF of the masked block text, and a
     pretrained sentence embedding (`all-MiniLM-L6-v2`, CPU) of a name-free
     profile, trained on exact-name pairs (coefficients in
     `models/match_scorer.json`). Name-blind (85 same-name pairs, grouped
     CV) it ranks the true counterpart first **41%** of the time, top 10
     **72%**, median rank 3 (chance: 0.3% / 2.7%); misses are mostly
     siblings (dragon colors, devil types).
   - **Name similarity** (`pipeline/name_similarity.py`): IDF-weighted fuzzy
     ROUGE-1 recall — the share of the 5e name's words found in the 4e
     name, rare words weighted up. Recall suits 4e's "base + extra words"
     naming, which BLEU's brevity penalty punishes; IDF separates
     "Ancient Red Dragon" from "Ancient Gold Dragon".
   - **Chosen on hand labels** (`manual/local/match_labels.csv`, 170
     candidates): on 79 4e monsters with a correct candidate, ROUGE-1/IDF
     at 0.65 over the top 5 gives **98% precision, 82% recall**; chrF
     reaches 91%/81%, BLEU-1 67%/73%; using only the single top candidate
     drops recall to ~40% (a sibling often outranks the base creature).
   - **Configurable** (`config/matching.toml`, or CLI flags on
     `pipeline.build_pairs`): `method` = `metadata` (above) or `legacy`
     (the original name-only fuzzy matcher, reproduces the first 117 pairs
     exactly); `cardinality` = how many 4e partners each 5e monster keeps:
     `many`, `one` (1-to-1, preferring the ~exact partner), or `elbow`
     (1-to-N: all ~exact anchors, then partners best-first until the score
     drops by more than `elbow_ratio`x). The settings used are saved with
     the outputs (`data/surrogate/match_config.json`).
   - **Comparison** (`pipeline/compare_matching.py`): all six
     configurations scored on one shared test set (79 same-creature pairs,
     5 folds grouped by 5e monster, leaking pairs removed from training).
     They differ in coverage — 97 to 154 pairs — but not in accuracy:
     median R² 0.42–0.44 and MAE 72–74% of the mean baseline throughout.
   - Default (`metadata`, `many`): **154 training pairs** (87 same + 67
     related; 154 unique 4e, 104 unique 5e monsters).
5. [x] **Modeling** — `models/fit_models.py` fits per-field linear
   regressions in both directions (4e→5e and 5e→4e): HP, AC, attack bonus,
   damage average, CR/level, all six ability scores, and saving throws (5e
   saves from 4e defenses; 4e defenses from 5e saves/abilities).
   - **Design goal: easy, explainable conversion.** Each stat converts by
     a short rule a GM can apply by hand — the matching stat in the other
     edition, the level/CR scale, and 4e role adjustments — e.g.
     *5e HP = 0.18·HP + 4.7·level − 2 (solo +72)*. All rules are listed in
     [`models/conversion_rules.md`](models/conversion_rules.md).
   - **How the rules were chosen** (`models/experiments.py`, all under the
     same grouped CV): adding **4e role** (minion / elite / solo; for
     5e→4e, the role the converted monster should play) was the largest
     gain (median MAE 70% → 59% of the mean baseline, R² 0.48 → 0.60).
     Dropping from many covariates per stat to counterpart + level/CR, and
     rounding coefficients to 2 significant figures, cost nothing
     measurable. Terms are then pruned: a counterpart stat with a negative
     coefficient is dropped, and any term whose removal raises CV error by
     under 1% goes (nested CV: median 59.8% → 60.2%). Log-scale HP/damage,
     ridge, and the original non-negative constraint did no better.
   - **Role interactions** (`models/experiments.py --interactions`, nested
     CV): 4e multiplies solo HP by 4, so when converting *to* 4e the solo HP
     bonus grows with level. A solo × CR slope replaces the flat bonus
     (*4e HP = 0.83·HP + 50.5, elite +130, solo +31·CR*), cutting 5e→4e HP
     error from 39% to 29% of the baseline (R² 0.86 → 0.93). 4e minions
     always have 1 HP, so that rule returns 1 for minions, and they are left
     out of its fit. The other candidates were combat roles (brute, lurker,
     soldier, ... vs. skirmisher), elite/solo slopes on level/CR for every
     stat, and slopes on the counterpart stat. Offered all at once or forward
     one at a time, they help some fields and hurt more (median error
     60.2% → 59.2–61.2%), mostly by chasing noise in the small 5e
     saving-throw sets. Minion slopes can't be scored: all 6 minion pairs are
     related, not same-creature.
   - **Not adopted: elite/solo slopes for 4e→5e saves.** They looked
     promising in that comparison, so they were rechecked on fresh folds
     (nested CV, ×10, two seeds). CON save error fell from 84% to 66% of
     the baseline and WIS from 77% to 72%, but CHA and DEX got worse. Each
     save has only 27–35 scored pairs, over half of DEX/WIS/CHA solos, so
     the slopes rest on a handful of monsters. The selected CON rule also
     dropped level for standard monsters (*CON save = 6, elite
     +0.15·level, solo +0.35·level*), a flat +6 at any level, with 59%
     coverage in the low third. Keeping level and adding the slopes gives a
     sensible shape (about +0.1 per level for standard, +0.2 elite, +0.27
     solo) and cuts WIS/CHA error by 4–6%. That variant was picked after
     seeing results, though, and still hurts DEX.
   - **Evaluation**: repeated grouped 5-fold CV (×10). Pairs sharing a 4e
     or 5e monster form one group and never straddle train and test;
     metrics are scored on held-out same-creature pairs and reported as
     mean ± sd over repeats. The final equations are fit on all pairs.
   - **Intervals**: 80% cross-conformal, `result ± q·(result + c)` with `c`
     the median target value — near-constant for small stats, proportional
     for large ones. Coverage is measured with nested CV and reported per
     third of the prediction range; it beat both purely absolute intervals
     (big monsters under-covered: 46% for 5e→4e HP) and purely relative
     ones (small values under-covered).
   - **Data checks**: grouped CV exposed OCR errors in the 5e parse (an
     Otyugh STR of 116, six ACs of 1); the parser now repairs split AC
     digits and checks each ability score against its printed modifier.
6. [x] **Web app** — `app/app.py` is a bidirectional Dash dashboard with a
   direction switch (4e → 5e / 5e → 4e), a **Convert** tab showing predicted
   stats with intervals, and an **Explain** tab listing every fitted equation,
   features, R², MAE, and interval width.
7. [x] **Difficulty scaler** — `models/fit_scale_models.py` fits within-edition
   scaling curves (stat ~ level for 4e, stat ~ CR for 5e; HP/damage
   multiplicative, everything else additive, pooled fallback for sparse 5e
   saves). `app/scale_app.py` (port 8051) scales a typed-in monster to a new
   level/CR: every stat shifts by the curve-implied difference, preserving the
   monster's own deviation from the average. A **Method** tab lists each
   curve's per-step change, R², and n.

## Design notes

- **Parser-first, per edition.** Each edition/source book gets its own
  parser feeding a common schema, so new editions are additive. Pre-scraped
  datasets were considered and rejected in favor of trusted, full-corpus
  local parsing.
- **Manual overrides.** Blocks that defeat parsing are corrected by hand
  in `manual/local/overrides_*.json` (gitignored; see the `.example.json`
  files for the format) rather than by more parser special-casing;
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
# build cross-edition matched pairs (training set + review CSV);
# needs requirements-match.txt
.venv/bin/python -m pipeline.build_pairs

# other matching settings (defaults in config/matching.toml)
.venv/bin/python -m pipeline.build_pairs --method legacy --cardinality one

# name-blind evaluation of the metadata matcher; compare all configurations
.venv/bin/python -m pipeline.eval_matching
.venv/bin/python -m pipeline.compare_matching

# refresh the shareable surrogate after re-parsing or re-matching
.venv/bin/python -m pipeline.make_surrogate

# fit per-field linear models and save equations
.venv/bin/python models/fit_models.py

# compare model designs under the same grouped CV; honest cost of pruning
.venv/bin/python -m models.experiments --fields
.venv/bin/python -m models.experiments --nested-selection
.venv/bin/python -m models.experiments --interactions

# fit within-edition scaling curves (level/CR -> stat)
.venv/bin/python models/fit_scale_models.py

# run the Dash converter/explanation app
.venv/bin/python app/app.py

# run the Dash difficulty scaler (port 8051)
.venv/bin/python app/scale_app.py
```

### Without the rule books

The fitted models can be rebuilt and audited from the tracked surrogate alone.
The surrogate keeps record order, so the seeded splits — and every equation —
come out identical to the fit on the real data (checked by
`tests/test_reproduce_models.py`):

```bash
.venv/bin/python models/fit_models.py --data surrogate
.venv/bin/python models/fit_scale_models.py --data surrogate
.venv/bin/python -m pytest
```

`--data auto` (the default) uses the real local data when present and falls
back to the surrogate otherwise.
