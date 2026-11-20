"""Name rules, name similarity, and metadata normalization on invented monsters."""

from pipeline.matcher import ages_conflict, name_tier
from pipeline.metadata import (
    damage_types, languages, moral_axis, split_name, type_tags_4e, type_tags_5e,
)
from pipeline.name_similarity import NameSimilarity


def prof(name):
    base, alts = split_name(name)
    return {"name": base, "alt_names": alts, "family": None}


def test_exact_including_ocr_splits_and_age_synonyms():
    assert name_tier(prof("Glimmer Wyrm"), prof("Glimmer Wyrm")) == "exact"
    assert name_tier(prof("Adult Glimmer Wyrm"), prof("Adult G Limmer Wyrm")) == "exact"
    assert name_tier(prof("Elder Glimmer Wyrm"), prof("Ancient Glimmer Wyrm")) == "exact"


def test_alias_and_ocr():
    assert name_tier(prof("Howling Shade (Moanling)"), prof("Moanling")) == "alias"
    assert name_tier(prof("Mire Eye of Vothrak"), prof("Mlre Eye of Vothrak")) == "ocr"


def test_variants_are_not_approximately_exact():
    assert name_tier(prof("Glimmer Wyrm Warlord"), prof("Glimmer Wyrm")) is None
    assert name_tier(prof("Hollow Knight Warrior"), prof("Hollow Knight Warlord")) is None


def test_ages_must_agree():
    assert ages_conflict("Young Glimmer Wyrm", "Adult Glimmer Wyrm")
    assert not ages_conflict("Glimmer Wyrm", "Adult Glimmer Wyrm")


def test_name_similarity_weights_rare_words():
    corpus = [f"{age} {color} Wyrm" for age in ("Young", "Adult", "Ancient")
              for color in ("Red", "Gold", "Teal", "Plum")] + ["Bog Troll", "Troll"]
    ns = NameSimilarity(corpus)
    assert ns.rouge1_recall("Bog Troll", "Troll") == 1.0           # variant: base fully covered
    assert ns.rouge1_recall("Ancient Red Wyrm", "Ancient Gold Wyrm") < 0.65  # sibling
    assert ns.rouge1_recall("Efreet Blazer", "Efreeti") == 1.0       # fuzzy word
    assert ns.rouge1_recall("Wailing Ghost", "Ghast") == 0.0         # too far apart
    assert ns.bleu1("Bog Troll", "Troll") < 0.5                       # why BLEU fits poorly


def test_metadata_normalization():
    assert type_tags_4e("Medium immortal humanoid (devil)") >= {"fiend", "humanoid"}
    assert type_tags_5e("Large fiend (devil), lawful evil") == {"fiend"}
    assert moral_axis("Chaotic evil") == moral_axis("lawful evil") == "evil"
    assert moral_axis("any alignment") == "any"
    assert languages("Supernal, telepathy 20") == {"celestial", "telepathy"}
    assert damage_types("2d6 fire damage; resist 10 cold") == {"fire", "cold"}
