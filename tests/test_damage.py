"""Damage measures read from stat-block text (invented monsters only)."""

from pipeline.damage import attacks_5e, damage_4e, damage_5e, ocr_fix_5e


def block5(actions):
    return {"meta": {"text": "Medium construct, unaligned Armor Class 15 Hit Points 40 (6d8 + 12) "
                             "Challenge 3 (700 XP) ACTIONS " + actions}}


def test_riders_count_toward_attack_damage():
    atk = attacks_5e("Lantern Rod. Melee Weapon Attack: +5 to hit, reach 5 ft., one target. "
                     "Hit: 6 (1d6 + 3) bludgeoning damage plus 9 (2d8) fire damage.")
    assert atk == [{"name": "lantern rod", "bonus": 5, "base": 6, "damage": 15, "melee": True}]


def test_ocr_slips_in_dice_are_repaired():
    assert ocr_fix_5e("Hit: 7 (ld6 + 4)") == "Hit: 7 (1d6 + 4)"
    assert ocr_fix_5e("Hit: 17 (2dl0 + 6)") == "Hit: 17 (2d10 + 6)"
    assert ocr_fix_5e("Hit: 17 (2d1 0 + 6)") == "Hit: 17 (2d10 + 6)"
    assert ocr_fix_5e("Weapon Attack:+ 13 to hit") == "Weapon Attack: +13 to hit"


def test_multiattack_named_counts():
    d = damage_5e(block5(
        "Multiattack. The golem makes three attacks: one with its gear and two with its pistons. "
        "Gear. Melee Weapon Attack: +6 to hit, reach 5 ft., one target. Hit: 10 (2d6 + 3) "
        "slashing damage. Piston. Melee Weapon Attack: +6 to hit, reach 5 ft., one target. "
        "Hit: 7 (1d8 + 3) bludgeoning damage."))
    assert d == {"damage_avg": 10.0, "attack_bonus": 6, "damage_per_round": 24.0}


def test_multiattack_generic_and_alternatives():
    d = damage_5e(block5(
        "Multiattack. The scout makes two melee attacks or two ranged attacks. "
        "Hatchet. Melee Weapon Attack: +4 to hit, reach 5 ft., one target. Hit: 5 (1d6 + 2) "
        "slashing damage. Sling. Ranged Weapon Attack: +4 to hit, range 30/120 ft., one target. "
        "Hit: 4 (1d4 + 2) bludgeoning damage."))
    assert d["damage_per_round"] == 10.0  # the better routine: two hatchets


def test_unresolved_multiattack_is_unknown():
    # the claw attack was lost to a page break: don't guess its damage
    d = damage_5e(block5(
        "Multiattack. The beast makes three attacks: one with its bite and two with its claws. "
        "Bite. Melee Weapon Attack: +7 to hit, reach 10 ft., one target. Hit: 14 (2d10 + 3) "
        "piercing damage."))
    assert d["damage_avg"] == 14.0 and d["damage_per_round"] is None


def test_attack_without_hit_damage_does_not_borrow_the_next():
    d = damage_5e(block5(
        "Snare. Ranged Weapon Attack: +3 to hit, range 5/15 ft., one target. Hit: The target "
        "is restrained. Club. Melee Weapon Attack: +3 to hit, reach 5 ft., one target. "
        "Hit: 4 (1d6 + 1) bludgeoning damage."))
    assert d == {"damage_avg": 4.0, "attack_bonus": 3, "damage_per_round": 4.0}


def test_no_text_falls_back_to_corrected_attacks():
    d = damage_5e({"attacks": [{"name": "Slam", "bonus": 4, "damage_avg": 9}]})
    assert d == {"damage_avg": 9.0, "attack_bonus": 4, "damage_per_round": None}


BLOCK4 = (
    "Large natural beast XP 300 Initiative +5 Senses Perception +6 HP 90; Bloodied 45 "
    "AC 21; Fortitude 20, Reflex 18, Will 17 Speed 6 "
    "m Gore (standard; at-will) +12 vs. AC; 1d10 + 5 damage. "
    "M Trample (standard; at-will) The beast makes two gore attacks. "
    "C Stone Breath (standard; recharge 5 6) Close blast 3; +10 vs. Reflex; 3d8 + 4 damage. "
    "Alignment Unaligned Languages - Str 20 (+8) Dex 14 (+5) Wis 12 (+4) "
    "Con 18 (+7) Int 2 (+0) Cha 8 (+2)"
)


def test_4e_basic_attack_not_breath_weapon():
    d = damage_4e({"meta": {"text": BLOCK4}})
    assert d == {"damage_avg": 10.5, "attack_bonus": 12, "damage_max": 17.5,
                 "attacks_per_round": 2}


def test_4e_minion_fixed_damage():
    text = ("Small humanoid XP 25 HP 1; a missed attack never damages a minion. "
            "m Spear (standard; at-will) +6 vs. AC; 4 damage. Str 12 (+2)")
    d = damage_4e({"meta": {"text": text}})
    assert d["damage_avg"] == 4.0 and d["attacks_per_round"] == 1
