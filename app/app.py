#!/usr/bin/env python3
"""Dash web app: bidirectional D&D stat block conversion with conformal intervals.

Tabs:
    Convert  — numeric input form, predicted output, and 80% scale-adaptive conformal interval.
    Explain  — fitted equations, features, and fit quality per model.
"""

import json
import math
import sys
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, ALL
from dash.exceptions import PreventUpdate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def data_file(name):
    """models/<name> in the repo, or <name> beside this file in the flat
    hosted bundle (deploy/build_pycafe.py)."""
    flat = HERE / name
    return flat if flat.exists() else ROOT / "models" / name


MODELS_PATH = data_file("models.json")


def load_models():
    try:
        with open(MODELS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


MODELS = load_models()

DIRECTION_LABELS = {
    "4e_to_5e": "4e → 5e",
    "5e_to_4e": "5e → 4e",
}

# The target edition's scale. Rules other than the scale rule itself are
# stated on it (5e CR, 4e level) when the GM types the level/CR they want;
# left blank, the scale rule converts it for display and every other stat
# comes from its "direct" rule on the source monster's own level/CR.
TARGET_SCALE = {"4e_to_5e": "cr_num", "5e_to_4e": "level"}

# Display grouping for the field grids: (group title, fields)
FIELD_GROUPS = [
    ("Core stats", ["level", "cr_num", "hp", "ac", "attack_bonus", "damage_avg",
                    "attacks_per_round", "damage_per_round"]),
    (
        "Ability scores",
        ["ability_str", "ability_dex", "ability_con",
         "ability_int", "ability_wis", "ability_cha"],
    ),
    ("Defenses", ["fort", "ref", "will"]),
    (
        "Saving throws",
        ["save_str", "save_dex", "save_con", "save_int", "save_wis", "save_cha"],
    ),
]

FIELD_ORDER = [f for _, fields in FIELD_GROUPS for f in fields]

# 4e role tiers; models take them as indicator features ("role_elite", ...)
# chosen with the role selector rather than typed in.
ROLE_OPTIONS = [
    {"label": "Standard", "value": "standard"},
    {"label": "Minion", "value": "minion"},
    {"label": "Elite", "value": "elite"},
    {"label": "Solo", "value": "solo"},
]


# 4e combat roles; skirmisher (the 4e baseline monster) is the reference, so
# it adds no term. Shown only when some rule in the direction uses one.
COMBAT_ROLE_OPTIONS = [
    {"label": "Skirmisher", "value": "skirmisher"},
    {"label": "Artillery", "value": "artillery"},
    {"label": "Brute", "value": "brute"},
    {"label": "Controller", "value": "controller"},
    {"label": "Lurker", "value": "lurker"},
    {"label": "Soldier", "value": "soldier"},
]
COMBAT_ROLES = [o["value"] for o in COMBAT_ROLE_OPTIONS if o["value"] != "skirmisher"]


def role_state(tier, combat):
    """Indicator inputs for the role selectors: role_minion/elite/solo and
    role_artillery/brute/... (skirmisher and standard are all zeros)."""
    state = {f"role_{t}": 1.0 if tier == t else 0.0 for t in ("minion", "elite", "solo")}
    state.update({f"role_{r}": 1.0 if combat == r else 0.0 for r in COMBAT_ROLES})
    return state


def uses_combat_role(direction):
    """True if any rule (or its direct twin) in the direction has a combat-role term."""
    return any(
        part[len("role_"):] in COMBAT_ROLES
        for m in MODELS.values() if m["direction"] == direction
        for rule in (m, m.get("direct", {}))
        for f in rule.get("features", []) for part in f.split("*"))


def is_role_feature(field):
    return field.startswith("role_")


def max_parts(part):
    """Stats inside "max(ability_str,ability_dex)", or None for a plain stat."""
    return part[4:-1].split(",") if part.startswith("max(") else None


def input_stats(feature):
    """Typed-in stats a feature needs: "role_solo*cr_num" -> ["cr_num"],
    "max(ability_str,ability_dex)" -> ["ability_str", "ability_dex"]."""
    out = []
    for f in feature.split("*"):
        if not is_role_feature(f):
            out += max_parts(f) or [f]
    return out


def feature_values(features, state):
    """{feature: float} from typed stats and role indicators in `state`;
    products for interaction terms, the larger stat for max(a,b). None if
    any input is missing."""
    values = {}
    for feature in features:
        x = 1.0
        for part in feature.split("*"):
            try:
                x *= max(float(state.get(f)) for f in max_parts(part) or [part])
            except (TypeError, ValueError):
                return None
        values[feature] = x
    return values


def predict_with_interval(model, values, role=None):
    """Point prediction and 80% interval for one model.

    values: {feature: float}. Mirrors models/fit_models.LinearDesign:
    optional log1p feature/target transforms, then the model's interval
    rule ("scaled", "relative", "absolute", or "log"). A stat the 4e role
    fixes (minion HP = 1) is returned exactly.
    """
    fixed = model.get("fixed_by_role", {})
    if role in fixed:
        return fixed[role], fixed[role], fixed[role]
    log_features = set(model.get("log_features", []))
    z = model["intercept"]
    for coef, feature in zip(model["coef"], model["features"]):
        x = values[feature]
        if feature in log_features:
            x = math.log1p(max(x, 0.0))
        z += coef * x
    clip = model.get("clip_min", 0.0)
    pred = max(math.expm1(z) if model.get("log_target") else z, clip)
    q = model.get("conformal_quantile", model.get("conformal_quantile_relative", 0.0))
    kind = model.get("interval", "relative")
    if kind == "scaled":
        half = q * (abs(pred) + model.get("interval_offset", 0.0))
        low, high = pred - half, pred + half
    elif kind == "absolute":
        low, high = pred - q, pred + q
    elif kind == "log":
        low, high = math.expm1(math.log1p(pred) - q), math.expm1(math.log1p(pred) + q)
    else:
        low, high = pred * (1 - q), pred * (1 + q)
    return pred, max(low, clip), high


def fields_for_direction(direction):
    """Return (input_fields, output_fields) for the chosen direction. The
    target level/CR is both: an optional input and a converted output."""
    input_fields = set()
    output_fields = set()
    for key, model in MODELS.items():
        if model["direction"] != direction:
            continue
        for m in (model, model.get("direct", {})):
            input_fields.update(x for f in m.get("features", []) for x in input_stats(f))
        output_fields.add(model["target"])
    # deterministic order
    input_fields = [f for f in FIELD_ORDER if f in input_fields]
    output_fields = [f for f in FIELD_ORDER if f in output_fields]
    return input_fields, output_fields


def group_fields(fields):
    """Group an ordered field list into titled sections, preserving order."""
    groups = []
    remaining = set(fields)
    for title, group_fields_ in FIELD_GROUPS:
        present = [f for f in fields if f in group_fields_]
        if present:
            groups.append((title, present))
            remaining.difference_update(present)
    if remaining:
        groups.append(("Other", [f for f in fields if f in remaining]))
    return groups


def input_label(field, direction):
    if field == TARGET_SCALE[direction]:
        return "Target " + ("CR" if field == "cr_num" else "level")
    return label_for(field)


def label_for(field):
    mapping = {
        "level": "Level",
        "cr_num": "CR",
        "hp": "HP",
        "ac": "AC",
        "attack_bonus": "Attack Bonus",
        "damage_avg": "Damage per Attack",
        "damage_per_round": "Damage per Round",
        "attacks_per_round": "Attacks per Round",
        "ability_str": "STR",
        "ability_dex": "DEX",
        "ability_con": "CON",
        "ability_int": "INT",
        "ability_wis": "WIS",
        "ability_cha": "CHA",
        "fort": "Fortitude",
        "ref": "Reflex",
        "will": "Will",
        "save_str": "STR Save",
        "save_dex": "DEX Save",
        "save_con": "CON Save",
        "save_int": "INT Save",
        "save_wis": "WIS Save",
        "save_cha": "CHA Save",
    }
    return mapping.get(field, field.replace("_", " ").title())


def feature_label(feature):
    """Readable feature name: "role_solo*cr_num" -> "solo × CR"."""
    parts = []
    for part in feature.split("*"):
        if is_role_feature(part):
            parts.append(part[len("role_"):])
        elif max_parts(part):
            parts.append(f"max({', '.join(label_for(f) for f in max_parts(part))})")
        else:
            parts.append(label_for(part))
    return " × ".join(parts)


# 5e challenge ratings, for showing a converted CR as the nearest legal one.
VALID_CRS = [0, 0.125, 0.25, 0.5] + list(range(1, 31))
CR_LABELS = {0.125: "1/8", 0.25: "1/4", 0.5: "1/2"}


def nearest_cr_label(x):
    cr = min(VALID_CRS, key=lambda c: abs(c - x))
    return CR_LABELS.get(cr, str(int(cr)))


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "D&D Monster Edition Converter"


def inline_css(dash_app):
    """Dash serves assets/ itself; the flat hosted bundle ships styles.css
    beside the app instead, so inline it into the page."""
    css = HERE / "styles.css"
    if not (HERE / "assets").is_dir() and css.exists():
        dash_app.index_string = dash_app.index_string.replace(
            "{%css%}", "{%css%}\n<style>\n" + css.read_text(encoding="utf-8") + "\n</style>")


inline_css(app)

FAN_CONTENT = ("Unofficial Fan Content permitted under the Wizards of the Coast Fan Content "
               "Policy. Not approved or endorsed by Wizards. Portions of the materials used "
               "are property of Wizards of the Coast. ©Wizards of the Coast LLC.")

# 4e minions have no 5e counterpart, and only 6 matched pairs are minions,
# too few to fit a minion term: the 4e -> 5e rules convert one as a standard
# monster of its level (with its 1 HP).
MINION_NOTE = ("4e minions have no 5e equivalent, and too few were matched to learn a "
               "minion rule. These numbers are for a standard monster of the same level: "
               "treat them as an upper bound and scale down to taste.")

# A converted monster has at least 1 HP (low-level 4e monsters can convert below that).
MIN_HP = 1.0


def hero(title, subtitle, toggle):
    return html.Header(
        className="hero",
        children=html.Div(
            className="hero-inner",
            children=[
                html.Div(
                    className="brand",
                    children=[html.H1(title), html.P(subtitle, className="subtitle")],
                ),
                toggle,
            ],
        ),
    )


def tab(label, value, children):
    return dcc.Tab(
        label=label,
        value=value,
        className="custom-tab",
        selected_className="custom-tab--selected",
        children=html.Div(className="tab-body", children=children),
    )


app.layout = html.Div(
    className="page",
    children=[
        hero(
            "D&D Monster Edition Converter",
            "Bidirectional stat-block conversion with 80% conformal prediction intervals.",
            dcc.RadioItems(
                id="direction",
                className="direction-toggle",
                options=[
                    {"label": label, "value": val}
                    for val, label in DIRECTION_LABELS.items()
                ],
                value="4e_to_5e",
            ),
        ),
        html.Main(
            className="container",
            children=[
                dcc.Tabs(
                    className="tabs-container",
                    parent_className="tabs-parent",
                    mobile_breakpoint=0,
                    value="convert",
                    children=[
                        tab("Convert", "convert", [
                            html.Div(
                                className="convert-grid",
                                children=[
                                    html.Section(
                                        className="card",
                                        children=[
                                            html.Span("Source", className="card-eyebrow"),
                                            html.H3(id="input-heading"),
                                            html.P(
                                                "Type in your monster's stats. Outputs update "
                                                "as soon as a rule has every stat it needs.",
                                                className="card-hint",
                                            ),
                                            html.Div(
                                                className="field-group",
                                                children=[
                                                    html.H4(id="role-label",
                                                            className="field-group-title"),
                                                    dcc.RadioItems(
                                                        id="role",
                                                        className="direction-toggle role-toggle",
                                                        options=ROLE_OPTIONS,
                                                        value="standard",
                                                    ),
                                                    html.P(id="role-note", className="role-note",
                                                           style={"display": "none"}),
                                                ],
                                            ),
                                            html.Div(
                                                id="combat-role-group",
                                                className="field-group",
                                                children=[
                                                    html.H4(id="combat-role-label",
                                                            className="field-group-title"),
                                                    dcc.RadioItems(
                                                        id="combat-role",
                                                        className="direction-toggle role-toggle",
                                                        options=COMBAT_ROLE_OPTIONS,
                                                        value="skirmisher",
                                                    ),
                                                ],
                                            ),
                                            html.Div(id="input-fields"),
                                        ],
                                    ),
                                    html.Section(
                                        className="card output-card",
                                        children=[
                                            html.Span("Converted", className="card-eyebrow"),
                                            html.H3(id="output-heading"),
                                            html.P(
                                                [
                                                    "Predicted value, with the ",
                                                    html.Span("80%", className="legend-chip"),
                                                    " range that real conversions fall in "
                                                    "about 4 times in 5.",
                                                ],
                                                className="card-hint",
                                            ),
                                            html.Div(id="output-fields"),
                                        ],
                                    ),
                                ],
                            ),
                        ]),
                        tab("Explain", "explain", [
                            html.Section(
                                className="card explain-intro",
                                children=[
                                    html.Span("Method", className="card-eyebrow"),
                                    html.H3("Fitted linear models"),
                                    html.P(
                                        "Each rule is a short linear equation meant to "
                                        "be usable by hand: the matching stat in the "
                                        "other edition, the target edition's level/CR, "
                                        "and 4e role adjustments, with terms kept only if they "
                                        "measurably help. Rules are fit with Huber loss "
                                        "(a few unusual monsters can't drag a rule), "
                                        "counting related pairs (a 4e variant matched to "
                                        "its 5e base creature) at half weight."
                                    ),
                                    html.P(
                                        "Fit quality is from repeated "
                                        "grouped cross-validation (variants of one "
                                        "monster never straddle train and test), scored "
                                        "on same-creature pairs; ± is the spread over "
                                        "repeats. 4e role (minion / elite / solo vs. "
                                        "standard) enters as indicator terms, and so "
                                        "does the combat role (artillery, brute, "
                                        "controller, lurker, soldier vs. skirmisher) "
                                        "where it measurably helps a rule; for 5e → 4e "
                                        "these are the roles the converted monster "
                                        "should play. 4e multiplies solo HP by 4, so the solo HP "
                                        "bonus is a slope on level, and 4e "
                                        "minions always have 1 HP."
                                    ),
                                    html.P(
                                        "Level/CR in a rule is the target edition's, "
                                        "the way each edition's own monster math works: "
                                        "type the level/CR you want and these rules "
                                        "apply. Leave it blank and the first rule "
                                        "converts it, while every other stat uses its "
                                        "direct rule (shown under each card) from the "
                                        "source monster's own level/CR."
                                    ),
                                    html.P(
                                        "The range is an 80% cross-conformal "
                                        "prediction interval (nominal miscoverage "
                                        "α = 0.20) whose width scales with |prediction| "
                                        "+ a typical value of the stat, so it is roughly "
                                        "constant for small stats and proportional for "
                                        "large ones."
                                    ),
                                ],
                            ),
                            html.Div(id="equations-list"),
                        ]),
                    ],
                ),
            ],
        ),
        html.Footer(
            className="footer",
            children=[
                html.P("Numeric input only — no rule-book text is hosted."),
                html.P(FAN_CONTENT),
            ],
        ),
    ],
)


@app.callback(
    Output("role-note", "children"),
    Output("role-note", "style"),
    Input("direction", "value"),
    Input("role", "value"),
)
def update_role_note(direction, role):
    if direction == "4e_to_5e" and role == "minion":
        return MINION_NOTE, {}
    return "", {"display": "none"}


@app.callback(
    Output("input-heading", "children"),
    Output("output-heading", "children"),
    Output("input-fields", "children"),
    Output("output-fields", "children"),
    Output("role-label", "children"),
    Output("combat-role-label", "children"),
    Output("combat-role-group", "style"),
    Input("direction", "value"),
)
def update_layout(direction):
    source_edition = "4e" if direction == "4e_to_5e" else "5e"
    target_edition = "5e" if direction == "4e_to_5e" else "4e"
    input_fields, output_fields = fields_for_direction(direction)

    scale = TARGET_SCALE[direction]
    groups = group_fields([f for f in input_fields if f != scale])
    if scale in input_fields:
        groups.append(("Target (optional — blank uses the direct rules)", [scale]))
    input_children = []
    for title, fields in groups:
        grid = []
        for f in fields:
            grid.append(
                html.Div(
                    className="field",
                    children=[
                        html.Label(input_label(f, direction), className="field-label"),
                        dcc.Input(
                            id={"type": "input", "field": f},
                            className="field-input",
                            type="number",
                            value="",
                            placeholder="auto" if f == scale else "—",
                        ),
                    ],
                )
            )
        input_children.append(
            html.Div(
                className="field-group",
                children=[
                    html.H4(title, className="field-group-title"),
                    html.Div(grid, className="fields-grid"),
                ],
            )
        )

    output_children = []
    for title, fields in group_fields(output_fields):
        rows = []
        for f in fields:
            rows.append(
                html.Div(
                    className="stat-row",
                    children=[
                        html.Span(label_for(f), className="stat-label"),
                        html.Div(
                            id={"type": "output", "field": f},
                            className="stat-value",
                            children=EMPTY_OUTPUT,
                        ),
                    ],
                )
            )
        output_children.append(
            html.Div(
                className="field-group",
                children=[
                    html.H4(title, className="field-group-title"),
                    html.Div(rows, className="stat-list"),
                ],
            )
        )

    role_label = "4e role" if source_edition == "4e" else "Target 4e role"
    combat_label = ("4e combat role" if source_edition == "4e"
                    else "Target 4e combat role")
    return (
        f"{source_edition} monster",
        f"{target_edition} conversion",
        input_children,
        output_children,
        role_label,
        combat_label,
        {} if uses_combat_role(direction) else {"display": "none"},
    )


def target_scale_given(direction, state):
    """True when the GM typed a target level/CR."""
    try:
        float(state.get(TARGET_SCALE[direction]))
        return True
    except (TypeError, ValueError):
        return False


def rule_for(model, scale_given):
    """The rule to apply: on the target level/CR when it is given, else the
    direct rule (source level/CR) when the model has one."""
    return model if scale_given or "direct" not in model else model["direct"]


# The interval slot is always present (empty when there is no range) so the
# predictions line up in one column.
NO_INTERVAL = html.Span(className="stat-interval stat-interval-none")
EMPTY_OUTPUT = [html.Span("—", className="stat-pred stat-empty"), NO_INTERVAL]


def format_output(target, pred, low, high):
    """Render a prediction and its interval as stat-value children."""
    if target in ("cr_num", "damage_avg", "damage_per_round"):
        pred_s, lo_s, hi_s = f"{pred:.2f}", f"{low:.2f}", f"{high:.2f}"
    else:
        pred_s, lo_s, hi_s = str(round(pred)), str(round(low)), str(round(high))
    children = [html.Span(pred_s, className="stat-pred")]
    if target == "cr_num":
        children.append(html.Span(f"≈ CR {nearest_cr_label(pred)}", className="stat-note",
                                  title="nearest legal challenge rating"))
    if (lo_s, hi_s) != (pred_s, pred_s):
        children.append(html.Span(f"{lo_s} – {hi_s}", className="stat-interval",
                                  title="80% prediction interval"))
    else:
        children.append(NO_INTERVAL)  # fixed by role (minion HP = 1)
    return children


@app.callback(
    Output({"type": "output", "field": ALL}, "children"),
    Input("direction", "value"),
    Input({"type": "input", "field": ALL}, "value"),
    Input("role", "value"),
    Input("combat-role", "value"),
    prevent_initial_call=False,
)
def compute_outputs(direction, input_values, role, combat_role="skirmisher"):
    # Match input values to their fields via the ALL pattern matching order.
    input_fields, output_fields = fields_for_direction(direction)
    ctx = dash.callback_context
    # A direction switch rebuilds the field components; this callback can fire
    # against the stale DOM (old field set, new direction). Skip those fires —
    # the callback refires once the new components are in place.
    output_ids = [entry["id"]["field"] for entry in ctx.outputs_list]
    if sorted(output_ids) != sorted(output_fields):
        raise PreventUpdate
    # Dash passes input_values in the order the components were created in the
    # layout callback. Rebuild the input state dictionary.
    # ctx.inputs_list contains the IDs for the pattern-matched inputs.
    inputs_list = ctx.inputs_list
    # inputs_list is a list; first element is direction (dict without pattern),
    # rest are the pattern-matched inputs.
    state = {}
    input_ids = set()
    for item in inputs_list:
        if isinstance(item, list):
            input_ids.update(entry["id"]["field"] for entry in item)
            for entry, val in zip(item, input_values or []):
                field = entry["id"]["field"]
                state[field] = val
        elif isinstance(item, dict) and item.get("id") == "direction":
            pass  # already known
    if input_ids != set(input_fields):
        raise PreventUpdate
    if not uses_combat_role(direction):
        combat_role = "skirmisher"  # selector hidden: no term to apply
    state.update(role_state(role, combat_role))

    empty = EMPTY_OUTPUT
    results = {f: empty for f in output_fields}
    given = target_scale_given(direction, state)
    for key, model in MODELS.items():
        if model["direction"] != direction:
            continue
        target = model["target"]
        rule = rule_for(model, given)
        values = feature_values(rule["features"], state)
        if values is None and role not in rule.get("fixed_by_role", {}):
            results[target] = empty
            continue
        pred, low, high = predict_with_interval(rule, values, role)
        if target == "hp":
            pred, low, high = max(pred, MIN_HP), max(low, MIN_HP), max(high, MIN_HP)
        results[target] = format_output(target, pred, low, high)

    # Dash fills pattern-matched outputs in ctx.outputs_list order, which after
    # a rebuild need not match the field order: key every result by its id.
    return [results[f] for f in output_ids]


def metric(label, value, detail=None, bar=None):
    """A labelled metric tile; bar is a 0–1 fill fraction."""
    children = [
        html.Span(label, className="metric-label"),
        html.Span(value, className="metric-value"),
    ]
    if bar is not None:
        children.append(html.Div(
            className="metric-bar",
            children=html.Div(style={"width": f"{max(0.0, min(bar, 1.0)) * 100:.0f}%"}),
        ))
    if detail:
        children.append(html.Span(detail, className="metric-detail"))
    return html.Div(children, className="metric")


def model_card(model):
    r2, mae, base = model["r2"], model["mae"], model["baseline_mae"]
    gain = 1 - mae / base if base else 0.0
    thirds = model.get("coverage_by_third")
    thirds_s = " · ".join(f"{c * 100:.0f}" for c in thirds) if thirds else "?"
    dropped = [feature_label(f) for f in model.get("dropped_features", [])]
    return html.Div(
        className="card model-card",
        children=[
            html.Div(
                className="model-head",
                children=[
                    html.H4(label_for(model["target"])),
                    html.Span(f"n = {model['n']}", className="badge",
                              title=f"{model['n']} pairs, {model.get('n_groups', '?')} groups, "
                                    f"{model.get('n_eval', '?')} same-creature pairs scored"),
                ],
            ),
            html.Code(model["equation"], className="equation"),
            html.Div(
                className="interval-rule",
                children=[html.Span("80% range", className="legend-chip"),
                          html.Code(model.get("interval_rule", model.get("interval", "")))],
            ),
            *([html.Div(
                className="interval-rule",
                children=[html.Span("target " + ("CR" if model["target_scale"] == "cr_num"
                                                 else "level") + " blank",
                                    className="legend-chip"),
                          html.Code(f"{model['direct']['equation']}  "
                                    f"{model['direct']['interval_rule']}  ·  error "
                                    f"{model['direct']['mae'] / model['direct']['baseline_mae']:.0%}"
                                    f", R² {model['direct']['r2']:.2f}")],
            )] if "direct" in model else []),
            html.Div(
                className="metric-grid",
                children=[
                    metric("R²", f"{r2:.2f}", f"± {model.get('r2_sd', 0)}", bar=r2),
                    metric("MAE", f"{mae:.2f}", f"{gain:.0%} below baseline {base:.2f}",
                           bar=gain),
                    metric("Coverage", f"{model['test_coverage']:.0%}",
                           f"by third: {thirds_s}", bar=model["test_coverage"]),
                ],
            ),
            html.P(
                className="model-foot",
                children=[
                    html.Span("Uses "),
                    *[html.Span(feature_label(f), className="tag") for f in model["features"]],
                    html.Span(" · pruned: " + (", ".join(dropped) or "none")),
                ],
            ),
        ],
    )


@app.callback(Output("equations-list", "children"), Input("equations-list", "id"))
def render_equations(_):
    if not MODELS:
        return html.P("No models found. Run models/fit_models.py first.", className="card-hint")
    children = []
    for direction in ["4e_to_5e", "5e_to_4e"]:
        children.append(html.H3(DIRECTION_LABELS[direction], className="direction-heading"))
        children.append(html.Div(
            className="model-grid",
            children=[model_card(m) for m in MODELS.values() if m["direction"] == direction],
        ))
    return children


if __name__ == "__main__" and sys.platform != "emscripten":  # PyCafe serves `app` itself
    app.run(debug=True)
