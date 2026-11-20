#!/usr/bin/env python3
"""Dash web app: bidirectional D&D stat block conversion with conformal intervals.

Tabs:
    Convert  — numeric input form, predicted output, and 80% scale-adaptive conformal interval.
    Explain  — fitted equations, features, and fit quality per model.
"""

import json
import math
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, ALL
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = ROOT / "models" / "models.json"


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

# Display grouping for the field grids: (group title, fields)
FIELD_GROUPS = [
    ("Core stats", ["level", "cr_num", "hp", "ac", "attack_bonus", "damage_avg"]),
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


def is_role_feature(field):
    return field.startswith("role_")


def input_stats(feature):
    """Typed-in stats a feature needs: "role_solo*cr_num" -> ["cr_num"]."""
    return [f for f in feature.split("*") if not is_role_feature(f)]


def feature_values(features, state):
    """{feature: float} from typed stats and role indicators in `state`;
    products for interaction terms. None if any input is missing."""
    values = {}
    for feature in features:
        x = 1.0
        for part in feature.split("*"):
            try:
                x *= float(state.get(part))
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
    """Return (input_fields, output_fields) for the chosen direction."""
    input_fields = set()
    output_fields = set()
    for key, model in MODELS.items():
        if model["direction"] != direction:
            continue
        input_fields.update(x for f in model["features"] for x in input_stats(f))
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


def label_for(field):
    mapping = {
        "level": "Level",
        "cr_num": "CR",
        "hp": "HP",
        "ac": "AC",
        "attack_bonus": "Attack Bonus",
        "damage_avg": "Damage Average",
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
        parts.append(part[len("role_"):] if is_role_feature(part) else label_for(part))
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


def hero(title, subtitle, toggle):
    return html.Header(
        className="hero",
        children=html.Div(
            className="hero-inner",
            children=[
                html.Div(
                    className="brand",
                    children=[
                        html.Img(src=app.get_asset_url("d20.svg"), className="brand-mark", alt=""),
                        html.Div([html.H1(title), html.P(subtitle, className="subtitle")]),
                    ],
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
                                        "other edition, the level/CR scale, and 4e role "
                                        "adjustments, with terms kept only if they "
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
                                        "standard) enters as indicator terms; for 5e → 4e "
                                        "it is the role the converted monster should "
                                        "play. 4e multiplies solo HP by 4, so the solo HP "
                                        "bonus is a slope on CR, and 4e "
                                        "minions always have 1 HP."
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
            children="Numeric input only — no rule-book text is hosted. "
                     "Models: models/models.json.",
        ),
    ],
)


@app.callback(
    Output("input-heading", "children"),
    Output("output-heading", "children"),
    Output("input-fields", "children"),
    Output("output-fields", "children"),
    Output("role-label", "children"),
    Input("direction", "value"),
)
def update_layout(direction):
    source_edition = "4e" if direction == "4e_to_5e" else "5e"
    target_edition = "5e" if direction == "4e_to_5e" else "4e"
    input_fields, output_fields = fields_for_direction(direction)

    input_children = []
    for title, fields in group_fields(input_fields):
        grid = []
        for f in fields:
            grid.append(
                html.Div(
                    className="field",
                    children=[
                        html.Label(label_for(f), className="field-label"),
                        dcc.Input(
                            id={"type": "input", "field": f},
                            className="field-input",
                            type="number",
                            value="",
                            placeholder="—",
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
    return (
        f"{source_edition} monster",
        f"{target_edition} conversion",
        input_children,
        output_children,
        role_label,
    )


# The interval slot is always present (empty when there is no range) so the
# predictions line up in one column.
NO_INTERVAL = html.Span(className="stat-interval stat-interval-none")
EMPTY_OUTPUT = [html.Span("—", className="stat-pred stat-empty"), NO_INTERVAL]


def format_output(target, pred, low, high):
    """Render a prediction and its interval as stat-value children."""
    if target in ("cr_num", "damage_avg"):
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
    prevent_initial_call=False,
)
def compute_outputs(direction, input_values, role):
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
    for tier in ("minion", "elite", "solo"):
        state[f"role_{tier}"] = 1.0 if role == tier else 0.0

    empty = EMPTY_OUTPUT
    results = {f: empty for f in output_fields}
    for key, model in MODELS.items():
        if model["direction"] != direction:
            continue
        target = model["target"]
        values = feature_values(model["features"], state)
        if values is None and role not in model.get("fixed_by_role", {}):
            results[target] = empty
            continue
        pred, low, high = predict_with_interval(model, values, role)
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


if __name__ == "__main__":
    app.run(debug=True)
