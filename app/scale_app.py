#!/usr/bin/env python3
"""Dash web app: scale a monster's difficulty within one edition.

Enter a monster's current level (4e) or CR (5e) and its stats, pick a new
level/CR, and every stat is shifted along the edition's own fitted scaling
curve (models/scale_models.json, built by models/fit_scale_models.py):

    additive stats       new = old + slope * Δaxis
    hp / damage_avg      new = old * ratio ** Δaxis

Tabs:
    Scale   — input form and scaled stat sheet with per-stat deltas.
    Method  — how scaling works and the fitted curve per stat.

Runs on port 8051 so it can live next to the converter app (port 8050).
"""

import json
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, ALL
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).resolve().parent.parent
SCALE_MODELS_PATH = ROOT / "models" / "scale_models.json"


def load_scale_models():
    try:
        with open(SCALE_MODELS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


SCALE_MODELS = load_scale_models()

EDITION_LABELS = {
    "4e": "4e (levels)",
    "5e": "5e (challenge rating)",
}

# 5e challenge ratings, in order.
VALID_CRS = [0, 0.125, 0.25, 0.5] + list(range(1, 31))

CR_LABELS = {0.125: "1/8", 0.25: "1/4", 0.5: "1/2"}

# Stats that scale multiplicatively and how to round each stat for display.
MULTIPLICATIVE = {"hp", "damage_avg"}
HALF_STEP = {"damage_avg"}  # dice averages land on .5 steps

FIELD_GROUPS = [
    ("Core stats", ["hp", "ac", "attack_bonus", "damage_avg"]),
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


def label_for(field):
    mapping = {
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


def stats_for_edition(edition):
    """Ordered stat list for the edition, from the fitted models."""
    if edition not in SCALE_MODELS:
        return []
    stats = set(SCALE_MODELS[edition]["stats"])
    return [f for f in FIELD_ORDER if f in stats]


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


def axis_label(edition):
    return "Level" if edition == "4e" else "CR"


def snap_axis(edition, value):
    """Snap a target axis value to a legal level/CR. Returns None if invalid."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if edition == "4e":
        return float(min(max(round(v), 1), 40))
    return float(min(VALID_CRS, key=lambda cr: abs(cr - v)))


def format_axis(edition, value):
    v = int(value) if float(value).is_integer() else value
    return CR_LABELS.get(v, str(v))


def round_stat(stat, x):
    if stat in HALF_STEP:
        return round(x * 2) / 2
    return int(round(x))


def format_number(stat, x):
    if stat in HALF_STEP and not float(x).is_integer():
        return f"{x:.1f}".rstrip("0").rstrip(".")
    return str(int(round(x)))


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "D&D Monster Difficulty Scaler"

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


def axis_hint(edition):
    return "1 – 40" if edition == "4e" else "0 – 30, or 1/8 = 0.125 …"


def axis_control(which, edition):
    """Level/CR input. update_layout rebuilds it on every edition switch (a
    fresh, empty input); the initial layout holds the 4e version so callbacks
    listening to it can find it from the first render."""
    return html.Div(
        className="field axis-field",
        children=[
            html.Label(f"{'Current' if which == 'current' else 'New'} {axis_label(edition)}",
                       className="field-label"),
            dcc.Input(
                id=f"axis-{which}",
                className="field-input axis-input",
                type="number",
                value="",
                placeholder=axis_hint(edition),
                min=0,
            ),
        ],
    )


app.layout = html.Div(
    className="page",
    children=[
        hero(
            "D&D Monster Difficulty Scaler",
            "Strengthen or weaken a monster within its edition: "
            "stats shift along the edition's own scaling curve.",
            dcc.RadioItems(
                id="scale-edition",
                className="direction-toggle",
                options=[
                    {"label": label, "value": val}
                    for val, label in EDITION_LABELS.items()
                ],
                value="4e",
            ),
        ),
        html.Main(
            className="container",
            children=[
                dcc.Tabs(
                    className="tabs-container",
                    parent_className="tabs-parent",
                    mobile_breakpoint=0,
                    value="scale",
                    children=[
                        tab("Scale", "scale", [
                            html.Div(
                                className="convert-grid",
                                children=[
                                    html.Section(
                                        className="card",
                                        children=[
                                            html.Span("Source", className="card-eyebrow"),
                                            html.H3(id="scale-heading-current"),
                                            html.P(
                                                "Enter the current level/CR and any stats "
                                                "you want scaled.",
                                                className="card-hint",
                                            ),
                                            html.Div(axis_control("current", "4e"), id="axis-current-wrap",
                                                     className="axis-row"),
                                            html.Div(id="scale-input-fields"),
                                        ],
                                    ),
                                    html.Section(
                                        className="card output-card",
                                        children=[
                                            html.Span("Scaled", className="card-eyebrow"),
                                            html.H3(id="scale-heading-target"),
                                            html.P(
                                                "Pick a new level/CR; each stat shows its "
                                                "change from the original.",
                                                className="card-hint",
                                            ),
                                            html.Div(axis_control("target", "4e"), id="axis-target-wrap",
                                                     className="axis-row"),
                                            html.Div(id="scale-output-fields"),
                                        ],
                                    ),
                                ],
                            ),
                        ]),
                        tab("Method", "method", [
                            html.Section(
                                className="card explain-intro",
                                children=[
                                    html.Span("Method", className="card-eyebrow"),
                                    html.H3("Scaling curves"),
                                    html.P(
                                        "Per-edition ordinary least-squares fits of each "
                                        "stat against level (4e) or CR (5e), from the parsed "
                                        "Monster Manual data. Scaling a monster shifts every "
                                        "stat by the curve-implied difference between the old "
                                        "and new level/CR, so the monster keeps its own "
                                        "above- or below-average character: additive stats "
                                        "move by slope × Δ, while HP and damage scale by "
                                        "ratio^Δ. R² shows how tightly each stat actually "
                                        "tracks level/CR in the data."
                                    ),
                                ],
                            ),
                            html.Div(id="scale-curves-list"),
                        ]),
                    ],
                ),
            ],
        ),
        html.Footer(
            className="footer",
            children="Numeric input only — no rule-book text is hosted. "
                     "Curves: models/scale_models.json.",
        ),
    ],
)


@app.callback(
    Output("scale-heading-current", "children"),
    Output("scale-heading-target", "children"),
    Output("axis-current-wrap", "children"),
    Output("axis-target-wrap", "children"),
    Output("scale-input-fields", "children"),
    Output("scale-output-fields", "children"),
    Input("scale-edition", "value"),
)
def update_layout(edition):
    stats = stats_for_edition(edition)

    input_children = []
    for title, fields in group_fields(stats):
        grid = []
        for f in fields:
            grid.append(
                html.Div(
                    className="field",
                    children=[
                        html.Label(label_for(f), className="field-label"),
                        dcc.Input(
                            id={"type": "scale-input", "field": f},
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
    for title, fields in group_fields(stats):
        rows = []
        for f in fields:
            rows.append(
                html.Div(
                    className="stat-row",
                    children=[
                        html.Span(label_for(f), className="stat-label"),
                        html.Div(
                            id={"type": "scale-output", "field": f},
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

    return (
        "Current monster",
        "Scaled monster",
        axis_control("current", edition),
        axis_control("target", edition),
        input_children,
        output_children,
    )


def scale_stat(edition, stat, value, delta_axis):
    """Shift one stat along the edition's curve by delta_axis steps."""
    model = SCALE_MODELS[edition]["stats"].get(stat)
    if model is None or delta_axis == 0:
        return value
    if model["kind"] == "multiplicative":
        return value * (model["per_step_ratio"] ** delta_axis)
    return value + model["slope"] * delta_axis


EMPTY_OUTPUT = [html.Span("—", className="stat-pred stat-empty")]


def format_scaled(stat, old, new):
    """Render a scaled stat with its delta as stat-value children."""
    new_r = round_stat(stat, max(new, 1.0 if stat != "ac" else 0.0))
    delta = new_r - old
    if abs(delta) < 1e-9:
        delta_class, delta_text = "stat-delta delta-flat", "±0"
    elif delta > 0:
        delta_class = "stat-delta delta-up"
        delta_text = f"+{format_number(stat, delta)}"
    else:
        delta_class = "stat-delta delta-down"
        delta_text = f"−{format_number(stat, abs(delta))}"
    return [
        html.Span(delta_text, className=delta_class, title="change from original"),
        html.Span(format_number(stat, new_r), className="stat-pred"),
    ]


@app.callback(
    Output({"type": "scale-output", "field": ALL}, "children"),
    Input("scale-edition", "value"),
    Input("axis-current", "value"),
    Input("axis-target", "value"),
    Input({"type": "scale-input", "field": ALL}, "value"),
    prevent_initial_call=False,
)
def compute_scaled(edition, axis_current, axis_target, input_values):
    stats = stats_for_edition(edition)
    ctx = dash.callback_context
    # An edition switch rebuilds the field components; this callback can fire
    # against the stale DOM (old field set, new edition). Skip those fires —
    # the callback refires once the new components are in place.
    output_ids = [entry["id"]["field"] for entry in ctx.outputs_list]
    if sorted(output_ids) != sorted(stats):
        raise PreventUpdate
    state = {}
    input_ids = set()
    for item in ctx.inputs_list:
        if isinstance(item, list):
            input_ids.update(entry["id"]["field"] for entry in item)
            for entry, val in zip(item, input_values or []):
                state[entry["id"]["field"]] = val
    if input_ids != set(stats):
        raise PreventUpdate

    t0 = snap_axis(edition, axis_current)
    t1 = snap_axis(edition, axis_target)
    delta_axis = (t1 - t0) if (t0 is not None and t1 is not None) else None

    results = {f: EMPTY_OUTPUT for f in stats}
    if delta_axis is not None:
        for stat in stats:
            val = state.get(stat)
            if val in (None, ""):
                continue
            try:
                old = float(val)
            except (TypeError, ValueError):
                continue
            new = scale_stat(edition, stat, old, delta_axis)
            results[stat] = format_scaled(stat, old, new)

    # Dash fills pattern-matched outputs in ctx.outputs_list order, which after
    # a rebuild need not match the stat order: key every result by its id.
    return [results[f] for f in output_ids]


@app.callback(
    Output("scale-heading-target", "children", allow_duplicate=True),
    Input("scale-edition", "value"),
    Input("axis-target", "value"),
    prevent_initial_call=True,
)
def update_target_heading(edition, axis_target):
    t1 = snap_axis(edition, axis_target)
    if t1 is None:
        return "Scaled monster"
    return f"Scaled to {axis_label(edition)} {format_axis(edition, t1)}"


@app.callback(Output("scale-curves-list", "children"), Input("scale-curves-list", "id"))
def render_curves(_):
    if not SCALE_MODELS:
        return html.P("No scaling models found. Run models/fit_scale_models.py first.")
    children = []
    for edition in ["4e", "5e"]:
        spec = SCALE_MODELS[edition]
        children.append(
            html.H3([EDITION_LABELS[edition],
                     html.Span(f"{spec['n_monsters']} monsters", className="badge")],
                    className="direction-heading")
        )
        header = html.Thead(html.Tr([
            html.Th("Stat"), html.Th("Kind"), html.Th("Per step"), html.Th("R²"),
            html.Th("n", className="num"),
        ]))
        rows = []
        for stat in stats_for_edition(edition):
            m = spec["stats"][stat]
            if m["kind"] == "multiplicative":
                per_step = f"×{m['per_step_ratio']:.3f}"
            else:
                per_step = f"+{m['slope']:.3f}"
            rows.append(html.Tr([
                html.Td(label_for(stat)),
                html.Td([
                    html.Span(m["kind"], className=f"tag kind-{m['kind']}"),
                    html.Span(" pooled", className="muted") if m.get("pooled") else None,
                ]),
                html.Td(per_step, className="mono"),
                html.Td(html.Div(className="r2-cell", children=[
                    html.Div(className="r2-bar", children=html.Div(
                        style={"width": f"{max(0.0, min(m['r2'], 1.0)) * 100:.0f}%"})),
                    html.Span(f"{m['r2']:.3f}", className="mono"),
                ])),
                html.Td(m["n"], className="mono num"),
            ]))
        children.append(
            html.Div(
                className="card model-card",
                children=html.Div(html.Table([header, html.Tbody(rows)], className="method-table"),
                                  className="table-scroll"),
            )
        )
    return children


if __name__ == "__main__":
    app.run(debug=True, port=8051)
