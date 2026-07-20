#!/usr/bin/env python3
"""Dash web app: bidirectional D&D stat block conversion with conformal intervals.

Tabs:
    Convert  — numeric input form, predicted output, and 80% scale-adaptive conformal interval.
    Explain  — fitted equations, features, and fit quality per model.
"""

import json
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


def fields_for_direction(direction):
    """Return (input_fields, output_fields) for the chosen direction."""
    input_fields = set()
    output_fields = set()
    for key, model in MODELS.items():
        if model["direction"] != direction:
            continue
        input_fields.update(model["features"])
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


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "D&D Monster Edition Converter"

app.layout = html.Div(
    [
        html.Header(
            className="hero",
            children=[
                html.H1("D&D Monster Edition Converter"),
                html.P(
                    "Bidirectional stat-block conversion with conformal prediction intervals.",
                    className="subtitle",
                ),
                dcc.RadioItems(
                    id="direction",
                    className="direction-toggle",
                    options=[
                        {"label": label, "value": val}
                        for val, label in DIRECTION_LABELS.items()
                    ],
                    value="4e_to_5e",
                ),
            ],
        ),
        html.Main(
            className="container",
            children=[
                dcc.Tabs(
                    className="tabs-container",
                    parent_className="tabs-parent",
                    value="convert",
                    children=[
                        dcc.Tab(
                            label="Convert",
                            value="convert",
                            className="custom-tab",
                            selected_className="custom-tab custom-tab--selected",
                            children=[
                                html.Div(
                                    className="tab-content",
                                    children=[
                                        html.Div(
                                            className="convert-grid",
                                            children=[
                                                html.Section(
                                                    className="card",
                                                    children=[
                                                        html.Span("Source", className="card-eyebrow"),
                                                        html.H3(id="input-heading"),
                                                        html.P(
                                                            "Fill in the stats of your monster.",
                                                            className="card-hint",
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
                                                            "Prediction with 80% interval.",
                                                            className="card-hint",
                                                        ),
                                                        html.Div(id="output-fields"),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Tab(
                            label="Explain",
                            value="explain",
                            className="custom-tab",
                            selected_className="custom-tab custom-tab--selected",
                            children=[
                                html.Div(
                                    className="tab-content",
                                    children=[
                                        html.Section(
                                            className="card explain-intro",
                                            children=[
                                                html.H3("Fitted linear models"),
                                                html.P(
                                                    "Each model is ordinary least-squares on the "
                                                    "matched training pairs. The interval is an "
                                                    "80% scale-adaptive split-conformal prediction "
                                                    "interval (nominal miscoverage α = 0.20). It "
                                                    "grows or shrinks with the size of the prediction."
                                                ),
                                            ],
                                        ),
                                        html.Div(id="equations-list"),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("input-heading", "children"),
    Output("output-heading", "children"),
    Output("input-fields", "children"),
    Output("output-fields", "children"),
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
                            children=[
                                html.Span("—", className="stat-pred stat-empty"),
                            ],
                        ),
                    ],
                )
            )
        output_children.append(
            html.Div(
                className="field-group",
                children=[
                    html.H4(title, className="field-group-title"),
                    html.Div(rows),
                ],
            )
        )

    return (
        f"Input ({source_edition})",
        f"Output ({target_edition})",
        input_children,
        output_children,
    )


def format_output(target, pred, low, high):
    """Render a prediction and its interval as stat-value children."""
    if target in ("cr_num", "damage_avg"):
        pred_s, lo_s, hi_s = f"{pred:.2f}", f"{low:.2f}", f"{high:.2f}"
    else:
        pred_s, lo_s, hi_s = str(round(pred)), str(round(low)), str(round(high))
    return [
        html.Span(pred_s, className="stat-pred"),
        html.Span(f"{lo_s} – {hi_s}", className="stat-interval",
                  title="80% prediction interval"),
    ]


@app.callback(
    Output({"type": "output", "field": ALL}, "children"),
    Input("direction", "value"),
    Input({"type": "input", "field": ALL}, "value"),
    prevent_initial_call=False,
)
def compute_outputs(direction, input_values):
    # Match input values to their fields via the ALL pattern matching order.
    input_fields, output_fields = fields_for_direction(direction)
    ctx = dash.callback_context
    # A direction switch rebuilds the field components; this callback can fire
    # against the stale DOM (old field set, new direction) and would return
    # the wrong number of outputs. Skip those fires — the callback refires
    # once the new components are in place.
    if len(ctx.outputs_list) != len(output_fields):
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

    empty = [html.Span("—", className="stat-pred stat-empty")]
    results = {f: empty for f in output_fields}
    for key, model in MODELS.items():
        if model["direction"] != direction:
            continue
        target = model["target"]
        features = model["features"]
        # collect feature values
        xs = []
        missing = False
        for feat in features:
            val = state.get(feat)
            if val in (None, ""):
                missing = True
                break
            try:
                xs.append(float(val))
            except (TypeError, ValueError):
                missing = True
                break
        if missing:
            results[target] = empty
            continue
        pred = sum(c * x for c, x in zip(model["coef"], xs)) + model["intercept"]
        q_rel = model.get("conformal_quantile_relative", model.get("conformal_quantile", 0))
        low, high = pred * (1 - q_rel), pred * (1 + q_rel)
        low = max(low, 0.0)
        results[target] = format_output(target, pred, low, high)

    return [results[f] for f in output_fields]


@app.callback(Output("equations-list", "children"), Input("equations-list", "id"))
def render_equations(_):
    if not MODELS:
        return html.P("No models found. Run models/fit_models.py first.")
    children = []
    for direction in ["4e_to_5e", "5e_to_4e"]:
        children.append(html.H3(DIRECTION_LABELS[direction], className="direction-heading"))
        for key, model in MODELS.items():
            if model["direction"] != direction:
                continue
            children.append(
                html.Div(
                    className="card model-card",
                    children=[
                        html.H4(label_for(model["target"])),
                        html.Code(model["equation"], className="equation"),
                        html.Ul(
                            className="metrics",
                            children=[
                                html.Li(f"features: {', '.join(model['features'])}"),
                                html.Li(f"R² = {model['r2']}"),
                                html.Li(f"MAE = {model['mae']}"),
                                html.Li(f"baseline MAE = {model['baseline_mae']}"),
                                html.Li(
                                    f"80% rel half-width = {model.get('conformal_quantile_relative', model.get('conformal_quantile', 0))} "
                                    f"(coverage {model['test_coverage']})"
                                ),
                                html.Li(
                                    f"n = {model['n']} (train {model['n_train']} / cal {model['n_cal']} / test {model['n_test']})"
                                ),
                            ],
                        ),
                    ],
                )
            )
    return children


if __name__ == "__main__":
    app.run(debug=True)
