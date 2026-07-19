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
    order = [
        "level", "cr_num", "hp", "ac",
        "attack_bonus", "damage_avg",
        "ability_str", "ability_dex", "ability_con",
        "ability_int", "ability_wis", "ability_cha",
        "fort", "ref", "will",
        "save_str", "save_dex", "save_con", "save_int", "save_wis", "save_cha",
    ]
    input_fields = [f for f in order if f in input_fields]
    output_fields = [f for f in order if f in output_fields]
    return input_fields, output_fields


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


app = dash.Dash(__name__)
app.title = "D&D Monster Edition Converter"

app.layout = html.Div(
    [
        html.H1("D&D Monster Edition Converter"),
        html.P("Bidirectional numeric stat-block conversion with prediction intervals."),
        dcc.RadioItems(
            id="direction",
            options=[
                {"label": label, "value": val}
                for val, label in DIRECTION_LABELS.items()
            ],
            value="4e_to_5e",
            labelStyle={"display": "inline-block", "marginRight": "15px"},
        ),
        html.Hr(),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Convert",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3(id="input-heading"),
                                        html.Div(id="input-fields"),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "verticalAlign": "top",
                                        "width": "45%",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.H3(id="output-heading"),
                                        html.Div(id="output-fields"),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "verticalAlign": "top",
                                        "width": "50%",
                                        "marginLeft": "5%",
                                    },
                                ),
                            ],
                            style={"padding": "10px"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Explain",
                    children=[
                        html.Div(
                            [
                                html.H3("Fitted linear models"),
                                html.P(
                                    "Each model is ordinary least-squares on the "
                                    "matched training pairs. The interval is an "
                                    "80% scale-adaptive split-conformal prediction "
                                    "interval (nominal miscoverage α = 0.20). It "
                                    "grows or shrinks with the size of the prediction."
                                ),
                                html.Div(id="equations-list"),
                            ],
                            style={"padding": "10px"},
                        )
                    ],
                ),
            ]
        ),
    ],
    style={"maxWidth": "1000px", "margin": "0 auto"},
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
    for f in input_fields:
        input_children.append(
            html.Div(
                [
                    html.Label(label_for(f)),
                    dcc.Input(
                        id={"type": "input", "field": f},
                        type="number",
                        value="",
                        placeholder=label_for(f),
                        style={"width": "120px"},
                    ),
                ],
                style={"margin": "6px 0"},
            )
        )

    output_children = []
    for f in output_fields:
        output_children.append(
            html.Div(
                [
                    html.Label(label_for(f)),
                    dcc.Input(
                        id={"type": "output", "field": f},
                        type="text",
                        value="",
                        readOnly=True,
                        style={"width": "220px"},
                    ),
                ],
                style={"margin": "6px 0"},
            )
        )

    return (
        f"Input ({source_edition})",
        f"Output ({target_edition})",
        input_children,
        output_children,
    )


@app.callback(
    Output({"type": "output", "field": ALL}, "value"),
    Input("direction", "value"),
    Input({"type": "input", "field": ALL}, "value"),
    prevent_initial_call=False,
)
def compute_outputs(direction, input_values):
    # Match input values to their fields via the ALL pattern matching order.
    input_fields, output_fields = fields_for_direction(direction)
    # Dash passes input_values in the order the components were created in the
    # layout callback. Rebuild the input state dictionary.
    ctx = dash.callback_context
    # The first trigger is direction; remaining are inputs in order.
    values = [direction] + (input_values or [])
    state = {}
    # ctx.inputs_list contains the IDs for the pattern-matched inputs.
    inputs_list = ctx.inputs_list
    # inputs_list is a list; first element is direction (dict without pattern),
    # rest are the pattern-matched inputs.
    for item in inputs_list:
        if isinstance(item, list):
            for entry, val in zip(item, input_values or []):
                field = entry["id"]["field"]
                state[field] = val
        elif isinstance(item, dict) and item.get("id") == "direction":
            pass  # already known

    results = {f: "" for f in output_fields}
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
            results[target] = ""
            continue
        pred = sum(c * x for c, x in zip(model["coef"], xs)) + model["intercept"]
        q_rel = model.get("conformal_quantile_relative", model.get("conformal_quantile", 0))
        low, high = pred * (1 - q_rel), pred * (1 + q_rel)
        low = max(low, 0.0)
        if target in ("cr_num", "damage_avg"):
            results[target] = f"{pred:.2f}  ({low:.2f} – {high:.2f})"
        else:
            results[target] = f"{round(pred)}  ({round(low)} – {round(high)})"

    return [results[f] for f in output_fields]


@app.callback(Output("equations-list", "children"), Input("equations-list", "id"))
def render_equations(_):
    if not MODELS:
        return html.P("No models found. Run models/fit_models.py first.")
    children = []
    for direction in ["4e_to_5e", "5e_to_4e"]:
        children.append(html.H3(DIRECTION_LABELS[direction]))
        for key, model in MODELS.items():
            if model["direction"] != direction:
                continue
            children.append(html.H4(label_for(model["target"])))
            children.append(html.P(html.Code(model["equation"])))
            children.append(
                html.Ul(
                    [
                        html.Li(f"Features: {', '.join(model['features'])}"),
                        html.Li(f"R² = {model['r2']}"),
                        html.Li(f"MAE = {model['mae']}"),
                        html.Li(f"Baseline MAE = {model['baseline_mae']}"),
                        html.Li(
                            f"80% relative half-width = {model.get('conformal_quantile_relative', model.get('conformal_quantile', 0))} "
                            f"(test coverage {model['test_coverage']})"
                        ),
                        html.Li(
                            f"n={model['n']} (train {model['n_train']}, cal {model['n_cal']}, test {model['n_test']})"
                        ),
                    ]
                )
            )
    return children


if __name__ == "__main__":
    app.run(debug=True)
