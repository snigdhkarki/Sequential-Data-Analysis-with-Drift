"""
Plot temp, pressure, alarm, and predicted alarm-probability from
sensor_data.csv (+ alarm_probability_predictions.csv, if present) in one
wide, scrollable figure.

- temp and pressure are min-max scaled to 0-1 so they sit on the same visual
  level despite having very different units/ranges.
- the predicted "chance of alarm in the next 100 rows" (from
  train_alarm_predictor.py) is overlaid on that same 0-1 axis -- it's
  already a 0-100% probability, so it's just divided by 100. It only
  exists for the test rows (70,000+), so the line simply starts there.
- alarms are drawn as red vertical lines across the full height of the plot.
- a dashed gray line marks the train/test split (where the probability
  curve begins).
- the figure is wide and has a range-slider + zoom/pan enabled, so you can
  scroll/zoom through all rows smoothly (works in any browser).

Usage:
    pip install pandas plotly
    python plot_sensor_data.py
    (opens sensor_data_plot.html in your browser)
"""

import os

import pandas as pd
import plotly.graph_objects as go

CSV_PATH = "sensor_data.csv"
PRED_PATH = "alarm_probability_predictions.csv"   # optional; from train_alarm_predictor.py
OUT_HTML = "sensor_data_plot.html"

INITIAL_ROWS_VISIBLE = 500   # how many rows to show before you start scrolling


def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())


def main():
    df = pd.read_csv(CSV_PATH)
    df["row"] = df.index

    temp_scaled = min_max_scale(df["temp"])
    pressure_scaled = min_max_scale(df["pressure"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["row"], y=temp_scaled,
        mode="lines", name="temp",
        line=dict(color="royalblue", width=1),
        customdata=df["temp"],
        hovertemplate="row %{x}<br>temp: %{customdata:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df["row"], y=pressure_scaled,
        mode="lines", name="pressure",
        line=dict(color="darkorange", width=1),
        customdata=df["pressure"],
        hovertemplate="row %{x}<br>pressure: %{customdata:.1f}<extra></extra>",
    ))

    # ---- overlay the predicted alarm probability, if available ----
    split_row = None
    if os.path.exists(PRED_PATH):
        preds = pd.read_csv(PRED_PATH)[["row_index", "alarm_within_next_100_rows_pct"]]
        preds = preds.rename(columns={"row_index": "row"})
        df = df.merge(preds, on="row", how="left")
        prob_scaled = df["alarm_within_next_100_rows_pct"] / 100.0  # already 0-100 -> 0-1

        fig.add_trace(go.Scatter(
            x=df["row"], y=prob_scaled,
            mode="lines", name="predicted alarm probability",
            line=dict(color="seagreen", width=1.5),
            customdata=df["alarm_within_next_100_rows_pct"],
            hovertemplate="row %{x}<br>alarm chance: %{customdata:.1f}%<extra></extra>",
        ))

        split_row = int(preds["row"].min())
    else:
        print(f"Note: {PRED_PATH} not found -- plotting temp/pressure/alarm only.")

    # if we have predictions, start the visible window there so the probability
    # curve is on-screen immediately instead of the empty (NaN) training region
    initial_start = split_row if split_row is not None else 0

    # red vertical lines for every alarm row (full-height, drawn behind the traces)
    alarm_rows = df.loc[df["alarm"] == 1, "row"]
    shapes = [
        dict(
            type="line",
            xref="x", yref="paper",
            x0=r, x1=r, y0=0, y1=1,
            line=dict(color="red", width=1.2),
            opacity=0.6,
            layer="below",
        )
        for r in alarm_rows
    ]

    # dashed line marking where the train/test split happens (probability curve starts here)
    if split_row is not None:
        shapes.append(dict(
            type="line",
            xref="x", yref="paper",
            x0=split_row, x1=split_row, y0=0, y1=1,
            line=dict(color="gray", width=1.5, dash="dash"),
            layer="above",
        ))

    fig.update_layout(
        shapes=shapes,
        title="temp & pressure (scaled 0-1), predicted alarm probability, and alarms (red lines)",
        xaxis=dict(
            title="row index",
            rangeslider=dict(visible=True),   # <- lets you scroll/pan across all rows
            range=[initial_start, initial_start + INITIAL_ROWS_VISIBLE],  # initial visible window
        ),
        yaxis=dict(title="scaled value (0-1)"),
        width=1800,
        height=550,
        margin=dict(l=60, r=30, t=60, b=40),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )

    fig.write_html(OUT_HTML)
    print(f"Saved {OUT_HTML} ({len(df)} rows, {len(alarm_rows)} alarms)")


if __name__ == "__main__":
    main()