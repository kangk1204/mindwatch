"""
Interactive data coverage visualizations (anonymized).

Outputs:
1) mindwatch_coverage_interactive.html  - coverage bars + survey markers for all participants
2) mindwatch_dashboard.html             - summary plots (PHQ-9 distributions, durations, timeline subset)
3) mindwatch_coverage_all_participants.png (best effort, requires kaleido)
"""
from __future__ import annotations

import sys
from typing import Dict, List

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from train_tft import load_label_frames, load_hourly_sensor_frames
from pathlib import Path


def _anonymize_ids(ids: List[str]) -> Dict[str, str]:
    return {pid: f"P{idx+1:03d}" for idx, pid in enumerate(ids)}


def _aggregate_sensor_coverage(sensor_frames: Dict[str, pd.DataFrame], pid: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []
    counts: List[int] = []
    for df in sensor_frames.values():
        if pid not in df.columns:
            continue
        series = df[pid].dropna()
        if series.empty:
            continue
        starts.append(series.index.min())
        ends.append(series.index.max())
        counts.append(len(series))
    if not starts:
        return None, None, 0
    return min(starts), max(ends), int(sum(counts))


def build_coverage(labels: pd.DataFrame, sensor_frames: Dict[str, pd.DataFrame]) -> tuple[list[dict], Dict[str, str]]:
    participant_ids = labels["ID"].unique()
    id_map = _anonymize_ids(participant_ids.tolist())
    coverage: List[dict] = []

    for idx, pid in enumerate(participant_ids):
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(participant_ids)} participants...")

        participant_labels = labels[labels["ID"] == pid].sort_values("survey_wave")
        surveys = [
            {"wave": int(row["survey_wave"]), "timestamp": row["survey_timestamp"], "phq9": float(row["PHQ9_Score"])}
            for _, row in participant_labels.iterrows()
        ]

        sensor_start, sensor_end, sensor_count = _aggregate_sensor_coverage(sensor_frames, pid)
        coverage.append(
            {
                "participant_id": pid,
                "anon_id": id_map[pid],
                "surveys": surveys,
                "sensor_start": sensor_start,
                "sensor_end": sensor_end,
                "sensor_count": sensor_count,
            }
        )

    coverage.sort(key=lambda x: x["surveys"][0]["timestamp"] if x["surveys"] else pd.Timestamp.max)
    return coverage, id_map


def build_coverage_figure(coverage_data: List[dict]) -> go.Figure:
    fig = go.Figure()

    sensor_y = []
    sensor_width = []
    sensor_base = []
    sensor_hover = []
    for idx, pdata in enumerate(coverage_data):
        if pdata["sensor_start"] and pdata["sensor_end"]:
            duration_hours = (pdata["sensor_end"] - pdata["sensor_start"]).total_seconds() / 3600
            sensor_y.append(idx)
            sensor_width.append(duration_hours / 24)
            base_ts = pdata["sensor_start"] + (pdata["sensor_end"] - pdata["sensor_start"]) / 2
            sensor_base.append(base_ts - pd.Timedelta(days=sensor_width[-1] / 2))
            duration_days = (pdata["sensor_end"] - pdata["sensor_start"]).days
            sensor_hover.append(
                f"{pdata['anon_id']}<br>"
                f"Sensor: {pdata['sensor_start'].date()} to {pdata['sensor_end'].date()}<br>"
                f"Duration: {duration_days} days<br>"
                f"Data points (all sensors): {pdata['sensor_count']}"
            )

    fig.add_trace(
        go.Bar(
            x=sensor_width,
            y=sensor_y,
            base=sensor_base,
            orientation="h",
            marker=dict(color="rgba(52, 152, 219, 0.3)", line=dict(color="rgb(52, 152, 219)", width=1)),
            name="Sensor Data (any sensor)",
            text=sensor_hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        )
    )

    wave1_x = []
    wave1_y = []
    wave1_text = []
    wave2_x = []
    wave2_y = []
    wave2_text = []
    for idx, pdata in enumerate(coverage_data):
        for survey in pdata["surveys"]:
            hover = (
                f"{pdata['anon_id']}<br>"
                f"Wave {survey['wave']}<br>"
                f"Date: {survey['timestamp'].date()}<br>"
                f"PHQ-9: {survey['phq9']:.0f}"
            )
            if survey["wave"] == 1:
                wave1_x.append(survey["timestamp"])
                wave1_y.append(idx)
                wave1_text.append(hover)
            else:
                wave2_x.append(survey["timestamp"])
                wave2_y.append(idx)
                wave2_text.append(hover)

    fig.add_trace(
        go.Scatter(
            x=wave1_x,
            y=wave1_y,
            mode="markers",
            marker=dict(size=8, color="red", symbol="circle", line=dict(color="black", width=1)),
            name="Wave 1 Survey",
            text=wave1_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=wave2_x,
            y=wave2_y,
            mode="markers",
            marker=dict(size=8, color="orange", symbol="square", line=dict(color="black", width=1)),
            name="Wave 2 Survey",
            text=wave2_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "<b>MindWatch Data Coverage - All Participants</b><br>"
                "<sub>Interactive: Zoom, Pan, Hover | Blue: Any Sensor Data | Red: Wave 1 | Orange: Wave 2</sub>"
            ),
        x=0.5,
        xanchor="center",
        ),
        xaxis=dict(title="<b>Date</b>", showgrid=True, gridcolor="lightgray", gridwidth=0.5),
        yaxis=dict(
            title="<b>Participants (sorted by survey date)</b>",
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
            range=[-5, len(coverage_data) + 5],
            tickmode="array",
            tickvals=list(range(len(coverage_data))),
            ticktext=[pdata["anon_id"] for pdata in coverage_data],
        ),
        hovermode="closest",
        height=max(800, len(coverage_data) * 0.6),
        width=1400,
        showlegend=True,
        legend=dict(x=1.01, y=1, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_dashboard(coverage_data: List[dict]) -> go.Figure:
    fig_dashboard = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "PHQ-9 Score Distribution by Wave",
            "Sensor Data Duration Distribution",
            "Timeline Overview (first 200)",
            "Wave Participation",
        ),
        specs=[[{"type": "box"}, {"type": "histogram"}], [{"type": "scatter"}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    wave1_scores = [s["phq9"] for p in coverage_data for s in p["surveys"] if s["wave"] == 1]
    wave2_scores = [s["phq9"] for p in coverage_data for s in p["surveys"] if s["wave"] == 2]
    fig_dashboard.add_trace(go.Box(y=wave1_scores, name="Wave 1", marker_color="red", boxmean="sd"), row=1, col=1)
    fig_dashboard.add_trace(go.Box(y=wave2_scores, name="Wave 2", marker_color="orange", boxmean="sd"), row=1, col=1)

    durations = [(p["sensor_end"] - p["sensor_start"]).days for p in coverage_data if p["sensor_start"] and p["sensor_end"]]
    fig_dashboard.add_trace(go.Histogram(x=durations, nbinsx=30, marker_color="steelblue", name="Duration"), row=1, col=2)

    display_count = min(200, len(coverage_data))
    for idx in range(display_count):
        pdata = coverage_data[idx]
        if pdata["sensor_start"] and pdata["sensor_end"]:
            fig_dashboard.add_trace(
                go.Scatter(
                    x=[pdata["sensor_start"], pdata["sensor_end"]],
                    y=[idx, idx],
                    mode="lines",
                    line=dict(color="lightblue", width=3),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )

    wave_counts = {
        "Wave 1 only": sum(1 for p in coverage_data if len(p["surveys"]) == 1),
        "Both Waves": sum(1 for p in coverage_data if len(p["surveys"]) == 2),
    }
    fig_dashboard.add_trace(
        go.Bar(
            x=list(wave_counts.keys()),
            y=list(wave_counts.values()),
            marker_color=["lightcoral", "lightsalmon"],
            text=list(wave_counts.values()),
            textposition="auto",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig_dashboard.update_layout(title_text="<b>MindWatch Data Summary Dashboard</b>", title_x=0.5, height=900, width=1400, showlegend=True)
    fig_dashboard.update_xaxes(title_text="Wave", row=1, col=1)
    fig_dashboard.update_yaxes(title_text="PHQ-9 Score", row=1, col=1)
    fig_dashboard.update_xaxes(title_text="Duration (days)", row=1, col=2)
    fig_dashboard.update_yaxes(title_text="Count", row=1, col=2)
    fig_dashboard.update_xaxes(title_text="Date", row=2, col=1)
    fig_dashboard.update_yaxes(title_text="Participant", row=2, col=1)
    fig_dashboard.update_yaxes(title_text="Count", row=2, col=2)
    return fig_dashboard


def main(output_dir: str = "plots") -> None:
    print("Loading data...")
    labels = load_label_frames()
    sensor_frames = load_hourly_sensor_frames()

    print("Analyzing coverage for all participants...")
    coverage_data, _ = build_coverage(labels, sensor_frames)
    print(f"✓ Analyzed {len(coverage_data)} participants")

    print("\nCreating interactive visualization...")
    coverage_fig = build_coverage_figure(coverage_data)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    html_file = outdir / "mindwatch_coverage_interactive.html"
    coverage_fig.write_html(html_file)
    print(f"✓ Interactive HTML saved: {html_file}")

    print("\nCreating summary dashboard...")
    dashboard = build_dashboard(coverage_data)
    dashboard_file = outdir / "mindwatch_dashboard.html"
    dashboard.write_html(dashboard_file)
    print(f"✓ Summary dashboard saved: {dashboard_file}")

    print("\nCreating high-quality static PNG...")
    try:
        import plotly.io as pio  # noqa: F401

        fig_static = go.Figure(coverage_fig)
        fig_static.update_layout(
            height=max(1200, len(coverage_data) * 0.3),
            width=2000,
            font=dict(size=10),
        )
        png_file = outdir / "mindwatch_coverage_all_participants.png"
        fig_static.write_image(png_file, scale=2)
        print(f"✓ High-quality PNG saved: {png_file}")
    except Exception as e:
        print(f"⚠ Could not create PNG (kaleido issue): {e}")
        print("  (HTML files are still available)")

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("Generated files:")
    print(f"  1. {html_file} (interactive coverage)")
    print(f"  2. {dashboard_file} (summary plots)")
    print(f"  3. {outdir / 'mindwatch_coverage_all_participants.png'} (if generated)")
    print("=" * 70)


if __name__ == "__main__":
    main()
