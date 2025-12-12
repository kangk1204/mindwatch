"""
Interactive visualization focused on PHQ-9 survey timepoints and scores.
Analyzes patterns, changes, and distributions of PHQ-9 assessments.
"""
from __future__ import annotations

import sys
from typing import Dict, List

sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from train_tft import load_label_frames


def _anonymize_ids(ids: List[str]) -> Dict[str, str]:
    return {pid: f"P{idx+1:03d}" for idx, pid in enumerate(ids)}


def _severity_color(score: float) -> tuple[str, str]:
    if score < 5:
        return "Minimal (0-4)", "#2ecc71"
    elif score < 10:
        return "Mild (5-9)", "#f39c12"
    elif score < 15:
        return "Moderate (10-14)", "#e67e22"
    elif score < 20:
        return "Moderately Severe (15-19)", "#e74c3c"
    else:
        return "Severe (20-27)", "#8e44ad"


def prepare_participants(labels: pd.DataFrame) -> tuple[list[dict], Dict[str, str]]:
    ids = labels["ID"].unique()
    id_map = _anonymize_ids(ids.tolist())
    participants: List[dict] = []
    for pid in ids:
        person_data = labels[labels["ID"] == pid].sort_values("survey_wave")
        surveys = [
            {
                "wave": int(row["survey_wave"]),
                "timestamp": row["survey_timestamp"],
                "phq9": float(row["PHQ9_Score"]),
                "gad7": float(row.get("GAD7_Score", np.nan)),
                "isi": float(row.get("ISI_Score", np.nan)),
            }
            for _, row in person_data.iterrows()
        ]
        participants.append({"id": pid, "anon_id": id_map[pid], "surveys": surveys, "n_surveys": len(surveys)})
    participants.sort(key=lambda x: x["surveys"][0]["timestamp"])
    return participants, id_map


def build_figures(participants_data: List[dict]) -> list[go.Figure]:
    figs: List[go.Figure] = []

    # Timeline figure
    fig1 = go.Figure()
    wave1_data = []
    wave2_data = []
    for idx, pdata in enumerate(participants_data):
        for survey in pdata["surveys"]:
            severity, color = _severity_color(survey["phq9"])
            gad7_str = f"{survey['gad7']:.0f}" if not np.isnan(survey["gad7"]) else "N/A"
            isi_str = f"{survey['isi']:.0f}" if not np.isnan(survey["isi"]) else "N/A"
            hover = (
                f"{pdata['anon_id']}<br>"
                f"Wave {survey['wave']} | Date: {survey['timestamp'].strftime('%Y-%m-%d')}<br>"
                f"PHQ-9: {survey['phq9']:.0f} ({severity})<br>"
                f"GAD-7: {gad7_str} | ISI: {isi_str}"
            )
            point = {"x": survey["timestamp"], "y": idx, "color": color, "hover": hover, "wave": survey["wave"]}
            if survey["wave"] == 1:
                wave1_data.append(point)
            else:
                wave2_data.append(point)

        if len(pdata["surveys"]) == 2:
            s1, s2 = pdata["surveys"][0], pdata["surveys"][1]
            change = s2["phq9"] - s1["phq9"]
            if change > 2:
                line_color = "rgba(231, 76, 60, 0.3)"
            elif change < -2:
                line_color = "rgba(46, 204, 113, 0.3)"
            else:
                line_color = "rgba(149, 165, 166, 0.2)"
            fig1.add_trace(
                go.Scatter(
                    x=[s1["timestamp"], s2["timestamp"]],
                    y=[idx, idx],
                    mode="lines",
                    line=dict(color=line_color, width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if wave1_data:
        fig1.add_trace(
            go.Scatter(
                x=[d["x"] for d in wave1_data],
                y=[d["y"] for d in wave1_data],
                mode="markers",
                marker=dict(size=12, color=[d["color"] for d in wave1_data], symbol="circle", line=dict(color="black", width=1.5)),
                name="Wave 1",
                text=[d["hover"] for d in wave1_data],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )
    if wave2_data:
        fig1.add_trace(
            go.Scatter(
                x=[d["x"] for d in wave2_data],
                y=[d["y"] for d in wave2_data],
                mode="markers",
                marker=dict(size=12, color=[d["color"] for d in wave2_data], symbol="square", line=dict(color="black", width=1.5)),
                name="Wave 2",
                text=[d["hover"] for d in wave2_data],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )

    severity_colors = [
        ("Minimal (0-4)", "#2ecc71"),
        ("Mild (5-9)", "#f39c12"),
        ("Moderate (10-14)", "#e67e22"),
        ("Moderately Severe (15-19)", "#e74c3c"),
        ("Severe (20-27)", "#8e44ad"),
    ]
    for label, color in severity_colors:
        fig1.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color, symbol="circle", line=dict(color="black", width=1)),
                name=label,
                showlegend=True,
            )
        )

    fig1.update_layout(
        title=dict(
            text="<b>PHQ-9 Survey Timeline (Anonymized)</b><br><sub>Markers colored by severity; hover shows PHQ/GAD/ISI</sub>",
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(title="<b>Date</b>", showgrid=True, gridcolor="lightgray", gridwidth=0.5),
        yaxis=dict(
            title="<b>Participants</b>",
            tickmode="array",
            tickvals=list(range(len(participants_data))),
            ticktext=[p["anon_id"] for p in participants_data],
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
        ),
        hovermode="closest",
        height=max(800, len(participants_data) * 0.6),
        width=1200,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, bgcolor="white", bordercolor="black", borderwidth=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    figs.append(fig1)
    return figs


def main(output_dir: str = "plots") -> None:
    print("Loading data...")
    labels = load_label_frames()
    print(f"Total survey responses: {len(labels)}")
    print(f"Unique participants: {len(labels['ID'].unique())}")

    participants_data, _ = prepare_participants(labels)
    print(f"\nAnalyzed {len(participants_data)} participants")

    figs = build_figures(participants_data)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    html_path = outdir / "phq9_timeline_interactive.html"
    figs[0].write_html(html_path)
    print(f"\n✓ Interactive PHQ-9 timeline saved: {html_path}")


if __name__ == "__main__":
    from pathlib import Path
    main()

# ============================================================================
# VISUALIZATION 1: PHQ-9 Timeline with Score Colors
# ============================================================================
print("\nCreating PHQ-9 timeline visualization...")

# Sort by first survey date
participants_data.sort(key=lambda x: x['surveys'][0]['timestamp'])

fig1 = go.Figure()

# Define PHQ-9 severity colors
def get_phq9_color(score):
    if score < 5:
        return '#2ecc71'  # Green: Minimal
    elif score < 10:
        return '#f39c12'  # Yellow: Mild
    elif score < 15:
        return '#e67e22'  # Orange: Moderate
    elif score < 20:
        return '#e74c3c'  # Red: Moderately severe
    else:
        return '#8e44ad'  # Purple: Severe

def get_phq9_severity(score):
    if score < 5:
        return 'Minimal (0-4)'
    elif score < 10:
        return 'Mild (5-9)'
    elif score < 15:
        return 'Moderate (10-14)'
    elif score < 20:
        return 'Moderately Severe (15-19)'
    else:
        return 'Severe (20-27)'

# Collect all survey points
wave1_data = []
wave2_data = []

for idx, pdata in enumerate(participants_data):
    for survey in pdata['surveys']:
        color = get_phq9_color(survey['phq9'])
        severity = get_phq9_severity(survey['phq9'])

        gad7_str = f"{survey['gad7']:.0f}" if not np.isnan(survey['gad7']) else 'N/A'
        isi_str = f"{survey['isi']:.0f}" if not np.isnan(survey['isi']) else 'N/A'

        hover_text = (
            f"<b>Participant {idx+1}</b><br>" +
            f"ID: {pdata['id']}<br>" +
            f"Wave: {survey['wave']}<br>" +
            f"Date: {survey['timestamp'].strftime('%Y-%m-%d')}<br>" +
            f"<b>PHQ-9: {survey['phq9']:.0f}</b><br>" +
            f"Severity: {severity}<br>" +
            f"GAD-7: {gad7_str}<br>" +
            f"ISI: {isi_str}"
        )

        point_data = {
            'x': survey['timestamp'],
            'y': idx,
            'color': color,
            'phq9': survey['phq9'],
            'text': hover_text,
            'wave': survey['wave']
        }

        if survey['wave'] == 1:
            wave1_data.append(point_data)
        else:
            wave2_data.append(point_data)

# Add Wave 1 markers
if wave1_data:
    fig1.add_trace(go.Scatter(
        x=[d['x'] for d in wave1_data],
        y=[d['y'] for d in wave1_data],
        mode='markers',
        marker=dict(
            size=12,
            color=[d['color'] for d in wave1_data],
            symbol='circle',
            line=dict(color='black', width=1.5),
            opacity=0.9
        ),
        name='Wave 1',
        text=[d['text'] for d in wave1_data],
        hovertemplate='%{text}<extra></extra>',
        showlegend=True
    ))

# Add Wave 2 markers
if wave2_data:
    fig1.add_trace(go.Scatter(
        x=[d['x'] for d in wave2_data],
        y=[d['y'] for d in wave2_data],
        mode='markers',
        marker=dict(
            size=12,
            color=[d['color'] for d in wave2_data],
            symbol='square',
            line=dict(color='black', width=1.5),
            opacity=0.9
        ),
        name='Wave 2',
        text=[d['text'] for d in wave2_data],
        hovertemplate='%{text}<extra></extra>',
        showlegend=True
    ))

# Add connecting lines for participants with both waves
for idx, pdata in enumerate(participants_data):
    if len(pdata['surveys']) == 2:
        s1, s2 = pdata['surveys'][0], pdata['surveys'][1]

        # Color line by change direction
        change = s2['phq9'] - s1['phq9']
        if change > 2:
            line_color = 'rgba(231, 76, 60, 0.3)'  # Red: Worsened
        elif change < -2:
            line_color = 'rgba(46, 204, 113, 0.3)'  # Green: Improved
        else:
            line_color = 'rgba(149, 165, 166, 0.2)'  # Gray: Stable

        fig1.add_trace(go.Scatter(
            x=[s1['timestamp'], s2['timestamp']],
            y=[idx, idx],
            mode='lines',
            line=dict(color=line_color, width=1.5),
            showlegend=False,
            hoverinfo='skip'
        ))

# Add severity color legend manually
severity_colors = [
    ('Minimal (0-4)', '#2ecc71'),
    ('Mild (5-9)', '#f39c12'),
    ('Moderate (10-14)', '#e67e22'),
    ('Moderately Severe (15-19)', '#e74c3c'),
    ('Severe (20-27)', '#8e44ad')
]

for severity, color in severity_colors:
    fig1.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=color, symbol='circle', line=dict(color='black', width=1)),
        name=severity,
        showlegend=True
    ))

fig1.update_layout(
    title=dict(
        text='<b>PHQ-9 Survey Timeline - All Participants</b><br>' +
             '<sub>Circle: Wave 1 | Square: Wave 2 | Colors: PHQ-9 Severity</sub>',
        x=0.5,
        xanchor='center'
    ),
    xaxis=dict(
        title='<b>Survey Date</b>',
        showgrid=True,
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title='<b>Participants (sorted by first survey date)</b>',
        showgrid=True,
        gridcolor='lightgray',
        range=[-10, len(participants_data) + 10]
    ),
    hovermode='closest',
    height=max(1000, len(participants_data) * 0.4),
    width=1600,
    legend=dict(
        x=1.01,
        y=1,
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='black',
        borderwidth=1
    ),
    plot_bgcolor='white'
)

html_file1 = 'phq9_timeline_interactive.html'
fig1.write_html(html_file1)
print(f"✓ PHQ-9 timeline saved: {html_file1}")

# ============================================================================
# VISUALIZATION 2: PHQ-9 Change Analysis (Wave 1 → Wave 2)
# ============================================================================
print("\nCreating PHQ-9 change analysis...")

# Get participants with both waves
both_waves = [p for p in participants_data if len(p['surveys']) == 2]

wave1_scores = [p['surveys'][0]['phq9'] for p in both_waves]
wave2_scores = [p['surveys'][1]['phq9'] for p in both_waves]
changes = [w2 - w1 for w1, w2 in zip(wave1_scores, wave2_scores)]
participant_ids = [p['id'] for p in both_waves]

fig2 = go.Figure()

# Scatter plot: Wave 1 vs Wave 2
colors_change = ['red' if c > 2 else 'green' if c < -2 else 'gray' for c in changes]

fig2.add_trace(go.Scatter(
    x=wave1_scores,
    y=wave2_scores,
    mode='markers',
    marker=dict(
        size=8,
        color=colors_change,
        line=dict(color='black', width=0.5),
        opacity=0.6
    ),
    text=[
        f"Participant {i+1}<br>" +
        f"ID: {pid}<br>" +
        f"Wave 1: {w1:.0f}<br>" +
        f"Wave 2: {w2:.0f}<br>" +
        f"Change: {c:+.0f}"
        for i, (pid, w1, w2, c) in enumerate(zip(participant_ids, wave1_scores, wave2_scores, changes))
    ],
    hovertemplate='%{text}<extra></extra>',
    showlegend=False
))

# Add diagonal line (no change)
max_score = max(max(wave1_scores), max(wave2_scores))
fig2.add_trace(go.Scatter(
    x=[0, max_score],
    y=[0, max_score],
    mode='lines',
    line=dict(color='black', dash='dash', width=2),
    name='No Change',
    hoverinfo='skip'
))

# Add clinical threshold lines
fig2.add_hline(y=10, line_dash="dot", line_color="orange",
               annotation_text="Wave 2 Threshold (PHQ-9≥10)",
               annotation_position="right")
fig2.add_vline(x=10, line_dash="dot", line_color="orange",
               annotation_text="Wave 1 Threshold (PHQ-9≥10)",
               annotation_position="top")

fig2.update_layout(
    title='<b>PHQ-9 Score Change: Wave 1 → Wave 2</b><br>' +
          '<sub>Red: Worsened (Δ>2) | Green: Improved (Δ<-2) | Gray: Stable</sub>',
    xaxis_title='<b>Wave 1 PHQ-9 Score</b>',
    yaxis_title='<b>Wave 2 PHQ-9 Score</b>',
    hovermode='closest',
    height=700,
    width=800,
    plot_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='lightgray', range=[-1, max_score+1]),
    yaxis=dict(showgrid=True, gridcolor='lightgray', range=[-1, max_score+1])
)

html_file2 = 'phq9_change_analysis.html'
fig2.write_html(html_file2)
print(f"✓ PHQ-9 change analysis saved: {html_file2}")

# ============================================================================
# VISUALIZATION 3: Comprehensive PHQ-9 Dashboard
# ============================================================================
print("\nCreating comprehensive PHQ-9 dashboard...")

fig3 = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'PHQ-9 Distribution by Wave',
        'PHQ-9 Severity Categories',
        'PHQ-9 Change Distribution (Wave 1→2)',
        'Time Between Waves (Days)',
        'PHQ-9 vs GAD-7 Correlation',
        'PHQ-9 Temporal Trend'
    ),
    specs=[
        [{"type": "violin"}, {"type": "bar"}],
        [{"type": "histogram"}, {"type": "histogram"}],
        [{"type": "scatter"}, {"type": "scatter"}]
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.15
)

# 1. Violin plot: PHQ-9 by Wave
all_wave1 = labels[labels['survey_wave'] == 1]['PHQ9_Score'].dropna()
all_wave2 = labels[labels['survey_wave'] == 2]['PHQ9_Score'].dropna()

fig3.add_trace(
    go.Violin(y=all_wave1, name='Wave 1', box_visible=True, meanline_visible=True,
              fillcolor='rgba(231, 76, 60, 0.5)', line_color='red'),
    row=1, col=1
)
fig3.add_trace(
    go.Violin(y=all_wave2, name='Wave 2', box_visible=True, meanline_visible=True,
              fillcolor='rgba(243, 156, 18, 0.5)', line_color='orange'),
    row=1, col=1
)

# 2. Bar: Severity categories
def categorize_severity(score):
    if score < 5:
        return 'Minimal'
    elif score < 10:
        return 'Mild'
    elif score < 15:
        return 'Moderate'
    elif score < 20:
        return 'Mod. Severe'
    else:
        return 'Severe'

wave1_categories = all_wave1.apply(categorize_severity).value_counts()
wave2_categories = all_wave2.apply(categorize_severity).value_counts()

categories_order = ['Minimal', 'Mild', 'Moderate', 'Mod. Severe', 'Severe']
wave1_counts = [wave1_categories.get(c, 0) for c in categories_order]
wave2_counts = [wave2_categories.get(c, 0) for c in categories_order]

fig3.add_trace(
    go.Bar(x=categories_order, y=wave1_counts, name='Wave 1',
           marker_color='rgba(231, 76, 60, 0.7)'),
    row=1, col=2
)
fig3.add_trace(
    go.Bar(x=categories_order, y=wave2_counts, name='Wave 2',
           marker_color='rgba(243, 156, 18, 0.7)'),
    row=1, col=2
)

# 3. Histogram: PHQ-9 change
fig3.add_trace(
    go.Histogram(x=changes, nbinsx=30, marker_color='steelblue',
                name='Change Distribution', showlegend=False),
    row=2, col=1
)

# 4. Time between waves (only for those with both)
time_diffs = []
for p in both_waves:
    if len(p['surveys']) == 2:
        diff = (p['surveys'][1]['timestamp'] - p['surveys'][0]['timestamp']).days
        time_diffs.append(diff)

fig3.add_trace(
    go.Histogram(x=time_diffs, nbinsx=30, marker_color='coral',
                name='Time Diff', showlegend=False),
    row=2, col=2
)

# 5. PHQ-9 vs GAD-7
phq9_all = labels['PHQ9_Score'].dropna()
gad7_all = labels['GAD7_Score'].dropna()
common_idx = phq9_all.index.intersection(gad7_all.index)

fig3.add_trace(
    go.Scatter(
        x=phq9_all[common_idx],
        y=gad7_all[common_idx],
        mode='markers',
        marker=dict(size=4, color='purple', opacity=0.5),
        name='PHQ-9 vs GAD-7',
        showlegend=False
    ),
    row=3, col=1
)

# 6. Temporal trend
daily_mean = labels.groupby(labels['survey_timestamp'].dt.date)['PHQ9_Score'].mean()

fig3.add_trace(
    go.Scatter(
        x=daily_mean.index,
        y=daily_mean.values,
        mode='lines+markers',
        marker=dict(size=4, color='teal'),
        line=dict(color='teal', width=2),
        name='Daily Mean',
        showlegend=False
    ),
    row=3, col=2
)

# Update axes labels
fig3.update_xaxes(title_text="Wave", row=1, col=1)
fig3.update_yaxes(title_text="PHQ-9 Score", row=1, col=1)
fig3.update_xaxes(title_text="Severity", row=1, col=2)
fig3.update_yaxes(title_text="Count", row=1, col=2)
fig3.update_xaxes(title_text="PHQ-9 Change", row=2, col=1)
fig3.update_yaxes(title_text="Count", row=2, col=1)
fig3.update_xaxes(title_text="Days Between Waves", row=2, col=2)
fig3.update_yaxes(title_text="Count", row=2, col=2)
fig3.update_xaxes(title_text="PHQ-9", row=3, col=1)
fig3.update_yaxes(title_text="GAD-7", row=3, col=1)
fig3.update_xaxes(title_text="Date", row=3, col=2)
fig3.update_yaxes(title_text="Mean PHQ-9", row=3, col=2)

fig3.update_layout(
    title_text='<b>PHQ-9 Comprehensive Analysis Dashboard</b>',
    height=1200,
    width=1400,
    showlegend=True
)

html_file3 = 'phq9_dashboard_comprehensive.html'
fig3.write_html(html_file3)
print(f"✓ Comprehensive dashboard saved: {html_file3}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("PHQ-9 ANALYSIS SUMMARY")
print("="*70)

print(f"\nTotal Surveys: {len(labels)}")
print(f"  Wave 1: {len(labels[labels['survey_wave'] == 1])}")
print(f"  Wave 2: {len(labels[labels['survey_wave'] == 2])}")

print(f"\nParticipants:")
print(f"  Total: {len(participants_data)}")
print(f"  With 1 wave: {sum(1 for p in participants_data if p['n_surveys'] == 1)}")
print(f"  With 2 waves: {sum(1 for p in participants_data if p['n_surveys'] == 2)}")

print(f"\nPHQ-9 Scores:")
print(f"  Wave 1: {all_wave1.mean():.2f} ± {all_wave1.std():.2f} (median: {all_wave1.median():.1f})")
print(f"  Wave 2: {all_wave2.mean():.2f} ± {all_wave2.std():.2f} (median: {all_wave2.median():.1f})")

if changes:
    print(f"\nPHQ-9 Changes (Wave 1→2, n={len(changes)}):")
    print(f"  Mean change: {np.mean(changes):.2f} ± {np.std(changes):.2f}")
    print(f"  Improved (Δ≤-2): {sum(1 for c in changes if c <= -2)} ({100*sum(1 for c in changes if c <= -2)/len(changes):.1f}%)")
    print(f"  Stable (-2<Δ<2): {sum(1 for c in changes if -2 < c < 2)} ({100*sum(1 for c in changes if -2 < c < 2)/len(changes):.1f}%)")
    print(f"  Worsened (Δ≥2): {sum(1 for c in changes if c >= 2)} ({100*sum(1 for c in changes if c >= 2)/len(changes):.1f}%)")

print(f"\nClinical Threshold (PHQ-9 ≥ 10):")
print(f"  Wave 1: {sum(all_wave1 >= 10)} / {len(all_wave1)} ({100*sum(all_wave1 >= 10)/len(all_wave1):.1f}%)")
print(f"  Wave 2: {sum(all_wave2 >= 10)} / {len(all_wave2)} ({100*sum(all_wave2 >= 10)/len(all_wave2):.1f}%)")

if time_diffs:
    print(f"\nTime Between Waves:")
    print(f"  Mean: {np.mean(time_diffs):.1f} days")
    print(f"  Median: {np.median(time_diffs):.1f} days")
    print(f"  Range: {min(time_diffs)} - {max(time_diffs)} days")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print("\nGenerated files:")
print(f"  1. {html_file1}")
print(f"     → Timeline of all PHQ-9 surveys with severity colors")
print(f"\n  2. {html_file2}")
print(f"     → Wave 1 vs Wave 2 change analysis")
print(f"\n  3. {html_file3}")
print(f"     → Comprehensive 6-panel dashboard")
print("\nAll files are interactive HTML - open in any web browser!")
print("="*70)
