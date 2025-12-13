"""
Visualize data coverage timeline for MindWatch participants.
Shows sensor data availability and PHQ-9 survey timepoints.
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from train_tft import load_label_frames, load_hourly_sensor_frames, SENSOR_FEATURES

# Korean font setup
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_all_data():
    """Load labels and sensor data."""
    print("Loading label data...")
    labels = load_label_frames()

    print("Loading sensor data...")
    sensor_frames = load_hourly_sensor_frames()

    return labels, sensor_frames

def analyze_participant_coverage(labels, sensor_frames):
    """Analyze data coverage for each participant."""

    # Get all participant IDs from labels
    participant_ids = labels['ID'].unique()

    coverage_data = []

    for pid in participant_ids:
        participant_labels = labels[labels['ID'] == pid].sort_values('survey_wave')

        # Survey information
        surveys = []
        for _, row in participant_labels.iterrows():
            surveys.append({
                'wave': row['survey_wave'],
                'timestamp': row['survey_timestamp'],
                'phq9': row['PHQ9_Score']
            })

        # Sensor data coverage (using heart_rate as representative)
        sensor_coverage = {}
        for sensor_name, sensor_df in sensor_frames.items():
            if pid in sensor_df.columns:
                valid_data = sensor_df[pid].dropna()
                if len(valid_data) > 0:
                    sensor_coverage[sensor_name] = {
                        'start': valid_data.index.min(),
                        'end': valid_data.index.max(),
                        'count': len(valid_data)
                    }

        # Overall sensor coverage
        if sensor_coverage:
            all_starts = [info['start'] for info in sensor_coverage.values()]
            all_ends = [info['end'] for info in sensor_coverage.values()]
            sensor_start = min(all_starts)
            sensor_end = max(all_ends)
        else:
            sensor_start = None
            sensor_end = None

        coverage_data.append({
            'participant_id': pid,
            'surveys': surveys,
            'sensor_start': sensor_start,
            'sensor_end': sensor_end,
            'sensor_coverage': sensor_coverage
        })

    return coverage_data

def plot_participant_timeline(labels, sensor_frames, max_participants=50, output_file='participant_timeline.png'):
    """Create a Gantt-chart style visualization of participant data coverage."""

    print(f"Analyzing data coverage for {len(labels['ID'].unique())} participants...")
    coverage_data = analyze_participant_coverage(labels, sensor_frames)

    # Sort by earliest survey time
    coverage_data.sort(key=lambda x: x['surveys'][0]['timestamp'] if x['surveys'] else pd.Timestamp.max)

    # Select participants to display
    if len(coverage_data) > max_participants:
        print(f"Selecting {max_participants} representative participants...")
        # Select evenly spaced participants
        indices = np.linspace(0, len(coverage_data) - 1, max_participants, dtype=int)
        coverage_data = [coverage_data[i] for i in indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(10, len(coverage_data) * 0.3)))

    # Color scheme
    sensor_color = '#3498db'  # Blue for sensor data
    wave1_color = '#e74c3c'   # Red for Wave 1
    wave2_color = '#f39c12'   # Orange for Wave 2

    # Plot each participant
    for idx, pdata in enumerate(coverage_data):
        y_pos = idx

        # Plot sensor data coverage
        if pdata['sensor_start'] and pdata['sensor_end']:
            ax.barh(y_pos,
                   (pdata['sensor_end'] - pdata['sensor_start']).total_seconds() / 3600,
                   left=pdata['sensor_start'],
                   height=0.6,
                   color=sensor_color,
                   alpha=0.3,
                   edgecolor=sensor_color,
                   linewidth=1)

        # Plot survey markers
        for survey in pdata['surveys']:
            wave = survey['wave']
            ts = survey['timestamp']
            phq9 = survey['phq9']

            if wave == 1:
                color = wave1_color
                marker = 'o'
                label = f"W1: {phq9:.0f}"
            else:
                color = wave2_color
                marker = 's'
                label = f"W2: {phq9:.0f}"

            ax.scatter(ts, y_pos,
                      s=100,
                      c=color,
                      marker=marker,
                      edgecolors='black',
                      linewidths=1,
                      zorder=10)

            # Add PHQ-9 score label
            ax.text(ts, y_pos + 0.3, label,
                   fontsize=7,
                   ha='center',
                   va='bottom')

    # Formatting
    ax.set_yticks(range(len(coverage_data)))
    ax.set_yticklabels([f"P{i+1}" for i in range(len(coverage_data))], fontsize=8)
    ax.set_ylabel('Participants', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_title('MindWatch Data Collection Timeline\nSensor Coverage and PHQ-9 Survey Timepoints',
                fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=sensor_color, alpha=0.3, edgecolor=sensor_color,
                      label='Sensor Data Coverage'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=wave1_color,
               markersize=10, markeredgecolor='black', label='Wave 1 Survey'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=wave2_color,
               markersize=10, markeredgecolor='black', label='Wave 2 Survey'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Timeline plot saved: {output_file}")

    return fig, ax

def plot_detailed_sensor_coverage(labels, sensor_frames, n_participants=10, output_file='sensor_coverage_detail.png'):
    """Create detailed sensor-by-sensor coverage for selected participants."""

    coverage_data = analyze_participant_coverage(labels, sensor_frames)

    # Select participants with both waves
    both_waves = [p for p in coverage_data if len(p['surveys']) == 2]
    if len(both_waves) > n_participants:
        both_waves = both_waves[:n_participants]
    elif len(both_waves) == 0:
        print("No participants with both waves found, using any participants...")
        both_waves = coverage_data[:n_participants]

    # Create figure with subplots
    n_sensors = len(SENSOR_FEATURES)
    fig, axes = plt.subplots(n_participants, 1, figsize=(16, n_participants * 1.5), sharex=True)

    if n_participants == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, n_sensors))

    for idx, pdata in enumerate(both_waves):
        ax = axes[idx]

        # Plot each sensor
        sensor_y_positions = {}
        for sensor_idx, sensor_name in enumerate(SENSOR_FEATURES):
            sensor_y_positions[sensor_name] = sensor_idx

            if sensor_name in pdata['sensor_coverage']:
                info = pdata['sensor_coverage'][sensor_name]
                duration_hours = (info['end'] - info['start']).total_seconds() / 3600

                ax.barh(sensor_idx,
                       duration_hours,
                       left=info['start'],
                       height=0.7,
                       color=colors[sensor_idx],
                       alpha=0.6,
                       edgecolor='black',
                       linewidth=0.5)

        # Plot survey markers
        for survey in pdata['surveys']:
            wave = survey['wave']
            ts = survey['timestamp']
            phq9 = survey['phq9']

            if wave == 1:
                ax.axvline(ts, color='red', linestyle='--', linewidth=2, alpha=0.7, zorder=5)
                ax.text(ts, n_sensors - 0.5, f'W1: PHQ-9={phq9:.0f}',
                       rotation=90, va='bottom', ha='right', fontsize=9, color='red', fontweight='bold')
            else:
                ax.axvline(ts, color='orange', linestyle='--', linewidth=2, alpha=0.7, zorder=5)
                ax.text(ts, n_sensors - 0.5, f'W2: PHQ-9={phq9:.0f}',
                       rotation=90, va='bottom', ha='left', fontsize=9, color='orange', fontweight='bold')

        ax.set_yticks(range(n_sensors))
        ax.set_yticklabels([s.replace('_', ' ').title() for s in SENSOR_FEATURES], fontsize=7)
        ax.set_ylabel(f'P{idx+1}', fontsize=10, fontweight='bold', rotation=0, labelpad=20)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_ylim(-0.5, n_sensors - 0.5)

    axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
    fig.suptitle('Sensor-by-Sensor Data Coverage for Selected Participants\nRed: Wave 1 Survey | Orange: Wave 2 Survey',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed coverage plot saved: {output_file}")

    return fig, axes

def print_summary_statistics(labels, sensor_frames):
    """Print summary statistics about data coverage."""

    coverage_data = analyze_participant_coverage(labels, sensor_frames)

    print("\n" + "="*70)
    print("DATA COVERAGE SUMMARY")
    print("="*70)

    # Survey statistics
    wave_counts = labels.groupby('ID')['survey_wave'].count()
    print(f"\nSurvey Coverage:")
    print(f"  Total participants: {len(labels['ID'].unique())}")
    print(f"  Participants with 1 wave: {(wave_counts == 1).sum()}")
    print(f"  Participants with 2 waves: {(wave_counts == 2).sum()}")

    # Sensor statistics
    participants_with_sensor = sum(1 for p in coverage_data if p['sensor_start'] is not None)
    print(f"\nSensor Coverage:")
    print(f"  Participants with sensor data: {participants_with_sensor}")
    print(f"  Participants without sensor data: {len(coverage_data) - participants_with_sensor}")

    # Sensor duration
    durations = []
    for p in coverage_data:
        if p['sensor_start'] and p['sensor_end']:
            duration_days = (p['sensor_end'] - p['sensor_start']).days
            durations.append(duration_days)

    if durations:
        print(f"\nSensor Data Duration (days):")
        print(f"  Mean: {np.mean(durations):.1f}")
        print(f"  Median: {np.median(durations):.1f}")
        print(f"  Min: {np.min(durations):.1f}")
        print(f"  Max: {np.max(durations):.1f}")

    # Wave timing
    print(f"\nWave Timing:")
    wave1_data = labels[labels['survey_wave'] == 1]
    wave2_data = labels[labels['survey_wave'] == 2]

    if len(wave1_data) > 0:
        print(f"  Wave 1: {wave1_data['survey_timestamp'].min()} to {wave1_data['survey_timestamp'].max()}")
        print(f"          PHQ-9 mean: {wave1_data['PHQ9_Score'].mean():.2f} ± {wave1_data['PHQ9_Score'].std():.2f}")

    if len(wave2_data) > 0:
        print(f"  Wave 2: {wave2_data['survey_timestamp'].min()} to {wave2_data['survey_timestamp'].max()}")
        print(f"          PHQ-9 mean: {wave2_data['PHQ9_Score'].mean():.2f} ± {wave2_data['PHQ9_Score'].std():.2f}")

    # Same-day waves
    same_day_count = 0
    for pid in labels['ID'].unique():
        person_data = labels[labels['ID'] == pid]
        if len(person_data) == 2:
            dates = person_data['survey_timestamp'].dt.date.unique()
            if len(dates) == 1:
                same_day_count += 1

    print(f"\n  Participants with same-day Wave 1 and 2: {same_day_count}")

    print("="*70 + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize MindWatch data coverage')
    parser.add_argument('--n-timeline', type=int, default=50,
                       help='Number of participants in timeline plot (default: 50)')
    parser.add_argument('--n-detail', type=int, default=10,
                       help='Number of participants in detailed plot (default: 10)')
    parser.add_argument('--all', action='store_true',
                       help='Show all participants (use with caution, may be slow)')
    args = parser.parse_args()

    n_timeline = None if args.all else args.n_timeline
    n_detail = args.n_detail

    print("MindWatch Data Coverage Visualization")
    print("=" * 70)

    # Load data
    labels, sensor_frames = load_all_data()

    # Print statistics
    print_summary_statistics(labels, sensor_frames)

    # Create visualizations
    print("\nGenerating visualizations...\n")

    # Timeline plot
    if n_timeline is None:
        print(f"Plotting ALL {len(labels['ID'].unique())} participants (this may take a while)...")
        plot_participant_timeline(labels, sensor_frames,
                                 max_participants=len(labels['ID'].unique()),
                                 output_file='participant_timeline_all.png')
    else:
        plot_participant_timeline(labels, sensor_frames, max_participants=n_timeline)

    # Detailed sensor coverage
    plot_detailed_sensor_coverage(labels, sensor_frames, n_participants=n_detail)

    print("\n✓ All visualizations complete!")
    if n_timeline is None:
        print("\nGenerated files:")
        print("  - participant_timeline_all.png (overview of ALL participants)")
        print(f"  - sensor_coverage_detail.png (detailed view of {n_detail} participants)")
    else:
        print("\nGenerated files:")
        print(f"  - participant_timeline.png (overview of {n_timeline} participants)")
        print(f"  - sensor_coverage_detail.png (detailed view of {n_detail} participants)")
