# ============================================================================
# MOVEMENT PHASE ANALYSIS: CCA Trajectories Separated by Trial Period
#
# PHASES:
# - Pre-movement: -1000 to -200 ms (stimulus, decision planning)
# - Movement: -200 to +200 ms (movement execution)
# - Post-movement: +200 to +1000 ms (feedback, monitoring)
#
# QUESTION: When do hemispheres coordinate vs. specialize?
# - Decision phase: need bilateral coordination
# - Execution phase: might show asymmetry (left vs right paw)
# - Feedback phase: integration of outcome
# ============================================================================

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results"
results_dir_2 = ROOT / "results_2"

def bytes_to_array(b):
    return np.load(io.BytesIO(b), allow_pickle=False)

# ============================================================================
# LOAD CCA DATA
# ============================================================================

print("Loading CCA data...")

cca_df = pd.read_parquet(results_dir_2 / "cca_results_extended-2.parquet")

# Deserialize
for col in ["U_time", "V_time"]:
    if col in cca_df.columns:
        first_val = cca_df[col].iloc[0]
        if isinstance(first_val, bytes):
            cca_df[col] = cca_df[col].apply(bytes_to_array)

r_peak_per_region = cca_df.groupby('region')['r_peak'].mean().to_dict()

print(f"Loaded: {len(cca_df)} sessions, {cca_df['region'].nunique()} regions")

# ============================================================================
# DEFINE TIME PHASES
# ============================================================================

time_bins = np.arange(80) * 25 - 1000  # -1000 to +1000 ms

# Phase indices (in time bins, not ms)
# time_bins[20] = -500, time_bins[32] = -200, time_bins[40] = 0, time_bins[48] = +200, time_bins[60] = +500
pre_move_idx = (time_bins >= -1000) & (time_bins < -200)    # bins 0-33
move_idx = (time_bins >= -200) & (time_bins <= +200)        # bins 33-48
post_move_idx = (time_bins > +200) & (time_bins <= +1000)   # bins 48-79

phases = {
    'Pre-Movement\n(-1000 to -200ms)': pre_move_idx,
    'Movement\n(-200 to +200ms)': move_idx,
    'Post-Movement\n(+200 to +1000ms)': post_move_idx,
}

print(f"\nPhase definitions:")
for phase_name, phase_mask in phases.items():
    n_bins = phase_mask.sum()
    time_range = f"{time_bins[phase_mask].min():.0f} to {time_bins[phase_mask].max():.0f}ms"
    print(f"  {phase_name.replace(chr(10), ' ')}: {n_bins} bins ({time_range})")

# ============================================================================
# HELPER: Extract U_time and V_time for a region and phase
# ============================================================================

def get_uv_by_phase(region_name):
    """
    Returns: dict with keys 'Pre-Movement', 'Movement', 'Post-Movement'
    Each contains: (U_avg, V_avg) for that phase
    """
    region_data = cca_df[cca_df['region'] == region_name]

    U_time_list = []
    V_time_list = []

    for _, row in region_data.iterrows():
        u = row['U_time']
        v = row['V_time']

        if isinstance(u, bytes):
            u = bytes_to_array(u)
        if isinstance(v, bytes):
            v = bytes_to_array(v)

        if u.ndim == 1:
            u = u.reshape(-1, 1)
        if v.ndim == 1:
            v = v.reshape(-1, 1)

        U_time_list.append(u.mean(axis=1))
        V_time_list.append(v.mean(axis=1))

    U_avg = np.mean(U_time_list, axis=0)
    V_avg = np.mean(V_time_list, axis=0)

    # Extract phases
    phase_data = {}
    for phase_name, phase_mask in phases.items():
        U_phase = U_avg[phase_mask]
        V_phase = V_avg[phase_mask]
        phase_data[phase_name] = (U_phase, V_phase)

    return phase_data

# ============================================================================
# COMPUTE ASYNC METRIC BY PHASE
# ============================================================================

print("\nComputing asymmetry by phase...")

phase_results = []

for region in sorted(cca_df['region'].unique()):
    phase_data = get_uv_by_phase(region)
    r_peak = r_peak_per_region.get(region, 0.5)

    for phase_name, (U_phase, V_phase) in phase_data.items():
        if len(U_phase) == 0 or len(V_phase) == 0:
            continue

        # Compute asynchrony (distance from diagonal)
        async_dist = np.mean(np.abs(U_phase - V_phase))

        # Compute trajectory length in this phase
        U_diff = np.diff(U_phase)
        V_diff = np.diff(V_phase)
        trajectory_length = np.sum(np.sqrt(U_diff**2 + V_diff**2))

        # Compute correlation in this phase
        if np.std(U_phase) > 0 and np.std(V_phase) > 0:
            phase_corr = np.corrcoef(U_phase, V_phase)[0, 1]
        else:
            phase_corr = np.nan

        phase_results.append({
            'region': region,
            'phase': phase_name.split('\n')[0],  # Clean name
            'async_distance': async_dist,
            'trajectory_length': trajectory_length,
            'phase_correlation': phase_corr,
            'r_peak': r_peak,
        })

phase_df = pd.DataFrame(phase_results)

print(f"\nComputed {len(phase_df)} phase metrics\n")

# ============================================================================
# SUMMARY TABLE: Async by Phase
# ============================================================================

print("="*80)
print("ASYMMETRY BY MOVEMENT PHASE")
print("="*80)

pivot_async = phase_df.pivot_table(
    values='async_distance',
    index='region',
    columns='phase'
)

print("\nAsync Distance (lower = more synchronized):")
print(pivot_async.round(4).to_string())

print("\n\nMean async by phase (across all regions):")
async_by_phase = phase_df.groupby('phase')['async_distance'].agg(['mean', 'std', 'count'])
print(async_by_phase.round(4).to_string())

# ============================================================================
# VISUALIZATION 1: Async Changes Across Phases
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

phase_order = ['Pre-Movement', 'Movement', 'Post-Movement']

for ax_idx, region in enumerate(sorted(phase_df['region'].unique())[:18]):  # Show 18 regions
    region_data = phase_df[phase_df['region'] == region]
    r_peak = region_data['r_peak'].iloc[0]

    for phase in phase_order:
        phase_row = region_data[region_data['phase'] == phase]
        if len(phase_row) > 0:
            async_val = phase_row['async_distance'].values[0]

            # Map phase to x-position and color
            if phase == 'Pre-Movement':
                x_pos, color = 0, 'blue'
            elif phase == 'Movement':
                x_pos, color = 1, 'orange'
            else:
                x_pos, color = 2, 'red'

            # Size by r_peak (coupling strength)
            size = (r_peak - min(phase_df['r_peak'])) / (max(phase_df['r_peak']) - min(phase_df['r_peak'])) * 200 + 50

axes[0].scatter([], [], s=50, c='blue', label='Pre-Movement', alpha=0.6)
axes[0].scatter([], [], s=100, c='orange', label='Movement', alpha=0.6)
axes[0].scatter([], [], s=150, c='red', label='Post-Movement', alpha=0.6)
axes[0].legend(fontsize=11, loc='upper left')

# Actually plot the data
for region in sorted(phase_df['region'].unique()):
    region_data = phase_df[phase_df['region'] == region]
    r_peak = region_data['r_peak'].iloc[0]
    size = (r_peak - 0.25) / 0.7 * 150 + 30

    for phase in phase_order:
        phase_row = region_data[region_data['phase'] == phase]
        if len(phase_row) > 0:
            async_val = phase_row['async_distance'].values[0]
            if phase == 'Pre-Movement':
                x_pos, color = 0, 'blue'
            elif phase == 'Movement':
                x_pos, color = 1, 'orange'
            else:
                x_pos, color = 2, 'red'

            axes[0].scatter(x_pos, async_val, s=size, c=color, alpha=0.6, edgecolors='black', linewidth=0.5)

axes[0].set_xticks([0, 1, 2])
axes[0].set_xticklabels(phase_order, fontsize=11, fontweight='bold')
axes[0].set_ylabel('Asymmetry Distance (Async)', fontsize=11, fontweight='bold')
axes[0].set_title('How Hemispheric Sync Changes Across Trial Phases\n(Size = coupling strength r_peak)', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# ============================================================================
# VISUALIZATION 2: Trajectory Length by Phase
# ============================================================================

pivot_length = phase_df.pivot_table(
    values='trajectory_length',
    index='region',
    columns='phase'
)

x = np.arange(len(phase_order))
width = 0.25

for region_idx, region in enumerate(sorted(phase_df['region'].unique())[:10]):
    region_data = phase_df[phase_df['region'] == region]
    lengths = [region_data[region_data['phase'] == p]['trajectory_length'].values[0]
              if len(region_data[region_data['phase'] == p]) > 0 else 0
              for p in phase_order]

    axes[1].plot(x, lengths, marker='o', label=region, linewidth=2, markersize=6, alpha=0.7)

axes[1].set_xticks(x)
axes[1].set_xticklabels(phase_order, fontsize=11, fontweight='bold')
axes[1].set_ylabel('Trajectory Length in Phase', fontsize=11, fontweight='bold')
axes[1].set_title('Neural Dynamics Speed by Phase\n(Top 10 regions)', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9, loc='best')
axes[1].grid(True, alpha=0.3)

# ============================================================================
# VISUALIZATION 3: Bar plot - Mean Async by Phase
# ============================================================================

async_means = phase_df.groupby('phase')['async_distance'].mean()
async_std = phase_df.groupby('phase')['async_distance'].std()

colors_bar = ['blue', 'orange', 'red']
axes[2].bar(range(len(phase_order)), [async_means[p] for p in phase_order],
           yerr=[async_std[p] for p in phase_order],
           color=colors_bar, alpha=0.7, capsize=10, edgecolor='black', linewidth=2)

axes[2].set_xticks(range(len(phase_order)))
axes[2].set_xticklabels(phase_order, fontsize=11, fontweight='bold')
axes[2].set_ylabel('Mean Asymmetry ± Std', fontsize=11, fontweight='bold')
axes[2].set_title('Population-Level Hemispheric Synchrony\nby Phase', fontsize=11, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_fig = OUTPUT_DIR / "phase_analysis_async_by_movement.png"
plt.savefig(str(out_fig), dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print(f"\n\nSaved: {out_fig}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

out_csv = OUTPUT_DIR / "phase_analysis_results.csv"
phase_df.to_csv(str(out_csv), index=False)
print(f"Saved: {out_csv}")

print("\n✓ Movement phase analysis complete.")
print("\nKey Question Answered:")
print("  Do hemispheres stay synchronized throughout the trial?")
print("  Or do they diverge during movement execution?")
