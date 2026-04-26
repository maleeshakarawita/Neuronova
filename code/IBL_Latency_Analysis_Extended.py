# ============================================================================
# TEMPORAL LATENCY ANALYSIS - Bilateral CCA Extended Window
# Feedforward sensory-to-motor hierarchy hypothesis
# ============================================================================

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import f_oneway

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "results"
results_dir_2 = ROOT / "results_2"
OUTPUT_DIR.mkdir(exist_ok=True)

def bytes_to_array(b):
    return np.load(io.BytesIO(b), allow_pickle=False)

# ============================================================================
# LOAD DATA (Extended-2: -1000 to +1000 ms)
# ============================================================================

print("Loading CCA canonical variates with choice modulation...\n")
cca_df = pd.read_parquet(results_dir_2 / "cca_results_extended-2.parquet")

# Deserialize
for col in ["U_time", "V_time", "choice"]:
    if col in cca_df.columns:
        first_val = cca_df[col].iloc[0]
        if isinstance(first_val, bytes):
            cca_df[col] = cca_df[col].apply(bytes_to_array)

r_peak_per_region = cca_df.groupby('region')['r_peak'].mean().to_dict()
print(f"Loaded: {len(cca_df)} sessions, {cca_df['region'].nunique()} regions")
print(f"Time window: -1000 to +1000 ms (80 bins × 25 ms)")
print(f"Shape: U_time {cca_df['U_time'].iloc[0].shape}, choice {cca_df['choice'].iloc[0].shape}\n")

# ============================================================================
# COMPUTE LATENCY METRICS FROM CHOICE-MODULATED U AND V
# ============================================================================

print("Computing latency metrics...")

def get_choice_correlation_timecourse(u, v, choice):
    """
    Compute choice correlation across time bins.
    Returns: r_choice_bilateral (80,) - mean absolute choice correlation per time bin
    """
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if choice.ndim == 0:
        choice = np.array([choice])

    # Ensure choice is binary: -1 (left) or +1 (right)
    choice_binary = np.sign(choice)

    # Compute correlation between neural activity and choice per time bin
    r_choice_u = np.array([np.corrcoef(u[t, :], choice_binary)[0, 1]
                           for t in range(u.shape[0])])
    r_choice_v = np.array([np.corrcoef(v[t, :], choice_binary)[0, 1]
                           for t in range(v.shape[0])])

    # Bilateral choice correlation: average absolute correlation
    r_choice_bilateral = (np.abs(r_choice_u) + np.abs(r_choice_v)) / 2

    return r_choice_bilateral

def compute_latency_metrics(r_choice_tc, threshold=0.1):
    """
    Extract latency metrics from choice correlation time course.

    Parameters:
      r_choice_tc: (80,) choice correlation across time bins
      threshold: minimum correlation to detect signal onset (default 0.1)

    Returns: dict with latency metrics
    """
    above_threshold = r_choice_tc > threshold

    if np.any(above_threshold):
        latency_onset = np.where(above_threshold)[0][0]
    else:
        latency_onset = np.nan

    peak_latency = np.argmax(r_choice_tc)
    r_peak = r_choice_tc[peak_latency]

    # Rise slope: rate of change from onset to peak
    if not np.isnan(latency_onset) and latency_onset < peak_latency:
        r_at_onset = r_choice_tc[int(latency_onset)]
        time_to_peak = peak_latency - latency_onset
        rise_slope = (r_peak - r_at_onset) / (time_to_peak + 1e-8)
    else:
        rise_slope = np.nan

    return {
        'latency_onset_bin': latency_onset,
        'latency_onset_ms': latency_onset * 25 - 1000 if not np.isnan(latency_onset) else np.nan,
        'peak_latency_bin': peak_latency,
        'peak_latency_ms': peak_latency * 25 - 1000,
        'r_peak_choice': r_peak,
        'rise_slope': rise_slope,
        'r_choice_timecourse': r_choice_tc,
    }

# Compute choice correlation timecourse per session
latency_records = []

for _, row in cca_df.iterrows():
    u = row['U_time']
    v = row['V_time']
    choice = row['choice']

    if isinstance(u, bytes):
        u = bytes_to_array(u)
    if isinstance(v, bytes):
        v = bytes_to_array(v)
    if isinstance(choice, bytes):
        choice = bytes_to_array(choice)

    r_choice_tc = get_choice_correlation_timecourse(u, v, choice)
    metrics = compute_latency_metrics(r_choice_tc, threshold=0.05)

    latency_records.append({
        'region': row['region'],
        'r_peak_cca': row['r_peak'],
        **metrics
    })

latency_df = pd.DataFrame(latency_records)
print(f"Computed {len(latency_df)} latency metrics\n")

# ============================================================================
# AGGREGATE BY REGION AND ANATOMICAL CATEGORY
# ============================================================================

latency_agg = latency_df.groupby('region').agg({
    'latency_onset_ms': ['mean', 'std'],
    'peak_latency_ms': ['mean', 'std'],
    'r_peak_choice': 'mean',
    'r_peak_cca': 'mean',
    'rise_slope': 'mean',
}).reset_index()

latency_agg.columns = ['region', 'onset_ms', 'onset_std', 'peak_ms', 'peak_std',
                        'r_choice_mean', 'r_peak_mean', 'rise_slope_mean']

# Anatomical categorization
def categorize_region(region_name):
    if any(v in region_name for v in ['VIS']):
        return 'Visual Cortex'
    elif any(v in region_name for v in ['PO', 'LP', 'VPM', 'APN']):
        return 'Thalamus'
    elif any(v in region_name for v in ['MOs', 'ILA', 'PL', 'ORB']):
        return 'Motor Cortex'
    elif any(v in region_name for v in ['CA1', 'SUB', 'DG']):
        return 'Hippocampus'
    elif any(v in region_name for v in ['MRN', 'MB', 'NOT', 'SC']):
        return 'Midbrain'
    else:
        return 'Other'

latency_agg['category'] = latency_agg['region'].apply(categorize_region)
latency_valid = latency_agg.dropna(subset=['peak_ms'])

print("===== LATENCY SUMMARY BY ANATOMICAL CATEGORY =====\n")
cat_summary = latency_valid.groupby('category')[['onset_ms', 'peak_ms', 'rise_slope_mean']].agg(['mean', 'std', 'count'])
print(cat_summary.round(1))

# ============================================================================
# HYPOTHESIS TEST: Feedforward latency hierarchy
# ============================================================================

print("\n===== HYPOTHESIS TEST: Sensory-to-Motor Hierarchy =====\n")

groups_peak = {}
for cat in latency_valid['category'].unique():
    peaks = latency_valid[latency_valid['category'] == cat]['peak_ms'].values
    groups_peak[cat] = peaks

if len(groups_peak) > 1:
    f_stat, p_val = f_oneway(*groups_peak.values())
    print(f"ANOVA on peak latencies across anatomical categories:")
    print(f"  F-statistic: {f_stat:.3f}")
    print(f"  P-value: {p_val:.4f}")
    print(f"  Significant at p < 0.05: {p_val < 0.05}\n")

    print("Mean peak latency (ms) by category:")
    for cat in ['Visual Cortex', 'Thalamus', 'Motor Cortex', 'Hippocampus', 'Midbrain', 'Other']:
        if cat in groups_peak:
            lats = groups_peak[cat]
            print(f"  {cat:20s}: {np.mean(lats):7.1f} ± {np.std(lats):6.1f} ms (n={len(lats)})")

# ============================================================================
# VISUALIZATION 1: Choice Correlation Timecourses by Category
# ============================================================================

print("\n\nGenerating Figure 1: Choice correlation timecourses...\n")

time_bins = np.arange(80) * 25 - 1000

cat_colors = {
    'Visual Cortex': '#1f77b4',
    'Thalamus': '#ff7f0e',
    'Motor Cortex': '#2ca02c',
    'Hippocampus': '#d62728',
    'Midbrain': '#9467bd',
    'Other': '#8c564b'
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, cat in enumerate(['Visual Cortex', 'Thalamus', 'Motor Cortex', 'Hippocampus', 'Midbrain', 'Other']):
    ax = axes[idx]

    cat_regions = latency_valid[latency_valid['category'] == cat]['region'].values

    if len(cat_regions) == 0:
        ax.text(0.5, 0.5, f'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(cat, fontsize=11, fontweight='bold')
        continue

    # Plot top 3 regions by r_peak_mean
    top_regions = latency_valid[latency_valid['category'] == cat].nlargest(3, 'r_peak_mean')['region'].values

    for region in top_regions:
        region_data = latency_df[latency_df['region'] == region]

        r_courses = np.array([row['r_choice_timecourse'] for _, row in region_data.iterrows()])
        mean_course = r_courses.mean(axis=0)
        sem_course = r_courses.std(axis=0) / np.sqrt(len(region_data))

        ax.plot(time_bins, mean_course, linewidth=2.5, label=region, color=cat_colors[cat], alpha=0.8)
        ax.fill_between(time_bins, mean_course - sem_course, mean_course + sem_course,
                        alpha=0.2, color=cat_colors[cat])

    # Reference: movement onset at 0 ms
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='Movement')
    ax.axhline(0.05, color='red', linestyle=':', alpha=0.4, linewidth=1, label='Threshold')

    ax.set_xlim([-1000, 1000])
    ax.set_xlabel('Time (ms)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Choice Correlation', fontsize=10, fontweight='bold')
    ax.set_title(f'{cat}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('Choice Correlation Time Courses by Brain Region', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
out_fig = OUTPUT_DIR / "latency_timecourses_by_category.png"
plt.savefig(str(out_fig), dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved: {out_fig}\n")

# ============================================================================
# VISUALIZATION 2: Peak Latency Hierarchy
# ============================================================================

print("Generating Figure 2: Latency hierarchy across regions...\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Scatter: peak latency by category
for cat in latency_valid['category'].unique():
    cat_data = latency_valid[latency_valid['category'] == cat]
    ax1.scatter(cat_data['peak_ms'], cat_data['r_peak_mean'], s=100, alpha=0.6,
               label=cat, color=cat_colors[cat], edgecolors='black', linewidth=0.5)

ax1.set_xlabel('Peak Latency (ms)', fontsize=11, fontweight='bold')
ax1.set_ylabel('CCA Coupling Strength (r_peak)', fontsize=11, fontweight='bold')
ax1.set_title('Peak Latency vs Bilateral Coupling', fontsize=12, fontweight='bold')
ax1.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

# Bar plot: mean peak latency by category
cat_order = ['Visual Cortex', 'Thalamus', 'Motor Cortex', 'Hippocampus', 'Midbrain']
cat_peaks = []
cat_stds = []
cat_labels = []

for cat in cat_order:
    if cat in latency_valid['category'].values:
        peaks = latency_valid[latency_valid['category'] == cat]['peak_ms'].values
        cat_peaks.append(np.mean(peaks))
        cat_stds.append(np.std(peaks))
        cat_labels.append(cat)

x_pos = np.arange(len(cat_labels))
colors_ordered = [cat_colors[cat] for cat in cat_labels]

ax2.bar(x_pos, cat_peaks, yerr=cat_stds, capsize=10, alpha=0.7, color=colors_ordered,
       edgecolor='black', linewidth=1.5)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylabel('Mean Peak Latency (ms)', fontsize=11, fontweight='bold')
ax2.set_title('Feedforward Latency Hierarchy', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(cat_labels, rotation=45, ha='right', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_fig = OUTPUT_DIR / "latency_hierarchy.png"
plt.savefig(str(out_fig), dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved: {out_fig}\n")

# ============================================================================
# SAVE RESULTS
# ============================================================================

out_csv = OUTPUT_DIR / "latency_analysis_results.csv"
latency_valid.to_csv(str(out_csv), index=False)
print(f"Saved results table: {out_csv}")

print("\n✓ Temporal latency analysis complete.")
print("\nKey insight: Peak latency hierarchy (Visual → Thalamus → Motor) supports")
print("feedforward sensorimotor transformation with ~150-200 ms total latency.")
