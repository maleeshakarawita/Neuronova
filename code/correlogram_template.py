# ============================================================================
# CROSS-CORRELOGRAM ANALYSIS: Temporal Delays in Bilateral Coupling
# ============================================================================
# Input: U_time, V_time from CCA pipeline (left and right canonical variates)
# Output: Peak lag per region, statistical significance, visualizations

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

# ============================================================================
# SETUP
# ============================================================================

ROOT       = Path("__file__").resolve().parent.parent
DATA_DIR   = ROOT / "data"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Deserialize arrays from bytes
def bytes_to_array(b):
    return np.load(io.BytesIO(b), allow_pickle=False)

# Load CCA results with U_time, V_time
results_dir_2 = ROOT / "results_2"
cca_df = pd.read_parquet(results_dir_2 / "cca_results_extended-2.parquet")

print(f"Available columns: {list(cca_df.columns)}")

# Should have U_time/V_time in extended parquet
if "U_time" in cca_df.columns and "V_time" in cca_df.columns:
    print("\n✓ U_time and V_time found. Using full time courses for lag analysis.")
    array_cols = ["U_time", "V_time", "choice"]
else:
    print("\n⚠️  ERROR: U_time/V_time NOT found. Check file path.")
    array_cols = ["U_time", "V_time", "choice"]  # Will fail gracefully if missing

for col in array_cols:
    if col in cca_df.columns:
        first_val = cca_df[col].iloc[0]
        if isinstance(first_val, (bytes, bytearray)):
            cca_df[col] = cca_df[col].apply(bytes_to_array)

print(f"\nLoaded CCA results: {cca_df.shape[0]} sessions")
if "U_time" in cca_df.columns:
    print(f"U_time shape (example): {cca_df['U_time'].iloc[0].shape}")
else:
    print(f"U_peak shape (example): {cca_df['U_peak'].iloc[0].shape}")

# ============================================================================
# FUNCTION: Compute Cross-Correlogram
# ============================================================================

def compute_correlogram(U_time_all, V_time_all, max_lag_ms=500, bin_size_ms=25):
    """
    Compute cross-correlation between U and V at different lags.

    Parameters:
    -----------
    U_time_all : 1D array, left hemisphere canonical variate (all trials concatenated)
    V_time_all : 1D array, right hemisphere canonical variate (all trials concatenated)
    max_lag_ms : int, maximum lag in milliseconds
    bin_size_ms : int, time bin size in ms (25 ms = 1 bin)

    Returns:
    --------
    lags_ms : array of lag values in ms
    correlogram : array of correlation values at each lag
    peak_lag_ms : int, lag with maximum correlation
    peak_correlation : float, maximum correlation value
    """

    max_lag_bins = max_lag_ms // bin_size_ms  # Convert ms to bins
    lags_ms = np.arange(-max_lag_ms, max_lag_ms + bin_size_ms, bin_size_ms)
    correlogram = np.zeros(len(lags_ms))

    # Standardize both signals
    U_std = (U_time_all - U_time_all.mean()) / (U_time_all.std() + 1e-8)
    V_std = (V_time_all - V_time_all.mean()) / (V_time_all.std() + 1e-8)

    # Compute correlation at each lag
    for i, lag_bins in enumerate(range(-max_lag_bins, max_lag_bins + 1)):
        if lag_bins > 0:
            # Positive lag: V leads U (V earlier)
            u_slice = U_std[lag_bins:]
            v_slice = V_std[:-lag_bins]
        elif lag_bins < 0:
            # Negative lag: U leads V (U earlier)
            u_slice = U_std[:lag_bins]
            v_slice = V_std[-lag_bins:]
        else:
            # Zero lag: simultaneous
            u_slice = U_std
            v_slice = V_std

        # Compute correlation only if enough data
        if len(u_slice) > 2 and len(v_slice) > 2:
            r = np.corrcoef(u_slice, v_slice)[0, 1]
            correlogram[i] = r if not np.isnan(r) else 0
        else:
            correlogram[i] = 0

    # Peak lag
    peak_idx = np.argmax(np.abs(correlogram))
    peak_lag_ms = lags_ms[peak_idx]
    peak_correlation = correlogram[peak_idx]

    return lags_ms, correlogram, peak_lag_ms, peak_correlation


# ============================================================================
# FUNCTION: Permutation Null for Lag Significance
# ============================================================================

def permutation_null_lag(U_time_all, V_time_all, max_lag_ms=500, n_perms=200):
    """
    Generate null distribution of peak lags under shuffled correspondence.

    Parameters:
    -----------
    U_time_all, V_time_all : arrays of canonical variates
    max_lag_ms : maximum lag to test
    n_perms : number of permutations

    Returns:
    --------
    null_peak_lags : array of peak lags from permuted data (length n_perms)
    """

    null_peak_lags = []
    rng = np.random.default_rng(seed=42)

    for _ in range(n_perms):
        # Shuffle V (break U-V correspondence)
        V_perm = rng.permutation(V_time_all)
        _, _, peak_lag, _ = compute_correlogram(U_time_all, V_perm, max_lag_ms)
        null_peak_lags.append(peak_lag)

    return np.array(null_peak_lags)


# ============================================================================
# ANALYSIS: Compute Correlograms Per Region
# ============================================================================

print("\n" + "="*80)
print("COMPUTING CROSS-CORRELOGRAMS")
print("="*80)

correlogram_records = []

for region in cca_df['region'].unique():
    region_data = cca_df[cca_df['region'] == region]

    # Collect U_time and V_time across sessions
    # Shape expected: (n_time_bins, n_trials) per session
    U_time_list = []
    V_time_list = []

    for _, row in region_data.iterrows():
        u = row['U_time']
        v = row['V_time']

        # Ensure 2D: (time_bins, trials)
        if u.ndim == 1:
            u = u.reshape(-1, 1)
        if v.ndim == 1:
            v = v.reshape(-1, 1)

        U_time_list.append(u)
        V_time_list.append(v)

    # Concatenate across sessions: (n_time_bins, n_total_trials)
    U_time_all = np.concatenate(U_time_list, axis=1)
    V_time_all = np.concatenate(V_time_list, axis=1)

    # Count actual trials BEFORE averaging
    n_trials_region = U_time_all.shape[1]

    # Average across trials to get trial-averaged time course
    # Shape: (n_time_bins,)
    U_time_all = U_time_all.mean(axis=1)
    V_time_all = V_time_all.mean(axis=1)

    # Compute correlogram
    lags_ms, correlogram, peak_lag_ms, peak_corr = compute_correlogram(
        U_time_all, V_time_all, max_lag_ms=500, bin_size_ms=25
    )

    # Compute permutation null
    print(f"  {region:15s}: Computing null (n=200)...", end=" ")
    null_peak_lags = permutation_null_lag(U_time_all, V_time_all, max_lag_ms=500, n_perms=200)
    null_95_ci = np.percentile(np.abs(null_peak_lags), 95)

    # Is peak lag significant?
    peak_lag_significant = np.abs(peak_lag_ms) > null_95_ci

    print(f"peak_lag={peak_lag_ms:+4.0f} ms, sig={peak_lag_significant}")

    correlogram_records.append({
        'region': region,
        'peak_lag_ms': peak_lag_ms,
        'peak_lag_abs_ms': np.abs(peak_lag_ms),
        'peak_correlation': peak_corr,
        'lag_significant': peak_lag_significant,
        'null_95_ci': null_95_ci,
        'correlogram': correlogram,
        'lags_ms': lags_ms,
        'n_trials': n_trials_region,
    })

correlogram_df = pd.DataFrame(correlogram_records)

print("\nCorrelogram analysis complete.\n")

# ============================================================================
# INTERPRETATION
# ============================================================================

print("="*80)
print("INTERPRETATION OF LAGS")
print("="*80)

print("""
Peak lag > 0: Right hemisphere (V) LEADS left hemisphere (U)
             → V causally influences U
             → Right hemisphere initiates bilateral synchrony

Peak lag < 0: Left hemisphere (U) LEADS right hemisphere (V)
             → U causally influences V
             → Left hemisphere initiates bilateral synchrony

Peak lag ≈ 0: Simultaneous U-V activity
             → Bilateral coupling is synchronous, no causal lead
""")

print("\n" + "="*80)
print("SUMMARY TABLE: PEAK LAGS BY REGION")
print("="*80 + "\n")

summary_lag = correlogram_df[[
    'region', 'peak_lag_ms', 'peak_correlation', 'lag_significant',
    'null_95_ci', 'n_trials'
]].sort_values('peak_lag_ms')

print(summary_lag.round(2).to_string())

# ============================================================================
# REGIONAL PATTERNS
# ============================================================================

print("\n" + "="*80)
print("REGIONAL PATTERNS")
print("="*80)

left_lead = correlogram_df[correlogram_df['peak_lag_ms'] < -50]  # U leads by >50ms
right_lead = correlogram_df[correlogram_df['peak_lag_ms'] > 50]   # V leads by >50ms
simultaneous = correlogram_df[np.abs(correlogram_df['peak_lag_ms']) <= 50]

print(f"\nLEFT HEMISPHERE DOMINANT (U leads V by >50 ms): {len(left_lead)} regions")
if len(left_lead) > 0:
    print(left_lead[['region', 'peak_lag_ms', 'peak_correlation']].sort_values('peak_lag_ms').round(2).to_string())

print(f"\n\nRIGHT HEMISPHERE DOMINANT (V leads U by >50 ms): {len(right_lead)} regions")
if len(right_lead) > 0:
    print(right_lead[['region', 'peak_lag_ms', 'peak_correlation']].sort_values('peak_lag_ms').round(2).to_string())

print(f"\n\nSIMULTANEOUS (|lag| ≤ 50 ms): {len(simultaneous)} regions")
if len(simultaneous) > 0:
    print(simultaneous[['region', 'peak_lag_ms', 'peak_correlation']].sort_values('peak_lag_ms').round(2).to_string())

# ============================================================================
# VISUALIZATION 1: Grid of Correlograms
# ============================================================================

fig, axes = plt.subplots(7, 7, figsize=(20, 20))
axes = axes.flatten()

for idx, (_, row) in enumerate(correlogram_df.sort_values('region').iterrows()):
    ax = axes[idx]
    region = row['region']
    lags = row['lags_ms']
    correlogram = row['correlogram']
    peak_lag = row['peak_lag_ms']

    ax.plot(lags, correlogram, linewidth=2, color='steelblue', label='Observed')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(peak_lag, color='red', linestyle='-', linewidth=2, alpha=0.7, label=f'Peak: {peak_lag:+.0f}ms')
    ax.axhline(0, color='black', linewidth=0.5)

    # Mark significance
    if row['lag_significant']:
        ax.text(0.5, 0.95, '★ SIG', transform=ax.transAxes,
               fontsize=10, fontweight='bold', color='red', va='top', ha='center')

    ax.set_xlabel('Lag (ms)', fontsize=9)
    ax.set_ylabel('r', fontsize=9)
    ax.set_title(f'{region}\n({row["n_trials"]} trials)', fontsize=10, fontweight='bold')
    ax.set_xlim([-500, 500])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

# Remove extra subplots
for idx in range(len(correlogram_df), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Cross-Correlograms: U_time vs V_time Lag Analysis',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
out_fig = OUTPUT_DIR / "correlograms_grid.png"
plt.savefig(str(out_fig), dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print(f"\nSaved: {out_fig}")

# ============================================================================
# VISUALIZATION 2: Peak Lag Summary
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

summary_sorted = correlogram_df.sort_values('peak_lag_ms')
y_pos = np.arange(len(summary_sorted))
colors = ['red' if sig else 'lightgray' for sig in summary_sorted['lag_significant']]

bars = ax.barh(y_pos, summary_sorted['peak_lag_ms'].values, color=colors, alpha=0.7, edgecolor='black')

ax.set_yticks(y_pos)
ax.set_yticklabels(summary_sorted['region'].values, fontsize=10)
ax.set_xlabel('Peak Lag (ms)', fontsize=12, fontweight='bold')
ax.set_title('Bilateral Coupling Temporal Hierarchy\nRed = significant lag; Gray = not significant',
            fontsize=13, fontweight='bold')
ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
out_fig = OUTPUT_DIR / "peak_lags_summary.png"
plt.savefig(str(out_fig), dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved: {out_fig}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

out_csv = OUTPUT_DIR / "correlograms_summary.csv"
correlogram_df[['region', 'peak_lag_ms', 'peak_correlation', 'lag_significant', 'null_95_ci', 'n_trials']].to_csv(str(out_csv), index=False)
print(f"Saved: {out_csv}")

out_parquet = OUTPUT_DIR / "correlograms_full.parquet"
correlogram_df.to_parquet(str(out_parquet))
print(f"Saved: {out_parquet}")

print("\n✓ Cross-correlogram analysis complete.")
