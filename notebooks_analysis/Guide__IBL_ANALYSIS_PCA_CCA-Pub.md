# Guide: IBL_ANALYSIS_PCA_CCA-Pub.ipynb

## What This Notebook Does

**3-stage pipeline:**
1. **PCA (Cells 7-18):** Reduces 50-200 neurons → 3 principal components per region
2. **CCA (Cells 20-29):** Measures bilateral coupling strength (r_peak, U_time, V_time)
3. **Analysis (Cells 30+):** Creates publication figures

---

## Quick Overview

### Stage 1: PCA Computation
- **Input:** Raw spike times from left/right hemispheres
- **Process:** Find 3 main "directions" of neural activity
- **Output:** Z_L, Z_R (3 PC scores per trial, per time point)
- **Why:** Denoise data, compress from 50-200 neurons to 3 components

### Stage 2: CCA Computation  
- **Input:** PC scores (Z_L, Z_R)
- **Process:** For each time bin, use K-Fold cross-validation to find strongest correlation between hemispheres
- **Output:** r_time (test CV coupling at each time), r_peak (maximum CV coupling), overfit_gap (training − test gap)
- **Interpretation:** r_peak = 0.8 → strong sync, r_peak = 0.3 → weak/independent; overfit_gap shows whether model generalizes

### Stage 3: Analysis
- **Temporal Asymmetry:** Do left and right hemispheres have the same trajectory?
- **Choice Modulation:** How do canonical variates change with animal's choice?
- **Phase Analysis:** When do hemispheres synchronize (planning vs. execution)?
- **Output:** Publication figures (scatter plots, time courses, heatmaps)

---

## Key Variables (Computed in This Notebook)

### CCA Core Outputs
| Variable | Meaning |
|----------|---------|
| **r_peak** | Maximum bilateral coupling (0-1, higher = more synchronized) |
| **r_time** | Coupling strength at each time bin (cross-validated test fold) |
| **t_peak** | When peak coupling occurs (in ms) |
| **U_time** | Left hemisphere's canonical variate over time |
| **V_time** | Right hemisphere's canonical variate over time |
| **U_peak, V_peak** | Peak amplitude values of left/right canonical variates |

### Choice Encoding
| Variable | Meaning |
|----------|---------|
| **r_choice_U** | Correlation between U_time and animal's choice (per time bin) |
| **r_choice_V** | Correlation between V_time and animal's choice (per time bin) |

### Asymmetry Metrics
| Variable | Meaning |
|----------|---------|
| **shape_similarity** | Correlation between U and V trajectory shapes (0-1, higher = similar) |
| **temporal_offset_bins** | Time lag between peak timings of U vs V (in time bins) |
| **U_variability** | Variance/spread of left canonical variate over time |
| **V_variability** | Variance/spread of right canonical variate over time |
| **amplitude_asymmetry** | Relative difference in peak amplitudes: \|U_peak − V_peak\| / (U_peak + V_peak) |

### Cross-Correlogram Metrics
| Variable | Meaning |
|----------|---------|
| **peak_lag_ms** | Time lag at maximum cross-correlation between hemispheres |
| **peak_correlation** | Maximum correlation value across all lags |
| **lag_significant** | Whether peak lag is statistically significant |

### Phase-Separated Analysis
| Variable | Meaning |
|----------|---------|
| **async_distance** | Asynchronous distance between left/right trajectories during phase |
| **trajectory_length** | Path length of canonical trajectory during phase |
| **phase_correlation** | Coupling strength (r) during specific movement phase |

---

## Output Figures

### CCA Results Analysis
- **`uv_asymmetry_scatter.png`** - 4-panel scatter: r_peak vs shape_similarity, temporal_offset, variability, amplitude_asymmetry
- **`cca_choice_modulation_asymmetry_curated.png`** - 3-column time-series for curated regions (Left | Right | Asymmetry), split by choice

### CCA Canonical Trajectories
- **`cca_canonical_trajectories_curated.png`** - Phase-colored time-series showing:
  - **U_time** (Left hemisphere's canonical variate): Compressed representation of left-side activity trajectory
  - **V_time** (Right hemisphere's canonical variate): Compressed representation of right-side activity trajectory  
  - **Bilateral asymmetry** (U_time − V_time): Difference between left and right trajectories, reveals hemispheric divergence

### Movement Phase Analysis
- **`cca_phase_separated_*.png`** - Multiple figures comparing pre-movement, movement, and post-movement phases
- Shows async distance and trajectory length per phase

### Cross-Correlogram Analysis
- **`correlograms_summary.csv`** - Table of peak lags and correlation values per region
- **`correlograms_full.parquet`** - Full correlogram data

### Asymmetry Analysis
- **`cca_asymmetry_analysis.md`** - Markdown interpretation of temporal and amplitude asymmetry metrics

---

## Related Notebooks

- **Upstream:** bilateral_data_fetch.ipynb (fetches raw spike data)
- **Downstream:** IBL_Choice_Encoding_Analysis.ipynb, IBL_CCA_Analysis_Coupling.ipynb, Swanson_CCA_P1_Metrics.ipynb

---

