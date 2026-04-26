# Methodology: Bilateral Neural Coupling Analysis via Canonical Correlation Analysis

## Data Source
International Brain Laboratory (IBL) dataset. N=152 bilateral region-pairs across 46 homologous brain regions (left and right hemispheres) from 25 male and female mice performing a visual discrimination task. Recorded regions include motor cortex, visual cortex, thalamus, hippocampus, and midbrain. All procedures approved by institutional animal care and use committees.

---

## Step 1: Data Preprocessing and PCA Dimensionality Reduction

**Temporal binning:** Raw spike times were binned at 10 ms resolution during the decision-making epoch (−1000 to +1000 ms relative to stimulus onset). Spike counts were smoothed with a Gaussian kernel (σ = 1.5 bins ≈ 37 ms) to reduce Poisson noise.

**PCA computation:** For each session × hemisphere × region combination, Principal Component Analysis (PCA) was fitted to compress neural population activity to K = 3 principal components. The PCA was trained on all (trial × time-point) combinations, preserving ~80% variance. Output: Z_L, Z_R (3D tensors of shape n_trials × n_time × 3 components per hemisphere).

**Rationale:** PCA denoise the high-dimensional spike data and enables tractable CCA computation across thousands of trials and time bins.

---

## Step 2: Canonical Correlation Analysis with 3-Fold Cross-Validation

**Time binning for CCA:** PCA scores were re-sampled into 80 time bins spanning −1000 to +1000 ms relative to stimulus onset (25 ms bin width).

**CCA formulation:** For each time bin independently, we computed linear combinations U (left) and V (right) that maximize the Pearson correlation r between canonical variates:

```
U = w_L · Z_L  (left hemisphere canonical variate)
V = w_R · Z_R  (right hemisphere canonical variate)
r = Pearson(U, V)
```

**Cross-validation:** To prevent overfitting, we applied 3-fold KFold cross-validation on trials:
- Training: fit CCA on 2/3 of trials, record r_train
- Testing: evaluate r on held-out 1/3, record r_test (primary metric)
- Repeat 3 times; average to obtain r_time per time bin

**Output metrics:**
- **r_time:** Cross-validated coupling strength at each of 80 time bins
- **overfit_gap:** r_time_train − r_time (measures generalization)
- **r_peak:** Maximum r_test across all time bins
- **t_peak:** Time bin index where r_peak occurs
- **U_peak, V_peak:** Canonical variate amplitudes at t_peak

**Rationale:** Cross-validation on trials ensures reported r values reflect true effect sizes, not overfitting artifacts. Peak selection identifies the time window of strongest bilateral synchrony.

---

## Step 3: Permutation Null Distribution and Significance Testing

**Null generation:** For each region, 500 permutations were generated:
1. Shuffle right-hemisphere trial order (break U-V correspondence)
2. Recompute CCA at each time bin with shuffled data
3. Record r_peak from permuted data

**Significance threshold:** null_95 = 95th percentile of permuted r_peak distribution. Reported r_peak values are flagged as significant if r_peak > null_95.

**Rationale:** Permutation testing establishes whether observed bilateral coupling exceeds chance correlation levels.

---

## Step 4: Bilateral Asymmetry Metrics

Computed per region across all sessions to characterize left-right hemispheric differences:

**Canonical variate shape similarity:**
```
shape_similarity = Pearson(U_normalized, V_normalized)  ∈ [0, 1]
```
Correlation between normalized U and V trajectories. High values (>0.7) indicate parallel activity patterns; low values (<0.3) indicate independent trajectories.

**Temporal offset:**
```
temporal_offset_bins = |argmax(U) − argmax(V)|
```
Time lag in samples (bins) between peak amplitudes. Asymmetric timing suggests sequential rather than synchronized processing.

**Amplitude asymmetry:**
```
amplitude_asymmetry = |U_peak − V_peak| / (U_peak + V_peak)  ∈ [0, 1]
```
Normalized difference in canonical variate magnitudes. Values close to 0 indicate matched amplitudes; close to 1 indicate strong asymmetry.

**Variability metrics:**
```
U_variability = Var(U_time)
V_variability = Var(V_time)
```
Variance of canonical variates over time. Reflects activity stability across the trial window.

**Fréchet distance:** Computed between mean trial-averaged left and right choice trajectories in 3D PC space:
```
Fréchet = geodesic distance between trajectory curves
```
Captures overall shape dissimilarity independent of trial-by-trial variability.

---

## Step 5: Cross-Correlogram Analysis and Lag Detection

**Cross-correlogram computation:** For each region, trial-averaged canonical variates U_time and V_time were cross-correlated across lags (−500 to +500 ms):
```
r(lag) = Pearson(U_time, V_time_shifted(lag))
```

**Peak lag identification:**
```
peak_lag_ms = argmax(|r(lag)|)
peak_correlation = r(peak_lag_ms)
```

**Lag significance testing:** 200 permutations with shuffled trial order to generate null distribution of peak lags. Significance threshold: null_95_lag = 95th percentile of |null peak lags|. A lag is flagged as significant if |peak_lag| > null_95_lag.

**Interpretation:**
- Positive peak_lag: right hemisphere leads (V peaks before U)
- Negative peak_lag: left hemisphere leads (U peaks before V)
- peak_lag = 0: simultaneous peak correlation

---

## Step 6: Choice Encoding Analysis

**Choice variable normalization:** Animal choice (left=0, right=1) normalized to ±1 for correlation analysis.

**Per time-bin choice correlation:** For each time bin and region, computed Pearson correlation between canonical variates and normalized choice:
```
r_choice_U(t) = Pearson(U_time(t), choice_norm)
r_choice_V(t) = Pearson(V_time(t), choice_norm)
```

These metrics quantify how well left and right hemisphere canonical variates individually encode the animal's decision at each time point.

**Bilateral choice encoding metric:** To combine hemispheric choice signals without cancellation:
```
r_choice_bilateral(t) = (|r_choice_U(t)| + |r_choice_V(t)|) / 2
```

**Choice encoding summary metrics per region:**
- **r_choice_peak:** Maximum r_choice_bilateral across all 80 time bins
- **r_choice_mean:** Mean r_choice_bilateral across entire trial
- **peak_latency_ms:** Time bin where choice signal is strongest

**Phase-specific choice encoding:** For each movement phase, computed mean and peak r_choice_bilateral to identify when choice signals emerge relative to task phase (planning vs. execution vs. learning).

**Coupling-choice relationship:** Computed Spearman correlation between r_choice_peak and r_peak (bilateral coupling strength) across regions to test whether bilateral synchrony and choice encoding are independent or coupled properties.

---

## Step 7: Phase-Separated Analysis

**Movement phases:** Time window divided into three non-overlapping phases relative to movement onset:
- **Pre-movement:** −1000 to −500 ms (stimulus epoch, decision formation)
- **Movement:** −200 to +100 ms (motor execution)
- **Post-movement:** +500 to +1000 ms (outcome/feedback period)

**Phase-specific coupling metrics:**

**Asynchrony distance:**
```
async_distance = mean(Euclidean_distance((U, V)))
```
Average Euclidean distance between canonical variate pairs during the phase. Low values indicate tight bilateral coupling; high values indicate hemispheric divergence.

**Trajectory length:**
```
trajectory_length = sum(distances between consecutive (U, V) points)
```
Arc length in 2D canonical variate space during phase. Reflects the amount of neural state change.

**Phase correlation:**
```
phase_correlation = Pearson(U_phase, V_phase)
```
Coupling strength within each movement phase. Reveals whether bilateral synchrony is phase-dependent.

---

## Statistical Confidence and Reporting

**Region-level summary:** For each region, statistics were aggregated across sessions. Canonical correlations r were Fisher z-transformed before pooling:

```
z_i = 0.5 × log((1 + r_i) / (1 − r_i))
z_mean = mean(z_i)
SE_z = SD(z_i) / sqrt(n_sessions)
95% CI: lower = tanh(z_mean − 1.96×SE_z), upper = tanh(z_mean + 1.96×SE_z)
```

**Effective sample size:** Harmonic mean of (n_trials × n_neurons) across sessions, accounting for variability in neural population quality and sample size.

---

## Step 8: Coupling Strength Temporal Dynamics

**Temporal stability analysis:** For each region, computed metrics describing how coupling strength evolves across the trial:

```
coupling_range = max(r_time) − min(r_time)
```

Regions classified as:
- **Stable coupling:** range < 0.10 (5 regions)
- **Dynamic coupling:** range ≥ 0.10 (41 regions)

**Phase-specific coupling:** Bilateral coupling r averaged separately for three non-overlapping phases:
- **Early coupling:** −1000 to −500 ms (stimulus presentation)
- **Movement coupling:** −200 to +200 ms (motor execution window)
- **Late coupling:** +500 to +1000 ms (post-movement feedback)

**Coupling change metric:**
```
coupling_change = late_coupling − early_coupling
```

Regions show either increasing coupling (coupling_change > 0) or decreasing coupling (coupling_change < 0) across trial phases, revealing whether bilateral synchrony is task-phase dependent.

---

## Output Summary

**Primary outputs:**
- **cca_results_extended.parquet:** Per-region, per-time-bin coupling metrics (r_time, r_train, U_time, V_time, r_choice_U, r_choice_V)
- **bilateral_asymmetry_metrics.csv:** Region-level asymmetry statistics (shape_similarity, temporal_offset, amplitude_asymmetry, variability)
- **coupling_temporal_dynamics.csv:** Temporal coupling metrics (coupling_range, peak_coupling_time_ms, early/movement/late_coupling, coupling_change)
- **choice_encoding_analysis.csv:** Choice encoding metrics (r_choice_peak, r_choice_mean, peak_latency_ms, phase-specific encoding)
- **correlograms_summary.csv:** Cross-correlogram peaks (peak_lag_ms, peak_correlation, lag_significant)
- **phase_separation_analysis.csv:** Phase-specific async_distance, trajectory_length, phase_correlation

**Visualization outputs:**
- Time-series plots of r(t), coupling dynamics grouped by brain system
- Choice encoding landscapes (r_choice by region, relation to coupling)
- Choice encoding time courses (r_choice_U, r_choice_V, bilateral average per region)
- Scatter plots linking coupling strength to asymmetry and choice metrics
- Movement-phase-separated coupling and choice encoding plots
- Brain atlas maps colored by bilateral coupling strength and choice encoding
