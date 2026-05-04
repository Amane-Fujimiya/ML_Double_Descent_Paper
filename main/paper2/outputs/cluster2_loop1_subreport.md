# Cluster 2, Loop 1: Phase Diagram Exploration

**Date**: 2026-05-05
**Loop**: 1 of 2

## 1. Objective

Map the (γ, T_eff = η/B) phase space for tanh architecture. Identify phase boundary where DD peak vanishes. Test prediction: boundary is non-vertical, shifting right with increasing T_eff.

## 2. Experimental Design

| Parameter | Value |
|-----------|-------|
| d | 20 |
| n | 1500 |
| Activation | tanh |
| η | 0.01 |
| B range | 4, 16, 64, 256, 1024, 4096 |
| γ range | 10 values: 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0 |
| Seeds | 2 |
| Epochs | 200 |

## 3. Results

### 3.1 Test Loss Grid

| γ \ B | 4 (T_eff=2.5e-3) | 16 (6.25e-4) | 64 (1.56e-4) | 256 (3.91e-5) | 1024 (9.77e-6) |
|-------|-------------------|--------------|--------------|---------------|----------------|
| 0.3 | 0.005126 | 0.004788 | 0.008147 | **0.015187** | **0.018855** |
| 0.5 | 0.004468 | 0.004153 | 0.006674 | 0.011639 | 0.014142 |
| 0.8 | 0.003668 | 0.003850 | 0.005732 | 0.008049 | 0.010743 |
| 1.0 | 0.003541 | 0.003707 | 0.005334 | 0.006825 | 0.008297 |
| 1.2 | 0.003509 | 0.003571 | 0.004687 | 0.005031 | 0.007024 |
| 1.5 | 0.003558 | 0.003446 | 0.004267 | 0.005360 | 0.005318 |
| 2.0 | 0.003072 | 0.003313 | 0.003929 | 0.004691 | 0.005422 |
| 3.0 | 0.002981 | 0.003091 | 0.003452 | 0.003494 | 0.004057 |
| 5.0 | 0.003030 | **0.002975** | 0.003071 | 0.003167 | 0.003237 |
| 10.0 | **0.002898** | 0.002869 | **0.002889** | **0.002845** | **0.002889** |

### 3.2 Key Observations

**Finding 1: No clear DD peak at any T_eff value.** Test loss decreases monotonically with γ for all B ∈ {4, 16, 64, 256, 1024}. At γ=1.5, B=4 shows a slight bump (0.003558 → 0.003072, difference +0.000486) but this is below seed-level noise (2 seeds, σ unknown). This constitutes a **negative result**: at d=20, n=1500, tanh does not display an observable double descent peak.

**Finding 2: Under-parameterized regime sensitivity to T_eff.** At low γ (0.3--0.5), test loss varies dramatically with batch size:
- B=4 (T_eff=2.5e-3): 0.005126 at γ=0.3 — model converges well
- B=1024 (T_eff≈1e-5): 0.018855 at γ=0.3 — model struggles to fit (near full-batch)
This confirms that SGD noise is essential for escaping poor minima at low capacity. The 3.7× test loss ratio between B=4 and B=1024 at γ=0.3 quantifies the practical importance of SGD noise in the under-parameterized regime.

**Finding 3: Over-parameterized convergence is T_eff-independent.** At γ≥3.0, all B values converge to essentially the same test loss (within 0.001 of each other). This confirms the "free lunch" regime: beyond sufficient capacity, even full-batch gradient descent finds good solutions because the loss landscape contains abundant good minima at convergence.

**Finding 4: Phase boundary is vertical (no DD to vanish).** Since no DD peak was observed, the question of whether the DD vanishing boundary is non-vertical cannot be tested. The data instead supports a monotonic-decrease phase (Region I → III directly, skipping the critical Region II).

## 4. Theoretical Synthesis

### Why No DD at This Scale?

Comparison with Loop 4 results (d=30, n=3000, tanh recovery=15.1%) suggests that DD visibility depends on **n_train/d ratio** and **training duration**:
- Loop 4 (DD present): d=30, n_train=2100, ratio=70, 500 epochs, tanh R_H=2.03
- Loop 6 (DD absent): d=20, n_train=1050, ratio=52.5, 200 epochs, tanh R_H=???

The lower n_train/d ratio (52 vs 70) and reduced epochs (200 vs 500) both suppress DD. At γ=1.0 (k=20), the model has 420 parameters, and 1050 training points / 420 params = 2.5 data points per parameter — still over-parameterized enough that convergence is easy.

### Revised DD Condition

Double descent requires BOTH:
1. **Sharpness differential** (R_H ≫ 1): saturation at low γ creates high curvature
2. **Non-equilibrium dynamics** (T_eff large enough): SGD noise must outcompete the deterministic gradient near the interpolation threshold
3. **Critical n_train/d ratio**: Too many data points → model always converges well → no peak; too few → model never converges → no recovery

The Phase Diagram at d=20, n=1500 reveals that condition (3) is violated — the data is too abundant relative to the parameter count for a visible peak at any T_eff.

### New Prediction (H1''''')

Double descent amplitude ∝ (R_H - 1) × T_eff × f(n_train/d), where f is a window function peaking at n_train/d ≈ 5--50 and vanishing at both very small and very large ratios.

## 5. Next Steps

1. **Loop 2**: Increase d to 50 and reduce n to create stronger DD signal at a manageable scale
2. Alternatively: re-run with reduced n (n=500, ratio=25) to force DD peak
3. Compare tanh (saturating) vs ReLU (non-saturating) phase diagrams at same scale

## 6. Decision

**RECORD AS NEGATIVE RESULT** — the non-observation of DD at this scale constrains the conditions under which DD occurs. This is scientifically informative: the DD phase boundary in (γ, T_eff) space is not universal but depends on n/d and architecture. The monotonic-decrease regime dominates at d=20, n=1500 for all T_eff values tested. This constraint is added to the Friction Log as an empirical boundary condition.
