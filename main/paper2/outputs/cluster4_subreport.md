# Cluster 4: Lyapunov Spectrum — Sub-Research Report

**Date**: 2026-05-04
**Loop**: 1 of 3
**Status**: FRICTION POINT — FTLE computation requires algorithmic correction

---

## 1. Objective & Hypothesis

**Primary Hypothesis (L1):** The maximal Finite-Time Lyapunov Exponent (FTLE) \(\lambda_1\) peaks near the interpolation threshold \(\gamma=1\), reflecting the dynamical criticality of SGD at the phase boundary. The FTLE spectrum width should correlate with the Sharpness Ratio \(R_H\).

**Physical Mechanism:** At \(\gamma \approx 1\), the Hessian is maximally ill-conditioned (\(\kappa(H) \to \infty\)), producing large anisotropic noise. This noise amplifies trajectory divergence in sharp directions, increasing the Lyapunov exponents. As \(\gamma\) moves away from 1 (in either direction), the dynamics become more regular → lower FTLE.

---

## 2. Experimental Design

| Parameter | Value |
|-----------|-------|
| Model | Two-layer tanh network |
| \(d\) | 20 |
| \(n\) | 800 |
| Pretraining epochs | 150 |
| FTLE steps | 150 |
| Perturbation \(\varepsilon\) | \(10^{-6}\) |
| \(n_{\text{exponents}}\) | 5 |
| \(\gamma\) values | {0.5, 1.0, 1.5, 2.0, 3.0} |

---

## 3. Results — FRICTION POINT

### 3.1 Observed Values

| \(\gamma\) | \(\mathrm{Tr}(H)\) | \(\lambda_1\) | \(\lambda_2\) | \(\lambda_3\) |
|-----------|-------------------|-------------|-------------|-------------|
| 0.50 | 115.2 | -55.2620 | -55.2620 | -55.2620 |
| 1.00 | 31.6 | -55.2620 | -55.2620 | -55.2620 |
| 1.50 | 38.3 | -55.2620 | -55.2620 | -55.2620 |
| 2.00 | 27.7 | -55.2620 | -55.2620 | -55.2620 |
| 3.00 | 21.0 | -55.2620 | -55.2620 | -55.2620 |

**All FTLE values are IDENTICAL (-55.2620)** across all \(\gamma\) values and all exponent indices. This is physically impossible — FTLE should vary with the dynamics — and indicates an algorithmic bug.

### 3.2 Root Cause Analysis

Investigation of `lyapunov_spectrum.py:compute_lyapunov_spectrum()` reveals the issue:

1. **Shadow trajectories share the reference gradient**: In the loop at line ~120, shadow models apply `ref_grads` (the reference gradient) instead of computing their own gradients. While this is correct for maintaining the same mini-batch noise realization, the shadow parameters are not being properly evolved.

2. **The perturbation decays**: The shadow displacement \(\|W_t^{\text{shadow}} - W_t^{\text{ref}}\|\) likely decays to zero because:
   - Both trajectories use the same gradient updates
   - The tanh network at convergence has small gradients
   - The perturbation \(\varepsilon = 10^{-6}\) is absorbed by gradient descent

3. **The log-divergence saturates**: \(\log(\|\text{displacement}\| / \varepsilon)\) hits a numerical floor when the displacement shrinks below machine precision.

### 3.3 Diagnostic Confirmation

The constant value \(-55.2620 \approx \log(10^{-6} / 10^{-30}) \approx \log(10^{24})\) suggests the code is computing \(\log(\varepsilon / \varepsilon_{\text{machine}})\), i.e., the displacement norm is zero to machine precision.

---

## 4. Proposed Fix (Loop 2)

### Algorithmic Correction

```python
# CORRECT FTLE algorithm:
# 1. Shadow = reference (copy parameters)
# 2. Perturb shadow: W_shadow = W_ref + ε * u  (u random unit vector)
# 3. FOR each step:
#    a. SAME mini-batch for BOTH trajectories
#    b. Reference: W_ref -= η * g_ref
#    c. Shadow:   W_shadow -= η * g_shadow  (NOT g_ref! Shadow evolves freely)
#    d. displacement = W_shadow - W_ref
#    e. Renormalize: W_shadow = W_ref + ε * displacement / ||displacement||
#    f. Log: λ_step = log(||displacement|| / ε)
```

The critical change: **line (c)** must use the shadow's own gradient, computed on the same mini-batch but using the shadow's current parameters. The shadow must evolve freely under the same data stream; only the renormalization keeps it near the reference trajectory.

### Additional Improvements
1. Use larger ε (e.g., \(10^{-4}\)) to avoid numerical underflow
2. Use double precision (float64) for the shadow trajectory
3. Verify the Gram-Schmidt orthogonalization preserves the perturbation norm
4. Add diagnostic: track displacement norm at each step to confirm divergence

---

## 5. FTLE Theory for SGD (Section 7.X Draft)

Despite the computational issue, the theoretical connection between FTLE and DD is well-founded:

### Definition (FTLE for SGD)
The Finite-Time Lyapunov Exponent for the SGD map \(\Phi_t: W_0 \mapsto W_t\) is:

\[
\lambda_i(\Phi_T, W_0) = \frac{1}{T} \log \sigma_i(D\Phi_T(W_0))
\]

where \(\sigma_i\) are the singular values of the Jacobian of the T-step SGD map.

### Conjecture (FTLE-DD Connection)
1. \(\lambda_1(\gamma) \text{ peaks at } \gamma \approx 1\) — maximal dynamical instability at the interpolation threshold
2. \(\lambda_1 - \lambda_k\) (spectrum width) \(\propto R_H\) — the sharpness ratio predicts the anisotropy of the dynamics
3. The Kaplan-Yorke dimension \(D_{KY}\) transitions from \(\sim d\) (under-parameterized) to \(\sim k-d\) (over-parameterized), reflecting the dimensionality of the effective dynamics

### Testable Prediction
If the FTLE algorithm is corrected, we expect:
- \(\lambda_1(\gamma=1) > \lambda_1(\gamma=0.5)\) and \(\lambda_1(\gamma=1) > \lambda_1(\gamma=3.0)\)
- \(\text{Spearman } \rho(\lambda_1, R_H) > 0.5\)
- The full spectrum shows a gap opening at γ > 1, corresponding to the emergence of the zero-loss manifold

---

## 6. Next Steps

1. **Loop 2 (CRITICAL)**: Fix the FTLE algorithm — shadow must compute its own gradients
2. **Loop 2b**: Test on linear model first (analytically tractable FTLE)
3. **Loop 3**: Full γ sweep with corrected algorithm and multi-seed bootstrapping
4. **Manuscript**: Draft Section 7.X (FTLE-DD Connection) with corrected results

**Decision**: PAUSE FTLE experiments until algorithm is corrected. Redirect computational resources to Clusters 2 and 5 in the interim. FTLE algorithm fix is the highest-priority code task for the next iteration.
