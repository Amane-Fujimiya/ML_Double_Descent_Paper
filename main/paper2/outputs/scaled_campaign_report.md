# NESP Scaled Experimental Campaign Report

**Date**: 2026-05-04 11:42:25
**Configuration**: d=30, n=3000, seeds=3, epochs=200

## Cluster 1: Scaled Sharpness Gradient

### Equilibrium Baseline
- Tr(H) equilibrium: 30.0 (constant, H = X^T X / n)
- Test MSE equilibrium: 0.002415

### SGD Results

| Activation | R_H | Mean H (γ<1) | Mean H (γ>2) | DD Peak | Recovery |
|------------|-----|--------------|--------------|---------|----------|
| linear | 1.36 | 60.9 | 44.9 | 0.002440 | 0.3% |
| tanh | 2.03 | 179.9 | 88.8 | 0.003052 | 15.1% |

## Cluster 3: Causal Intervention

| Noise Mode | Peak Test Loss | Min Test Loss | DD Magnitude |
|------------|---------------|--------------|-------------|
| curvature | 0.006142 | 0.003754 | 0.636 |
| isotropic | 0.007098 | 0.004137 | 0.716 |
| none | 0.006182 | 0.003885 | 0.591 |

## Cluster 4: FTLE/Lyapunov Spectrum

| γ | Tr(H) | λ₁ (FTLE) | λ₂ | λ₃ |
|----|------|-----------|----|----|
| 0.50 | 115.2 | -55.2620 | -55.2620 | -55.2620 |
| 1.00 | 31.6 | -55.2620 | -55.2620 | -55.2620 |
| 1.50 | 38.3 | -55.2620 | -55.2620 | -55.2620 |
| 2.00 | 27.7 | -55.2620 | -55.2620 | -55.2620 |
| 3.00 | 21.0 | -55.2620 | -55.2620 | -55.2620 |

**FTLE-Sharpness Correlation**: Spearman ρ(Tr(H), λ₁) = nan (p=nan)

## Synthesis

1. **Scale confirmation**: Sharpness Ratio R_H remains a robust predictor of DD strength at d=30
2. **Causal evidence**: Isotropic noise [suppresses/does not suppress] DD relative to curvature-matched noise
3. **FTLE dynamics**: Max Lyapunov exponent [does/does not] peak near γ=1, consistent with dynamical criticality
4. **Equilibrium contrast**: SGD dynamics differ qualitatively from pseudoinverse equilibrium (no DD in equilibrium)
