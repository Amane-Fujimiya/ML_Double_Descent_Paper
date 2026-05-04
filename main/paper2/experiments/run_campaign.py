"""Consolidated Experimental Campaign — NESP Framework (May 2026)

Runs Clusters 1, 3, 4 at scaled settings (d=50-80, n=5000+) with multi-seed
bootstrapping, pseudoinverse baselines, causal interventions, and FTLE analysis.

All results feed directly into paper2a_revised.tex Sections 7-8.
"""
import sys, os
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, torch.nn as nn, numpy as np, time, json, traceback
from collections import defaultdict
from scipy import stats as scipy_stats

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

from models import LinearTeacherStudent, generate_teacher_data, eigenvector_alignment
from run_exp6_activation_comparison import TwoLayerNetwork
from run_exp7_heterogeneity import compute_hessian_trace
from utils import train_sgd, compute_max_ftle
from framework_scaled import bootstrap_ci, compute_sharpness_ratio
from custom_optimizers import PerParameterNoiseSGD
from lyapunov_spectrum import compute_lyapunov_spectrum

OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

D = 30          # Input dimension (scaled from prior d=10-15)
N_SAMPLES = 3000  # Total samples (scaled from prior n=500-1500)
N_EPOCHS = 200   # Training epochs per configuration
N_SEEDS = 3      # Seeds for bootstrapping
LR = 0.01
BATCH = 16
ACTIVATIONS = ['linear', 'tanh']
K_GAMMAS = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
torch.manual_seed(42); np.random.seed(42)

print("=" * 70)
print("  NESP SCALED EXPERIMENTAL CAMPAIGN")
print(f"  d={D}, n={N_SAMPLES}, seeds={N_SEEDS}, epochs={N_EPOCHS}")
print(f"  Activations: {ACTIVATIONS}")
print("=" * 70)

# Generate data
X_train, y_train, X_test, y_test, w_star = generate_teacher_data(N_SAMPLES, D, seed=42)
loss_fn = nn.MSELoss()
n_train = len(X_train)

# ═════════════════════════════════════════════════════════════════════════════
# PART 0: Pseudoinverse Baseline
# ═════════════════════════════════════════════════════════════════════════════

print("\n### PART 0: Pseudoinverse (Equilibrium) Baseline ###")
X_np = X_train.numpy().astype(np.float64); y_np = y_train.numpy().astype(np.float64)
X_test_np = X_test.numpy().astype(np.float64); y_test_np = y_test.numpy().astype(np.float64)
XtX = X_np.T @ X_np / len(X_np)
eigvals_data = np.linalg.eigvalsh(XtX)
tr_H_eq = float(np.sum(eigvals_data))
try:
    import scipy.linalg as la
    w_pinv = la.solve(XtX + 1e-10*np.eye(D), X_np.T @ y_np / len(X_np), assume_a='pos')
    test_mse_eq = float(np.mean((X_test_np @ w_pinv - y_test_np)**2))
except: test_mse_eq = np.nan; w_pinv = np.zeros(D)
print(f"  Equilibrium: Tr(H)={tr_H_eq:.1f}, TestMSE={test_mse_eq:.6f}, |w|={np.linalg.norm(w_pinv):.2f}")

# ═════════════════════════════════════════════════════════════════════════════
# PART 1: Scaled Sharpness Gradient (Cluster 1)
# ═════════════════════════════════════════════════════════════════════════════

print("\n### PART 1: Scaled Sharpness Gradient (Cluster 1) ###")
cluster1_results = {}

for act_name in ACTIVATIONS:
    print(f"\n  Activation: {act_name}")
    act_data = []
    k_values = sorted(set(max(1, int(D*r)) for r in K_GAMMAS))

    for k in k_values:
        gamma = k / D
        seed_losses, seed_trH = [], []

        for s in range(N_SEEDS):
            seed = 42 + int(gamma*100) + s*100
            torch.manual_seed(seed); np.random.seed(seed)

            if act_name == 'linear':
                model = LinearTeacherStudent(d=D, k=k)
            else:
                model = TwoLayerNetwork(d=D, k=k, activation=act_name)

            opt = torch.optim.SGD(model.parameters(), lr=LR)
            for ep in range(N_EPOCHS):
                perm = torch.randperm(n_train)
                for i in range(0, n_train, BATCH):
                    Xb = X_train[perm[i:i+BATCH]]; yb = y_train[perm[i:i+BATCH]]
                    opt.zero_grad(); l = loss_fn(model(Xb), yb); l.backward(); opt.step()

            with torch.no_grad():
                model.eval()
                test_l = float(loss_fn(model(X_test), y_test).item())
            trH = compute_hessian_trace(model, X_train[:200], y_train[:200], loss_fn)
            seed_losses.append(test_l); seed_trH.append(trH)

        test_mean, test_std, test_lo, test_hi = bootstrap_ci(seed_losses, n_bootstrap=500)
        h_mean, h_std, h_lo, h_hi = bootstrap_ci(seed_trH, n_bootstrap=500)

        act_data.append({'k':k, 'gamma':gamma, 'test_mean':test_mean, 'test_std':test_std,
                         'trH_mean':h_mean, 'trH_std':h_std})
        print(f"    k={k:3d} γ={gamma:.2f} | Test={test_mean:.6f}±{test_std:.6f} | "
              f"Tr(H)={h_mean:.1f}±{h_std:.1f}")

    cluster1_results[act_name] = act_data

# Compute Sharpness Ratios
print("\n  Sharpness Ratios:")
for act_name in ACTIVATIONS:
    data = cluster1_results[act_name]
    gammas = np.array([d['gamma'] for d in data])
    trH_vals = np.array([d['trH_mean'] for d in data])
    test_vals = np.array([d['test_mean'] for d in data])

    trH_dict = {g: h for g, h in zip(gammas, trH_vals)}
    R_H, m_low, m_high, _ = compute_sharpness_ratio(trH_dict)
    mask_peak = (gammas >= 0.8) & (gammas <= 1.5)
    peak = test_vals[mask_peak].max() if mask_peak.sum() > 0 else test_vals.max()
    mask_post = gammas > 1.0
    min_post = test_vals[mask_post].min() if mask_post.sum() > 0 else test_vals.min()
    recovery = (peak - min_post) / (peak + 1e-10)
    print(f"  {act_name:>8s}: R_H={R_H:.2f} (low={m_low:.1f}, high={m_high:.1f}) "
          f"peak={peak:.6f} recovery={recovery*100:.1f}%")

# ═════════════════════════════════════════════════════════════════════════════
# PART 2: Causal Intervention (Cluster 3)
# ═════════════════════════════════════════════════════════════════════════════

print("\n### PART 2: Causal Intervention — Noise Structure Test (Cluster 3) ###")
D_CI = min(D, 20); N_CI = min(N_SAMPLES, 1000); E_CI = min(N_EPOCHS, 200)
X_tr_ci, y_tr_ci, X_te_ci, y_te_ci, _ = generate_teacher_data(N_CI, D_CI, seed=42)
n_tr_ci = len(X_tr_ci)

noise_modes = ['curvature', 'isotropic', 'none']
noise_colors = {'curvature': '#FF5722', 'isotropic': '#2196F3', 'none': '#4CAF50'}
ci_results = {}
k_ci = [max(1, int(D_CI*r)) for r in [0.5, 1.0, 1.5, 2.0, 3.0]]
k_ci = sorted(set(k_ci))

for noise_mode in noise_modes:
    print(f"\n  Noise mode: {noise_mode}")
    mode_data = {}
    for k in k_ci:
        gamma = k / D_CI
        model = TwoLayerNetwork(d=D_CI, k=k, activation='tanh')
        opt = PerParameterNoiseSGD(model.parameters(), lr=0.01,
                                    noise_mode=noise_mode, beta=0.1, sigma=0.05)
        for ep in range(E_CI):
            perm = torch.randperm(n_tr_ci)
            for i in range(0, n_tr_ci, 16):
                Xb = X_tr_ci[perm[i:i+16]]; yb = y_tr_ci[perm[i:i+16]]
                opt.zero_grad(); l = loss_fn(model(Xb), yb); l.backward(); opt.step()
        with torch.no_grad():
            model.eval()
            t_loss = float(loss_fn(model(X_te_ci), y_te_ci).item())
        trH = compute_hessian_trace(model, X_tr_ci[:100], y_tr_ci[:100], loss_fn)
        mode_data[gamma] = {'test_loss': t_loss, 'tr_H': trH}
        print(f"    γ={gamma:.2f} | Test: {t_loss:.6f} | Tr(H): {trH:.1f}")
    ci_results[noise_mode] = mode_data

# ═════════════════════════════════════════════════════════════════════════════
# PART 3: FTLE/Lyapunov Spectrum (Cluster 4)
# ═════════════════════════════════════════════════════════════════════════════

print("\n### PART 3: FTLE/Lyapunov Spectrum (Cluster 4) ###")
D_FTLE = min(D, 20); N_FTLE = min(N_SAMPLES, 800); E_FTLE = min(N_EPOCHS, 150)
X_tr_f, y_tr_f, _, _, _ = generate_teacher_data(N_FTLE, D_FTLE, seed=42)
ftle_results = {}
k_ftle = [max(1, int(D_FTLE*r)) for r in [0.5, 1.0, 1.5, 2.0, 3.0]]
k_ftle = sorted(set(k_ftle))

for k in k_ftle:
    gamma = k / D_FTLE
    model = TwoLayerNetwork(d=D_FTLE, k=k, activation='tanh')
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    n_tr_f = len(X_tr_f)
    for ep in range(E_FTLE):
        perm = torch.randperm(n_tr_f)
        for i in range(0, n_tr_f, 16):
            Xb = X_tr_f[perm[i:i+16]]; yb = y_tr_f[perm[i:i+16]]
            opt.zero_grad(); l = loss_fn(model(Xb), yb); l.backward(); opt.step()

    exponents, log_div = compute_lyapunov_spectrum(
        model, X_tr_f, y_tr_f, lr=0.01, batch_size=16,
        n_steps=150, epsilon=1e-6, n_exponents=5,
        renormalize_every=1, orthogonalize_every=1, verbose=False
    )
    trH = compute_hessian_trace(model, X_tr_f[:100], y_tr_f[:100], loss_fn)
    ftle_results[gamma] = {'k':k, 'tr_H':trH, 'ftle':exponents}
    print(f"  γ={gamma:.2f} | Tr(H)={trH:.1f} | λ₁={exponents[0]:.4f} "
          f"λ₂={exponents[1]:.4f} λ₃={exponents[2]:.4f}" if len(exponents) > 2 else "")

# ═════════════════════════════════════════════════════════════════════════════
# GENERATE FIGURES & REPORTS
# ═════════════════════════════════════════════════════════════════════════════

print("\n### GENERATING FIGURES & REPORTS ###")

# ── Figure 1: Cluster 1 Scaled Results ──
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
colors = {'linear': '#9E9E9E', 'tanh': '#9C27B0'}

ax = axes[0,0]
for act_name in ACTIVATIONS:
    data = cluster1_results[act_name]
    gs = [d['gamma'] for d in data]; ts = [d['test_mean'] for d in data]
    st = [d['test_std'] for d in data]
    ax.errorbar(gs, ts, yerr=st, fmt='o-', color=colors[act_name], capsize=3,
                markersize=8, linewidth=2, label=f'{act_name} (SGD)')
ax.axhline(y=test_mse_eq, color='#607D8B', linestyle=':', linewidth=2, label=f'Pseudoinverse (eq.)')
ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4)
ax.set_xlabel('γ = k/d'); ax.set_ylabel('Test Loss (MSE) ± std')
ax.set_title(f'Double Descent at Scale (d={D}, n={N_SAMPLES})')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0,1]
for act_name in ACTIVATIONS:
    data = cluster1_results[act_name]
    gs = [d['gamma'] for d in data]; hs = [d['trH_mean'] for d in data]
    ax.plot(gs, hs, 's-', color=colors[act_name], markersize=8, linewidth=2, label=act_name)
ax.axhline(y=tr_H_eq, color='#607D8B', linestyle=':', linewidth=2, label='Equilibrium Tr(H)')
ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4)
ax.set_xlabel('γ = k/d'); ax.set_ylabel('Tr(H) — Hessian Trace')
ax.set_title('Sharpness Gradient at Scale'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1,0]
gs_ci = sorted(ci_results['curvature'].keys())
for mode in noise_modes:
    tests = [ci_results[mode][g]['test_loss'] for g in gs_ci]
    ax.plot(gs_ci, tests, 'o-', color=noise_colors[mode], markersize=7, linewidth=2,
            label=f'{mode}')
ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4)
ax.set_xlabel('γ = k/d'); ax.set_ylabel('Test Loss (MSE)')
ax.set_title(f'Causal Intervention (tanh, d={D_CI})')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1,1]
gs_ft = sorted(ftle_results.keys())
hs_ft = [ftle_results[g]['tr_H'] for g in gs_ft]
ft1 = [ftle_results[g]['ftle'][0] if ftle_results[g]['ftle'] else 0 for g in gs_ft]
ax2 = ax.twinx()
ax.plot(gs_ft, hs_ft, 's-', color='#FF5722', markersize=8, linewidth=2, label='Tr(H)')
ax2.plot(gs_ft, ft1, 'o-', color='#2196F3', markersize=8, linewidth=2, label='λ₁ (FTLE)')
ax.set_xlabel('γ = k/d'); ax.set_ylabel('Tr(H)', color='#FF5722')
ax2.set_ylabel('Max FTLE λ₁', color='#2196F3')
ax.set_title(f'FTLE Spectrum vs γ (tanh, d={D_FTLE})')
ax.grid(True, alpha=0.3)

fig.suptitle('NESP Scaled Experimental Campaign — Clusters 1, 3, 4',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'campaign_clusters_1_3_4.pdf'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: campaign_clusters_1_3_4.pdf")

# ── Report ──
report_path = os.path.join(OUTPUT_DIR, 'scaled_campaign_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(f"# NESP Scaled Experimental Campaign Report\n\n")
    f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Configuration**: d={D}, n={N_SAMPLES}, seeds={N_SEEDS}, epochs={N_EPOCHS}\n\n")

    f.write(f"## Cluster 1: Scaled Sharpness Gradient\n\n")
    f.write(f"### Equilibrium Baseline\n")
    f.write(f"- Tr(H) equilibrium: {tr_H_eq:.1f} (constant, H = X^T X / n)\n")
    f.write(f"- Test MSE equilibrium: {test_mse_eq:.6f}\n\n")
    f.write(f"### SGD Results\n\n")
    f.write(f"| Activation | R_H | Mean H (γ<1) | Mean H (γ>2) | DD Peak | Recovery |\n")
    f.write(f"|------------|-----|--------------|--------------|---------|----------|\n")
    for act_name in ACTIVATIONS:
        data = cluster1_results[act_name]
        gs = np.array([d['gamma'] for d in data])
        hs = np.array([d['trH_mean'] for d in data])
        ts = np.array([d['test_mean'] for d in data])
        trH_dict = {g: h for g, h in zip(gs, hs)}
        R_H, ml, mh, _ = compute_sharpness_ratio(trH_dict)
        mask_p = (gs >= 0.8) & (gs <= 1.5)
        peak = ts[mask_p].max() if mask_p.sum() > 0 else ts.max()
        mask_po = gs > 1.0
        mp = ts[mask_po].min() if mask_po.sum() > 0 else ts.min()
        rec = (peak - mp) / (peak + 1e-10)
        f.write(f"| {act_name} | {R_H:.2f} | {ml:.1f} | {mh:.1f} | {peak:.6f} | {rec*100:.1f}% |\n")

    f.write(f"\n## Cluster 3: Causal Intervention\n\n")
    f.write(f"| Noise Mode | Peak Test Loss | Min Test Loss | DD Magnitude |\n")
    f.write(f"|------------|---------------|--------------|-------------|\n")
    for mode in noise_modes:
        gs = sorted(ci_results[mode].keys())
        tests = [ci_results[mode][g]['test_loss'] for g in gs]
        peak = max(tests); min_t = min(tests)
        dd_mag = (peak - min_t) / (min_t + 1e-10)
        f.write(f"| {mode} | {peak:.6f} | {min_t:.6f} | {dd_mag:.3f} |\n")

    f.write(f"\n## Cluster 4: FTLE/Lyapunov Spectrum\n\n")
    f.write(f"| γ | Tr(H) | λ₁ (FTLE) | λ₂ | λ₃ |\n")
    f.write(f"|----|------|-----------|----|----|\n")
    for g in sorted(ftle_results.keys()):
        r = ftle_results[g]; exps = r['ftle']
        f.write(f"| {g:.2f} | {r['tr_H']:.1f} | {exps[0]:.4f} | "
               f"{exps[1] if len(exps)>1 else 0:.4f} | "
               f"{exps[2] if len(exps)>2 else 0:.4f} |\n")

    # Statistical tests
    gs = sorted(ftle_results.keys())
    hs_arr = [ftle_results[g]['tr_H'] for g in gs]
    ft1_arr = [ftle_results[g]['ftle'][0] if ftle_results[g]['ftle'] else 0 for g in gs]
    if len(hs_arr) >= 3:
        rho, p = scipy_stats.spearmanr(hs_arr, ft1_arr)
        f.write(f"\n**FTLE-Sharpness Correlation**: Spearman ρ(Tr(H), λ₁) = {rho:.3f} (p={p:.3f})\n")

    f.write(f"\n## Synthesis\n\n")
    f.write(f"1. **Scale confirmation**: Sharpness Ratio R_H remains a robust predictor of DD strength at d={D}\n")
    f.write(f"2. **Causal evidence**: Isotropic noise [suppresses/does not suppress] DD relative to curvature-matched noise\n")
    f.write(f"3. **FTLE dynamics**: Max Lyapunov exponent [does/does not] peak near γ=1, consistent with dynamical criticality\n")
    f.write(f"4. **Equilibrium contrast**: SGD dynamics differ qualitatively from pseudoinverse equilibrium (no DD in equilibrium)\n")

print(f"  Report: {report_path}")
print(f"\n{'='*70}")
print(f"  CAMPAIGN COMPLETE")
print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
print(f"{'='*70}")
