"""Experiment 7: Landscape Heterogeneity vs Double Descent.

Tests the revised hypothesis H1': Curvature-noise coupling exists in all
models, but DOUBLE DESCENT requires landscape heterogeneity — the
coexistence of sharp and flat minima with distinct curvature.

Protocol:
1. For each activation (linear, ReLU, tanh), sweep γ through interpolation
2. At each γ, train N=5 models from different random seeds
3. Measure:
   - Test error (double descent curve)
   - Tr(H) at convergence for each seed
   - Landscape heterogeneity: CV(H) = std(Tr(H)) / mean(Tr(H)) across seeds
   - Coupling strength: alignment ratio
4. Key analysis: correlate double descent peak height with heterogeneity

Prediction (H1'):
  - Linear models: HIGH coupling, LOW heterogeneity → WEAK double descent
  - ReLU networks: MODERATE coupling, HIGH heterogeneity → STRONG double descent
  - Tanh networks: MODERATE coupling, MODERATE heterogeneity → MODERATE double descent
"""
import sys
import os
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import generate_teacher_data, eigenvector_alignment
from run_exp6_activation_comparison import TwoLayerNetwork
from utils import train_sgd


def compute_hessian_trace(model, X, y, loss_fn):
    """Estimate Tr(H) using Hutchinson's estimator."""
    params = list(model.parameters())

    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grads])

    # Single Hutchinson sample
    n_params = grad_flat.shape[0]
    v = torch.randn(n_params)
    hv = torch.autograd.grad(grad_flat, params, grad_outputs=v, retain_graph=True)
    hv_flat = torch.cat([h.flatten() for h in hv])

    return torch.dot(v, hv_flat).item()


def run_experiment_7(
    d: int = 12,
    n_samples: int = 800,
    k_values: list = None,
    activations: list = None,
    lr: float = 0.01,
    batch_size: int = 16,
    n_epochs: int = 600,
    n_seeds: int = 5,
    output_dir: str = './outputs',
):
    """Landscape heterogeneity experiment."""
    if k_values is None:
        k_values = [int(d * r) for r in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]]
        k_values = sorted(set(max(1, k) for k in k_values))

    if activations is None:
        activations = ['linear', 'relu', 'tanh']

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  EXPERIMENT 7: Landscape Heterogeneity vs Double Descent")
    print(f"  d={d}, n={n_samples}, {n_seeds} seeds per point")
    print(f"  Activations: {activations}")
    print("=" * 60)

    loss_fn = nn.MSELoss()

    # Results structure: {activation: {gamma: [{seed_data}, ...]}}
    all_results = {}

    for act_name in activations:
        print(f"\n{'─'*50}")
        print(f"  Activation: {act_name}")
        print(f"{'─'*50}")

        act_results = {}  # gamma -> list of seed results

        for k in k_values:
            gamma = k / d
            n_params = k * d + k
            seed_results = []

            for seed_idx in range(n_seeds):
                seed = 42 + seed_idx * 100
                torch.manual_seed(seed)
                np.random.seed(seed)

                X_train, y_train, X_test, y_test, w_star = generate_teacher_data(
                    n_samples, d, seed=seed
                )

                model = TwoLayerNetwork(d=d, k=k, activation=act_name)

                def dummy_spectra(m, X, y):
                    return {}

                log = train_sgd(
                    model, X_train, y_train, X_test, y_test,
                    lr=lr, batch_size=batch_size,
                    n_epochs=n_epochs,
                    record_every=200,
                    record_spectra_every=100000,
                    compute_spectra_fn=dummy_spectra,
                    verbose=False,
                )

                # Compute Hessian trace
                tr_H = compute_hessian_trace(model, X_train[:100], y_train[:100], loss_fn)

                # Compute alignment (only for first seed to save time)
                if seed_idx == 0:
                    k_eig = min(10, n_params)
                    hess_eigvals, hess_eigvecs = model.compute_hessian_topk(
                        X_train[:200], y_train[:200], loss_fn, k_eig=k_eig
                    )
                    sigma_eigvals, sigma_eigvecs = model.compute_noise_cov_topk(
                        X_train[:500], y_train[:500], loss_fn, k_eig=k_eig
                    )
                    alignment = eigenvector_alignment(hess_eigvecs, sigma_eigvecs)
                    alignment_ratio = alignment / (1.0 / n_params)
                else:
                    alignment_ratio = None

                seed_results.append({
                    'seed': seed,
                    'test_loss': log['test_loss'][-1],
                    'min_test_loss': min(log['test_loss']),
                    'tr_H': tr_H,
                    'alignment_ratio': alignment_ratio,
                })

                if (seed_idx + 1) % 3 == 0 or seed_idx == 0:
                    print(f"    k={k:3d} (γ={gamma:.2f}) | seed {seed_idx+1}/{n_seeds} "
                          f"| test={log['test_loss'][-1]:.6f} | Tr(H)={tr_H:.2f}")

            act_results[gamma] = seed_results

        all_results[act_name] = act_results

    # ── Compute heterogeneity metrics ──
    heterogeneity = {}  # {activation: {gamma: cv_H}}
    double_descent = {}  # {activation: {gamma: mean_test_loss}}

    for act_name in activations:
        het_dict = {}
        dd_dict = {}
        for gamma, seed_results in all_results[act_name].items():
            tr_H_vals = [s['tr_H'] for s in seed_results]
            test_vals = [s['min_test_loss'] for s in seed_results]

            mean_tr_H = np.mean(tr_H_vals)
            std_tr_H = np.std(tr_H_vals)
            cv_H = std_tr_H / (mean_tr_H + 1e-10)  # coefficient of variation

            het_dict[gamma] = cv_H
            dd_dict[gamma] = np.mean(test_vals)

        heterogeneity[act_name] = het_dict
        double_descent[act_name] = dd_dict

    # ── Generate Figures ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    colors = {'linear': '#9E9E9E', 'relu': '#4CAF50', 'tanh': '#9C27B0'}

    # Row 1: Double descent and heterogeneity by activation
    for idx, act_name in enumerate(activations):
        ax = axes[0, idx]
        gammas = sorted(double_descent[act_name].keys())
        test_losses = [double_descent[act_name][g] for g in gammas]

        # Twin axis for heterogeneity
        ax2 = ax.twinx()

        ax.plot(gammas, test_losses, 'o-', color=colors[act_name],
                markersize=8, linewidth=2, label='Test error')
        het_vals = [heterogeneity[act_name][g] for g in gammas]
        ax2.plot(gammas, het_vals, 's--', color='#FF5722',
                 markersize=6, linewidth=1.5, alpha=0.7, label='CV(H)')

        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4)
        ax.set_xlabel('γ = k/d')
        ax.set_ylabel('Test Loss (MSE)', color=colors[act_name])
        ax2.set_ylabel('CV(H) = std/mean', color='#FF5722')
        ax.set_title(f'{act_name}')
        ax.grid(True, alpha=0.3)

    # Row 2: Cross-activation comparison
    # Subplot 4: Double descent peak vs heterogeneity
    ax = axes[1, 0]
    for act_name in activations:
        gammas = sorted(heterogeneity[act_name].keys())
        dd_vals = [double_descent[act_name][g] for g in gammas]
        het_vals = [heterogeneity[act_name][g] for g in gammas]

        ax.scatter(het_vals, dd_vals, color=colors[act_name], s=80,
                   alpha=0.7, edgecolors='black', linewidth=0.5, label=act_name)

        # Fit line
        if len(het_vals) >= 4:
            coeffs = np.polyfit(het_vals, dd_vals, 1)
            x_fit = np.linspace(min(het_vals), max(het_vals), 20)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, '--', color=colors[act_name], linewidth=1, alpha=0.5)

    ax.set_xlabel('Landscape Heterogeneity CV(H)')
    ax.set_ylabel('Mean Test Loss (MSE)')
    ax.set_title('Test Error vs Landscape Heterogeneity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 5: Coupling strength vs heterogeneity
    ax = axes[1, 1]
    for act_name in activations:
        gammas = sorted(heterogeneity[act_name].keys())
        het_vals = [heterogeneity[act_name][g] for g in gammas]
        # Get alignment ratio (only computed for first seed)
        align_vals = []
        for g in gammas:
            ar = all_results[act_name][g][0].get('alignment_ratio')
            align_vals.append(ar if ar is not None else 0)

        ax.scatter(het_vals, align_vals, color=colors[act_name], s=80,
                   alpha=0.7, edgecolors='black', linewidth=0.5, label=act_name)

    ax.set_xlabel('Landscape Heterogeneity CV(H)')
    ax.set_ylabel('Alignment Ratio')
    ax.set_title('Coupling Strength vs Heterogeneity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 6: Summary bar chart
    ax = axes[1, 2]
    x_pos = np.arange(len(activations))
    width = 0.25

    mean_het = [np.mean(list(heterogeneity[a].values())) for a in activations]
    mean_align = [np.mean([all_results[a][g][0].get('alignment_ratio', 0)
                           for g in all_results[a].keys()
                           if all_results[a][g][0].get('alignment_ratio') is not None])
                  for a in activations]

    # Normalize for display
    norm_het = [h / max(mean_het) if max(mean_het) > 0 else 0 for h in mean_het]
    norm_align = [a / max(mean_align) if max(mean_align) > 0 else 0 for a in mean_align]

    ax.bar(x_pos - width/2, norm_het, width, color='#FF5722', alpha=0.7,
           edgecolor='black', linewidth=0.5, label='Norm. Heterogeneity')
    ax.bar(x_pos + width/2, norm_align, width, color='#2196F3', alpha=0.7,
           edgecolor='black', linewidth=0.5, label='Norm. Coupling')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(activations)
    ax.set_ylabel('Normalized Value')
    ax.set_title('Heterogeneity vs Coupling by Activation')
    ax.legend()

    fig.suptitle('Experiment 7: Landscape Heterogeneity Enables Double Descent',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exp7_heterogeneity.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: exp7_heterogeneity.pdf")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 7 SUMMARY")
    print(f"{'='*60}")

    for act_name in activations:
        gammas = sorted(heterogeneity[act_name].keys())
        mean_het_val = np.mean([heterogeneity[act_name][g] for g in gammas])
        max_het_val = max([heterogeneity[act_name][g] for g in gammas])
        mean_dd = np.mean([double_descent[act_name][g] for g in gammas])

        align_vals = []
        for g in gammas:
            ar = all_results[act_name][g][0].get('alignment_ratio')
            if ar is not None:
                align_vals.append(ar)
        mean_align_val = np.mean(align_vals) if align_vals else 0

        print(f"  {act_name:>6s}: <CV(H)>={mean_het_val:.3f}, "
              f"max CV(H)={max_het_val:.3f}, "
              f"<test>={mean_dd:.6f}, "
              f"<align>={mean_align_val:.1f}x")

    # Key test of H1'
    # Prediction: ReLU should have higher heterogeneity than Linear
    # If true, this explains double descent differences
    linear_het = np.mean(list(heterogeneity['linear'].values()))
    relu_het = np.mean(list(heterogeneity['relu'].values()))

    if relu_het > 1.5 * linear_het:
        print(f"\n  STRONG evidence for H1': ReLU heterogeneity ({relu_het:.3f}) "
              f">> Linear ({linear_het:.3f})")
    elif relu_het > linear_het:
        print(f"\n  MODERATE evidence for H1': ReLU somewhat more heterogeneous")
    else:
        print(f"\n  AGAINST H1': Linear model MORE heterogeneous than ReLU")
        print(f"  Linear CV(H)={linear_het:.3f}, ReLU CV(H)={relu_het:.3f}")
        print(f"  → Landscape heterogeneity may not be the key differentiator")

    return all_results


if __name__ == '__main__':
    results = run_experiment_7(
        d=12,
        n_samples=800,
        n_epochs=600,
        n_seeds=5,
        output_dir='./outputs',
    )
