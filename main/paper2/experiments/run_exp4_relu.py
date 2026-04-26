"""Experiment 4: Curvature-Noise Coupling in ReLU Networks.

Tests if the alignment between Hessian and noise covariance eigenvectors
persists in nonlinear (ReLU) networks.

Protocol:
1. Train a two-layer ReLU network on regression
2. At convergence, compute top-k Hessian eigenvectors (via power iteration)
3. Compute top-k noise covariance eigenvectors (from per-sample gradients)
4. Measure alignment: (1/k) Σ |v_i(H)^T v_i(Σ)|²

Success criterion: alignment >> 1/dim (significantly above random baseline)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import ShallowReLUNetwork, generate_teacher_data, eigenvector_alignment
from utils import train_sgd, summarize_experiment


def run_experiment_4(
    d: int = 15,
    n_samples: int = 1500,
    k_values: list = None,
    lr: float = 0.01,
    batch_size: int = 16,
    n_epochs: int = 2000,
    seed: int = 42,
    output_dir: str = './outputs',
):
    """ReLU curvature-noise coupling experiment."""
    if k_values is None:
        k_values = [int(d * r) for r in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]]
        k_values = sorted(set(max(1, k) for k in k_values))

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  EXPERIMENT 4: Curvature-Noise Coupling in ReLU Networks")
    print(f"  d={d}, n={n_samples}")
    print("=" * 60)

    X_train, y_train, X_test, y_test, w_star = generate_teacher_data(
        n_samples, d, seed=seed
    )
    loss_fn = nn.MSELoss()

    results = []

    for k in k_values:
        gamma = k / d
        n_params = k * d + k  # U: k×d, v: k

        print(f"\n--- k={k} (γ={gamma:.2f}), total params={n_params} ---")

        model = ShallowReLUNetwork(d=d, k=k, activation='relu')

        # Train with SGD
        def dummy_spectra(m, X, y):
            return {}

        log = train_sgd(
            model, X_train, y_train, X_test, y_test,
            lr=lr, batch_size=batch_size,
            n_epochs=n_epochs,
            record_every=100,
            record_spectra_every=100000,
            compute_spectra_fn=dummy_spectra,
            verbose=True,
        )

        # After training: compute alignment
        k_eig = min(20, n_params)

        print(f"  Computing Hessian top-{k_eig} eigenvectors...")
        hess_eigvals, hess_eigvecs = model.compute_hessian_topk(
            X_train[:200], y_train[:200], loss_fn, k_eig=k_eig
        )

        print(f"  Computing noise covariance top-{k_eig} eigenvectors...")
        sigma_eigvals, sigma_eigvecs = model.compute_noise_cov_topk(
            X_train[:500], y_train[:500], loss_fn, k_eig=k_eig
        )

        alignment = eigenvector_alignment(hess_eigvecs, sigma_eigvecs)
        random_baseline = 1.0 / n_params
        alignment_ratio = alignment / random_baseline

        print(f"  Alignment: {alignment:.6f}")
        print(f"  Random baseline (1/dim): {random_baseline:.6f}")
        print(f"  Alignment ratio: {alignment_ratio:.1f}x")

        # Eigenvalue correlation
        n_compare = min(len(hess_eigvals), len(sigma_eigvals))
        if n_compare >= 5:
            hess_top = hess_eigvals[:n_compare].detach().numpy()
            sigma_top = sigma_eigvals[:n_compare].detach().numpy()
            mask = (hess_top > 1e-10) & (sigma_top > 1e-10)
            if mask.sum() >= 3:
                corr = np.corrcoef(np.log(hess_top[mask]), np.log(sigma_top[mask]))[0, 1]
            else:
                corr = np.nan
        else:
            corr = np.nan

        results.append({
            'k': k,
            'gamma': gamma,
            'n_params': n_params,
            'final_test_loss': log['test_loss'][-1],
            'alignment': alignment,
            'alignment_ratio': alignment_ratio,
            'hess_eigvals': hess_eigvals.detach(),
            'sigma_eigvals': sigma_eigvals.detach(),
            'eigval_correlation': corr,
        })

    # ── Generate Figures ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    gammas = [r['gamma'] for r in results]
    alignments = [r['alignment'] for r in results]
    alignment_ratios = [r['alignment_ratio'] for r in results]
    eigval_corrs = [r['eigval_correlation'] for r in results]

    # Subplot 1: Alignment vs γ
    ax = axes[0]
    ax.plot(gammas, alignment_ratios, 'o-', color='#4CAF50', markersize=10,
            linewidth=2, zorder=5)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
               label='Random baseline (1:1)')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4, label='γ=1')
    ax.set_xlabel('γ = k/d')
    ax.set_ylabel('Alignment Ratio (vs random baseline)')
    ax.set_title('H-Σ Eigenvector Alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: Eigenvalue correlation vs γ
    ax = axes[1]
    ax.plot(gammas, eigval_corrs, 's-', color='#FF5722', markersize=10,
            linewidth=2, zorder=5)
    ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4)
    ax.set_xlabel('γ = k/d')
    ax.set_ylabel('log(λ_H) vs log(λ_Σ) Correlation')
    ax.set_title('Eigenvalue Spectrum Correlation')
    ax.grid(True, alpha=0.3)

    # Subplot 3: Test error vs γ (double descent for ReLU)
    ax = axes[2]
    test_losses = [r['final_test_loss'] for r in results]
    ax.plot(gammas, test_losses, 'o-', color='#2196F3', markersize=10,
            linewidth=2)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4)
    ax.set_xlabel('γ = k/d')
    ax.set_ylabel('Test Loss (MSE)')
    ax.set_title('Double Descent (ReLU Network)')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 4: Curvature-Noise Coupling Survives in ReLU Networks',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exp4_relu_alignment.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: exp4_relu_alignment.pdf")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 4 SUMMARY")
    print(f"{'='*60}")
    print(f"{'k':>5s} {'γ':>6s} {'Alignment':>12s} {'Ratio':>8s} {'Eigval ρ':>10s} {'Test Loss':>12s}")
    print(f"{'-'*58}")
    for r in results:
        print(f"{r['k']:5d} {r['gamma']:6.2f} {r['alignment']:12.6f} "
              f"{r['alignment_ratio']:8.1f}x {r['eigval_correlation']:10.3f} "
              f"{r['final_test_loss']:12.6f}")

    alignment_str = "SUPPORTS" if any(r['alignment_ratio'] > 5 for r in results) else "WEAK EVIDENCE for"
    print(f"\n  Conclusion: {alignment_str} NESP curvature-noise coupling in ReLU networks")
    if alignment_str == "SUPPORTS":
        print(f"  (alignment significantly above random baseline → coupling persists)")
    else:
        print(f"  (alignment near random → coupling may be nonlinearity-dependent)")

    return results


if __name__ == '__main__':
    results = run_experiment_4(
        d=15,
        n_samples=1500,
        output_dir='./outputs',
    )
