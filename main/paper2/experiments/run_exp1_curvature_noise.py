"""Experiment 1: Curvature-Noise Coupling Validation.

Verifies Hypothesis 1 of the NESP framework:
  Σ(W) ≈ α(W) · H(W)  along the training trajectory.

Protocol:
- Linear teacher-student model: ŷ = v^T U x
- Vary hidden width k through interpolation threshold (k ≈ d)
- For each k: train with SGD, compute H eigenvalues, estimate Σ eigenvalues
- Key outputs:
  1. Tr(Σ) vs k/d and test error vs k/d  →  peaks should coincide at k ≈ d
  2. λ_i(H) vs λ_i(Σ) scatter plot  →  positive correlation
  3. Alignment between H and Σ eigenvectors
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from models import LinearTeacherStudent, generate_teacher_data
from utils import (
    train_sgd,
    compute_hessian_eigenvalues,
    estimate_noise_covariance_eigenvalues,
    compute_condition_number,
    compute_effective_rank,
    summarize_experiment,
)


def run_experiment_1(
    d: int = 50,
    n_samples: int = 10000,
    k_values: list = None,
    lr: float = 0.01,
    n_epochs: int = 10000,
    seed: int = 42,
    output_dir: str = './outputs',
):
    """Run curvature-noise coupling experiment."""
    if k_values is None:
        k_values = [int(d * r) for r in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]]
        k_values = sorted(set(max(1, k) for k in k_values))

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  EXPERIMENT 1: Curvature-Noise Coupling Validation")
    print("  Linear Teacher-Student Model")
    print(f"  d={d}, n={n_samples}, k values={k_values}")
    print("=" * 60)

    X_train, y_train, X_test, y_test, w_star = generate_teacher_data(
        n_samples, d, seed=seed
    )

    results = {}

    for k in k_values:
        gamma = k / d
        print(f"\n--- k={k} (γ={gamma:.2f}) ---")

        model = LinearTeacherStudent(d=d, k=k)
        loss_fn = torch.nn.MSELoss()

        def compute_spectra(m, X, y):
            """Compute H eigenvalues and Σ eigenvalues."""
            # Hessian eigenvalues (using exact Kronecker structure)
            hess_eig = m.compute_hessian(X, y)

            # Noise covariance eigenvalues
            sigma_eig = estimate_noise_covariance_eigenvalues(m, X, y, loss_fn,
                                                               n_grad_samples=min(500, X.shape[0]))

            return {
                'trace_H': hess_eig.sum().item(),
                'trace_Sigma': sigma_eig.sum().item(),
                'lambda_max_H': hess_eig.max().item(),
                'lambda_max_Sigma': sigma_eig.max().item() if len(sigma_eig) > 0 else 0,
                'kappa_H': compute_condition_number(hess_eig),
                'effective_rank_H': compute_effective_rank(hess_eig),
                'hess_eigvals': hess_eig,
                'sigma_eigvals': sigma_eig,
            }

        log = train_sgd(
            model, X_train, y_train, X_test, y_test,
            lr=lr, batch_size=1,
            n_epochs=n_epochs,
            record_every=50,
            record_spectra_every=500,
            w_star=w_star,
            compute_spectra_fn=compute_spectra,
            verbose=True,
        )

        results[k] = {
            'gamma': gamma,
            'log': log,
            'final_train_loss': log['train_loss'][-1],
            'final_test_loss': log['test_loss'][-1],
            'min_test_loss': min(log['test_loss']),
        }

        if 'trace_H' in log:
            results[k]['final_trace_H'] = log['trace_H'][-1] if log['trace_H'] else None
            results[k]['final_trace_Sigma'] = log['trace_Sigma'][-1] if log['trace_Sigma'] else None
            results[k]['final_lambda_max_H'] = log['lambda_max_H'][-1] if log['lambda_max_H'] else None
            results[k]['final_kappa_H'] = log['kappa_H'][-1] if log['kappa_H'] else None

            # Eigenvalue scatter data (last spectra measurement)
            hess_eig = log['hess_eigvals'][-1] if log['hess_eigvals'] else None
            sigma_eig = log['sigma_eigvals'][-1] if log['sigma_eigvals'] else None
            results[k]['hess_eigvals'] = hess_eig
            results[k]['sigma_eigvals'] = sigma_eig

    # ── Generate Figures ──

    # Figure 1: Test error and Tr(Σ) vs γ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    gammas = [results[k]['gamma'] for k in sorted(results.keys())]
    test_losses = [results[k]['min_test_loss'] for k in sorted(results.keys())]

    ax1.plot(gammas, test_losses, 'o-', color='#2196F3', markersize=8, linewidth=2)
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='γ=1 (interpolation)')
    ax1.set_xlabel('γ = k/d (over-parameterization ratio)')
    ax1.set_ylabel('Test Loss (MSE)')
    ax1.set_title('Double Descent: Test Error vs γ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Σ trace vs γ
    trace_sigma_vals = []
    for k in sorted(results.keys()):
        ts = results[k].get('final_trace_Sigma')
        trace_sigma_vals.append(ts if ts is not None else np.nan)

    ax2.plot(gammas, trace_sigma_vals, 's-', color='#FF5722', markersize=8, linewidth=2)
    ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('γ = k/d')
    ax2.set_ylabel('Tr(Σ) — Noise Covariance Trace')
    ax2.set_title('Noise Amplification at Interpolation Threshold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Experiment 1: Curvature-Noise Coupling in Linear Teacher-Student',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exp1_double_descent_noise.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: exp1_double_descent_noise.pdf")

    # Figure 2: λ_i(H) vs λ_i(Σ) scatter for a representative k
    k_above = max(k for k in results if results[k]['gamma'] >= 1.2)
    hess_eig = results[k_above].get('hess_eigvals')
    sigma_eig = results[k_above].get('sigma_eigvals')

    if hess_eig is not None and sigma_eig is not None:
        fig, ax = plt.subplots(figsize=(7, 6))

        # Trim to same length
        n_pts = min(len(hess_eig), len(sigma_eig))
        hess_arr = hess_eig[:n_pts].detach().numpy()
        sigma_arr = sigma_eig[:n_pts].detach().numpy()

        # Filter positive values for log scale
        mask = (hess_arr > 1e-12) & (sigma_arr > 1e-12)
        n_valid = mask.sum()

        if n_valid >= 3:
            ax.scatter(hess_arr[mask], sigma_arr[mask], alpha=0.6, s=20,
                       color='#4CAF50', edgecolors='black', linewidth=0.3)

            # Correlation
            corr = np.corrcoef(np.log(hess_arr[mask]), np.log(sigma_arr[mask]))[0, 1]

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('λ_i(H) — Hessian eigenvalue')
            ax.set_ylabel('λ_i(Σ) — Noise covariance eigenvalue')
            ax.set_title(f'Eigenvalue Correspondence (k={k_above}, γ={k_above/d:.1f})\n'
                         f'Log-log correlation: ρ = {corr:.3f}')
            ax.grid(True, alpha=0.3)

            # Diagonal reference
            lims = [max(min(hess_arr[mask]), 1e-12), min(max(hess_arr[mask]), 1e6)]
            ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
        else:
            ax.text(0.5, 0.5, 'Not enough data for scatter\n(try longer training or larger model)',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Eigenvalue Correspondence (k={k_above}, γ={k_above/d:.1f})\n'
                         f'(insufficient data)')
            corr = np.nan

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'exp1_eigenvalue_scatter.pdf'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: exp1_eigenvalue_scatter.pdf  (ρ = {corr:.3f})")

    # Figure 3: κ(H) and Tr(H) vs γ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    kappas = [results[k].get('final_kappa_H', np.nan) for k in sorted(results.keys())]
    traces_H = [results[k].get('final_trace_H', np.nan) for k in sorted(results.keys())]

    ax1.plot(gammas, kappas, 'D-', color='#9C27B0', markersize=8, linewidth=2)
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5)
    ax1.set_yscale('log')
    ax1.set_xlabel('γ = k/d')
    ax1.set_ylabel('Condition Number κ(H)')
    ax1.set_title('Hessian Ill-Conditioning at Criticality')
    ax1.grid(True, alpha=0.3)

    ax2.plot(gammas, traces_H, 'o-', color='#009688', markersize=8, linewidth=2)
    ax2.set_xlabel('γ = k/d')
    ax2.set_ylabel('Tr(H)')
    ax2.set_title('Total Curvature vs Over-parameterization')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Experiment 1: Hessian Geometry Across the Transition',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exp1_hessian_geometry.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: exp1_hessian_geometry.pdf")

    # ── Print Summary ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 1 SUMMARY")
    print(f"{'='*60}")
    print(f"{'k':>5s} {'γ':>6s} {'Test Loss':>12s} {'Tr(H)':>10s} {'Tr(Σ)':>10s} {'κ(H)':>10s}")
    print(f"{'-'*55}")
    for k in sorted(results.keys()):
        r = results[k]
        th = r.get('final_trace_H', 'N/A')
        ts = r.get('final_trace_Sigma', 'N/A')
        kh = r.get('final_kappa_H', 'N/A')
        th_str = f"{th:.2e}" if isinstance(th, (int, float)) else str(th)
        ts_str = f"{ts:.2e}" if isinstance(ts, (int, float)) else str(ts)
        kh_str = f"{kh:.2e}" if isinstance(kh, (int, float)) else str(kh)
        print(f"{k:5d} {r['gamma']:6.2f} {r['min_test_loss']:12.8f} {th_str:>10s} {ts_str:>10s} {kh_str:>10s}")

    return results


if __name__ == '__main__':
    results = run_experiment_1(
        d=30,
        n_samples=4000,
        k_values=[int(30*r) for r in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]],
        lr=0.01,
        n_epochs=4000,
        output_dir='./outputs',
    )
