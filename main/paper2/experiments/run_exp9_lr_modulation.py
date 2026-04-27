"""Experiment 9: Learning Rate Modulation of Sharpness Gradient.

Tests the mechanism: DD peak = f(η × R_H), where:
- R_H = Tr(H)_{gamma<1} / Tr(H)_{gamma>2} is a landscape property
- η controls noise amplification
- Together they determine double descent strength

Prediction:
  1. R_H should be INDEPENDENT of η (landscape property)
  2. DD peak should INCREASE with η (more noise → more amplification)
  3. DD peak ~ η × R_H should show stronger correlation than R_H alone
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
from scipy import stats as scipy_stats

from models import generate_teacher_data, eigenvector_alignment
from run_exp6_activation_comparison import TwoLayerNetwork
from run_exp7_heterogeneity import compute_hessian_trace
from utils import train_sgd


def run_experiment_9(
    d: int = 15,
    n_samples: int = 1200,
    k_values: list = None,
    activations: list = None,
    lr_values: list = None,
    batch_size: int = 16,
    n_epochs: int = 600,
    seed: int = 42,
    output_dir: str = './outputs',
):
    """Learning rate modulation experiment."""
    if k_values is None:
        k_values = [int(d * r) for r in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]]
        k_values = sorted(set(max(1, k) for k in k_values))

    if activations is None:
        activations = ['linear', 'relu', 'tanh']

    if lr_values is None:
        lr_values = [0.003, 0.01, 0.03]

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  EXPERIMENT 9: Learning Rate Modulation")
    print(f"  d={d}, activations={activations}")
    print(f"  lr_values={lr_values}")
    print("=" * 60)

    X_train, y_train, X_test, y_test, w_star = generate_teacher_data(
        n_samples, d, seed=seed
    )
    loss_fn = nn.MSELoss()

    # {lr: {act_name: {gamma: {test_loss, tr_H}}}}
    all_results = {}

    for lr in lr_values:
        print(f"\n{'#'*50}")
        print(f"  Learning Rate: {lr}")
        print(f"{'#'*50}")

        lr_results = {}

        for act_name in activations:
            print(f"\n  Activation: {act_name}")

            act_data = []
            for k in k_values:
                gamma = k / d
                n_params = k * d + k

                model = TwoLayerNetwork(d=d, k=k, activation=act_name)

                def dummy_spectra(m, X, y):
                    return {}

                log = train_sgd(
                    model, X_train, y_train, X_test, y_test,
                    lr=lr, batch_size=batch_size,
                    n_epochs=n_epochs,
                    record_every=300,
                    record_spectra_every=100000,
                    compute_spectra_fn=dummy_spectra,
                    verbose=False,
                )

                tr_H = compute_hessian_trace(model, X_train[:100], y_train[:100], loss_fn)

                act_data.append({
                    'k': k, 'gamma': gamma,
                    'test_loss': log['test_loss'][-1],
                    'min_test_loss': min(log['test_loss']),
                    'tr_H': tr_H,
                })

                print(f"    k={k:3d} (γ={gamma:.2f}) | "
                      f"test={log['test_loss'][-1]:.6f} | Tr(H)={tr_H:.1f}")

            lr_results[act_name] = act_data

        all_results[lr] = lr_results

    # ── Compute metrics for each (lr, activation) ──
    metrics = []  # list of {lr, act_name, R_H, peak_height, mean_H_low, mean_H_high}

    for lr in lr_values:
        for act_name in activations:
            data = all_results[lr][act_name]
            gammas = np.array([d['gamma'] for d in data])
            tr_H_vals = np.array([d['tr_H'] for d in data])
            test_vals = np.array([d['min_test_loss'] for d in data])

            mask_low = gammas < 1.0
            mask_high = gammas > 2.0

            mean_H_low = tr_H_vals[mask_low].mean() if mask_low.sum() > 0 else 1
            mean_H_high = tr_H_vals[mask_high].mean() if mask_high.sum() > 0 else 1
            R_H = mean_H_low / max(mean_H_high, 1e-10)

            mask_near = (gammas >= 0.8) & (gammas <= 1.5)
            if mask_near.sum() > 0:
                peak_height = test_vals[mask_near].max()
            else:
                peak_height = test_vals.max()

            metrics.append({
                'lr': lr,
                'act_name': act_name,
                'R_H': R_H,
                'peak_height': peak_height,
                'mean_H_low': mean_H_low,
                'mean_H_high': mean_H_high,
                'eta_R_H': lr * R_H,
            })

    # ── Generate Figures ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    colors = {'linear': '#9E9E9E', 'relu': '#4CAF50', 'tanh': '#9C27B0'}
    markers = {0.003: 'o', 0.01: 's', 0.03: 'D'}

    # Row 1: Double descent curves for each activation at different η
    for idx, act_name in enumerate(activations):
        ax = axes[0, idx]
        for lr in lr_values:
            data = all_results[lr][act_name]
            gammas = [d['gamma'] for d in data]
            test_losses = [d['min_test_loss'] for d in data]
            ax.plot(gammas, test_losses, f'{markers[lr]}-',
                    color=colors[act_name], markersize=6, linewidth=1.5,
                    alpha=0.8 if lr == 0.01 else 0.5,
                    label=f'η={lr}')

        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4)
        ax.set_xlabel('γ = k/d')
        ax.set_ylabel('Test Loss (MSE)')
        ax.set_title(f'{act_name} — DD by Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2, Subplot 1: DD Peak vs R_H (colored by η)
    ax = axes[1, 0]
    for m in metrics:
        ax.scatter(m['R_H'], m['peak_height'],
                   color=colors[m['act_name']],
                   marker=markers[m['lr']], s=100, alpha=0.7,
                   edgecolors='black', linewidth=0.5,
                   label=f"{m['act_name']} η={m['lr']}" if m['lr'] == 0.01 else "")

    ax.set_xlabel('Sharpness Ratio R_H')
    ax.set_ylabel('DD Peak Height (MSE)')
    ax.set_title('DD Peak vs R_H by Learning Rate')
    ax.grid(True, alpha=0.3)

    # Subplot 2: DD Peak vs η × R_H (combined metric)
    ax = axes[1, 1]
    for m in metrics:
        ax.scatter(m['eta_R_H'], m['peak_height'],
                   color=colors[m['act_name']],
                   marker=markers[m['lr']], s=100, alpha=0.7,
                   edgecolors='black', linewidth=0.5)

    # Spearman correlation
    eta_rh_vals = [m['eta_R_H'] for m in metrics]
    peak_vals = [m['peak_height'] for m in metrics]
    rho_combined, p_combined = scipy_stats.spearmanr(eta_rh_vals, peak_vals)

    # Linear fit
    coeffs = np.polyfit(eta_rh_vals, peak_vals, 1)
    x_fit = np.linspace(0, max(eta_rh_vals) * 1.1, 50)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, 'k--', linewidth=1, alpha=0.5,
            label=f'fit (ρ={rho_combined:.3f})')

    ax.set_xlabel('η × R_H')
    ax.set_ylabel('DD Peak Height (MSE)')
    ax.set_title(f'DD Peak vs η×R_H (ρ={rho_combined:.3f}, p={p_combined:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 3: R_H stability across η
    ax = axes[1, 2]
    for act_name in activations:
        act_rh = [m['R_H'] for m in metrics if m['act_name'] == act_name]
        act_lr = [m['lr'] for m in metrics if m['act_name'] == act_name]
        ax.plot(act_lr, act_rh, 'o-', color=colors[act_name],
                markersize=10, linewidth=2, label=act_name)

    ax.set_xlabel('Learning Rate η')
    ax.set_ylabel('Sharpness Ratio R_H')
    ax.set_title('R_H Stability Across Learning Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 9: Learning Rate Amplifies Sharpness Gradient Signal',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exp9_lr_modulation.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: exp9_lr_modulation.pdf")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 9 SUMMARY")
    print(f"{'='*60}")

    # Test predictions
    print(f"\n  Prediction 1: R_H independent of η?")
    for act_name in activations:
        act_rhs = [m['R_H'] for m in metrics if m['act_name'] == act_name]
        cv_rh = np.std(act_rhs) / (np.mean(act_rhs) + 1e-10)
        print(f"    {act_name}: R_H across η = {act_rhs}, CV={cv_rh:.2f}")

    print(f"\n  Prediction 2: Spearman ρ(η×R_H, DD peak) = {rho_combined:.3f}")
    print(f"    vs ρ(R_H, DD peak) alone — should improve with η modulation")

    # Per-η correlation
    for lr in lr_values:
        lr_metrics = [m for m in metrics if m['lr'] == lr]
        if len(lr_metrics) >= 3:
            rh = [m['R_H'] for m in lr_metrics]
            pk = [m['peak_height'] for m in lr_metrics]
            rho_lr, p_lr = scipy_stats.spearmanr(rh, pk)
            print(f"    η={lr}: ρ(R_H, peak) = {rho_lr:.3f}, p={p_lr:.3f}")

    print(f"\n  Combined η×R_H metric: ρ={rho_combined:.3f}, p={p_combined:.3f}")

    if rho_combined > 0.7:
        print(f"  STRONG evidence: η×R_H is a robust predictor of DD peak")
    elif rho_combined > 0.5:
        print(f"  MODERATE evidence for combined metric")
    else:
        print(f"  WEAK evidence for combined metric")

    return all_results, metrics


if __name__ == '__main__':
    results, metrics = run_experiment_9(
        d=15,
        n_samples=1200,
        n_epochs=600,
        output_dir='./outputs',
    )
