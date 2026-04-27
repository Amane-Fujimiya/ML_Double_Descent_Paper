"""Experiment 8: Sharpness Gradient vs Double Descent Strength.

Tests H1'': Double descent peak height correlates with Sharpness Ratio
R_H = Tr(H)_{gamma<1} / Tr(H)_{gamma>>1}.

Protocol:
1. For 5 activations, sweep gamma densely through interpolation threshold
2. At each gamma, measure: Tr(H), alignment ratio, test error
3. Compute R_H for each activation
4. Correlate R_H with double descent peak height

Prediction (H1''): Spearman rho(R_H, peak) > 0.7
  - tanh: highest R_H (strong saturation) → strongest DD
  - relu: lowest R_H (no saturation) → weakest DD
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


def run_experiment_8(
    d: int = 15,
    n_samples: int = 1500,
    k_values: list = None,
    activations: list = None,
    lr: float = 0.01,
    batch_size: int = 16,
    n_epochs: int = 800,
    seed: int = 42,
    output_dir: str = './outputs',
):
    """Sharpness gradient experiment."""
    if k_values is None:
        # Dense gamma sweep
        k_values = [int(d * r) for r in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]]
        k_values = sorted(set(max(1, k) for k in k_values))

    if activations is None:
        activations = ['linear', 'leaky_relu', 'relu', 'gelu', 'tanh']

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  EXPERIMENT 8: Sharpness Gradient vs Double Descent")
    print(f"  d={d}, n={n_samples}, dense gamma sweep")
    print(f"  Activations: {activations}")
    print("=" * 60)

    X_train, y_train, X_test, y_test, w_star = generate_teacher_data(
        n_samples, d, seed=seed
    )
    loss_fn = nn.MSELoss()

    all_results = {}  # {act_name: [{gamma, k, test_loss, tr_H, align_ratio}, ...]}

    for act_name in activations:
        print(f"\n{'─'*50}")
        print(f"  Activation: {act_name}")
        print(f"{'─'*50}")

        act_results = []

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
                record_every=400,
                record_spectra_every=100000,
                compute_spectra_fn=dummy_spectra,
                verbose=False,
            )

            # Hessian trace
            tr_H = compute_hessian_trace(model, X_train[:100], y_train[:100], loss_fn)

            # Alignment
            k_eig = min(10, n_params)
            hess_eigvals, hess_eigvecs = model.compute_hessian_topk(
                X_train[:200], y_train[:200], loss_fn, k_eig=k_eig
            )
            sigma_eigvals, sigma_eigvecs = model.compute_noise_cov_topk(
                X_train[:500], y_train[:500], loss_fn, k_eig=k_eig
            )
            alignment = eigenvector_alignment(hess_eigvecs, sigma_eigvecs)
            alignment_ratio = alignment / (1.0 / n_params)

            # Eigenvalue correlation
            n_compare = min(len(hess_eigvals), len(sigma_eigvals))
            if n_compare >= 3:
                hess_top = hess_eigvals[:n_compare].detach().numpy()
                sigma_top = sigma_eigvals[:n_compare].detach().numpy()
                mask = (hess_top > 1e-10) & (sigma_top > 1e-10)
                if mask.sum() >= 3:
                    corr = np.corrcoef(np.log(hess_top[mask]), np.log(sigma_top[mask]))[0, 1]
                else:
                    corr = np.nan
            else:
                corr = np.nan

            act_results.append({
                'k': k, 'gamma': gamma, 'n_params': n_params,
                'test_loss': log['test_loss'][-1],
                'min_test_loss': min(log['test_loss']),
                'tr_H': tr_H,
                'alignment_ratio': alignment_ratio,
                'eigval_corr': corr,
            })

            print(f"    k={k:3d} (γ={gamma:.2f}) | test={log['test_loss'][-1]:.6f} | "
                  f"Tr(H)={tr_H:.1f} | align={alignment_ratio:.1f}x | ρ={corr:.3f}")

        all_results[act_name] = act_results

    # ── Compute Sharpness Ratio and Double Descent Metrics ──
    metrics = {}  # {act_name: {R_H, peak_height, peak_gamma, min_test, mean_align}}

    for act_name in activations:
        results = all_results[act_name]
        gammas = np.array([r['gamma'] for r in results])
        tr_H_vals = np.array([r['tr_H'] for r in results])
        test_vals = np.array([r['min_test_loss'] for r in results])

        # Sharpness Ratio: mean Tr(H) at gamma<1 / mean Tr(H) at gamma>2
        mask_low = gammas < 1.0
        mask_high = gammas > 2.0

        mean_H_low = tr_H_vals[mask_low].mean() if mask_low.sum() > 0 else 1
        mean_H_high = tr_H_vals[mask_high].mean() if mask_high.sum() > 0 else 1
        R_H = mean_H_low / max(mean_H_high, 1e-10)

        # Double descent peak: max test error near gamma=1
        mask_near = (gammas >= 0.8) & (gammas <= 1.5)
        if mask_near.sum() > 0:
            peak_height = test_vals[mask_near].max()
            peak_gamma = gammas[mask_near][test_vals[mask_near].argmax()]
        else:
            peak_height = test_vals.max()
            peak_gamma = gammas[test_vals.argmax()]

        # Second descent: min test after threshold
        mask_post = gammas > 1.0
        if mask_post.sum() > 0:
            min_post = test_vals[mask_post].min()
        else:
            min_post = test_vals.min()

        recovery = (peak_height - min_post) / (peak_height + 1e-10)

        # Mean alignment
        align_vals = [r['alignment_ratio'] for r in results]
        mean_align = np.mean(align_vals)

        metrics[act_name] = {
            'R_H': R_H,
            'peak_height': peak_height,
            'peak_gamma': peak_gamma,
            'min_post': min_post,
            'recovery': recovery,
            'mean_align': mean_align,
            'mean_H_low': mean_H_low,
            'mean_H_high': mean_H_high,
        }

    # ── Generate Figures ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    colors = {
        'linear': '#9E9E9E', 'leaky_relu': '#FF9800',
        'relu': '#4CAF50', 'gelu': '#2196F3', 'tanh': '#9C27B0'
    }

    # Subplot 1: Double descent curves for all activations
    ax = axes[0, 0]
    for act_name in activations:
        gammas = [r['gamma'] for r in all_results[act_name]]
        test_losses = [r['min_test_loss'] for r in all_results[act_name]]
        ax.plot(gammas, test_losses, 'o-', color=colors[act_name],
                markersize=7, linewidth=2, label=act_name)

    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4, label='γ=1')
    ax.set_xlabel('γ = k/d')
    ax.set_ylabel('Test Loss (MSE)')
    ax.set_title('Double Descent by Activation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: Tr(H) vs gamma (sharpness gradient)
    ax = axes[0, 1]
    for act_name in activations:
        gammas = [r['gamma'] for r in all_results[act_name]]
        tr_Hs = [r['tr_H'] for r in all_results[act_name]]
        ax.plot(gammas, tr_Hs, 's-', color=colors[act_name],
                markersize=7, linewidth=2, label=act_name)

    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4)
    ax.set_xlabel('γ = k/d')
    ax.set_ylabel('Tr(H) — Hessian Trace')
    ax.set_title('Sharpness Gradient by Activation')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 3: R_H vs Peak Height (THE KEY PLOT)
    ax = axes[0, 2]
    for act_name in activations:
        m = metrics[act_name]
        ax.scatter(m['R_H'], m['peak_height'], color=colors[act_name],
                   s=200, alpha=0.8, edgecolors='black', linewidth=1,
                   label=act_name, zorder=5)

    # Spearman correlation
    rh_vals = [metrics[a]['R_H'] for a in activations]
    peak_vals = [metrics[a]['peak_height'] for a in activations]
    if len(rh_vals) >= 3:
        rho, p_val = scipy_stats.spearmanr(rh_vals, peak_vals)
    else:
        rho, p_val = np.nan, np.nan

    ax.set_xlabel('Sharpness Ratio R_H = <Tr(H)>_{γ<1} / <Tr(H)>_{γ>2}')
    ax.set_ylabel('Double Descent Peak Height (MSE)')
    ax.set_title(f'R_H vs DD Peak (Spearman ρ={rho:.3f}, p={p_val:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 4: Alignment vs R_H
    ax = axes[1, 0]
    for act_name in activations:
        m = metrics[act_name]
        ax.scatter(m['R_H'], m['mean_align'], color=colors[act_name],
                   s=200, alpha=0.8, edgecolors='black', linewidth=1,
                   label=act_name, zorder=5)

    ax_rh = [metrics[a]['R_H'] for a in activations]
    ax_align = [metrics[a]['mean_align'] for a in activations]
    if len(ax_rh) >= 3:
        rho2, p_val2 = scipy_stats.spearmanr(ax_rh, ax_align)
    else:
        rho2, p_val2 = np.nan, np.nan

    ax.set_xlabel('Sharpness Ratio R_H')
    ax.set_ylabel('Mean Alignment Ratio')
    ax.set_title(f'Coupling Strength vs R_H (ρ={rho2:.3f}, p={p_val2:.3f})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 5: Recovery rate vs R_H
    ax = axes[1, 1]
    for act_name in activations:
        m = metrics[act_name]
        ax.scatter(m['R_H'], m['recovery'] * 100, color=colors[act_name],
                   s=200, alpha=0.8, edgecolors='black', linewidth=1,
                   label=act_name, zorder=5)

    ax.set_xlabel('Sharpness Ratio R_H')
    ax.set_ylabel('Recovery Rate (%)')
    ax.set_title('Post-Peak Recovery vs R_H')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 6: Summary table as bar chart
    ax = axes[1, 2]
    x_pos = np.arange(len(activations))
    width = 0.3

    rh_norm = [metrics[a]['R_H'] / max(rh_vals) if max(rh_vals) > 0 else 0
               for a in activations]
    peak_norm = [metrics[a]['peak_height'] / max(peak_vals) if max(peak_vals) > 0 else 0
                 for a in activations]

    ax.bar(x_pos - width/2, rh_norm, width, color='#FF5722', alpha=0.7,
           edgecolor='black', linewidth=0.5, label='Sharpness Ratio (norm)')
    ax.bar(x_pos + width/2, peak_norm, width, color='#2196F3', alpha=0.7,
           edgecolor='black', linewidth=0.5, label='DD Peak (norm)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(activations, rotation=15)
    ax.set_ylabel('Normalized Value')
    ax.set_title('Sharpness Ratio vs DD Peak')
    ax.legend(fontsize=8)

    fig.suptitle('Experiment 8: Sharpness Gradient Predicts Double Descent Strength',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exp8_sharpness_gradient.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: exp8_sharpness_gradient.pdf")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 8 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Activation':>12s} {'R_H':>8s} {'Peak H':>10s} {'DD Peak':>10s} "
          f"{'Recovery':>10s} {'<Align>':>8s}")
    print(f"{'-'*60}")
    for act_name in activations:
        m = metrics[act_name]
        print(f"{act_name:>12s} {m['R_H']:8.2f} {m['mean_H_low']:10.1f} "
              f"{m['peak_height']:10.6f} {m['recovery']*100:9.1f}% "
              f"{m['mean_align']:8.1f}x")

    # ── Test H1'' ──
    if not np.isnan(rho):
        if rho > 0.7 and p_val < 0.1:
            print(f"\n  STRONG evidence for H1'': R_H predicts DD peak "
                  f"(ρ={rho:.3f}, p={p_val:.3f})")
        elif rho > 0.4:
            print(f"\n  MODERATE evidence for H1'': Positive correlation "
                  f"(ρ={rho:.3f}, p={p_val:.3f})")
        elif rho > 0:
            print(f"\n  WEAK evidence for H1'': Weak positive correlation "
                  f"(ρ={rho:.3f}, p={p_val:.3f})")
        else:
            print(f"\n  AGAINST H1'': Negative or zero correlation "
                  f"(ρ={rho:.3f}, p={p_val:.3f})")

    # Ranking
    sorted_by_rh = sorted(activations, key=lambda a: metrics[a]['R_H'], reverse=True)
    sorted_by_peak = sorted(activations, key=lambda a: metrics[a]['peak_height'], reverse=True)
    print(f"\n  By Sharpness Ratio: {' > '.join(sorted_by_rh)}")
    print(f"  By DD Peak Height: {' > '.join(sorted_by_peak)}")

    return all_results, metrics


if __name__ == '__main__':
    results, metrics = run_experiment_8(
        d=15,
        n_samples=1500,
        n_epochs=800,
        output_dir='./outputs',
    )
