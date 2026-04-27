"""Experiment 5: Equilibrium Erosion at t → ∞.

Tests the prediction that once the curvature-noise coupling vanishes
(at L(W) ≈ 0), the kinetic selection pressure erodes and test error
slowly increases as weights diffuse along the zero-loss manifold.

Protocol:
1. Train linear teacher-student model well past convergence
2. Continue training for 10-100x additional epochs
3. Record test error, weight displacement, and Tr(H) throughout
4. Vary batch size B: larger noise → faster erosion
5. Control: full-batch GD (B=n) should show NO erosion

NESP prediction:
  After convergence: dW_t ≈ √(2D_res) · dB_t  (diffusion on manifold)
  → test error should slowly increase
  → Tr(H) should increase (drift toward sharper regions)
  → rate ∝ η/B
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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

from models import LinearTeacherStudent, generate_teacher_data
from utils import train_sgd_erosion


def run_experiment_5(
    d: int = 20,
    n_samples: int = 2000,
    k: int = 30,  # over-parameterized regime
    batch_sizes: list = None,
    lr: float = 0.01,
    n_epochs: int = 3000,  # epochs for initial convergence
    n_epochs_converge: int = None,  # if None, uses n_epochs
    n_epochs_extended: int = 15000,
    convergence_threshold: float = 1e-7,
    seed: int = 42,
    output_dir: str = './outputs',
):
    if n_epochs_converge is None:
        n_epochs_converge = n_epochs
    """Equilibrium erosion experiment."""
    n_train = int(0.7 * n_samples)
    if batch_sizes is None:
        batch_sizes = [16, 64, n_train]  # n_train = full-batch (control)

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  EXPERIMENT 5: Equilibrium Erosion at t → ∞")
    print(f"  d={d}, k={k} (γ={k/d:.1f}, over-parameterized)")
    print(f"  Extended training: {n_epochs_converge} + {n_epochs_extended} epochs")
    print("=" * 60)

    X_train, y_train, X_test, y_test, w_star = generate_teacher_data(
        n_samples, d, seed=seed
    )

    all_results = {}

    for B in batch_sizes:
        is_full = (B == n_train)
        print(f"\n{'─'*50}")
        print(f"  Batch size B = {B}" + (" (FULL-BATCH CONTROL)" if is_full else ""))
        print(f"{'─'*50}")

        model = LinearTeacherStudent(d=d, k=k)

        log = train_sgd_erosion(
            model, X_train, y_train, X_test, y_test,
            lr=lr, batch_size=B,
            n_epochs_converge=n_epochs_converge,
            n_epochs_extended=n_epochs_extended,
            record_every=200,
            convergence_threshold=convergence_threshold,
            verbose=True,
        )

        all_results[B] = log

        # Analyze erosion
        conv_epoch = log.get('convergence_epoch')
        if conv_epoch is not None:
            post_conv_mask = np.array(log['epoch']) >= conv_epoch
            post_test_losses = np.array(log['test_loss'])[post_conv_mask]
            post_epochs = np.array(log['epoch'])[post_conv_mask]

            if len(post_test_losses) >= 3:
                # Linear fit to post-convergence test error
                coeffs = np.polyfit(post_epochs, post_test_losses, 1)
                erosion_rate = coeffs[0]  # slope

                # t-test for significance
                from scipy import stats
                if len(post_test_losses) >= 10:
                    mid = len(post_test_losses) // 2
                    first_half = post_test_losses[:mid]
                    second_half = post_test_losses[mid:]
                    t_stat, p_val = stats.ttest_ind(second_half, first_half)
                else:
                    p_val = 1.0

                print(f"\n  Convergence at epoch {conv_epoch}")
                print(f"  Post-convergence epochs: {len(post_test_losses)}")
                print(f"  Erosion rate: {erosion_rate:.2e} per epoch")
                print(f"  p-value (first vs second half): {p_val:.4f}")

                all_results[B]['erosion_rate'] = erosion_rate
                all_results[B]['p_value'] = p_val
                all_results[B]['significant'] = (p_val < 0.05 and erosion_rate > 0)
            else:
                all_results[B]['erosion_rate'] = 0.0
                all_results[B]['significant'] = False
        else:
            print(f"\n  Model did not reach convergence threshold")

    # ── Generate Figures ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(batch_sizes)))

    # Subplot 1: Test error over extended training
    ax = axes[0]
    for idx, B in enumerate(batch_sizes):
        log = all_results[B]
        epochs = log['epoch']
        test_losses = log['test_loss']
        conv_ep = log.get('convergence_epoch')

        label = f'B={B}' + (' (full-batch)' if B == n_train else '')
        ax.plot(epochs, test_losses, linewidth=1, color=colors[idx],
                alpha=0.85, label=label)

        if conv_ep is not None:
            ax.axvline(x=conv_ep, color=colors[idx], linestyle='--',
                       alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss (MSE)')
    ax.set_title('Equilibrium Erosion: Test Error Over Extended Training')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Subplot 2: Zoom into post-convergence region (normalized)
    ax = axes[1]
    for idx, B in enumerate(batch_sizes):
        log = all_results[B]
        conv_ep = log.get('convergence_epoch')
        if conv_ep is None:
            continue

        post_epochs = np.array(log['epoch']) - conv_ep
        post_mask = post_epochs >= 0
        post_e = post_epochs[post_mask]
        post_l = np.array(log['test_loss'])[post_mask]

        # Normalize to first post-convergence value
        if len(post_l) > 0:
            norm_loss = post_l / post_l[0]
            label = f'B={B}' + (' (control)' if B == n_train else '')
            ax.plot(post_e, norm_loss, linewidth=1.5, color=colors[idx],
                    alpha=0.85, label=label)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Epochs since convergence')
    ax.set_ylabel('Normalized Test Loss (relative to convergence)')
    ax.set_title('Post-Convergence Drift (Normalized)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Erosion rate vs batch size
    ax = axes[2]
    B_vals = []
    erosion_rates = []
    p_vals = []
    significances = []

    for B in batch_sizes:
        if 'erosion_rate' in all_results[B]:
            B_vals.append(B)
            erosion_rates.append(all_results[B]['erosion_rate'])
            p_vals.append(all_results[B].get('p_value', 1.0))
            significances.append(all_results[B].get('significant', False))

    if B_vals:
        bar_colors = ['#4CAF50' if s else '#F44336' for s in significances]
        bars = ax.bar(range(len(B_vals)), erosion_rates, color=bar_colors,
                      alpha=0.7, edgecolor='black', linewidth=0.5)

        # Annotate with p-values
        for i, (rate, p) in enumerate(zip(erosion_rates, p_vals)):
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            y_pos = rate + max(abs(r) for r in erosion_rates) * 0.05
            ax.text(i, y_pos, sig, ha='center', fontsize=10, fontweight='bold')

        ax.set_xticks(range(len(B_vals)))
        ax.set_xticklabels([str(b) for b in B_vals])
        ax.set_xlabel('Batch Size B')
        ax.set_ylabel('Erosion Rate (Δ test loss / epoch)')
        ax.set_title('Erosion Rate vs Batch Size\n(* p<.05  ** p<.01  *** p<.001)')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Experiment 5: Equilibrium Erosion — NESP Kinetic Memory Effect',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exp5_equilibrium_erosion.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: exp5_equilibrium_erosion.pdf")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 5 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Batch B':>10s} {'Erosion Rate':>14s} {'p-value':>10s} {'Significant':>12s}")
    print(f"{'-'*48}")
    for B in batch_sizes:
        er = all_results[B].get('erosion_rate', None)
        pv = all_results[B].get('p_value', None)
        sig = all_results[B].get('significant', False)
        er_str = f"{er:.2e}" if er is not None else "N/A"
        pv_str = f"{pv:.4f}" if pv is not None else "N/A"
        sig_str = "YES" if sig else "no"
        print(f"{B:10d} {er_str:>14s} {pv_str:>10s} {sig_str:>12s}")

    # Conclusion
    full_batch_B = n_train
    if 'erosion_rate' in all_results.get(full_batch_B, {}):
        fb_erosion = all_results[full_batch_B].get('erosion_rate', 0)
        fb_sig = all_results[full_batch_B].get('significant', False)
        if not fb_sig:
            print(f"\n  ✓ Full-batch control shows no significant erosion (as predicted)")
        else:
            print(f"\n  ⚠ Full-batch also shows erosion — may indicate other effects")

    # Check if any small-batch shows significant erosion
    small_B_significant = any(
        all_results[B].get('significant', False)
        for B in batch_sizes if B != n_train
    )
    if small_B_significant:
        print(f"  ✓ Small-batch SGD shows significant erosion → supports NESP kinetic memory")
    else:
        print(f"  ⚠ No significant erosion detected → may need longer training or smaller B")

    return all_results


if __name__ == '__main__':
    results = run_experiment_5(
        d=20,
        n_samples=2000,
        k=30,
        output_dir='./outputs',
    )
