"""Experiment 3: Batch Size Dependence of Double Descent.

Tests the prediction that batch size modulates the double descent peak.

Protocol:
- Fix architecture (linear teacher-student), vary batch size B
- For each B, sweep hidden width k through interpolation threshold
- Key output: test error vs k/d for each B
  - Smaller B (larger noise) → amplified peak + faster second descent

NESP prediction:
  T_eff = η/B  →  Smaller B means larger effective temperature.
  Larger T_eff makes the system explore more, escaping sharp minima faster
  but also creating larger fluctuations at the interpolation threshold.
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
from utils import train_sgd, summarize_experiment


def run_experiment_3(
    d: int = 30,
    n_samples: int = 3000,
    k_values: list = None,
    batch_sizes: list = None,
    lr: float = 0.01,
    n_epochs: int = 1500,
    seed: int = 42,
    output_dir: str = './outputs',
):
    """Batch size dependence experiment."""
    if k_values is None:
        k_values = [int(d * r) for r in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]]
        k_values = sorted(set(max(1, k) for k in k_values))

    if batch_sizes is None:
        n_train = int(0.7 * n_samples)
        batch_sizes = [1, 4, 16, 64, n_train]  # last = full batch
        batch_sizes = [b for b in batch_sizes if b <= n_train]

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  EXPERIMENT 3: Batch Size Dependence of Double Descent")
    print(f"  d={d}, n={n_samples}, k values={k_values}")
    print(f"  Batch sizes: {batch_sizes}")
    print("=" * 60)

    X_train, y_train, X_test, y_test, w_star = generate_teacher_data(
        n_samples, d, seed=seed
    )

    all_results = {}  # {batch_size: {k: {...}}}

    for B in batch_sizes:
        print(f"\n{'─'*50}")
        print(f"  Batch size B = {B}")
        if B == X_train.shape[0]:
            print(f"  (Full-batch / deterministic GD)")
        print(f"{'─'*50}")

        B_results = {}

        for k in k_values:
            gamma = k / d
            model = LinearTeacherStudent(d=d, k=k)

            def dummy_spectra(m, X, y):
                return {}

            log = train_sgd(
                model, X_train, y_train, X_test, y_test,
                lr=lr, batch_size=B,
                n_epochs=n_epochs,
                record_every=50,
                record_spectra_every=100000,  # never compute spectra
                w_star=w_star,
                compute_spectra_fn=dummy_spectra,
                verbose=False,
            )

            B_results[k] = {
                'gamma': gamma,
                'final_test_loss': log['test_loss'][-1],
                'min_test_loss': min(log['test_loss']),
                'final_train_loss': log['train_loss'][-1],
                'test_history': list(zip(log['epoch'], log['test_loss'])),
            }

            print(f"    k={k:3d} (γ={gamma:.2f}) | "
                  f"train={B_results[k]['final_train_loss']:.6f} | "
                  f"test={B_results[k]['min_test_loss']:.6f}")

        all_results[B] = B_results

    # ── Generate Figures ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color map for batch sizes
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(batch_sizes)))

    # Subplot 1: Test error vs γ for each batch size
    for idx, B in enumerate(batch_sizes):
        gammas = [all_results[B][k]['gamma'] for k in sorted(all_results[B].keys())]
        test_losses = [all_results[B][k]['min_test_loss'] for k in sorted(all_results[B].keys())]

        label = f'B={B}' + (' (full-batch)' if B == X_train.shape[0] else '')
        ax1.plot(gammas, test_losses, 'o-', color=colors[idx],
                 markersize=6, linewidth=1.5, label=label, alpha=0.85)

    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.4, label='γ=1')
    ax1.set_xlabel('γ = k/d (over-parameterization ratio)')
    ax1.set_ylabel('Test Loss (MSE)')
    ax1.set_title('Double Descent Curves by Batch Size')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Peak height and recovery rate vs B
    peak_heights = []
    recovery_rates = []
    B_labels = []

    for B in batch_sizes:
        losses = np.array([all_results[B][k]['min_test_loss']
                           for k in sorted(all_results[B].keys())])
        gammas = np.array([all_results[B][k]['gamma']
                           for k in sorted(all_results[B].keys())])

        if len(losses) >= 4:
            # Peak = max near γ=1
            near_threshold = np.where((gammas >= 0.8) & (gammas <= 1.5))[0]
            if len(near_threshold) > 0:
                peak = losses[near_threshold].max()
            else:
                peak = losses.max()

            peak_heights.append(peak)

            # Recovery rate: (post-peak min) / peak
            post_peak = losses[gammas >= 1.0]
            if len(post_peak) > 0:
                recovery = 1.0 - post_peak.min() / peak
            else:
                recovery = 0.0
            recovery_rates.append(recovery)
            B_labels.append(str(B))

    ax2_twin = ax2.twinx()
    bars = ax2.bar(range(len(B_labels)), peak_heights, color='#FF9800', alpha=0.7,
                   edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(B_labels)))
    ax2.set_xticklabels(B_labels)
    ax2.set_xlabel('Batch Size B')
    ax2.set_ylabel('Peak Test Loss (near γ=1)', color='#FF9800')

    ax2_twin.plot(range(len(B_labels)), recovery_rates, 'D-',
                  color='#2196F3', linewidth=2, markersize=8)
    ax2_twin.set_ylabel('Recovery Rate (post-peak descent)', color='#2196F3')
    ax2.set_title('Peak Amplitude & Recovery vs Batch Size')

    fig.suptitle('Experiment 3: Batch Size Modulates Double Descent',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exp3_batch_size_dependence.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: exp3_batch_size_dependence.pdf")

    # Figure 2: Training trajectories for a representative k
    k_rep = k_values[len(k_values) // 2]  # middle k
    fig, ax = plt.subplots(figsize=(10, 5))
    for B in batch_sizes:
        hist = all_results[B][k_rep]['test_history']
        epochs, losses = zip(*hist)
        ax.plot(epochs, losses, linewidth=1, alpha=0.8, label=f'B={B}')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title(f'Training Trajectories: k={k_rep} (γ={k_rep/d:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exp3_trajectories.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: exp3_trajectories.pdf")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 3 SUMMARY")
    print(f"{'='*60}")
    for B in batch_sizes:
        losses = [all_results[B][k]['min_test_loss']
                  for k in sorted(all_results[B].keys())]
        gammas = [all_results[B][k]['gamma']
                  for k in sorted(all_results[B].keys())]
        peak_idx = np.argmax(losses)
        print(f"  B={B:4d}: peak={losses[peak_idx]:.6f} at γ={gammas[peak_idx]:.2f}, "
              f"min_test={min(losses):.6f}")

    return all_results


if __name__ == '__main__':
    results = run_experiment_3(
        d=30,
        n_samples=3000,
        output_dir='./outputs',
    )
