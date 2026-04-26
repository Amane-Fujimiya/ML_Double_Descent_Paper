"""Experiment 2: Escape Time Measurement.

Quantifies the relationship between curvature and escape time,
testing the "survival of the flattest" mechanism.

Protocol:
1. Create linear teacher-student model at interpolation threshold (k ≈ d)
2. Train to a "sharp" minimum using very small learning rate (η_small)
3. Restart with normal learning rate, measure time until escape
4. Repeat for "flat" minima (train with large η or add noise)
5. Key output: τ_escape vs λ_max(H) showing inverse relationship

Prediction from NESP:
  τ ∼ exp(ΔE / T_eff),  where T_eff ∝ η/B · λ_max
  → log(τ) should decrease with λ_max
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
    summarize_experiment,
)


def create_sharp_minimum(model, X_train, y_train, lr_small=0.001, n_epochs=3000):
    """Train to a sharp minimum using very small learning rate."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_small)
    loss_fn = torch.nn.MSELoss()
    n_train = X_train.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 1):  # B=1 for SGD
            X_batch = X_train[perm[i:i+1]]
            y_batch = y_train[perm[i:i+1]]
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    return model


def create_flat_minimum(model, X_train, y_train, lr_large=0.05, n_epochs=3000):
    """Train to a (relatively) flat minimum using larger learning rate.

    Larger LR + SGD noise pushes the system toward flatter basins.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_large)
    loss_fn = torch.nn.MSELoss()
    n_train = X_train.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 1):
            X_batch = X_train[perm[i:i+1]]
            y_batch = y_train[perm[i:i+1]]
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    return model


def measure_escape_time(model, X_train, y_train, X_test, y_test,
                        lr=0.01, escape_threshold=0.01, max_epochs=5000,
                        record_every=10):
    """Measure epochs until test loss drops below escape_threshold.

    If the model is already at a sharp minimum (high test loss),
    we train with normal lr and count epochs until escape.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    n_train = X_train.shape[0]

    test_loss_history = []
    escape_epoch = None

    for epoch in range(max_epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 1):
            X_batch = X_train[perm[i:i+1]]
            y_batch = y_train[perm[i:i+1]]
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        if epoch % record_every == 0:
            with torch.no_grad():
                model.eval()
                test_loss = loss_fn(model(X_test), y_test).item()
                model.train()
            test_loss_history.append((epoch, test_loss))

            if escape_epoch is None and test_loss < escape_threshold:
                escape_epoch = epoch

    return escape_epoch, test_loss_history


def run_experiment_2(
    d: int = 20,
    n_samples: int = 2000,
    k: int = 20,  # at interpolation threshold
    n_trials: int = 8,
    lr_values: list = None,
    seed: int = 42,
    output_dir: str = './outputs',
):
    """Run escape time experiment."""
    if lr_values is None:
        lr_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  EXPERIMENT 2: Escape Time Measurement")
    print(f"  Linear Teacher-Student (d={d}, k={k}, γ={k/d:.1f})")
    print("=" * 60)

    X_train, y_train, X_test, y_test, w_star = generate_teacher_data(
        n_samples, d, seed=seed
    )
    loss_fn = torch.nn.MSELoss()

    results = []
    all_histories = []

    for trial, lr_init in enumerate(lr_values):
        print(f"\n--- Trial {trial+1}/{len(lr_values)}: initial LR = {lr_init} ---")

        # Phase 1: Create minimum with given LR
        model = LinearTeacherStudent(d=d, k=k)
        model = create_sharp_minimum(model, X_train, y_train,
                                     lr_small=lr_init, n_epochs=3000)

        # Measure sharpness (λ_max of Hessian)
        with torch.no_grad():
            hess_eig = model.compute_hessian(X_train, y_train)
            lambda_max = hess_eig.max().item()

        # Measure initial test loss
        with torch.no_grad():
            test_loss_initial = loss_fn(model(X_test), y_test).item()

        print(f"  λ_max(H) = {lambda_max:.6f}  |  Initial test loss = {test_loss_initial:.6f}")

        # Phase 2: Measure escape time with normal LR
        escape_epoch, hist = measure_escape_time(
            model, X_train, y_train, X_test, y_test,
            lr=0.01,
            escape_threshold=0.05,
            max_epochs=5000,
        )

        results.append({
            'init_lr': lr_init,
            'lambda_max': lambda_max,
            'initial_test_loss': test_loss_initial,
            'escape_epoch': escape_epoch,
            'escaped': escape_epoch is not None,
        })
        all_histories.append(hist)

        status = f"escaped at epoch {escape_epoch}" if escape_epoch else "DID NOT ESCAPE"
        print(f"  → {status}")

    # ── Generate Figures ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot 1: Escape time vs λ_max
    ax = axes[0]
    lmax_vals = [r['lambda_max'] for r in results]
    escape_vals = [r['escape_epoch'] if r['escape_epoch'] else 5000 for r in results]
    escaped_flags = [r['escaped'] for r in results]

    colors = ['#4CAF50' if e else '#F44336' for e in escaped_flags]
    ax.scatter(lmax_vals, escape_vals, c=colors, s=80, edgecolors='black', linewidth=0.5, zorder=5)

    # Fit line for escaped points
    escaped_lmax = [lmax_vals[i] for i in range(len(lmax_vals)) if escaped_flags[i]]
    escaped_epochs = [escape_vals[i] for i in range(len(lmax_vals)) if escaped_flags[i]]
    if len(escaped_lmax) >= 3:
        coeffs = np.polyfit(np.log(escaped_lmax), np.log(escaped_epochs), 1)
        x_fit = np.logspace(np.log10(min(lmax_vals)*0.8), np.log10(max(lmax_vals)*1.2), 50)
        y_fit = np.exp(coeffs[1]) * x_fit ** coeffs[0]
        ax.plot(x_fit, y_fit, 'k--', linewidth=1,
                label=f'τ ∝ λ_max^{coeffs[0]:.2f}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('λ_max(H) — Maximum Hessian eigenvalue')
    ax.set_ylabel('Escape Time (epochs)')
    ax.set_title('Escape Time vs Curvature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: Initial LR vs λ_max (progressive sharpening signature)
    ax = axes[1]
    ax.plot(lr_values, lmax_vals, 'o-', color='#FF5722', markersize=8, linewidth=2)
    ax.set_xlabel('Initial Learning Rate')
    ax.set_ylabel('λ_max(H) at Convergence')
    ax.set_title('Smaller LR → Sharper Minima')
    ax.grid(True, alpha=0.3)

    # Subplot 3: Example escape trajectories
    ax = axes[2]
    for i, hist in enumerate(all_histories[:4]):  # first 4 trials
        epochs, losses = zip(*hist)
        ax.plot(epochs, losses, linewidth=1, alpha=0.8,
                label=f'init LR={lr_values[i]:.3f}')

    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='escape threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Escape Trajectories (selected trials)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 2: Escape Time and Curvature-Noise Coupling',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exp2_escape_time.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: exp2_escape_time.pdf")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 2 SUMMARY")
    print(f"{'='*60}")
    print(f"{'Init LR':>10s} {'λ_max(H)':>12s} {'Escape Epoch':>14s} {'Status':>12s}")
    print(f"{'-'*50}")
    for r in results:
        esc_str = str(r['escape_epoch']) if r['escaped'] else 'NOT ESCAPED'
        print(f"{r['init_lr']:10.4f} {r['lambda_max']:12.6f} {esc_str:>14s} {'✓' if r['escaped'] else '✗':>12s}")

    if len(escaped_lmax) >= 3:
        print(f"\n  Power-law fit: τ ∝ λ_max^{coeffs[0]:.2f}")
        print(f"  (NESP predicts negative exponent — sharper minima escape faster)")

    return results


if __name__ == '__main__':
    results = run_experiment_2(
        d=20,
        n_samples=2000,
        k=20,
        lr_values=[0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05],
        output_dir='./outputs',
    )
