"""Experiment 6: Activation Function Comparison.

Tests whether curvature-noise coupling strength correlates with
nonlinearity degree of the activation function.

Protocol:
1. Train two-layer networks with different activations:
   - Linear (σ(x) = x) — baseline, expected weakest coupling
   - LeakyReLU — mild nonlinearity
   - ReLU — standard nonlinearity
   - GELU — smooth nonlinearity
   - Tanh — saturating nonlinearity
2. For each activation, sweep k through interpolation threshold
3. Measure alignment ratio between H and Σ eigenvectors
4. Key output: alignment ratio vs activation function vs γ

Prediction (H5): Alignment strength increases with nonlinearity degree.
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

from models import ShallowReLUNetwork, generate_teacher_data, eigenvector_alignment
from utils import train_sgd


class TwoLayerNetwork(nn.Module):
    """Flexible two-layer network with configurable activation."""

    def __init__(self, d: int, k: int, activation: str = 'relu'):
        super().__init__()
        self.d = d
        self.k = k

        self.U = nn.Parameter(torch.randn(k, d) * 0.01)
        self.v = nn.Parameter(torch.randn(k, 1) * 0.01)

        if activation == 'linear':
            self.activation = nn.Identity()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.activation_name = activation

    def forward(self, x):
        h = self.activation(x @ self.U.T)  # (batch, k)
        return h @ self.v                   # (batch, 1)

    def compute_hessian_topk(self, X_sample, y_sample, loss_fn, k_eig=20):
        """Compute top-k Hessian eigenvalues via power iteration."""
        params = list(self.parameters())

        def hvp(vec):
            vec = vec.detach()
            y_hat = self.forward(X_sample)
            loss = loss_fn(y_hat, y_sample)
            grad = torch.autograd.grad(loss, params, create_graph=True)
            grad_flat = torch.cat([g.flatten() for g in grad])
            hvp_flat = torch.autograd.grad(
                grad_flat, params, grad_outputs=vec, retain_graph=True
            )
            return torch.cat([h.flatten() for h in hvp_flat])

        n_params = sum(p.numel() for p in params)
        k_eig = min(k_eig, n_params)

        eigvals = []
        eigvecs = []
        v = torch.randn(n_params)

        for _ in range(k_eig):
            for _ in range(10):
                hv = hvp(v)
                v = hv / (hv.norm() + 1e-10)

            hv = hvp(v)
            lam = torch.dot(v, hv)
            eigvals.append(lam.item())
            eigvecs.append(v.clone())

            v = torch.randn(n_params)
            for ev in eigvecs:
                v = v - torch.dot(v, ev) * ev
            v = v / (v.norm() + 1e-10)

        return torch.tensor(eigvals), torch.stack(eigvecs, dim=0)

    def compute_noise_cov_topk(self, X_batch, y_batch, loss_fn, k_eig=20):
        """Estimate top-k eigenvalues/vectors of noise covariance."""
        batch_size = X_batch.shape[0]
        params = list(self.parameters())
        n_params = sum(p.numel() for p in params)
        k_eig = min(k_eig, min(n_params, batch_size))

        grads = []
        for i in range(batch_size):
            self.zero_grad()
            x_i = X_batch[i:i+1]
            y_i = y_batch[i:i+1]
            y_hat = self.forward(x_i)
            loss = loss_fn(y_hat, y_i)
            loss.backward()
            g = torch.cat([p.grad.flatten() for p in params])
            grads.append(g)

        grads = torch.stack(grads, dim=0)
        mean_g = grads.mean(dim=0)
        centered = grads - mean_g
        cov = centered.T @ centered / batch_size

        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            return eigvals[-k_eig:].flip(0), eigvecs[:, -k_eig:].T.flip(0)
        except Exception:
            return torch.zeros(k_eig), torch.zeros(k_eig, n_params)


def run_experiment_6(
    d: int = 15,
    n_samples: int = 1500,
    k_values: list = None,
    activations: list = None,
    lr: float = 0.01,
    batch_size: int = 16,
    n_epochs: int = 1000,
    seed: int = 42,
    output_dir: str = './outputs',
):
    """Activation function comparison experiment."""
    if k_values is None:
        k_values = [int(d * r) for r in [0.5, 1.0, 1.5, 2.0, 3.0]]
        k_values = sorted(set(max(1, k) for k in k_values))

    if activations is None:
        activations = ['linear', 'leaky_relu', 'relu', 'gelu', 'tanh']

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("  EXPERIMENT 6: Activation Function Comparison")
    print(f"  d={d}, n={n_samples}")
    print(f"  Activations: {activations}")
    print(f"  k values: {k_values}")
    print("=" * 60)

    X_train, y_train, X_test, y_test, w_star = generate_teacher_data(
        n_samples, d, seed=seed
    )
    loss_fn = nn.MSELoss()

    # Store results: {activation_name: [{k, gamma, alignment, ...}]}
    all_results = {}

    for act_name in activations:
        print(f"\n{'─'*50}")
        print(f"  Activation: {act_name}")
        print(f"{'─'*50}")

        act_results = []

        for k in k_values:
            gamma = k / d
            n_params = k * d + k
            k_eig = min(15, n_params)

            print(f"    k={k} (γ={gamma:.2f}), n_params={n_params}")

            model = TwoLayerNetwork(d=d, k=k, activation=act_name)

            # Train
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

            # Compute alignment
            hess_eigvals, hess_eigvecs = model.compute_hessian_topk(
                X_train[:200], y_train[:200], loss_fn, k_eig=k_eig
            )
            sigma_eigvals, sigma_eigvecs = model.compute_noise_cov_topk(
                X_train[:500], y_train[:500], loss_fn, k_eig=k_eig
            )

            alignment = eigenvector_alignment(hess_eigvecs, sigma_eigvecs)
            random_baseline = 1.0 / n_params
            alignment_ratio = alignment / random_baseline

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
                'alignment': alignment, 'alignment_ratio': alignment_ratio,
                'eigval_corr': corr,
                'final_test_loss': log['test_loss'][-1],
            })

            print(f"      Alignment={alignment:.6f}, Ratio={alignment_ratio:.1f}x, "
                  f"Eigval corr={corr:.3f}, Test loss={log['test_loss'][-1]:.6f}")

        all_results[act_name] = act_results

    # ── Generate Figures ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Color map for activations
    colors = {
        'linear': '#9E9E9E',
        'leaky_relu': '#FF9800',
        'relu': '#4CAF50',
        'gelu': '#2196F3',
        'tanh': '#9C27B0',
    }

    # Subplot 1: Alignment Ratio vs γ for each activation
    ax = axes[0, 0]
    for act_name in activations:
        gammas = [r['gamma'] for r in all_results[act_name]]
        ratios = [r['alignment_ratio'] for r in all_results[act_name]]
        ax.plot(gammas, ratios, 'o-', color=colors.get(act_name, '#000000'),
                markersize=8, linewidth=2, label=act_name)

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
               label='Random baseline')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4, label='γ=1')
    ax.set_xlabel('γ = k/d')
    ax.set_ylabel('Alignment Ratio (vs random baseline)')
    ax.set_title('H-Σ Eigenvector Alignment by Activation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: Eigenvalue Correlation vs γ
    ax = axes[0, 1]
    for act_name in activations:
        gammas = [r['gamma'] for r in all_results[act_name]]
        corrs = [r['eigval_corr'] for r in all_results[act_name]]
        ax.plot(gammas, corrs, 's-', color=colors.get(act_name, '#000000'),
                markersize=8, linewidth=2, label=act_name)

    ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4)
    ax.set_xlabel('γ = k/d')
    ax.set_ylabel('log(λ_H) vs log(λ_Σ) Correlation')
    ax.set_title('Eigenvalue Spectrum Correlation by Activation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 3: Max alignment ratio by activation (bar chart)
    ax = axes[1, 0]
    max_ratios = []
    avg_ratios = []
    for act_name in activations:
        ratios = [r['alignment_ratio'] for r in all_results[act_name]]
        max_ratios.append(max(ratios))
        avg_ratios.append(np.mean(ratios) if ratios else 0)

    x_pos = np.arange(len(activations))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, max_ratios, width, color='#FF5722', alpha=0.7,
                   edgecolor='black', linewidth=0.5, label='Max')
    bars2 = ax.bar(x_pos + width/2, avg_ratios, width, color='#2196F3', alpha=0.7,
                   edgecolor='black', linewidth=0.5, label='Mean')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(activations, rotation=15)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Alignment Ratio')
    ax.set_title('Max & Mean Alignment Ratio by Activation')
    ax.legend()

    # Subplot 4: Test Error vs γ by activation
    ax = axes[1, 1]
    for act_name in activations:
        gammas = [r['gamma'] for r in all_results[act_name]]
        test_losses = [r['final_test_loss'] for r in all_results[act_name]]
        ax.plot(gammas, test_losses, 'o-', color=colors.get(act_name, '#000000'),
                markersize=8, linewidth=2, label=act_name)

    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.4)
    ax.set_xlabel('γ = k/d')
    ax.set_ylabel('Test Loss (MSE)')
    ax.set_title('Generalization Error by Activation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment 6: Curvature-Noise Coupling vs Activation Function',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'exp6_activation_comparison.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: exp6_activation_comparison.pdf")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  EXPERIMENT 6 SUMMARY")
    print(f"{'='*60}")
    for act_name in activations:
        ratios = [r['alignment_ratio'] for r in all_results[act_name]]
        max_r = max(ratios) if ratios else 0
        avg_r = np.mean(ratios) if ratios else 0
        print(f"  {act_name:>12s}: max ratio={max_r:.1f}x, mean ratio={avg_r:.1f}x")

    # Check H5: Does alignment correlate with nonlinearity?
    # We expect: linear < leaky_relu < relu < gelu < tanh
    linear_max = max(r['alignment_ratio'] for r in all_results.get('linear', [{'alignment_ratio':1}]))
    nonlinear_max = max(
        max(r['alignment_ratio'] for r in all_results.get(a, [{'alignment_ratio':1}]))
        for a in activations if a != 'linear'
    )

    if nonlinear_max > 5 * linear_max:
        print(f"\n  STRONG evidence for H5: Nonlinear activations show >>5x stronger coupling than linear")
    elif nonlinear_max > 2 * linear_max:
        print(f"\n  MODERATE evidence for H5: Nonlinear activations show >2x stronger coupling")
    elif nonlinear_max > linear_max:
        print(f"\n  WEAK evidence for H5: Nonlinear activations slightly stronger")
    else:
        print(f"\n  NO evidence for H5: Coupling strength similar across activations")

    # Sort by mean alignment ratio
    sorted_activations = sorted(activations, key=lambda a: np.mean(
        [r['alignment_ratio'] for r in all_results[a]]
    ) if all_results[a] else 0)

    print(f"\n  Activation ranking (by mean alignment):")
    for i, a in enumerate(sorted_activations):
        mean_r = np.mean([r['alignment_ratio'] for r in all_results[a]]) if all_results[a] else 0
        print(f"    {i+1}. {a}: {mean_r:.1f}x")

    return all_results


if __name__ == '__main__':
    results = run_experiment_6(
        d=15,
        n_samples=1500,
        output_dir='./outputs',
    )
