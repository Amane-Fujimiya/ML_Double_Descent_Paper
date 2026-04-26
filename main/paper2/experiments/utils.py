"""Shared utilities for NESP experiments.

- Hessian spectrum computation
- SGD noise covariance estimation
- Finite-Time Lyapunov Exponent (FTLE) computation
- Training loops and metrics tracking
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Optional, Callable


# ── Training Loop ───────────────────────────────────────────────────────────

def train_sgd(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    lr: float = 0.01,
    batch_size: int = 1,
    n_epochs: int = 1000,
    record_every: int = 20,
    record_spectra_every: int = 100,
    w_star: Optional[torch.Tensor] = None,
    compute_spectra_fn: Optional[Callable] = None,
    verbose: bool = False,
):
    """Train with SGD, tracking loss, Hessian spectrum, and noise covariance.

    Returns a dict of logged quantities at each record step.
    """
    n_train = X_train.shape[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    log = defaultdict(list)
    step = 0

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_train)
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            X_batch = X_shuf[i:i+batch_size]
            y_batch = y_shuf[i:i+batch_size]

            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

        avg_loss = epoch_loss / n_batches

        # Record metrics
        if epoch % record_every == 0:
            with torch.no_grad():
                model.eval()
                y_hat_train = model(X_train)
                train_loss = loss_fn(y_hat_train, y_train).item()
                y_hat_test = model(X_test)
                test_loss = loss_fn(y_hat_test, y_test).item()
                model.train()

            log['epoch'].append(epoch)
            log['train_loss'].append(train_loss)
            log['test_loss'].append(test_loss)
            log['step'].append(step)

            if w_star is not None and hasattr(model, 'effective_weight'):
                w_eff = model.effective_weight()
                alignment = (w_eff.T @ w_star).item() / (
                    w_eff.norm().item() * w_star.norm().item() + 1e-10
                )
                log['teacher_alignment'].append(alignment)

            if verbose and epoch % 200 == 0:
                print(f"  Epoch {epoch:5d} | Train: {train_loss:.6f} | Test: {test_loss:.6f}")

        # Compute spectra (expensive, do infrequently)
        if compute_spectra_fn is not None and epoch % record_spectra_every == 0:
            spec_data = compute_spectra_fn(model, X_train, y_train)
            log['spectra_epoch'].append(epoch)
            for key, val in spec_data.items():
                log[key].append(val)

    return dict(log)


def train_sgd_erosion(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    lr: float = 0.01,
    batch_size: int = 1,
    n_epochs_converge: int = 2000,
    n_epochs_extended: int = 10000,
    record_every: int = 100,
    convergence_threshold: float = 1e-8,
    verbose: bool = False,
):
    """Extended training for equilibrium erosion experiment (Experiment 5).

    Trains to convergence, then continues for extended period.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_train = X_train.shape[0]

    log = defaultdict(list)

    total_epochs = n_epochs_converge + n_epochs_extended
    convergence_epoch = None

    for epoch in range(total_epochs):
        perm = torch.randperm(n_train)
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        for i in range(0, n_train, batch_size):
            X_batch = X_shuf[i:i+batch_size]
            y_batch = y_shuf[i:i+batch_size]

            optimizer.zero_grad()
            y_hat = model(X_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            optimizer.step()

        # Record
        if epoch % record_every == 0:
            with torch.no_grad():
                model.eval()
                train_loss = loss_fn(model(X_train), y_train).item()
                test_loss = loss_fn(model(X_test), y_test).item()
                model.train()

            log['epoch'].append(epoch)
            log['train_loss'].append(train_loss)
            log['test_loss'].append(test_loss)

            # Track weight movement
            if hasattr(model, 'U'):
                w = model.U.data.clone().flatten()
                if 'weight_at_convergence' not in log:
                    log['weight_norm'].append(w.norm().item())
                else:
                    w0 = log['weight_at_convergence']
                    log['weight_displacement'].append(
                        (w - w0).norm().item()
                    )

            # Mark convergence
            if convergence_epoch is None and train_loss < convergence_threshold:
                convergence_epoch = epoch
                if hasattr(model, 'U'):
                    log['weight_at_convergence'] = model.U.data.clone().flatten()
                if verbose:
                    print(f"  Converged at epoch {epoch}, starting extended monitoring")

    log['convergence_epoch'] = convergence_epoch
    return dict(log)


# ── Hessian Utilities ────────────────────────────────────────────────────────

def compute_hessian_eigenvalues(model: nn.Module, loss_fn, X, y):
    """Compute exact Hessian eigenvalues (small models only).

    Uses torch.autograd to build the full Hessian.
    Only practical for d_total < ~1000.
    """
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)

    y_hat = model(X)
    loss = loss_fn(y_hat, y)

    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grads])

    # Build Hessian row by row
    H = torch.zeros(n_params, n_params)
    for i in range(n_params):
        h_i = torch.autograd.grad(
            grad_flat[i], params, retain_graph=(i < n_params - 1)
        )
        H[i] = torch.cat([h.flatten() for h in h_i])

    eigvals = torch.linalg.eigvalsh(H)
    return eigvals


def compute_hessian_trace_diag(model: nn.Module, loss_fn, X, y):
    """Estimate Tr(H) and diag(H) using Hutchinson's estimator.

    Tr(H) ≈ (1/m) Σ_k v_k^T H v_k where v_k ~ N(0,I).
    """
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)

    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grads])

    # Hutchinson trace estimator: single sample
    v = torch.randn(n_params)
    hv = torch.autograd.grad(grad_flat, params, grad_outputs=v, retain_graph=True)
    hv_flat = torch.cat([h.flatten() for h in hv])

    trace_est = torch.dot(v, hv_flat).item()
    return trace_est


# ── Noise Covariance Utilities ───────────────────────────────────────────────

def estimate_noise_covariance_eigenvalues(model, X, y, loss_fn, n_grad_samples=500):
    """Estimate noise covariance eigenvalues from per-sample gradient variance.

    For SGD noise: ξ = g - ∇L, Σ = E[ξ ⊗ ξ] ≈ E[g ⊗ g] (near optimum).

    Returns eigenvalues of the estimated covariance matrix.
    """
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    n_data = min(n_grad_samples, X.shape[0])

    # Collect per-sample gradients
    grad_list = []
    for i in range(n_data):
        model.zero_grad()
        x_i = X[i:i+1]
        y_i = y[i:i+1]
        y_hat = model(x_i)
        loss = loss_fn(y_hat, y_i)
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in params])
        grad_list.append(g.detach())

    grads = torch.stack(grad_list, dim=0)  # (n_data, n_params)
    mean_g = grads.mean(dim=0, keepdim=True)
    centered = grads - mean_g

    # Σ ≈ (1/n) Σ_i g_i g_i^T  (near optimum, E[g] ≈ 0)
    # Use SVD for eigenvalues (more stable for rank-deficient matrices)
    try:
        _, S, _ = torch.linalg.svd(centered / np.sqrt(n_data - 1), full_matrices=False)
        eigvals = S ** 2
    except Exception:
        eigvals = torch.zeros(min(n_data, n_params))

    return eigvals


# ── Finite-Time Lyapunov Exponent (FTLE) ─────────────────────────────────────

def compute_max_ftle(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    lr: float = 0.01,
    batch_size: int = 1,
    n_steps: int = 500,
    epsilon: float = 1e-6,
    renormalize_every: int = 1,
):
    """Compute maximal Finite-Time Lyapunov Exponent.

    Algorithm from Section 7.X of paper2a.tex:
    1. Evolve reference trajectory W_t via SGD
    2. Evolve shadow W̃_t = W_t + ε·u_t (same mini-batches!)
    3. Renormalize after each step
    4. Λ_max ≈ (1/T) Σ_t ln(||W̃_t - W_t|| / ε)
    """
    n_train = X_train.shape[0]
    params = list(model.parameters())

    # Deep copy for shadow model
    import copy
    shadow_model = copy.deepcopy(model)
    shadow_params = list(shadow_model.parameters())

    # Perturb shadow
    for p, sp in zip(params, shadow_params):
        direction = torch.randn_like(p)
        direction /= direction.norm()
        sp.data = p.data + epsilon * direction

    log_divergence = []

    for step in range(n_steps):
        # Same mini-batch for both trajectories
        idx = torch.randint(0, n_train, (batch_size,))
        X_batch, y_batch = X_train[idx], y_train[idx]

        # --- Reference step ---
        model.zero_grad()
        loss_ref = nn.MSELoss()(model(X_batch), y_batch)
        loss_ref.backward()
        for p in params:
            p.data -= lr * p.grad

        # --- Shadow step (same batch!) ---
        shadow_model.zero_grad()
        loss_shadow = nn.MSELoss()(shadow_model(X_batch), y_batch)
        loss_shadow.backward()
        for sp, p in zip(shadow_params, params):
            sp.data -= lr * sp.grad  # same lr, same batch gradient

        # Renormalize
        if step % renormalize_every == 0:
            diff = 0.0
            for p, sp in zip(params, shadow_params):
                diff += (sp.data - p.data).pow(2).sum()
            diff_norm = diff.sqrt().item()

            log_divergence.append(np.log(diff_norm / epsilon + 1e-30))

            # Rescale shadow
            for p, sp in zip(params, shadow_params):
                direction = (sp.data - p.data) / (diff_norm + 1e-30)
                sp.data = p.data + epsilon * direction

    ftle = np.mean(log_divergence) if log_divergence else 0.0
    return ftle, log_divergence


# ── Condition Number Utilities ───────────────────────────────────────────────

def compute_condition_number(eigvals):
    """λ_max / max(λ_min, eps)."""
    pos = eigvals[eigvals > 1e-10]
    if len(pos) == 0:
        return float('inf')
    return (eigvals.max() / pos.min()).item()


def compute_effective_rank(eigvals, threshold=0.99):
    """Number of eigenvalues needed to explain `threshold` fraction of trace."""
    total = eigvals.sum().item()
    if total == 0:
        return 0
    sorted_eig = eigvals.sort(descending=True).values
    cumsum = sorted_eig.cumsum(0)
    rank = (cumsum / total < threshold).sum().item() + 1
    return rank


# ── Pretty-print helpers ─────────────────────────────────────────────────────

def summarize_experiment(log, experiment_name=""):
    """Print a summary of experiment results."""
    if experiment_name:
        print(f"\n{'='*60}")
        print(f"  {experiment_name}")
        print(f"{'='*60}")

    if 'train_loss' in log and len(log['train_loss']) > 0:
        final_train = log['train_loss'][-1]
        final_test = log['test_loss'][-1]
        min_test = min(log['test_loss'])
        print(f"  Final train loss: {final_train:.8f}")
        print(f"  Final test loss:  {final_test:.8f}")
        print(f"  Min test loss:    {min_test:.8f}")

    if 'convergence_epoch' in log and log['convergence_epoch'] is not None:
        print(f"  Converged at epoch: {log['convergence_epoch']}")

    return log
