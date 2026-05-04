"""Lyapunov Spectrum and Epoch-wise Double Descent (Cluster 4).

Extends the FTLE computation in utils.py to compute the full Lyapunov spectrum
using shadow trajectories and orthogonalization (Benettin algorithm).

Core algorithm:
1. Evolve reference trajectory W_t via SGD
2. Maintain k shadow vectors u^{(i)}_t, each perturbed by ε
3. After each step, Gram-Schmidt orthogonalize the shadow displacements
4. Lyapunov exponents: λ_i = (1/T) Σ_t log(||ũ^{(i)}_t|| / ε)

Connection to double descent:
- λ_1 (max FTLE) should peak near γ=1 (when training dynamics are most chaotic)
- The FTLE spectrum width should correlate with Sharpness Ratio R_H
- Epoch-wise DD can be understood as the temporal evolution of λ_i(t)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
import copy


def compute_lyapunov_spectrum(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    lr: float = 0.01,
    batch_size: int = 1,
    n_steps: int = 1000,
    epsilon: float = 1e-6,
    renormalize_every: int = 1,
    n_exponents: int = 5,
    orthogonalize_every: int = 1,
    verbose: bool = False,
) -> Tuple[List[float], List[List[float]]]:
    """Compute the full FTLE spectrum using the Benettin algorithm.

    Algorithm (Benettin et al., 1980):
    1. Initialize k orthonormal perturbation vectors u^{(i)}_0
    2. Evolve reference + k shadows with SAME mini-batches
    3. After each step, Gram-Schmidt orthogonalize the displacement vectors
    4. λ_i = (1/T) Σ_t log(norm of i-th orthogonalized vector)

    Args:
        model: The neural network model
        X_train, y_train: Training data
        lr: Learning rate
        batch_size: SGD batch size
        n_steps: Number of SGD steps for FTLE estimation
        epsilon: Initial perturbation magnitude
        renormalize_every: Renormalize every N steps
        n_exponents: Number of Lyapunov exponents to compute
        orthogonalize_every: Gram-Schmidt every N steps

    Returns:
        (exponents, log_divergence_history)
            exponents: list of λ_1, λ_2, ..., λ_k (sorted descending)
            log_divergence_history: list per exponent of log(||u_i||/ε) values
    """
    n_train = X_train.shape[0]
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    n_exponents = min(n_exponents, n_params)

    # Flattened parameter view for reference model
    def get_flat_params(m):
        return torch.cat([p.data.flatten() for p in m.parameters()])

    def set_flat_params(m, flat):
        offset = 0
        for p in m.parameters():
            n = p.numel()
            p.data = flat[offset:offset+n].reshape(p.shape).clone()
            offset += n

    # Initialize k orthonormal perturbation vectors
    perturbation_vecs = []
    for i in range(n_exponents):
        v = torch.randn(n_params)
        for j in range(i):
            v -= torch.dot(v, perturbation_vecs[j]) * perturbation_vecs[j]
        v = v / (v.norm() + 1e-10)
        perturbation_vecs.append(v)

    # Log divergence accumulators
    log_div_accum = [[] for _ in range(n_exponents)]
    shadow_models = [copy.deepcopy(model) for _ in range(n_exponents)]

    ref_params_flat = get_flat_params(model)

    for step in range(n_steps):
        # Same mini-batch for ALL trajectories
        idx = torch.randint(0, n_train, (batch_size,))
        X_batch, y_batch = X_train[idx], y_train[idx]
        loss_fn = nn.MSELoss()

    # --- Reference step ---
    model.zero_grad()
    loss_ref = loss_fn(model(X_batch), y_batch)
    loss_ref.backward()

    # Update reference using its own gradient
    for p in params:
        if p.grad is not None:
            p.data -= lr * p.grad

    # --- Shadow steps with OWN gradients (same mini-batch) ---
    # CRITICAL: Each shadow computes its OWN gradient on the SAME mini-batch,
    # NOT copying the reference gradient. The shadow must evolve freely
    # under the same data stream to capture trajectory divergence.
    for i, sm in enumerate(shadow_models):
        sm.zero_grad()
        loss_s = loss_fn(sm(X_batch), y_batch)
        loss_s.backward()

        for p in sm.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad

        # --- Orthogonalize and renormalize ---
        if step % renormalize_every == 0:
            ref_flat = get_flat_params(model)
            displacements = []

            for i, sm in enumerate(shadow_models):
                sm_flat = get_flat_params(sm)
                disp = sm_flat - ref_flat
                displacements.append(disp)

            # Gram-Schmidt orthogonalization
            for i in range(n_exponents):
                for j in range(i):
                    # Remove projection onto previous vectors
                    proj = torch.dot(displacements[i], displacements[j])
                    displacements[i] -= proj * displacements[j]

                norm_i = displacements[i].norm() + 1e-30
                log_div_accum[i].append(np.log(norm_i.item() / epsilon))

                # Normalize
                displacements[i] = displacements[i] / (norm_i + 1e-30)

                # Rescale to epsilon
                set_flat_params(shadow_models[i],
                                ref_flat + epsilon * displacements[i])

        if verbose and step % 200 == 0:
            current_exps = [np.mean(logs[-min(50, len(logs)):]) if logs else 0
                           for logs in log_div_accum]
            print(f"  Step {step:5d} | FTLE: " +
                  " ".join([f"λ{i+1}={e:.4f}" for i, e in enumerate(current_exps)]))

    # Final exponents: time average
    exponents = [np.mean(logs) if logs else 0.0 for logs in log_div_accum]

    return exponents, log_div_accum


def compute_epochwise_ftle(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    lr: float = 0.01,
    batch_size: int = 1,
    n_epochs: int = 100,
    n_ftle_steps_per_window: int = 100,
    epsilon: float = 1e-6,
    n_exponents: int = 3,
) -> Dict[str, List[float]]:
    """Compute FTLE spectrum as a function of epoch to observe epoch-wise DD.

    Uses sliding windows: at each recording epoch, compute FTLE over
    the next n_ftle_steps_per_window steps without affecting the
    reference trajectory (snapshot-based computation).

    Returns:
        Dict with keys: 'epoch', 'ftle_1', 'ftle_2', ..., 'ftle_k',
        'spectrum_width' (λ_1 - λ_k), 'kaplan_yorke_dim'
    """
    n_train = X_train.shape[0]
    n_exponents = min(n_exponents, sum(p.numel() for p in model.parameters()))

    results = {
        'epoch': [],
    }
    for i in range(n_exponents):
        results[f'ftle_{i+1}'] = []
    results['spectrum_width'] = []
    results['kaplan_yorke_dim'] = []

    for epoch in range(n_epochs):
        # Snapshot the current model state
        snap_model = copy.deepcopy(model)

        # Compute FTLE over a short window from this snapshot
        exponents, _ = compute_lyapunov_spectrum(
            snap_model, X_train, y_train,
            lr=lr, batch_size=batch_size,
            n_steps=n_ftle_steps_per_window,
            epsilon=epsilon,
            n_exponents=n_exponents,
            renormalize_every=1,
            orthogonalize_every=1,
            verbose=False,
        )

        results['epoch'].append(epoch)
        for i, lam in enumerate(exponents):
            results[f'ftle_{i+1}'].append(lam)

        # Spectrum width
        results['spectrum_width'].append(exponents[0] - exponents[-1])

        # Kaplan-Yorke dimension
        ky_dim = _kaplan_yorke_dim(exponents)
        results['kaplan_yorke_dim'].append(ky_dim)

        # Advance reference model one epoch
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            X_batch = X_train[perm[i:i+batch_size]]
            y_batch = y_train[perm[i:i+batch_size]]
            model.zero_grad()
            y_hat = model(X_batch)
            loss = nn.MSELoss()(y_hat, y_batch)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    p.data -= lr * p.grad

    return results


def _kaplan_yorke_dim(exponents: List[float]) -> float:
    """Compute Kaplan-Yorke (Lyapunov) dimension.

    D_KY = j + Σ_{i=1}^{j} λ_i / |λ_{j+1}|
    where j is the largest integer such that Σ_{i=1}^{j} λ_i ≥ 0.
    """
    sorted_lam = sorted(exponents, reverse=True)
    cumsum = 0.0
    j = 0
    for i, lam in enumerate(sorted_lam):
        cumsum += lam
        if cumsum >= 0:
            j = i
    if j < len(sorted_lam) - 1 and abs(sorted_lam[j+1]) > 1e-10:
        return float(j + 1 + sum(sorted_lam[:j+1]) / abs(sorted_lam[j+1]))
    else:
        return float(j + 1)


def compute_ftle_sharpness_correlation(
    model_factory,
    X_train, y_train,
    k_values: List[int],
    d: int,
    lr: float = 0.01,
    batch_size: int = 1,
    n_epochs: int = 500,
    n_ftle_steps: int = 200,
    n_exponents: int = 5,
    seed: int = 42,
) -> Dict[str, List]:
    """Compute correlation between FTLE spectrum and Sharpness Ratio.

    For each width k:
    1. Train model to convergence
    2. Measure Tr(H) and compute Sharpness Ratio contribution
    3. Compute FTLE spectrum
    4. Record correlation

    Returns:
        Dict with 'gamma', 'R_H_contrib', 'ftle_max', 'ftle_spectrum_width',
        'kaplan_yorke_dim'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = {
        'gamma': [], 'k': [], 'tr_H': [],
        'ftle_max': [], 'ftle_spectrum_width': [],
        'kaplan_yorke_dim': [],
    }

    for k in k_values:
        gamma = k / d
        model = model_factory(d=d, k=k)

        # Train to convergence
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        n_train = X_train.shape[0]

        for epoch in range(n_epochs):
            perm = torch.randperm(n_train)
            for i in range(0, n_train, batch_size):
                X_batch = X_train[perm[i:i+batch_size]]
                y_batch = y_train[perm[i:i+batch_size]]
                optimizer.zero_grad()
                loss = loss_fn(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        # Measure Tr(H) using Hutchinson
        from .run_exp7_heterogeneity import compute_hessian_trace
        tr_H = compute_hessian_trace(model, X_train[:200], y_train[:200], loss_fn)

        # Measure FTLE
        exponents, _ = compute_lyapunov_spectrum(
            model, X_train, y_train,
            lr=lr, batch_size=batch_size,
            n_steps=n_ftle_steps,
            epsilon=1e-6,
            n_exponents=n_exponents,
            renormalize_every=1,
        )

        results['gamma'].append(gamma)
        results['k'].append(k)
        results['tr_H'].append(tr_H)
        results['ftle_max'].append(exponents[0] if exponents else 0)
        results['ftle_spectrum_width'].append(
            exponents[0] - exponents[-1] if len(exponents) >= 2 else 0
        )
        results['kaplan_yorke_dim'].append(_kaplan_yorke_dim(exponents))

    return results
