"""Scaled-up experimental framework for NESP submission-grade results.

Supports:
- d=50-100, n=5000-10000 regime
- Multi-seed averaging with bootstrap CIs
- Pseudo-inverse (equilibrium) baseline
- Fisher Information estimation
- Full observables: Tr(H), Tr(Sigma), alignment, Sharpness Ratio, FTLE
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Optional, List, Dict, Any, Callable, Tuple
import time
import warnings

# ── Bootstrap & Statistics ──────────────────────────────────────────────────

def bootstrap_ci(data: List[float], n_bootstrap: int = 2000, alpha: float = 0.05,
                 statistic_fn: Callable = np.mean) -> Tuple[float, float, float, float]:
    """Compute bootstrap confidence interval and standard error.

    Returns (estimate, std_err, ci_lower, ci_upper).
    """
    if len(data) < 3:
        return statistic_fn(data), 0.0, data, data
    data = np.array(data)
    estimate = statistic_fn(data)
    boot_stats = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(data), size=len(data), replace=True)
        boot_stats.append(statistic_fn(data[idx]))
    boot_stats = np.array(boot_stats)
    std_err = np.std(boot_stats)
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return estimate, std_err, ci_lower, ci_upper


def spearman_p_value(rho: float, n: int) -> float:
    """Approximate p-value for Spearman's rho using t-distribution."""
    if n <= 2:
        return 1.0
    from scipy import stats
    t_stat = rho * np.sqrt((n - 2) / (1 - rho ** 2 + 1e-10))
    return 2 * stats.t.sf(abs(t_stat), df=n - 2)


# ── Pseudo-inverse Baseline ────────────────────────────────────────────────

class PseudoInverseBaseline:
    """Equilibrium solution: linear least-squares via pseudoinverse.

    For a linear model f(x) = W x, the minimum-norm least-squares solution
    is W_pinv = Y X^T (X X^T)^†.  This represents the equilibrium (zero-noise)
    solution that SGD approaches in the limit t → ∞.

    Provides:
    - Equilibrium test error
    - Equilibrium Hessian eigenvalues
    - Sharpness Ratio for the equilibrium landscape
    """

    def __init__(self, X_train: torch.Tensor, y_train: torch.Tensor,
                 X_test: torch.Tensor, y_test: torch.Tensor, d: int):
        self.X_train = X_train.numpy().astype(np.float64)
        self.y_train = y_train.numpy().astype(np.float64)
        self.X_test = X_test.numpy().astype(np.float64)
        self.y_test = y_test.numpy().astype(np.float64)
        self.d = d
        self.n = X_train.shape[0]

    def solve(self, k: int, reg: float = 1e-10) -> Dict[str, float]:
        """Compute pseudoinverse solution for width k.

        The model is y = v^T U x where U ∈ R^{k×d}, v fixed = 1/√k.
        Effective weight: w_eff = U^T v ∈ R^d.

        Under the fixed-v parameterization, the optimization is on w_eff directly:
        min_w ||X w - y||² → w* = X^† y

        Hessian of this problem: H = X^T X / n (constant, independent of k).
        """
        X = self.X_train
        y = self.y_train
        n, d = X.shape

        # Pseudoinverse: w* = (X^T X + λI)^{-1} X^T y
        import scipy.linalg as la
        XtX = X.T @ X / n
        XtX_reg = XtX + reg * np.eye(d)
        Xty = X.T @ y / n
        w_star = la.solve(XtX_reg, Xty, assume_a='pos')

        # Test error
        y_pred_test = self.X_test @ w_star
        test_mse = np.mean((y_pred_test - self.y_test) ** 2)
        train_mse = np.mean((X @ w_star - y) ** 2)

        # Hessian eigenvalues: σ_i(X^T X / n)
        eigvals = la.eigvalsh(XtX)
        tr_H = np.sum(eigvals)
        lambda_max = np.max(eigvals)
        lambda_min = np.min(eigvals[eigvals > 1e-12]) if np.any(eigvals > 1e-12) else 1e-12
        kappa = lambda_max / lambda_min

        return {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'w_norm': float(np.linalg.norm(w_star)),
            'tr_H': float(tr_H),
            'lambda_max_H': float(lambda_max),
            'kappa_H': float(kappa),
            'eigvals': eigvals,
            'eff_params': k,  # effective: w ∈ R^d, projected from U via v
        }


def compute_pseudoinverse_sharpness_ratio(X, y, d, k_values, reg=1e-10):
    """Compute Sharpness Ratio for the equilibrium (pseudoinverse) landscape.

    R_H = Tr(H)_{γ<1} / Tr(H)_{γ>2}
    For the linear model with fixed v, H = X^T X / n is independent of k,
    so R_H ≡ 1 for the pseudo-inverse baseline.
    """
    baseline = PseudoInverseBaseline(X, y, X, y, d)
    results = {}
    for k in k_values:
        gamma = k / d
        res = baseline.solve(k, reg)
        results[gamma] = res
    return results


# ── Fisher Information Estimation ───────────────────────────────────────────

def estimate_fisher_information(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                                 loss_fn: Callable, n_mc: int = 100) -> Tuple[float, np.ndarray]:
    """Estimate Fisher Information Matrix trace and diagonal eigenvalues.

    F = E_{y~p(·|x)} [∇log p(y|x) ∇log p(y|x)^T]
    For MSE loss with Gaussian noise: F = H (the Hessian), since
    log p(y|x) ∝ -(y - f(x))² / (2σ²).

    We use Monte Carlo sampling of the output distribution.
    For regression with fixed variance, F = H (empirical Fisher = true Fisher).
    """
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)

    y_hat = model(X)
    loss = loss_fn(y_hat, y)

    # For MSE, F = H. We compute H trace via Hutchinson.
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grads])

    # Hutchinson trace estimator with m samples
    trace_estimates = []
    for _ in range(min(n_mc, 10)):
        v = torch.randn(n_params)
        hv = torch.autograd.grad(grad_flat, params, grad_outputs=v, retain_graph=True)
        hv_flat = torch.cat([h.flatten() for h in hv])
        trace_estimates.append(torch.dot(v, hv_flat).item())

    trace_F = np.mean(trace_estimates)
    trace_std = np.std(trace_estimates)
    return trace_F, trace_std, np.array(trace_estimates)


# ── Extended Training Loop with Multi-Seed Support ──────────────────────────

def train_sgd_multiseed(
    model_factory: Callable[[], nn.Module],
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    lr: float = 0.01,
    batch_size: int = 1,
    n_epochs: int = 5000,
    n_seeds: int = 5,
    base_seed: int = 42,
    record_every: int = 200,
    compute_metrics_fn: Optional[Callable] = None,
    compute_metrics_every: int = 1000,
    verbose: bool = False,
    early_stop_loss: Optional[float] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """Train with multiple seeds, aggregating results with bootstrap CIs.

    Returns aggregated log with mean ± ci for each metric.
    """
    all_logs = []
    convergence_epochs = []
    training_times = []

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx * 100
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_factory()
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
            X_train_d = X_train.cuda()
            y_train_d = y_train.cuda()
            X_test_d = X_test.cuda()
            y_test_d = y_test.cuda()
        else:
            X_train_d, y_train_d = X_train, y_train
            X_test_d, y_test_d = X_test, y_test

        t0 = time.time()
        log = _train_sgd_single(
            model, X_train_d, y_train_d, X_test_d, y_test_d,
            lr=lr, batch_size=batch_size, n_epochs=n_epochs,
            record_every=record_every,
            compute_metrics_fn=compute_metrics_fn,
            compute_metrics_every=compute_metrics_every,
            verbose=verbose and seed_idx == 0,
            early_stop_loss=early_stop_loss,
        )
        t1 = time.time()
        training_times.append(t1 - t0)
        all_logs.append(log)
        if 'convergence_epoch' in log:
            convergence_epochs.append(log['convergence_epoch'])

    # Aggregate epoch-aligned metrics
    aggregated = _aggregate_logs(all_logs, n_seeds)

    if convergence_epochs:
        aggregated['mean_convergence_epoch'] = np.mean(convergence_epochs)
        aggregated['std_convergence_epoch'] = np.std(convergence_epochs)

    aggregated['training_time_mean'] = np.mean(training_times)
    aggregated['training_time_std'] = np.std(training_times)
    aggregated['all_logs'] = all_logs

    return aggregated


def _train_sgd_single(model, X_train, y_train, X_test, y_test,
                      lr, batch_size, n_epochs, record_every,
                      compute_metrics_fn, compute_metrics_every,
                      verbose, early_stop_loss):
    """Single-seed training loop."""
    n_train = X_train.shape[0]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    log = defaultdict(list)
    convergence_epoch = None

    for epoch in range(n_epochs):
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

        avg_loss = epoch_loss / n_batches

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
            log['batch_loss'].append(avg_loss)

        if compute_metrics_fn is not None and epoch % compute_metrics_every == 0:
            metrics = compute_metrics_fn(model, X_train, y_train, loss_fn)
            for key, val in metrics.items():
                log.setdefault(key, []).append(
                    (epoch, val) if not isinstance(val, tuple) else val
                )

        if early_stop_loss is not None and avg_loss < early_stop_loss and convergence_epoch is None:
            convergence_epoch = epoch

        if verbose and epoch % (record_every * 5) == 0:
            print(f"  Epoch {epoch:5d} | Train: {log['train_loss'][-1]:.6e} | "
                  f"Test: {log['test_loss'][-1]:.6e}")

    if convergence_epoch is not None:
        log['convergence_epoch'] = convergence_epoch

    return dict(log)


def _aggregate_logs(all_logs: List[Dict], n_seeds: int) -> Dict[str, Any]:
    """Aggregate multiple seed results with bootstrap statistics."""
    aggregated = {}
    # Find common metrics
    common_keys = set(all_logs[0].keys())
    for log in all_logs[1:]:
        common_keys &= set(log.keys())

    for key in common_keys:
        values_per_seed = [log[key] for log in all_logs]

        if isinstance(values_per_seed[0], list) and len(values_per_seed[0]) > 0:
            # Time series: align by epoch and compute mean/std across seeds
            max_len = min(len(v) for v in values_per_seed)
            aligned = [v[:max_len] for v in values_per_seed]

            if isinstance(aligned[0][0], (int, float, np.integer, np.floating)):
                aligned_arr = np.array(aligned)
                mean_vals = np.mean(aligned_arr, axis=0)
                std_vals = np.std(aligned_arr, axis=0, ddof=1)

                aggregated[f'{key}_mean'] = mean_vals.tolist()
                aggregated[f'{key}_std'] = std_vals.tolist()
                aggregated[f'{key}_raw'] = aligned
            else:
                aggregated[f'{key}_raw'] = aligned

        elif isinstance(values_per_seed[0], (int, float, np.integer, np.floating)):
            vals = [float(v) for v in values_per_seed]
            est, std_err, ci_lo, ci_hi = bootstrap_ci(vals)
            aggregated[f'{key}_mean'] = est
            aggregated[f'{key}_std'] = std_err
            aggregated[f'{key}_ci_lo'] = ci_lo
            aggregated[f'{key}_ci_hi'] = ci_hi
            aggregated[f'{key}_raw'] = vals

    return aggregated


# ── Enhanced Metrics Collection ─────────────────────────────────────────────

def compute_enhanced_metrics(model, X_train, y_train, loss_fn,
                              compute_fisher=True, compute_alignment=True,
                              k_eig=20, use_full_subset=0):
    """Compute comprehensive metrics at a checkpoint.

    Returns dict with:
    - tr_H, lambda_max_H, kappa_H: Hessian metrics
    - tr_Sigma, lambda_max_Sigma: Noise covariance metrics
    - alignment_ratio: H-Σ eigenvector alignment
    - fisher_trace: Fisher Information trace
    """
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    metrics = {}

    # Use subset for computational efficiency
    n_use = use_full_subset if use_full_subset > 0 else min(500, X_train.shape[0])
    X_sub = X_train[:n_use]
    y_sub = y_train[:n_use]

    # Hessian trace via Hutchinson
    k_eig_actual = min(k_eig, n_params)

    try:
        if hasattr(model, 'compute_hessian_topk'):
            hess_eigvals, hess_eigvecs = model.compute_hessian_topk(
                X_sub, y_sub, loss_fn, k_eig=k_eig_actual
            )
            hess_eigvals_np = hess_eigvals.detach().numpy()
            metrics['tr_H_est'] = float(np.sum(hess_eigvals_np))
            metrics['lambda_max_H'] = float(np.max(hess_eigvals_np))

            pos = hess_eigvals_np[hess_eigvals_np > 1e-10]
            metrics['kappa_H_est'] = float(hess_eigvals_np.max() / pos.min()) if len(pos) > 0 else float('inf')
        else:
            hess_eigvecs = None
    except Exception:
        hess_eigvecs = None
        metrics['tr_H_est'] = np.nan

    # Noise covariance
    try:
        if hasattr(model, 'compute_noise_cov_topk'):
            sigma_eigvals, sigma_eigvecs = model.compute_noise_cov_topk(
                X_sub, y_sub, loss_fn, k_eig=k_eig_actual
            )
            sigma_eigvals_np = sigma_eigvals.detach().numpy()
            metrics['tr_Sigma_est'] = float(np.sum(sigma_eigvals_np))
            metrics['lambda_max_Sigma'] = float(np.max(sigma_eigvals_np))
        else:
            sigma_eigvecs = None
    except Exception:
        sigma_eigvecs = None
        metrics['tr_Sigma_est'] = np.nan

    # Alignment
    if compute_alignment and hess_eigvecs is not None and sigma_eigvecs is not None:
        try:
            from models import eigenvector_alignment
            alignment = eigenvector_alignment(hess_eigvecs, sigma_eigvecs)
            random_baseline = 1.0 / n_params
            metrics['alignment'] = alignment
            metrics['alignment_ratio'] = alignment / random_baseline
        except Exception:
            metrics['alignment'] = np.nan
            metrics['alignment_ratio'] = np.nan
    else:
        metrics['alignment'] = np.nan
        metrics['alignment_ratio'] = np.nan

    # Fisher Information
    if compute_fisher:
        try:
            tr_F, std_F, _ = estimate_fisher_information(model, X_sub, y_sub, loss_fn)
            metrics['fisher_trace'] = tr_F
            metrics['fisher_std'] = std_F
        except Exception:
            metrics['fisher_trace'] = np.nan
            metrics['fisher_std'] = np.nan

    return metrics


# ── Sharpness Ratio Computation ─────────────────────────────────────────────

def compute_sharpness_ratio(tr_H_by_gamma: Dict[float, float],
                             low_gamma: float = 1.0,
                             high_gamma: float = 2.0) -> Tuple[float, float, float, Dict]:
    """Compute Sharpness Ratio R_H = <Tr(H)>_{γ<low} / <Tr(H)>_{γ>high}.

    Returns (R_H, mean_H_low, mean_H_high, details).
    """
    gammas = np.array(list(tr_H_by_gamma.keys()))
    tr_H_vals = np.array(list(tr_H_by_gamma.values()))

    mask_low = gammas < low_gamma
    mask_high = gammas > high_gamma

    mean_H_low = tr_H_vals[mask_low].mean() if mask_low.sum() > 0 else 1.0
    mean_H_high = tr_H_vals[mask_high].mean() if mask_high.sum() > 0 else 1.0
    R_H = mean_H_low / max(mean_H_high, 1e-10)

    details = {
        'gammas_low': gammas[mask_low].tolist(),
        'tr_H_low': tr_H_vals[mask_low].tolist(),
        'gammas_high': gammas[mask_high].tolist(),
        'tr_H_high': tr_H_vals[mask_high].tolist(),
    }

    return R_H, mean_H_low, mean_H_high, details
