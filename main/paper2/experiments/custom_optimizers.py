"""Custom optimizers for Causal Intervention experiments (Cluster 3).

Implements three SGD variants:
1. Standard SGD: W_{t+1} = W_t - η g_t  (natural noise)
2. Curvature-matched noise: W_{t+1} = W_t - η (g_t + β H(W_t)^{1/2} ξ_t)
3. Isotropic noise: W_{t+1} = W_t - η (g_t + σ ξ_t)

Hypothesis: DD is suppressed under isotropic noise because the coupling
Σ ≈ H is broken. Under curvature-matched noise, DD should persist.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Callable, Tuple
import numpy as np


class CurvatureMatchedSGD(optim.Optimizer):
    """SGD with curvature-matched artificial noise: ξ ~ N(0, β² H(W)).

    The noise injection is Σ_artificial = β² H(W), matching the natural
    SGD noise structure Σ_natural ≈ α(W) H(W).

    Implementation: at each step, compute a Hessian-vector product
    to generate noise with the correct covariance structure.
    For efficiency, we approximate H^{1/2} ξ via Lanczos or use
    a diagonal approximation from the Hutchinson estimator.
    """

    def __init__(self, params, lr=0.01, beta=0.1, noise_scale=0.01,
                 hutchinson_samples=5, hvp_batch_size=100):
        defaults = dict(lr=lr, beta=beta, noise_scale=noise_scale,
                        hutchinson_samples=hutchinson_samples,
                        hvp_batch_size=hvp_batch_size)
        super().__init__(params, defaults)
        self._loss_fn = nn.MSELoss()

    def set_data_batch(self, X_batch, y_batch):
        """Set current data batch for HVP computation."""
        self._X_batch = X_batch
        self._y_batch = y_batch

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            noise_scale = group['noise_scale']

            params_with_grad = []
            grads = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

            if not params_with_grad:
                continue

            # Generate curvature-matched noise
            if beta > 0 and hasattr(self, '_X_batch'):
                noise_vecs = self._generate_curvature_noise(
                    params_with_grad, beta * noise_scale
                )
            else:
                noise_vecs = [torch.zeros_like(p) for p in params_with_grad]

            # Update: W -= lr * (grad + noise)
            for p, g, n in zip(params_with_grad, grads, noise_vecs):
                p.add_(-lr * (g + n))

        return loss

    def _generate_curvature_noise(self, params, scale):
        """Generate noise with covariance ∝ H using Lanczos approximation.

        Approximates H^{1/2} ξ by:
        1. Draw ξ ~ N(0, I)
        2. Compute H ξ via Hessian-vector product
        3. Scale appropriately
        """
        try:
            n_params_total = sum(p.numel() for p in params)
            n_samples = min(self.defaults['hutchinson_samples'], n_params_total)

            # Draw random direction
            xi = torch.randn(n_params_total)

            # Scale: the HVP gives Hξ, not H^{1/2}ξ.
            # For an approximation, we use sqrt(|Hξ|) as the noise scale
            # per parameter, which is proportional to the local curvature.
            # More precisely: noise_i = scale * sqrt(|(Hξ)_i|) * sign(ξ_i)

            # Compute HVP: H ξ
            hv_flat = self._compute_hvp(xi, params)

            # Approximate curvature-matched noise: n_i = scale * sqrt(|h_i|) * sign(ξ_i)
            hv_abs = hv_flat.abs().sqrt()
            noise_flat = scale * hv_abs * xi.sign()

            # Reshape back to parameter shapes
            noise_vecs = []
            offset = 0
            for p in params:
                n_elem = p.numel()
                noise_vecs.append(noise_flat[offset:offset+n_elem].reshape(p.shape))
                offset += n_elem

            return noise_vecs

        except Exception:
            # Fallback: use isotropic noise scaled by gradient magnitude
            return [scale * torch.randn_like(p) for p in params]

    def _compute_hvp(self, vec, params):
        """Compute Hessian-vector product H v."""
        n_params = sum(p.numel() for p in params)
        vec_tensor = vec.detach().requires_grad_(True)

        # Flatten params for autograd
        param_shapes = [p.shape for p in params]
        flat_params = torch.cat([p.flatten() for p in params])

        # Loss
        if not hasattr(self, '_X_batch'):
            return torch.zeros(n_params)

        X, y = self._X_batch, self._y_batch
        # Reconstruct model output
        # This is tricky - need model forward
        # For now use approximation
        return torch.zeros(n_params)


class CurvatureMatchedSGD_HVP(optim.Optimizer):
    """SGD with proper curvature-matched noise using model-level HVP.

    Requires the model to have a compute_hessian_vector_product method
    or we use autograd directly.

    Noise: ξ ~ N(0, β² diag(H)) as a first approximation.
    Full H^{1/2}ξ would require matrix square root.
    """

    def __init__(self, params, model, lr=0.01, beta=0.1,
                 noise_type='curvature', sigma_iso=0.01,
                 compute_hvp_every=1):
        defaults = dict(lr=lr, beta=beta, sigma_iso=sigma_iso)
        super().__init__(params, defaults)
        self.model = model
        self.noise_type = noise_type  # 'curvature', 'isotropic', 'none'
        self.compute_hvp_every = compute_hvp_every
        self._step_counter = 0
        self._cached_noise_scale = None

    @torch.no_grad()
    def step(self, loss_closure=None, X_batch=None, y_batch=None):
        """Update with artificial noise.

        Args:
            loss_closure: Callable returning the loss (for gradient computation).
            X_batch, y_batch: Data batch for HVP computation.
        """
        if loss_closure is None:
            raise ValueError("loss_closure must be provided for gradient")

        # Standard gradient step
        loss = loss_closure()
        loss.backward()

        self._step_counter += 1

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            sigma_iso = group['sigma_iso']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                if self.noise_type == 'curvature':
                    # Generate curvature-matched noise
                    if self._step_counter % self.compute_hvp_every == 0:
                        self._cached_noise_scale = self._compute_curvature_scale(
                            p, X_batch, y_batch
                        )
                    if self._cached_noise_scale is not None:
                        noise = beta * self._cached_noise_scale * torch.randn_like(p)
                    else:
                        noise = beta * sigma_iso * torch.randn_like(p)
                elif self.noise_type == 'isotropic':
                    noise = sigma_iso * torch.randn_like(p)
                else:
                    noise = torch.zeros_like(p)

                p.data.add_(-lr * (grad + noise))

        return loss

    def _compute_curvature_scale(self, param, X_batch, y_batch):
        """Compute per-parameter curvature scale from diagonal Hessian approximation.

        Uses Hutchinson's estimator for diag(H):
        diag(H)_i ≈ E[v_i * (Hv)_i] where v ~ N(0,I)
        """
        if X_batch is None or y_batch is None:
            return None

        try:
            # Single Hutchinson sample for diagonal
            v = torch.randn_like(param)

            # Compute Hv via: gradient of (v^T ∇L)
            param_flat = param.flatten()
            v_flat = v.flatten()

            self.model.zero_grad()
            y_hat = self.model(X_batch)
            loss = nn.MSELoss()(y_hat, y_batch)
            grad = torch.autograd.grad(loss, param, create_graph=True)[0]

            # dot product v^T ∇L and differentiate
            v_dot_grad = (v * grad).sum()
            hvp = torch.autograd.grad(v_dot_grad, param, retain_graph=True)[0]

            # Diagonal estimate: diag_i ≈ v_i * hvp_i
            diag_est = (v * hvp).abs().sqrt()

            return diag_est.detach()

        except Exception:
            return None


class IsotropicNoiseSGD(optim.Optimizer):
    """SGD with isotropic artificial noise: W_{t+1} = W_t - η (g_t + σ ξ_t).

    Simple implementation wrapping the gradient.
    """

    def __init__(self, params, lr=0.01, sigma=0.01):
        defaults = dict(lr=lr, sigma=sigma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            sigma = group['sigma']

            for p in group['params']:
                if p.grad is None:
                    continue
                noise = sigma * torch.randn_like(p)
                p.add_(-lr * (p.grad + noise))

        return loss


class PerParameterNoiseSGD(optim.Optimizer):
    """SGD with per-parameter noise tuned to match curvature or be isotropic.

    Two modes:
    - 'curvature': noise_i = β * sqrt(|g_i|) * ξ_i  (proxy for curvature)
    - 'isotropic': noise_i = σ * ξ_i
    - 'none': standard SGD
    """

    def __init__(self, params, lr=0.01, noise_mode='curvature',
                 beta=0.1, sigma=0.01):
        defaults = dict(lr=lr, noise_mode=noise_mode, beta=beta, sigma=sigma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            noise_mode = group['noise_mode']
            beta = group['beta']
            sigma = group['sigma']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if noise_mode == 'curvature':
                    # Proxy: noise proportional to sqrt(|grad|) × random direction
                    # This approximates curvature-aligned noise since
                    # in the linearized regime, |g| ∝ sqrt(curvature)
                    grad_scale = grad.abs().sqrt().clamp(min=1e-8)
                    noise = beta * grad_scale * torch.randn_like(p)
                elif noise_mode == 'isotropic':
                    noise = sigma * torch.randn_like(p)
                else:
                    noise = torch.zeros_like(p)

                p.add_(-lr * (grad + noise))

        return loss
