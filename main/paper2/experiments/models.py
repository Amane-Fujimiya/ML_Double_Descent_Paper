"""Model definitions for NESP experiments.

Linear Teacher-Student model and shallow ReLU network,
matching the theoretical setup in paper2a.tex Section 7.
"""

import torch
import torch.nn as nn
import numpy as np


# ── Data Generation ──────────────────────────────────────────────────────────

def generate_teacher_data(n_samples: int, d: int, seed: int = 42, noise_std: float = 0.05):
    """Generate data from linear teacher: y = w*^T x + noise, x ~ N(0, I_d).

    Returns train/test splits with consistent teacher.
    noise_std adds label noise to prevent instant convergence.
    """
    rng = np.random.RandomState(seed)
    w_star = rng.randn(d, 1).astype(np.float32)
    w_star /= np.linalg.norm(w_star)  # unit-norm teacher

    X = rng.randn(n_samples, d).astype(np.float32)
    noise = rng.randn(n_samples, 1).astype(np.float32) * noise_std
    y = X @ w_star + noise

    # Split: 70% train, 30% test
    n_train = int(0.7 * n_samples)
    idx = rng.permutation(n_samples)
    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_test, y_test = X[idx[n_train:]], y[idx[n_train:]]
    return (torch.tensor(X_train), torch.tensor(y_train),
            torch.tensor(X_test), torch.tensor(y_test),
            torch.tensor(w_star))


# ── Linear Teacher-Student Model ─────────────────────────────────────────────

class LinearTeacherStudent(nn.Module):
    """Two-layer linear network: ŷ = v^T U x.

    Matches Section 7.6 of paper2a.tex.
    v is fixed to 1/sqrt(k), only U is trained.
    """

    def __init__(self, d: int, k: int):
        super().__init__()
        self.d = d          # input dimension
        self.k = k          # hidden width

        # Fixed second-layer weights: v_j = 1/sqrt(k)
        v = torch.ones(k, 1) / np.sqrt(k)
        self.register_buffer('v', v)

        # Trainable first-layer weights U ∈ R^{k × d}
        self.U = nn.Parameter(torch.randn(k, d) * 0.01)

    def forward(self, x):
        # x: (batch, d) -> output: (batch, 1)
        # ŷ = v^T U x
        return x @ self.U.T @ self.v

    def effective_weight(self):
        """Return w_eff = U^T v ∈ R^d (the effective linear predictor)."""
        return self.U.T @ self.v

    def compute_hessian(self, X_batch, y_batch):
        """Compute the exact per-sample Hessian H_sample = (v v^T) ⊗ (x x^T).

        For the population Hessian, average over the batch.
        The full Hessian is (k*d) × (k*d), but we compute its eigenvalues
        via the Kronecker structure.
        """
        batch_size = X_batch.shape[0]
        # Population Hessian (expectation over batch):
        # H = (v v^T) ⊗ (1/B Σ x_i x_i^T)
        vv = self.v @ self.v.T                               # (k, k)
        xx_avg = X_batch.T @ X_batch / batch_size            # (d, d)

        # Eigenvalues of Kronecker product = outer product of eigenvalues
        eig_vv = torch.linalg.eigvalsh(vv)                   # (k,)
        eig_xx = torch.linalg.eigvalsh(xx_avg)               # (d,)
        hessian_eigvals = torch.outer(eig_vv, eig_xx).flatten()  # (k*d,)
        return hessian_eigvals

    def compute_noise_covariance(self, X_batch, y_batch):
        """Estimate SGD noise covariance Σ from per-sample gradient variance.

        For B=1, Σ ≈ E_x[g ⊗ g] where g = e(x) · (v x^T).
        We compute per-sample gradients and estimate their covariance.
        """
        batch_size = X_batch.shape[0]
        k, d = self.k, self.d

        # Compute per-sample gradients
        grads = []
        for i in range(batch_size):
            self.zero_grad()
            x_i = X_batch[i:i+1]  # (1, d)
            y_i = y_batch[i:i+1]  # (1, 1)
            y_hat = self.forward(x_i)
            loss = 0.5 * (y_hat - y_i).pow(2).sum()
            loss.backward()
            g = self.U.grad.clone().flatten()  # (k*d,)
            grads.append(g)

        grads = torch.stack(grads, dim=0)  # (B, k*d)

        # Covariance estimate
        mean_g = grads.mean(dim=0, keepdim=True)
        centered = grads - mean_g
        cov = centered.T @ centered / (batch_size - 1)  # (k*d, k*d)

        # Eigenvalues of noise covariance
        try:
            eigvals = torch.linalg.eigvalsh(cov)
        except Exception:
            # Fallback for numerical issues
            eigvals = torch.zeros(k * d)
        return eigvals


# ── Shallow ReLU Network ─────────────────────────────────────────────────────

class ShallowReLUNetwork(nn.Module):
    """Two-layer ReLU network: f(x) = v^T σ(U x).

    Used in Experiment 4 to test if curvature-noise coupling
    survives in nonlinear networks.
    """

    def __init__(self, d: int, k: int, activation: str = 'relu'):
        super().__init__()
        self.d = d
        self.k = k

        # First layer: U ∈ R^{k × d}
        self.U = nn.Parameter(torch.randn(k, d) * 0.01)
        # Second layer: v ∈ R^{k × 1}, trainable
        self.v = nn.Parameter(torch.randn(k, 1) * 0.01)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        # x: (batch, d)
        h = self.activation(x @ self.U.T)  # (batch, k)
        return h @ self.v                   # (batch, 1)

    def compute_hessian_topk(self, X_sample, y_sample, loss_fn, k_eig=20):
        """Compute top-k Hessian eigenvalues via power iteration (Lanczos).

        This approximates the Hessian-vector product (HVP) using
        automatic differentiation.
        """
        def hvp(vec):
            # Hessian-vector product via double backward
            vec = vec.detach()
            y_hat = self.forward(X_sample)
            loss = loss_fn(y_hat, y_sample)
            grad = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            grad_flat = torch.cat([g.flatten() for g in grad])
            hvp_flat = torch.autograd.grad(
                grad_flat, self.parameters(), grad_outputs=vec, retain_graph=True
            )
            return torch.cat([h.flatten() for h in hvp_flat])

        n_params = sum(p.numel() for p in self.parameters())
        k_eig = min(k_eig, n_params)

        # Simple power iteration for top eigenvalues
        # (Full Lanczos would be more accurate but more complex)
        eigvals = []
        eigvecs = []
        v = torch.randn(n_params)

        for _ in range(k_eig):
            # Power iteration
            for _ in range(10):  # convergence iterations
                hv = hvp(v)
                v = hv / (hv.norm() + 1e-10)

            # Rayleigh quotient gives eigenvalue
            hv = hvp(v)
            lam = torch.dot(v, hv)
            eigvals.append(lam.item())
            eigvecs.append(v.clone())

            # Deflate: remove this component
            v = torch.randn(n_params)
            for ev in eigvecs:
                v = v - torch.dot(v, ev) * ev
            v = v / (v.norm() + 1e-10)

        return torch.tensor(eigvals), torch.stack(eigvecs, dim=0)

    def compute_noise_cov_topk(self, X_batch, y_batch, loss_fn, k_eig=20):
        """Estimate top-k eigenvalues/vectors of noise covariance.

        Uses per-sample gradient outer products.
        """
        batch_size = X_batch.shape[0]
        n_params = sum(p.numel() for p in self.parameters())
        k_eig = min(k_eig, min(n_params, batch_size))

        # Collect per-sample gradients
        grads = []
        for i in range(batch_size):
            self.zero_grad()
            x_i = X_batch[i:i+1]
            y_i = y_batch[i:i+1]
            y_hat = self.forward(x_i)
            loss = loss_fn(y_hat, y_i)
            loss.backward()
            g = torch.cat([p.grad.flatten() for p in self.parameters()])
            grads.append(g)

        grads = torch.stack(grads, dim=0)  # (B, n_params)
        mean_g = grads.mean(dim=0)
        centered = grads - mean_g
        cov = centered.T @ centered / batch_size

        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            # Return top-k (largest)
            return eigvals[-k_eig:].flip(0), eigvecs[:, -k_eig:].T.flip(0)
        except Exception:
            return torch.zeros(k_eig), torch.zeros(k_eig, n_params)


# ── Utility: compute eigenvector alignment ───────────────────────────────────

def eigenvector_alignment(eigvecs_H, eigvecs_Sigma):
    """Measure alignment between two sets of eigenvectors.

    Alignment = (1/k) Σ_i |v_i(H)^T v_i(Σ)|^2
    Returns value in [0, 1]. Random expectation ≈ 1/dim.
    """
    k = eigvecs_H.shape[0]
    dim = eigvecs_H.shape[1]
    align = 0.0
    for i in range(k):
        dot = torch.dot(eigvecs_H[i], eigvecs_Sigma[i])
        align += (dot ** 2).item()
    return align / k
