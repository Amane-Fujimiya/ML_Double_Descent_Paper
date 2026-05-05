"""Deep Network DD: 5-layer tanh sweep at d=30.

Tests whether the Sharpness Ratio R_H mechanism survives in deep architectures.
Protocol: 
  1. 5-layer tanh network with pyramid width (k1→k2→k3→k4→k5)
  2. Sweep gamma = k1/d through interpolation
  3. 5 seeds per gamma, 2000 epochs each
  4. Measure test loss + Tr(H) at convergence

Usage:
  python experiments/run_deep5_tanh.py [--d 30] [--n 3000] [--seeds 5] [--epochs 2000]
  Output: ./outputs/deep5_tanh_d30.json
"""
import sys, os, json, argparse, time
import torch, numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from models import generate_teacher_data

# Import compute_hessian_trace from run_exp7_heterogeneity
from run_exp7_heterogeneity import compute_hessian_trace


class Deep5LayerTanh(torch.nn.Module):
    def __init__(self, d, k1, k2, k3, k4, k5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, k1), torch.nn.Tanh(),
            torch.nn.Linear(k1, k2), torch.nn.Tanh(),
            torch.nn.Linear(k2, k3), torch.nn.Tanh(),
            torch.nn.Linear(k3, k4), torch.nn.Tanh(),
            torch.nn.Linear(k4, k5),
        )
        self.v = torch.ones(k5, 1) / np.sqrt(k5)

    def forward(self, x):
        h = self.net(x)
        return (h @ self.v.to(h.device)).squeeze(-1)


def run_deep5(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Deep5] device={device}, d={args.d}, n={args.n}, seeds={args.seeds}, epochs={args.epochs}')

    X_tr, y_tr, X_te, y_te, _ = generate_teacher_data(args.n, args.d, seed=42)
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_te, y_te = X_te.to(device), y_te.to(device)
    loss_fn = torch.nn.MSELoss()

    gammas = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    results = []
    t_start = time.time()

    for gamma in gammas:
        k1 = max(1, int(args.d * gamma))
        k2 = max(1, int(k1 * 1.2))
        k3 = max(1, int(k2 * 1.2))
        k4 = max(1, int(k3 * 1.2))
        k5 = max(1, int(k4 * 1.2))

        seed_losses, seed_trHs = [], []
        for s in range(args.seeds):
            seed = args.base_seed + int(gamma * 100) + s * 100
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = Deep5LayerTanh(args.d, k1, k2, k3, k4, k5).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=args.lr)

            for ep in range(args.epochs):
                perm = torch.randperm(len(X_tr))
                for i in range(0, len(X_tr), args.batch_size):
                    xb = X_tr[perm[i:i+args.batch_size]]
                    yb = y_tr[perm[i:i+args.batch_size]]
                    opt.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    opt.step()

            with torch.no_grad():
                model.eval()
                test_l = float(loss_fn(model(X_te), y_te).item())

            # Hessian trace (on CPU to avoid OOM)
            model_cpu = model.cpu()
            X_cpu, y_cpu = X_tr[:500].cpu(), y_tr[:500].cpu()
            trH = compute_hessian_trace(model_cpu, X_cpu, y_cpu, loss_fn)
            model.to(device)

            seed_losses.append(test_l)
            seed_trHs.append(trH)
            elapsed = time.time() - t_start
            print(f'[Deep5] γ={gamma:.1f} seed={s+1}/{args.seeds} '
                  f'test={test_l:.6f} Tr(H)={trH:.1f} elapsed={elapsed:.0f}s')

        results.append({
            'gamma': gamma, 'k1': k1, 'k2': k2, 'k3': k3, 'k4': k4, 'k5': k5,
            'test_mean': float(np.mean(seed_losses)),
            'test_std': float(np.std(seed_losses, ddof=1)),
            'trH_mean': float(np.mean(seed_trHs)),
            'trH_std': float(np.std(seed_trHs, ddof=1)),
            'seeds_loss': seed_losses,
            'seeds_trH': seed_trHs
        })

    out_path = os.path.join(os.path.dirname(_here), '..', 'outputs', args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[Deep5] Saved to {out_path}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=30)
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_seed', type=int, default=42)
    parser.add_argument('--output', default='deep5_tanh_d30.json')
    args = parser.parse_args()
    run_deep5(args)
