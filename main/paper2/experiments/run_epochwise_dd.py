"""Epoch-wise Double Descent Experiment.

Tracks test loss at every epoch for a fixed architecture (gamma = k/d) 
over extended training. The NESP framework predicts epoch-wise DD 
mediated by progressive sharpening (Section 7.5).

Protocol:
  1. Fix d=50, k=60 (gamma=1.2, just past interpolation threshold)
  2. Train for 5000 epochs with full per-epoch test loss recording
  3. Output JSON with epoch-indexed test loss trajectory

Usage:
  python experiments/run_epochwise_dd.py [--d 50] [--k 60] [--n 5000] \
      [--epochs 5000] [--activation tanh]
  Output: ./outputs/epochwise_d50.json
"""
import sys, os, json, argparse, time
import torch, numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
from models import generate_teacher_data
from run_exp6_activation_comparison import TwoLayerNetwork


def run_epochwise(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[Epochwise] device={device}, d={args.d}, k={args.k}, n={args.n}, '
          f'epochs={args.epochs}, activation={args.activation}')

    X_tr, y_tr, X_te, y_te, _ = generate_teacher_data(args.n, args.d, seed=args.seed)
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_te, y_te = X_te.to(device), y_te.to(device)
    loss_fn = torch.nn.MSELoss()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = TwoLayerNetwork(d=args.d, k=args.k, activation=args.activation).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    results = []
    t_start = time.time()

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
            model.train()

        results.append({'epoch': ep, 'test_loss': test_l})

        if ep % 500 == 0:
            elapsed = time.time() - t_start
            print(f'[Epochwise] epoch {ep}/{args.epochs}, test={test_l:.6f}, elapsed={elapsed:.0f}s')

    elapsed = time.time() - t_start
    results.append({'_elapsed_seconds': elapsed, '_config': vars(args)})

    out_path = os.path.join(os.path.dirname(_here), '..', 'outputs', args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f'[Epochwise] Done. {args.epochs} epochs in {elapsed:.0f}s. Saved to {out_path}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=50)
    parser.add_argument('--k', type=int, default=60)
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--activation', default='tanh')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='epochwise_d50.json')
    args = parser.parse_args()
    run_epochwise(args)
