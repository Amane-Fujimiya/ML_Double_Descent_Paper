"""Loop 6: Phase Diagram Explorer (Cluster 2, Loop 1).

Sweeps (gamma, T_eff) = (gamma, eta/B) space for tanh architecture.
Goal: Identify DD peak vanishing boundary.

Usage:
    py experiments/run_loop6_phase.py          # Default: d=30, tanh
    py experiments/run_loop6_phase.py --d 30 --n 3000 --seeds 3
    py experiments/run_loop6_phase.py --resume outputs/checkpoint_phase.json
"""
import sys, os
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, torch.nn as nn, numpy as np, time, json, argparse
from collections import defaultdict

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from matplotlib.colors import LogNorm

from models import generate_teacher_data
from run_exp6_activation_comparison import TwoLayerNetwork
from framework_scaled import bootstrap_ci


def run_phase_diagram(d, n_samples, n_epochs, n_seeds, output_dir, resume_from=None):
    """Sweep (gamma, B) space for fixed eta=0.01 on tanh."""
    gammas = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]
    batch_sizes = [4, 16, 64, 256, 1024, 4096]
    lr = 0.01
    act_name = 'tanh'
    base_seed = 42

    print("=" * 70)
    print("  LOOP 6 - CLUSTER 2: Phase Diagram Exploration")
    print(f"  d={d}, n={n_samples}, eta={lr}, epochs={n_epochs}, seeds={n_seeds}")
    print(f"  gamma values: {len(gammas)}, batch sizes: {len(batch_sizes)}")
    print(f"  Total configs: {len(gammas) * len(batch_sizes) * n_seeds}")
    print("=" * 70)

    # Checkpoint with config validation
    ckpt_path = resume_from or os.path.join(output_dir, 'checkpoint_phase.json')
    config_id = f"d{d}_n{n_samples}_e{n_epochs}"
    checkpoint = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r') as f:
            ckpt_data = json.load(f)
        if ckpt_data.get('config_id', '') == config_id:
            checkpoint = ckpt_data.get('entries', {})
            print(f"  Resuming from checkpoint: {len(checkpoint)} valid entries (config: {config_id})")
        else:
            print(f"  Checkpoint config mismatch ({ckpt_data.get('config_id')} != {config_id}) — starting fresh")
            checkpoint = {}

    # Data
    X_train, y_train, X_test, y_test, _ = generate_teacher_data(n_samples, d, seed=42)
    loss_fn = nn.MSELoss()
    n_train = len(X_train)

    total = len(gammas) * len(batch_sizes) * n_seeds
    done = 0
    t_start = time.time()
    results = {}  # {(gamma, B): {'test_mean':, 'test_std':, 'test_raw':[]}}

    for gamma in gammas:
        k = max(1, int(d * gamma))
        gamma = k / d  # round to actual gamma

        for B in batch_sizes:
            if B > n_train:
                continue
            T_eff = lr / B
            key = f"{gamma:.3f}_{B}"

            if key in checkpoint:
                results[(gamma, B)] = checkpoint[key]
                print(f"  gamma={gamma:.2f} B={B:4d} T_eff={T_eff:.2e} — SKIPPED")
                continue

            seed_losses = []
            for s in range(n_seeds):
                seed = base_seed + int(gamma * 100) + int(np.log10(B+1)) * 10 + s * 50
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = TwoLayerNetwork(d=d, k=k, activation=act_name)
                opt = torch.optim.SGD(model.parameters(), lr=lr)

                for ep in range(n_epochs):
                    perm = torch.randperm(n_train)
                    for i in range(0, n_train, B):
                        Xb = X_train[perm[i:i+B]]
                        yb = y_train[perm[i:i+B]]
                        opt.zero_grad()
                        l = loss_fn(model(Xb), yb)
                        l.backward()
                        opt.step()

                with torch.no_grad():
                    model.eval()
                    test_l = float(loss_fn(model(X_test), y_test).item())
                seed_losses.append(test_l)
                done += 1

            # Bootstrap CI
            test_mean, test_std, test_lo, test_hi = bootstrap_ci(seed_losses, n_bootstrap=500)

            entry = {
                'gamma': round(gamma, 3), 'B': B, 'T_eff': T_eff,
                'test_mean': float(test_mean), 'test_std': float(test_std),
                'test_raw': [float(x) for x in seed_losses],
            }
            results[(gamma, B)] = entry
            checkpoint[key] = entry

            # Save checkpoint periodically
            if done % 10 == 0:
                with open(ckpt_path, 'w') as f:
                    json.dump({'config_id': config_id, 'entries': checkpoint}, f, indent=2)

            elapsed = time.time() - t_start
            eta_fraction = done / total
            eta_sec = elapsed / eta_fraction * (1 - eta_fraction) if eta_fraction > 0 else 0
            eta_str = f"{eta_sec/60:.0f}m" if eta_sec < 3600 else f"{eta_sec/3600:.1f}h"
            print(f"  gamma={gamma:.2f} B={B:4d} T_eff={T_eff:.2e} | "
                  f"test={test_mean:.6f}±{test_std:.6f} | done={done}/{total} ETA:{eta_str}")

    # Save final
    with open(ckpt_path, 'w') as f:
        json.dump({'config_id': config_id, 'entries': checkpoint}, f, indent=2)
    print(f"\n  Phase diagram checkpoint saved: {ckpt_path}")

    return results


def plot_phase_diagram(results, d, n_samples, output_dir):
    """Generate contour plot of test loss vs (gamma, T_eff)."""
    # Build data grid
    gammas_set = sorted(set(k[0] for k in results.keys()))
    B_set = sorted(set(k[1] for k in results.keys()))
    teff_set = [0.01 / B for B in B_set]

    grid = np.full((len(B_set), len(gammas_set)), np.nan)
    for i, B in enumerate(B_set):
        for j, gamma in enumerate(gammas_set):
            key = (gamma, B)
            if key in results:
                grid[i, j] = results[key]['test_mean']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (1) Contour plot
    ax = axes[0]
    X_plot, Y_plot = np.meshgrid(gammas_set, np.log10(teff_set))
    im = ax.pcolormesh(X_plot, Y_plot, grid, cmap='RdYlBu_r',
                       shading='auto', norm=LogNorm())
    ax.axvline(x=1.0, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('γ = k/d')
    ax.set_ylabel('log₁₀(T_eff) = log₁₀(η/B)')
    ax.set_title(f'Test Loss Heatmap (tanh, d={d}, n={n_samples})')
    plt.colorbar(im, ax=ax, label='Test MSE')

    # (2) DD peak height vs T_eff at each gamma
    ax = axes[1]
    cmap = plt.cm.viridis
    norm_g = plt.Normalize(min(gammas_set), max(gammas_set))
    for gamma in gammas_set:
        teffs_plot = []
        peaks_plot = []
        for B in B_set:
            if (gamma, B) in results:
                teffs_plot.append(0.01 / B)
                peaks_plot.append(results[(gamma, B)]['test_mean'])
        if len(teffs_plot) > 1:
            ax.plot(teffs_plot, peaks_plot, 'o-', color=cmap(norm_g(gamma)),
                    markersize=5, linewidth=1.5, alpha=0.7,
                    label=f'γ={gamma:.1f}')
    ax.set_xscale('log')
    ax.set_xlabel('T_eff = η/B')
    ax.set_ylabel('Test Loss (MSE)')
    ax.set_title('Test Loss vs T_eff by γ')
    ax.legend(fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)

    # (3) Phase boundary: where DD peak vanishes
    ax = axes[2]
    # For each T_eff, find the gamma where second descent is complete
    # (test loss monotonic or minimal after gamma=1)
    teff_boundary = []
    gamma_boundary = []
    for B in B_set:
        teff = 0.01 / B
        g_vals = []
        t_vals = []
        for gamma in gammas_set:
            if (gamma, B) in results:
                g_vals.append(gamma)
                t_vals.append(results[(gamma, B)]['test_mean'])
        if len(g_vals) >= 4:
            # Find gamma where test loss first drops below under-parameterized level
            g_arr = np.array(g_vals)
            t_arr = np.array(t_vals)
            # Simple heuristic: gamma at which slope becomes definitively negative
            mask_pre = g_arr < 1.0
            mask_post = g_arr > 1.5
            if mask_post.sum() >= 2:
                pre_mean = t_arr[mask_pre].mean() if mask_pre.sum() > 0 else t_arr.max()
                post_min = t_arr[mask_post].min()
                # Recovery threshold: 10% reduction from pre-interpolation
                threshold = pre_mean * 0.9
                for j in range(len(g_arr)):
                    if g_arr[j] > 1.0 and t_arr[j] < threshold:
                        teff_boundary.append(teff)
                        gamma_boundary.append(g_arr[j])
                        break

    if teff_boundary:
        # Sort by teff
        idx = np.argsort(teff_boundary)
        ax.plot([teff_boundary[i] for i in idx],
                [gamma_boundary[i] for i in idx],
                's-', color='#FF5722', markersize=10, linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('T_eff = η/B')
    ax.set_ylabel('γ where test drops below threshold')
    ax.set_title('DD Vanishing Boundary')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Phase Diagram: (γ, T_eff) Space — NESP Framework',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'phase_diagram.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: phase_diagram.pdf")


def generate_subreport(results, d, n_samples, n_seeds, n_epochs, output_dir):
    """Generate cluster2_loop1_subreport.md."""
    report_path = os.path.join(output_dir, 'cluster2_loop1_subreport.md')

    gammas_set = sorted(set(k[0] for k in results.keys()))
    B_set = sorted(set(k[1] for k in results.keys()))

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Cluster 2, Loop 1: Phase Diagram Exploration\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 1. Objective\n\n")
        f.write(f"Map the (γ, T_eff = η/B) phase space for tanh architecture.\n")
        f.write(f"Identify the phase boundary where DD peak vanishes.\n\n")

        f.write(f"## 2. Experimental Design\n\n")
        f.write(f"| Parameter | Value |\n")
        f.write(f"|-----------|-------|\n")
        f.write(f"| d | {d} |\n")
        f.write(f"| n | {n_samples} |\n")
        f.write(f"| Activation | tanh |\n")
        f.write(f"| η | 0.01 |\n")
        f.write(f"| B range | {B_set} |\n")
        f.write(f"| γ range | {len(gammas_set)} values ({min(gammas_set):.1f}–{max(gammas_set):.1f}) |\n")
        f.write(f"| Seeds | {n_seeds} |\n")
        f.write(f"| Epochs | {n_epochs} |\n\n")

        f.write(f"## 3. Results\n\n")
        f.write(f"### 3.1 Test Loss Grid\n\n")
        f.write(f"| γ \\ B | {' | '.join(str(b) for b in B_set)} |\n")
        f.write(f"|------|{'|'.join('---' for _ in B_set)}|\n")
        for gamma in gammas_set:
            row = f"| {gamma:.1f} "
            for B in B_set:
                val = results.get((gamma, B), {}).get('test_mean', np.nan)
                row += f"| {val:.6f} " if not np.isnan(val) else "| — "
            f.write(row + "|\n")

        f.write(f"\n### 3.2 Key Observations\n\n")

        # Find strongest DD
        dd_strengths = {}
        for B in B_set:
            teff = 0.01 / B
            g_vals = []
            t_vals = []
            for gamma in gammas_set:
                if (gamma, B) in results:
                    g_vals.append(gamma)
                    t_vals.append(results[(gamma, B)]['test_mean'])
            if len(g_vals) >= 4:
                g_arr = np.array(g_vals)
                t_arr = np.array(t_vals)
                mask_pre = g_arr < 1.0
                mask_post = g_arr > 1.5
                if mask_pre.sum() and mask_post.sum():
                    peak = max(t_arr)
                    min_p = t_arr[mask_post].min()
                    dd = (peak - min_p) / (min_p + 1e-10)
                    dd_strengths[B] = {
                        'teff': teff, 'peak': peak, 'min': min_p,
                        'dd_strength': dd
                    }

        f.write(f"| B | T_eff | DD Peak | DD Min | DD Strength |\n")
        f.write(f"|---|-------|---------|--------|-------------|\n")
        for B in B_set:
            if B in dd_strengths:
                ds = dd_strengths[B]
                f.write(f"| {B} | {ds['teff']:.2e} | {ds['peak']:.6f} | "
                       f"{ds['min']:.6f} | {ds['dd_strength']*100:.1f}% |\n")

        f.write(f"\n## 4. Theoretical Synthesis\n\n")
        f.write(f"1. Phase boundary where DD vanishes identified\n")
        f.write(f"2. Boundary is [vertical/non-vertical] — [confirms/refutes] prediction\n")
        f.write(f"3. Optimal T_eff for strongest DD: [TBD]\n\n")

        f.write(f"## 5. Decision\n\n")
        f.write(f"- If clear phase structure: Add to manuscript Section 8\n")
        f.write(f"- If no DD observed: Document as scale/clamping artifact\n")

    print(f"  Subreport: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type=int, default=20)
    parser.add_argument('--n', type=int, default=1500)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--output', type=str, default='./outputs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--noplot', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results = run_phase_diagram(
        d=args.d, n_samples=args.n, n_epochs=args.epochs,
        n_seeds=args.seeds, output_dir=args.output,
        resume_from=args.resume
    )

    if not args.noplot:
        plot_phase_diagram(results, args.d, args.n, args.output)
    generate_subreport(results, args.d, args.n, args.seeds, args.epochs, args.output)
    print("\n  LOOP 6 CLUSTER 2 COMPLETE")
