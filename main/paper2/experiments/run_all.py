#!/usr/bin/env python3
import sys
import os
# Fix Unicode output on Windows terminals
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

"""Master runner for all NESP experiments.

Usage:
    python run_all.py                    # Run all experiments
    python run_all.py --exp 1            # Run only experiment 1
    python run_all.py --exp 1,2,3        # Run experiments 1, 2, and 3
    python run_all.py --quick            # Quick mode (smaller models, fewer epochs)
    python run_all.py --output ./figs    # Custom output directory

Generates:
    - outputs/exp1_double_descent_noise.pdf
    - outputs/exp1_eigenvalue_scatter.pdf
    - outputs/exp1_hessian_geometry.pdf
    - outputs/exp2_escape_time.pdf
    - outputs/exp3_batch_size_dependence.pdf
    - outputs/exp3_trajectories.pdf
    - outputs/exp4_relu_alignment.pdf
    - outputs/exp5_equilibrium_erosion.pdf
    - outputs/summary_report.txt
"""
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Prevent interactive matplotlib issues
import matplotlib
matplotlib.use('Agg')


def run_all_experiments(exp_ids=None, quick=False, output_dir='./outputs'):
    """Run specified experiments and return all results."""
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    results = {}

    # Common parameters (reduced for quick mode)
    if quick:
        print("=" * 60)
        print("  QUICK MODE — reduced model sizes and epochs")
        print("=" * 60)
        params = dict(d=10, n_samples=500, n_epochs=300, seed=42)
    else:
        params = dict(d=20, n_samples=2000, n_epochs=1500, seed=42)

    experiments = []

    if exp_ids is None or 1 in exp_ids:
        experiments.append(('1', 'Curvature-Noise Coupling', run_exp1))

    if exp_ids is None or 2 in exp_ids:
        experiments.append(('2', 'Escape Time Measurement', run_exp2))

    if exp_ids is None or 3 in exp_ids:
        experiments.append(('3', 'Batch Size Dependence', run_exp3))

    if exp_ids is None or 4 in exp_ids:
        experiments.append(('4', 'ReLU Alignment', run_exp4))

    if exp_ids is None or 5 in exp_ids:
        experiments.append(('5', 'Equilibrium Erosion', run_exp5))

    for exp_id, exp_name, exp_fn in experiments:
        print(f"\n{'#'*60}")
        print(f"  STARTING EXPERIMENT {exp_id}: {exp_name}")
        print(f"{'#'*60}")
        t0 = time.time()

        try:
            result = exp_fn(output_dir=output_dir, **params)
            t1 = time.time()
            print(f"\n  ✓ Experiment {exp_id} completed in {t1 - t0:.0f}s")
            results[exp_id] = {'name': exp_name, 'data': result, 'status': 'OK'}
        except Exception as e:
            t1 = time.time()
            print(f"\n  ✗ Experiment {exp_id} FAILED after {t1 - t0:.0f}s")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results[exp_id] = {'name': exp_name, 'data': None, 'status': f'FAILED: {e}'}

    # ── Generate Summary Report ──
    total_time = time.time() - start_time
    report_path = os.path.join(output_dir, 'summary_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("  NESP EXPERIMENTAL SUITE — SUMMARY REPORT\n")
        f.write(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)\n")
        f.write("=" * 60 + "\n\n")

        for exp_id, info in sorted(results.items()):
            f.write(f"[Exp {exp_id}] {info['name']}: {info['status']}\n")

        f.write(f"\nOutput directory: {os.path.abspath(output_dir)}\n")
        f.write(f"Generated figures:\n")
        for fname in sorted(os.listdir(output_dir)):
            if fname.endswith('.pdf'):
                f.write(f"  - {fname}\n")

    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")

    for exp_id, info in sorted(results.items()):
        status_icon = '✓' if info['status'] == 'OK' else '✗'
        print(f"  {status_icon} Experiment {exp_id}: {info['name']} — {info['status']}")

    return results


def run_exp1(output_dir='./outputs', **kwargs):
    from run_exp1_curvature_noise import run_experiment_1
    return run_experiment_1(output_dir=output_dir, **kwargs)


def run_exp2(output_dir='./outputs', **kwargs):
    from run_exp2_escape_time import run_experiment_2
    return run_experiment_2(output_dir=output_dir, **kwargs)


def run_exp3(output_dir='./outputs', **kwargs):
    from run_exp3_batch_size import run_experiment_3
    return run_experiment_3(output_dir=output_dir, **kwargs)


def run_exp4(output_dir='./outputs', **kwargs):
    from run_exp4_relu import run_experiment_4
    return run_experiment_4(output_dir=output_dir, **kwargs)


def run_exp5(output_dir='./outputs', **kwargs):
    from run_exp5_erosion import run_experiment_5
    return run_experiment_5(output_dir=output_dir, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NESP Experimental Suite — Master Runner'
    )
    parser.add_argument('--exp', type=str, default=None,
                        help='Comma-separated experiment IDs to run (e.g., 1,2,3). Default: all.')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: smaller models, fewer epochs.')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Output directory for figures and report.')
    args = parser.parse_args()

    exp_ids = None
    if args.exp:
        exp_ids = [int(x.strip()) for x in args.exp.split(',')]

    run_all_experiments(
        exp_ids=exp_ids,
        quick=args.quick,
        output_dir=args.output,
    )
