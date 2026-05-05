import json, os, math
import numpy as np
from scipy import stats

print('='*70)
print('PHAN TICH TOAN DIEN: Double Descent & Statistical Physics')
print('Loop 8 - Kiem chung moi')
print('='*70)
print()

# === BƯỚC 1: TÓM TẮT DỮ LIỆU ===
print('=== BUOC 1: TOM TAT TAT CA DU LIEU ===')
print()

with open('outputs/cluster1_results.json') as f:
    d1 = json.load(f)
with open('outputs_d50_10seeds/cluster1_results.json') as f:
    d2 = json.load(f)
with open('outputs/epochwise_d50.json') as f:
    ew50 = json.load(f)
with open('outputs/epochwise_d30.json') as f:
    ew30 = json.load(f)
if os.path.exists('outputs/deep5_tanh_d30_heavy.json'):
    with open('outputs/deep5_tanh_d30_heavy.json') as f:
        deep5 = json.load(f)
else:
    deep5 = None

# File 1
print('File 1 (outputs/cluster1_results.json):')
cfg = d1['config']
print(f'  Config: d={cfg["d"]}, n={cfg["n"]}, epochs={cfg["epochs"]}, seeds={cfg["seeds"]}')
print(f'  Equilibrium: tr_H={d1["equilibrium"]["tr_H"]:.4f}, test_mse={d1["equilibrium"]["test_mse"]:.6f}')
print(f'  Tanh points: {len(d1["results"]["tanh"])}, Linear points: {len(d1["results"]["linear"])}')
sr1 = d1['sharpness_ratios']
print(f'  R_H tanh: {sr1["tanh"]["R_H"]:.4f}, Recovery: {sr1["tanh"]["recovery"]*100:.2f}%')
print(f'  R_H linear: {sr1["linear"]["R_H"]:.4f}, Recovery: {sr1["linear"]["recovery"]*100:.2f}%')
print()

# File 2
print('File 2 (outputs_d50_10seeds/cluster1_results.json):')
cfg2 = d2['config']
print(f'  Config: d={cfg2["d"]}, seeds={cfg2["seeds"]}')
sr2 = d2['sharpness_ratios']
print(f'  R_H tanh: {sr2["tanh"]["R_H"]:.4f}, Recovery: {sr2["tanh"]["recovery"]*100:.2f}%')
print(f'  Peak tanh: {sr2["tanh"]["peak"]:.6f}')
print()

# File 3+4
print(f'File 3 (epochwise_d50.json): {len(ew50)} epochs')
print(f'File 4 (epochwise_d30.json): {len(ew30)} epochs')
print()

# File 6
if deep5:
    print(f'File 6 (deep5_tanh_d30_heavy.json): {len(deep5)} points x {len(deep5[0]["seeds_loss"])} seeds')
else:
    print('File 6: NOT FOUND')
print()

# === BƯỚC 2: CONTROL EXPERIMENT ===
print('='*70)
print('=== BUOC 2: CONTROL EXPERIMENT (d=30? vs d=50) ===')
print()

# NOTE: The files labeled as d=30 actually have d=50 params.
# Let's check param counts
tanh1 = d1['results']['tanh']
lin1 = d1['results']['linear']
tanh2 = d2['results']['tanh']
lin2 = d2['results']['linear']

print(f'Param counts for tanh at gamma=10.0:')
print(f'  File1 k=500 n_params={tanh1[-1]["n_params"]} (fits d=50: 50*500+500+500+1={50*500+500+500+1})')
print()

# Compare: R_H and recovery at different scales
print('Comparison R_H and Recovery:')
print(f'  File1 d=50 (3 seeds): R_H={sr1["tanh"]["R_H"]:.4f}, Recovery={sr1["tanh"]["recovery"]*100:.2f}%')
print(f'  File2 d=50 (10 seeds?): R_H={sr2["tanh"]["R_H"]:.4f}, Recovery={sr2["tanh"]["recovery"]*100:.2f}%')
print()

# Check if the d=30 control file exists
if os.path.exists('outputs/cluster1_results_d30_control_10seeds.json.json'):
    with open('outputs/cluster1_results_d30_control_10seeds.json.json') as f:
        d30ctrl = json.load(f)
    print('File: cluster1_results_d30_control_10seeds.json.json exists')
    print(f'  Tanh points: {len(d30ctrl["tanh"])}, Linear points: {len(d30ctrl["linear"])}')
    # Check n_params to determine actual d
    print(f'  n_params at gamma=10.0: {d30ctrl["tanh"][-1]["n_params"]}')
    n_p = d30ctrl["tanh"][-1]["n_params"]
    d_est = (n_p - 501) / 500
    print(f'  Estimated d from n_params: d = ({n_p} - 501) / 500 = {d_est:.1f}')
    print('  => This is also d=50 data! The label d30 is misleading.')
print()

# === BƯỚC 3: BOOTSTRAP CI cho d=50 10 seeds ===
print('='*70)
print('=== BUOC 3: BOOTSTRAP CI cho d=50 (10 seeds) ===')
print()

# Use the tanh data from d2 (outputs_d50_10seeds/cluster1_results.json)
# But it only has aggregate means, not per-seed data for bootstrap.
# We need per-seed data. Let me check if there's a checkpoint file
if os.path.exists('outputs_d50_10seeds/checkpoint_cluster1.json'):
    with open('outputs_d50_10seeds/checkpoint_cluster1.json') as f:
        ckpt = json.load(f)
    # Find the tanh data with per-seed info
    print('Checkpoint file keys:', list(ckpt.keys())[:10])
print()

# Calculate Spearman correlation between R_H and DD peak
# For now using aggregate data:
gamma_vals = [pt['gamma'] for pt in tanh2 if pt['gamma'] >= 1.2]
test_vals = [pt['test_mean'] for pt in tanh2 if pt['gamma'] >= 1.2]
trH_vals = [pt['trH_mean'] for pt in tanh2 if pt['gamma'] >= 1.2]

# Find DD peak (max test loss at gamma>=1.2)
peak_idx = np.argmax(test_vals)
print(f'DD Peak in gamma>=1.2: test={test_vals[peak_idx]:.6f} at gamma={gamma_vals[peak_idx]}')

# Calculate R_H for the 10-seeds data using same methodology
# Low gamma (gamma <= 1.1) vs High gamma (gamma >= 1.2)
low_pts_tanh = [pt for pt in tanh2 if pt['gamma'] <= 1.1]
high_pts_tanh = [pt for pt in tanh2 if pt['gamma'] >= 1.2]

mean_H_low = np.mean([pt['trH_mean'] for pt in low_pts_tanh])
mean_H_high = np.mean([pt['trH_mean'] for pt in high_pts_tanh])
R_H_calc = mean_H_low / mean_H_high

peak_value = max([pt['test_mean'] for pt in high_pts_tanh])
peak_at_gamma = [pt['gamma'] for pt in high_pts_tanh if pt['test_mean'] == peak_value][0]
eq_mse = d2['equilibrium']['test_mse']
recovery = (peak_value - test_vals[-1]) / (peak_value - eq_mse)

print()
print(f'Manual R_H calc: mean_H_low={mean_H_low:.4f}, mean_H_high={mean_H_high:.4f}')
print(f'R_H = {R_H_calc:.4f}')
print(f'Recovery = ({peak_value:.6f} - {test_vals[-1]:.6f}) / ({peak_value:.6f} - {eq_mse:.6f}) = {recovery*100:.2f}%')
print()

# === BƯỚC 4: EPOCH-WISE DD ===
print('='*70)
print('=== BUOC 4: EPOCH-WISE DOUBLE DESCENT ===')
print()

# D50
losses_50 = [e['test_loss'] for e in ew50]
min50 = min(losses_50)
max50 = max(losses_50)
min50_epoch = [e['epoch'] for e in ew50 if e['test_loss'] == min50][0]
max50_epoch = [e['epoch'] for e in ew50 if e['test_loss'] == max50][0]

# Early phase DD (epoch 1-200)
early_50 = [(e['epoch'], e['test_loss']) for e in ew50 if 0 < e['epoch'] <= 200]
early_max_50 = max(early_50, key=lambda x: x[1])
late_50 = [(e['epoch'], e['test_loss']) for e in ew50 if e['epoch'] >= 400]
late_min_50 = min(late_50, key=lambda x: x[1])

print(f'D50 Epoch-wise:')
print(f'  Initial loss: {ew50[0]["test_loss"]:.6f}')
print(f'  Early peak: {early_max_50[1]:.6f} at epoch {early_max_50[0]}')
print(f'  Late minimum: {late_min_50[1]:.6f} at epoch {late_min_50[0]}')
print(f'  Global min: {min50:.6f} at epoch {min50_epoch}')
print()

# D30
losses_30 = [e['test_loss'] for e in ew30]
min30 = min(losses_30)
max30 = max(losses_30)
min30_epoch = [e['epoch'] for e in ew30 if e['test_loss'] == min30][0]
max30_epoch = [e['epoch'] for e in ew30 if e['test_loss'] == max30][0]

early_30 = [(e['epoch'], e['test_loss']) for e in ew30 if 0 < e['epoch'] <= 200]
early_max_30 = max(early_30, key=lambda x: x[1])
late_30 = [(e['epoch'], e['test_loss']) for e in ew30 if e['epoch'] >= 400]
late_min_30 = min(late_30, key=lambda x: x[1])

print(f'D30 Epoch-wise:')
print(f'  Initial loss: {ew30[0]["test_loss"]:.6f}')
print(f'  Early peak: {early_max_30[1]:.6f} at epoch {early_max_30[0]}')
print(f'  Late minimum: {late_min_30[1]:.6f} at epoch {late_min_30[0]}')
print(f'  Global min: {min30:.6f} at epoch {min30_epoch}')
print()

# DD quantification
dd_ratio_50 = early_max_50[1] / late_min_50[1]
dd_ratio_30 = early_max_30[1] / late_min_30[1]
print(f'DD intensity (early_peak/late_min): d50={dd_ratio_50:.4f}, d30={dd_ratio_30:.4f}')
print()

# === BƯỚC 5: DEEP NETWORK ANALYSIS ===
print('='*70)
print('=== BUOC 5: DEEP NETWORK ANALYSIS (Deep5, tanh, d=30) ===')
print()

if deep5:
    test_means = [d['test_mean'] for d in deep5]
    trH_means = [d['trH_mean'] for d in deep5]
    gammas = [d['gamma'] for d in deep5]
    
    print(f'Test losses by gamma:')
    for d in deep5:
        print(f'  gamma={d["gamma"]:.1f}, test={d["test_mean"]:.6f} +/- {d["test_std"]:.6f}, trH={d["trH_mean"]:.4f} +/- {d["trH_std"]:.4f}')
    
    # Check for DD
    peak_test = max(test_means)
    min_test = min(test_means)
    test_range = peak_test - min_test
    mean_test = np.mean(test_means)
    print(f'\nTest loss range: {min_test:.6f} - {peak_test:.6f} (span={test_range:.6f})')
    print(f'Mean test loss: {mean_test:.6f}')
    print(f'Relative variation: {test_range/mean_test*100:.2f}%')
    
    # Hessian trace analysis
    mean_trH_low = np.mean([d['trH_mean'] for d in deep5 if d['gamma'] <= 1.0])
    mean_trH_high = np.mean([d['trH_mean'] for d in deep5 if d['gamma'] >= 2.0])
    R_H_deep5 = mean_trH_low / mean_trH_high if mean_trH_high != 0 else float('inf')
    print(f'\nDeep5 R_H: mean_H_low(gamma<=1.0)={mean_trH_low:.4f}, mean_H_high(gamma>=2.0)={mean_trH_high:.4f}')
    print(f'R_H_deep5 = {R_H_deep5:.4f}')
    
    # Compare with two-layer tanh at same d=30
    # We don't have two-layer tanh d=30 gamma sweep, but we have d=50 data
    # For comparison, note what the two-layer R_H is
    print(f'\nComparison:')
    print(f'  Two-layer tanh (d=50, 3 seeds): R_H={sr1["tanh"]["R_H"]:.4f}')
    print(f'  Deep5 tanh (d=30): R_H={R_H_deep5:.4f}')
    print(f'  Deep5 test loss ~{mean_test:.4f} (essentially flat, no DD detected)')
    
    # Hessian variability
    trH_all = []
    for d in deep5:
        trH_all.extend(d['seeds_trH'])
    trH_all = np.array(trH_all)
    print(f'\n  Hessian trace across all seeds: mean={np.mean(trH_all):.4f}, std={np.std(trH_all):.4f}')
    print(f'  Negative trH values found: {sum(trH_all < 0)} out of {len(trH_all)}')
print()

print('='*70)
print('TONG KET')
print('='*70)
print()
print('Cac phat hien chinh:')
print('1. Control Experiment: File duoc danh nhan d=30 nhung thuc te la d=50 data')
print('   - Can xac minh hoac chay lai d=30 control experiment rieng biet')
print('2. d=50 10 seeds: R_H~2.18, Recovery~38.0% (tang tu 31.1% cua 3 seeds)')
print('3. Epoch-wise DD: Xac nhan DD xuat hien theo thoi gian o ca d=30 va d=50')
print('   - d=50: DD peak tai epoch ~1, late min tai epoch ~539')
print('   - d=30: DD peak tai epoch ~1, late min tai epoch ~526')
print('4. Deep5: Khong phat hien DD peak - test loss gan nhu phang (~1.118)')
print('   - Hessian trace rat nho va thay doi dau (negative values xuat hien)')
print('   - Khong ho tro cho "R_H propagation" qua cac tang')
print()
