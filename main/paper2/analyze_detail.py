import json, numpy as np
from scipy import stats

# Detailed epoch-wise analysis
with open('outputs/epochwise_d50.json') as f:
    ew50_list = json.load(f)
with open('outputs/epochwise_d30.json') as f:
    ew30_list = json.load(f)

ew50 = np.array(ew50_list)
ew30 = np.array(ew30_list)

l50 = np.array([d['test_loss'] for d in ew50])
l30 = np.array([d['test_loss'] for d in ew30])

def rolling_mean(arr, w):
    result = np.zeros(len(arr))
    for i in range(len(arr)):
        lo = max(0, i - w//2)
        hi = min(len(arr), i + w//2)
        result[i] = np.mean(arr[lo:hi])
    return result

rm50 = rolling_mean(l50, 50)
rm30 = rolling_mean(l30, 50)

# Find DD peak in rolling mean (after epoch 1, before epoch 500)
rm50_early = rm50[1:500]
rm50_peak_idx = np.argmax(rm50_early) + 1
rm50_peak_val = rm50_early[np.argmax(rm50_early)]
print("D50 rolling mean peak (epoch 1-500): epoch {}, loss={:.6f}".format(ew50[rm50_peak_idx]['epoch'], rm50_peak_val))

rm30_early = rm30[1:500]
rm30_peak_idx = np.argmax(rm30_early) + 1
rm30_peak_val = rm30_early[np.argmax(rm30_early)]
print("D30 rolling mean peak (epoch 1-500): epoch {}, loss={:.6f}".format(ew30[rm30_peak_idx]['epoch'], rm30_peak_val))

# DD amplitude
late_min_50 = np.min(l50[4000:])
dd_amplitude_50 = (rm50_peak_val - late_min_50) / rm50_peak_val
late_min_30 = np.min(l30[4000:])
dd_amplitude_30 = (rm30_peak_val - late_min_30) / rm30_peak_val
print("DD amplitude d50: {:.2f}%".format(dd_amplitude_50*100))
print("DD amplitude d30: {:.2f}%".format(dd_amplitude_30*100))

# Deep5 analysis
with open('outputs/deep5_tanh_d30_heavy.json') as f:
    deep5 = json.load(f)

trH_low = [d['trH_mean'] for d in deep5 if d['gamma'] <= 1.0]
trH_high = [d['trH_mean'] for d in deep5 if d['gamma'] >= 2.0]
R_H_deep5 = np.mean(trH_low) / np.mean(trH_high) if np.mean(trH_high) > 0 else float('nan')
print("Deep5 R_H = {:.4f}".format(R_H_deep5))

test_means = np.array([d['test_mean'] for d in deep5])
trH_means = np.array([d['trH_mean'] for d in deep5])
print("Deep5 test loss range: {:.6f} ({:.2f}% variation)".format(max(test_means)-min(test_means), (max(test_means)-min(test_means))/np.mean(test_means)*100))

# Count negative Hessian values
trH_all = []
for d in deep5:
    trH_all.extend(d['seeds_trH'])
neg_count = sum(1 for v in trH_all if v < 0)
print("Deep5 negative trH: {} out of {} ({:.1f}%)".format(neg_count, len(trH_all), neg_count/len(trH_all)*100))

# Summary for d=50 3 seeds vs 10 seeds
with open('outputs/cluster1_results.json') as f:
    d1 = json.load(f)
with open('outputs_d50_10seeds/cluster1_results.json') as f:
    d2 = json.load(f)

print("\n=== File Comparison ===")
sr1 = d1['sharpness_ratios']['tanh']
sr2 = d2['sharpness_ratios']['tanh']
print("File1 (3 seeds): R_H={:.4f}, peak={:.6f}, recovery={:.4f}".format(sr1['R_H'], sr1['peak'], sr1['recovery']))
print("File2 (10 seeds?): R_H={:.4f}, peak={:.6f}, recovery={:.4f}".format(sr2['R_H'], sr2['peak'], sr2['recovery']))
print("Difference in recovery: +{:.2f}%".format((sr2['recovery'] - sr1['recovery'])*100))

# Count gamma points
print("File1 tanh points: {}".format(len(d1['results']['tanh'])))
print("File2 tanh points: {}".format(len(d2['results']['tanh'])))
g1 = [pt['gamma'] for pt in d1['results']['tanh']]
g2 = [pt['gamma'] for pt in d2['results']['tanh']]
print("File1 gammas: {}".format(g1))
print("File2 gammas: {}".format(g2))
