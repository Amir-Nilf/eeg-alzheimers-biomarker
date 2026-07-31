"""
STEP 9: HUB CENTRALITY ANALYSIS - ALL THREE BANDS
Computes betweenness centrality statistics for all 19 electrodes across
Theta, Alpha, and Beta bands.

Total: 57 tests (19 electrodes × 3 bands)
Applies FDR correction across all 57 tests.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import sys
import os

print("="*80)
print("HUB CENTRALITY ANALYSIS - ALL THREE BANDS")
print("="*80)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading data...")

try:
    df_alpha_beta = pd.read_csv('final_isef_data_dualband.csv')
    theta_df = pd.read_csv('theta_band_data.csv')
    print("  ✓ Data loaded successfully")
except FileNotFoundError as e:
    print(f"\n✗ ERROR: Required data file not found: {e}")
    print("\nMake sure you've run:")
    print("  1. Your original analysis (creates final_isef_data_dualband.csv)")
    print("  2. GENERATE_THETA_DATA.py (creates theta_band_data.csv)")
    sys.exit(1)

# Separate bands
alpha_df = df_alpha_beta[df_alpha_beta['Band']=='Alpha'].copy()
beta_df = df_alpha_beta[df_alpha_beta['Band']=='Beta'].copy()

print(f"  • Alpha: {len(alpha_df)} subjects")
print(f"  • Beta: {len(beta_df)} subjects")
print(f"  • Theta: {len(theta_df)} subjects")

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def bootstrap_ci(a, b, n_boot=10000, ci=95):
    """Bootstrap confidence interval for Cohen's d"""
    boot_d = []
    for _ in range(n_boot):
        a_sample = np.random.choice(a, size=len(a), replace=True)
        b_sample = np.random.choice(b, size=len(b), replace=True)
        boot_d.append(cohens_d(a_sample, b_sample))
    
    lo = np.percentile(boot_d, (100 - ci) / 2)
    hi = np.percentile(boot_d, 100 - (100 - ci) / 2)
    return lo, hi

# ══════════════════════════════════════════════════════════════════════════════
# ANALYZE ALL ELECTRODES × ALL BANDS
# ══════════════════════════════════════════════════════════════════════════════

print("\nAnalyzing all electrode centralities...")

# Get electrode names from columns
cent_cols = [c for c in alpha_df.columns if c.startswith('Cent_')]
electrodes = [c.replace('Cent_', '') for c in cent_cols]

print(f"  Found {len(electrodes)} electrodes")

all_results = []

for band_name, df in [('Theta', theta_df), ('Alpha', alpha_df), ('Beta', beta_df)]:
    print(f"\n  Processing {band_name} band...")
    
    ad_group = df[df['Group']=='A']
    cn_group = df[df['Group']=='C']
    
    for elec in electrodes:
        col_name = f'Cent_{elec}'
        
        if col_name not in df.columns:
            print(f"    ⚠ Warning: {col_name} not found in {band_name} data, skipping")
            continue
        
        ad_vals = ad_group[col_name].dropna().values
        cn_vals = cn_group[col_name].dropna().values
        
        if len(ad_vals) < 2 or len(cn_vals) < 2:
            continue
        
        # Statistics
        u_stat, p_raw = stats.mannwhitneyu(ad_vals, cn_vals, alternative='two-sided')
        d = cohens_d(ad_vals, cn_vals)
        d_ci_lo, d_ci_hi = bootstrap_ci(ad_vals, cn_vals)
        
        all_results.append({
            'Band': band_name,
            'Electrode': elec,
            'AD_mean': np.mean(ad_vals),
            'AD_std': np.std(ad_vals, ddof=1),
            'CN_mean': np.mean(cn_vals),
            'CN_std': np.std(cn_vals, ddof=1),
            'p_raw': p_raw,
            'cohens_d': d,
            'd_CI_lo': d_ci_lo,
            'd_CI_hi': d_ci_hi,
            'U_statistic': u_stat
        })

# Create DataFrame
results_df = pd.DataFrame(all_results)

print(f"\n  Total tests: {len(results_df)}")

# ══════════════════════════════════════════════════════════════════════════════
# FDR CORRECTION ACROSS ALL TESTS
# ══════════════════════════════════════════════════════════════════════════════

print("\nApplying FDR correction across all 57 tests...")

# Get all p-values
p_values = results_df['p_raw'].values

# FDR correction
sig_fdr, p_fdr = fdrcorrection(p_values, alpha=0.05, method='indep')

# Add to DataFrame
results_df['sig_raw'] = results_df['p_raw'] < 0.05
results_df['sig_fdr'] = sig_fdr
results_df['p_fdr'] = p_fdr

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

output_file = 'hub_centrality_results_ALL_BANDS.csv'
results_df.to_csv(output_file, index=False)

print(f"\n✓ Saved: {output_file}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)

print(f"\nTotal tests performed: {len(results_df)}")
print(f"  • Theta: {len(results_df[results_df['Band']=='Theta'])}")
print(f"  • Alpha: {len(results_df[results_df['Band']=='Alpha'])}")
print(f"  • Beta: {len(results_df[results_df['Band']=='Beta'])}")

print(f"\nNominal significance (p<0.05): {sig_fdr.sum()}")
sig_raw_count = (results_df['p_raw'] < 0.05).sum()
print(f"  • Raw p<0.05: {sig_raw_count} tests")
print(f"  • FDR-corrected: {sig_fdr.sum()} tests")

if sig_fdr.sum() > 0:
    print("\nTests surviving FDR correction:")
    print(f"  {'Band':<8}  {'Electrode':<10}  {'p_raw':<10}  {'p_FDR':<10}  {'Cohens d':<10}")
    print(f"  {'-'*60}")
    
    fdr_sig = results_df[results_df['sig_fdr']].sort_values('p_raw')
    for _, row in fdr_sig.iterrows():
        print(f"  {row['Band']:<8}  {row['Electrode']:<10}  "
              f"{row['p_raw']:<10.4f}  {row['p_fdr']:<10.4f}  {row['cohens_d']:<10.3f}")
else:
    print("\nNo tests survived FDR correction at α=0.05")

# THETA-specific summary
print("\n" + "="*80)
print("THETA BAND SPECIFIC FINDINGS")
print("="*80)

theta_results = results_df[results_df['Band']=='Theta']
theta_sig = theta_results[theta_results['p_raw'] < 0.05]

if len(theta_sig) > 0:
    print(f"\nNominal significant results (p<0.05): {len(theta_sig)}")
    print(f"  {'Electrode':<10}  {'p_raw':<10}  {'Cohens d':<10}  {'FDR?'}")
    print(f"  {'-'*50}")
    for _, row in theta_sig.sort_values('p_raw').iterrows():
        fdr_status = '✓' if row['sig_fdr'] else '✗'
        print(f"  {row['Electrode']:<10}  {row['p_raw']:<10.4f}  "
              f"{row['cohens_d']:<10.3f}  {fdr_status}")
else:
    print("\n✓ NO significant electrode effects in Theta band")
    print("  This confirms prescreening results: Theta network topology preserved")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
