"""
STEP 9: Hub Centrality Deep-Dive with FDR Correction
Primary finding pivot: the Fz hub disruption (d=0.945, p<0.0001) and
Alpha-band network fragmentation are the headline results. This step
performs a rigorous nodal analysis across all electrodes in both bands.

What this step does:
  1. Loads the per-subject per-band data saved by Step 8.
  2. Tests every electrode's Betweenness Centrality for group differences
     (Mann-Whitney U) in both Alpha and Beta bands.
  3. Applies Benjamini-Hochberg FDR correction across all electrode × band
     combinations (38 tests total: 19 electrodes × 2 bands).
     This is the correct multiple comparison correction for nodal analysis —
     more statistically powerful than Bonferroni for this many tests.
  4. Reports effect sizes (Cohen's d) and bootstrap 95% CIs for every
     significant node.
  5. Builds a topographic summary table sorted by effect size.
  6. Saves hub_centrality_results.csv for Step 10.

Why FDR over Bonferroni here:
  Bonferroni at 38 tests requires p < 0.0013 to claim significance — too
  conservative for exploratory nodal mapping where we expect spatially
  correlated findings. Benjamini-Hochberg FDR controls the expected
  proportion of false discoveries while preserving more true positives.
  Both uncorrected and FDR-corrected significance are reported.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a, ddof=1)**2 +
                      (nb-1)*np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def bootstrap_ci_d(a, b, n=10_000, ci=95):
    boot = [cohens_d(np.random.choice(a, len(a), replace=True),
                     np.random.choice(b, len(b), replace=True))
            for _ in range(n)]
    return np.percentile(boot, (100-ci)/2), np.percentile(boot, 100-(100-ci)/2)


# ── Load data ─────────────────────────────────────────────────────────────────
long_df = pd.read_csv("final_isef_data_dualband.csv")

grp_a_ids = long_df[long_df['Group'] == 'A']['ID'].unique()
grp_c_ids = long_df[long_df['Group'] == 'C']['ID'].unique()

print(f"--- HUB CENTRALITY ANALYSIS WITH FDR CORRECTION ---")
print(f"    N_AD={len(grp_a_ids)}, N_CN={len(grp_c_ids)}")
print(f"    Bands: Alpha, Beta")

cent_cols = [c for c in long_df.columns if c.startswith('Cent_')]
electrodes = [c.replace('Cent_', '') for c in cent_cols]
bands = ['Alpha', 'Beta']

print(f"    Electrodes: {len(electrodes)}")
print(f"    Total tests (FDR pool): {len(electrodes) * len(bands)}\n")

# ── Collect all p-values for FDR ─────────────────────────────────────────────
all_rows = []

for band in bands:
    band_df = long_df[long_df['Band'] == band]
    grp_a   = band_df[band_df['Group'] == 'A']
    grp_c   = band_df[band_df['Group'] == 'C']

    for electrode, col in zip(electrodes, cent_cols):
        a_vals = grp_a[col].dropna().values
        c_vals = grp_c[col].dropna().values

        if len(a_vals) < 3 or len(c_vals) < 3:
            continue

        u_stat, p_raw = stats.mannwhitneyu(a_vals, c_vals,
                                            alternative='two-sided')
        d = cohens_d(a_vals, c_vals)

        all_rows.append({
            'Band':       band,
            'Electrode':  electrode,
            'AD_mean':    np.mean(a_vals),
            'CN_mean':    np.mean(c_vals),
            'U_stat':     u_stat,
            'p_raw':      p_raw,
            'cohens_d':   d,
            'AD_vals':    a_vals,
            'CN_vals':    c_vals,
        })

# ── Apply FDR correction ──────────────────────────────────────────────────────
p_raw_array = np.array([r['p_raw'] for r in all_rows])
reject, p_fdr, _, _ = multipletests(p_raw_array, alpha=0.05, method='fdr_bh')

for i, row in enumerate(all_rows):
    row['p_fdr']        = p_fdr[i]
    row['sig_raw']      = row['p_raw'] < 0.05
    row['sig_fdr']      = reject[i]

results_df = pd.DataFrame([{k: v for k, v in r.items()
                             if k not in ('AD_vals', 'CN_vals')}
                            for r in all_rows])

# ── Bootstrap CIs for significant nodes ──────────────────────────────────────
print("Computing bootstrap CIs for nominally significant nodes...")
ci_data = {}
for row in all_rows:
    key = (row['Band'], row['Electrode'])
    if row['sig_raw']:
        lo, hi = bootstrap_ci_d(row['AD_vals'], row['CN_vals'])
        ci_data[key] = (lo, hi)
    else:
        ci_data[key] = (np.nan, np.nan)

results_df['d_CI_lo'] = results_df.apply(
    lambda r: ci_data[(r['Band'], r['Electrode'])][0], axis=1)
results_df['d_CI_hi'] = results_df.apply(
    lambda r: ci_data[(r['Band'], r['Electrode'])][1], axis=1)

# ── Print report ──────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  NODAL HUB DISRUPTION — FULL REPORT")
print(f"  FDR method: Benjamini-Hochberg  |  α = 0.05")
print(f"{'='*72}")

for band in bands:
    band_res = results_df[results_df['Band'] == band].sort_values('p_raw')
    sig_raw  = band_res[band_res['sig_raw']]
    sig_fdr  = band_res[band_res['sig_fdr']]

    print(f"\n  [ {band} Band ]  "
          f"Nominally significant: {len(sig_raw)}  |  "
          f"FDR-corrected: {len(sig_fdr)}")
    print(f"  {'Electrode':<10} {'AD mean':>8} {'CN mean':>8} "
          f"{'p raw':>8} {'p FDR':>8} {'Cohen d':>9} "
          f"{'95% CI':>18}  Sig?")
    print(f"  {'-'*72}")

    # Show all nominally significant + top 3 trends
    show = band_res[band_res['p_raw'] < 0.15].head(10)
    for _, r in show.iterrows():
        fdr_flag = '✓FDR' if r['sig_fdr'] else ('✓raw' if r['sig_raw'] else '~')
        ci_str   = (f"[{r['d_CI_lo']:.3f}, {r['d_CI_hi']:.3f}]"
                    if not np.isnan(r['d_CI_lo']) else "  —  ")
        print(f"  {r['Electrode']:<10} {r['AD_mean']:>8.4f} {r['CN_mean']:>8.4f} "
              f"{r['p_raw']:>8.4f} {r['p_fdr']:>8.4f} {r['cohens_d']:>9.3f} "
              f"{ci_str:>18}  {fdr_flag}")

# ── Headline finding: Fz ──────────────────────────────────────────────────────
fz_row = results_df[(results_df['Band'] == 'Alpha') &
                    (results_df['Electrode'] == 'Fz')]
if len(fz_row) > 0:
    fz = fz_row.iloc[0]
    print(f"\n{'='*72}")
    print(f"  HEADLINE FINDING: Alpha-band Fz Hub Disruption")
    print(f"{'='*72}")
    print(f"  Fz Betweenness Centrality (Alpha band):")
    print(f"    AD mean : {fz['AD_mean']:.4f}")
    print(f"    CN mean : {fz['CN_mean']:.4f}")
    print(f"    p (raw) : {fz['p_raw']:.6f}")
    print(f"    p (FDR) : {fz['p_fdr']:.6f}")
    print(f"    Cohen's d = {fz['cohens_d']:.3f}  "
          f"(95% CI: [{fz['d_CI_lo']:.3f}, {fz['d_CI_hi']:.3f}])")
    mag = ("large" if abs(fz['cohens_d']) >= 0.8 else
           "medium" if abs(fz['cohens_d']) >= 0.5 else "small")
    print(f"    Effect magnitude: {mag}")
    print(f"\n  Interpretation:")
    print(f"  Fz is a frontal midline hub at the intersection of the default")
    print(f"  mode network and executive control network — circuits that fail")
    print(f"  earliest in AD. {'Higher' if fz['cohens_d'] > 0 else 'Lower'} "
          f"centrality in AD suggests this hub is")
    print(f"  {'absorbing more traffic as surrounding routes fail'if fz['cohens_d'] > 0 else 'losing its bridging role as surrounding connectivity degrades'}.")

# ── Summary table sorted by |d| ───────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  ALL SIGNIFICANT NODES (p_raw < 0.05) RANKED BY EFFECT SIZE")
print(f"{'='*72}")
sig_all = results_df[results_df['sig_raw']].copy()
sig_all['abs_d'] = sig_all['cohens_d'].abs()
sig_all = sig_all.sort_values('abs_d', ascending=False)

print(f"  {'Band':<7} {'Electrode':<10} {'Cohen d':>9} "
      f"{'p raw':>8} {'p FDR':>8}  FDR sig?")
print(f"  {'-'*55}")
for _, r in sig_all.iterrows():
    print(f"  {r['Band']:<7} {r['Electrode']:<10} {r['cohens_d']:>9.3f} "
          f"{r['p_raw']:>8.4f} {r['p_fdr']:>8.4f}  "
          f"{'YES' if r['sig_fdr'] else 'no'}")

# ── Save ──────────────────────────────────────────────────────────────────────
results_df.drop(columns=['AD_vals', 'CN_vals'], errors='ignore')
save_df = results_df[[c for c in results_df.columns
                       if c not in ('AD_vals', 'CN_vals')]]
save_df.to_csv("hub_centrality_results.csv", index=False)
print(f"\nSaved: hub_centrality_results.csv")
