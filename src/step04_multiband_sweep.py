"""
STEP 4: Multi-Band Sweep with Statistics
  - Mann-Whitney U replaces independent t-test for each band comparison
    (non-parametric; appropriate for non-normal EEG metrics with small N)
  - Cohen's d reported for each band so effect magnitude is visible,
    not just significance
  - Bonferroni correction applied across the 3 bands (α_corrected = 0.05/3 = 0.0167) because running 3 tests on the same data inflates Type I error. Both uncorrected and corrected significance flags are shown.
  - Global Efficiency AND Clustering Coefficient tested per band
"""

import mne
import numpy as np
import networkx as nx
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs
from scipy import stats
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
BANDS = {
    'Theta': (4,  8),
    'Alpha': (8,  13),
    'Beta':  (13, 30),
}
THRESHOLD_PC   = 80       # top 20 % edges retained
N_BANDS        = len(BANDS)
ALPHA_NOMINAL  = 0.05
ALPHA_BONF     = ALPHA_NOMINAL / N_BANDS   # = 0.0167

# ── Helpers ───────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a, ddof=1)**2 +
                      (nb-1)*np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


# ── Main pipeline ─────────────────────────────────────────────────────────────
results = []
df_participants = pd.read_csv('data/participants.tsv', sep='\t')
df_participants['Group'] = df_participants['Group'].str.strip()

available_subs = sorted([f for f in os.listdir('data/derivatives')
                         if f.startswith('sub-')])

# Filter to only AD ('A') and CN ('C') — skip FTD
valid_groups = {'A', 'C'}
available_subs = [s for s in available_subs
                  if df_participants.loc[
                      df_participants['participant_id'] == s, 'Group'
                  ].values[0] in valid_groups
                  if len(df_participants.loc[
                      df_participants['participant_id'] == s, 'Group'].values) > 0]

print(f"--- MULTI-BAND SWEEP (REVISED) | N = {len(available_subs)} ---")
print(f"    Bands tested      : {list(BANDS.keys())}")
print(f"    Bonferroni α      : {ALPHA_BONF:.4f}  (0.05 / {N_BANDS} bands)\n")

for sub_id in available_subs:
    try:
        group = df_participants.loc[
            df_participants['participant_id'] == sub_id, 'Group'].values[0]

        data_path = (f"data/derivatives/{sub_id}/eeg/"
                     f"{sub_id}_task-eyesclosed_eeg.set")
        raw = mne.io.read_raw_eeglab(data_path, preload=True, verbose=False)

        for band_name, (fmin, fmax) in BANDS.items():
            raw_band = raw.copy().filter(fmin, fmax,
                                         fir_design='firwin', verbose=False)
            epochs = mne.make_fixed_length_epochs(raw_band, duration=2.0,
                                                   preload=True, verbose=False)

            con = spectral_connectivity_epochs(
                epochs, method='pli', mode='multitaper',
                fmin=fmin, fmax=fmax, faverage=True, verbose=False)
            matrix = con.get_data(output='dense')[:, :, 0]

            threshold = np.percentile(matrix, THRESHOLD_PC)
            adj = (matrix > threshold).astype(int)
            G   = nx.from_numpy_array(adj)

            results.append({
                'ID':         sub_id,
                'Group':      group,
                'Band':       band_name,
                'Efficiency': nx.global_efficiency(G),
                'Clustering': nx.average_clustering(G),
            })

        print(f"  Processed {sub_id} ({group})")

    except Exception as e:
        print(f"  Skipping {sub_id}: {e}")

# ── Statistical analysis ──────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv("multiband_results.csv", index=False)

metrics = ['Efficiency', 'Clustering']

print(f"\n{'='*72}")
print(f"  MULTI-BAND STATISTICAL REPORT")
print(f"  (Bonferroni-corrected α = {ALPHA_BONF:.4f} | Nominal α = {ALPHA_NOMINAL})")
print(f"{'='*72}")

summary_rows = []

for band in BANDS:
    band_data = df[df['Band'] == band]
    group_a   = band_data[band_data['Group'] == 'A']
    group_c   = band_data[band_data['Group'] == 'C']

    print(f"\n  [ {band} Band ]  N_AD={len(group_a)}, N_CN={len(group_c)}")
    print(f"  {'Metric':<12}  {'AD mean':>8}  {'CN mean':>8}  "
          f"{'p (MWU)':>9}  {'Cohen d':>8}  Nominal  Bonferroni")
    print(f"  {'-'*68}")

    for metric in metrics:
        a_vals = group_a[metric].dropna().values
        c_vals = group_c[metric].dropna().values

        _, p = stats.mannwhitneyu(a_vals, c_vals, alternative='two-sided')
        d    = cohens_d(a_vals, c_vals)

        nom_sig  = '✓' if p < ALPHA_NOMINAL else ''
        bonf_sig = '✓' if p < ALPHA_BONF    else ''

        print(f"  {metric:<12}  {np.mean(a_vals):>8.4f}  {np.mean(c_vals):>8.4f}  "
              f"{p:>9.4f}  {d:>8.3f}  {nom_sig:^7}  {bonf_sig:^10}")

        summary_rows.append({
            'Band': band, 'Metric': metric,
            'AD_mean': np.mean(a_vals), 'CN_mean': np.mean(c_vals),
            'p_MWU': p, 'cohens_d': d,
            'sig_nominal': p < ALPHA_NOMINAL,
            'sig_bonferroni': p < ALPHA_BONF,
        })

print(f"\n{'='*72}")

# ── Recommendation ────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
bonf_hits  = summary_df[summary_df['sig_bonferroni']]

print("\n  BAND SELECTION RECOMMENDATION:")
if len(bonf_hits) == 0:
    nom_hits = summary_df[summary_df['sig_nominal']]
    if len(nom_hits) > 0:
        best = nom_hits.loc[nom_hits['p_MWU'].idxmin()]
        print(f"  No band survives Bonferroni correction.")
        print(f"  Strongest nominal signal: {best['Band']} / {best['Metric']} "
              f"(p={best['p_MWU']:.4f}, d={best['cohens_d']:.3f})")
        print(f"  → Report as exploratory. Do not claim confirmatory significance.")
    else:
        print(f"  No band reaches nominal significance. Increase N or review preprocessing.")
else:
    best = bonf_hits.loc[bonf_hits['p_MWU'].idxmin()]
    print(f"  Band surviving Bonferroni correction: {best['Band']} / {best['Metric']}")
    print(f"  p = {best['p_MWU']:.4f},  Cohen's d = {best['cohens_d']:.3f}")
    print(f"  → Use {best['Band']} band as primary analysis band in Steps 8–10.")

summary_df.to_csv("multiband_stats_summary.csv", index=False)
print(f"\nSaved: multiband_results.csv  |  multiband_stats_summary.csv")
