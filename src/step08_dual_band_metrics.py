"""
STEP 8 (REVISED v2) — Dual-Band Mechanistic Graph Metric Suite
===============================================================
Runs the full mechanistic metric suite for BOTH Alpha and Beta bands.

Alpha band (8-13 Hz):
  -> Tests the "disconnection" story: long-range efficiency degradation.
     Step 7 showed Alpha Efficiency has the strongest simple metric signal
     (p=0.060, d=0.52), motivating deeper mechanistic analysis here.

Beta band (13-30 Hz):
  -> Tests the "hypercompensation" story: local clustering increase despite
     preserved global efficiency — consistent with compensatory rewiring.

Metrics computed per band per subject:
  - Global Efficiency
  - Local Efficiency
  - Clustering Coefficient
  - Characteristic Path Length (LCC fallback for disconnected graphs)
  - Modularity Q (Louvain if available, greedy modularity fallback)
  - Betweenness Centrality per electrode (nodal hub analysis)

Statistics: Mann-Whitney U + Cohen's d for every metric x band.

Output files:
  - final_isef_data_dualband.csv  (long format: one row per subject per band)
  - final_isef_data.csv           (wide format: one row per subject, used by Step 9)
  - mechanistic_summary_dualband.csv
  - {sub_id}_Alpha_connectivity.npy  and  {sub_id}_Beta_connectivity.npy

Install note (run once if not already installed):
    pip install python-louvain
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

try:
    import community as community_louvain
    USE_LOUVAIN = True
except ImportError:
    USE_LOUVAIN = False
    print("[INFO] python-louvain not found — using NetworkX greedy modularity.")

# ── Configuration ─────────────────────────────────────────────────────────────
PRIMARY_BANDS = {
    'Alpha': (8,  13),
    'Beta':  (13, 30),
}
THRESHOLD_PC = 80          # retain top 20 % of edges
np.random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_path_length(G):
    """Average shortest path length; falls back to largest connected component."""
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.average_shortest_path_length(lcc) if len(lcc) >= 3 else np.nan


def compute_modularity(G):
    if USE_LOUVAIN:
        partition = community_louvain.best_partition(G)
        return community_louvain.modularity(partition, G)
    communities = nx.community.greedy_modularity_communities(G)
    return nx.community.modularity(G, communities)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 +
                      (nb - 1) * np.std(b, ddof=1) ** 2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


# ── Subject list ──────────────────────────────────────────────────────────────
df_participants = pd.read_csv('data/participants.tsv', sep='\t')
df_participants['Group'] = df_participants['Group'].str.strip()

available_subs = sorted([f for f in os.listdir('data/derivatives')
                         if f.startswith('sub-')])
available_subs = [
    s for s in available_subs
    if len(df_participants.loc[
        df_participants['participant_id'] == s, 'Group'].values) > 0
    and df_participants.loc[
        df_participants['participant_id'] == s, 'Group'].values[0] in {'A', 'C'}
]

print(f"--- DUAL-BAND MECHANISTIC SUITE | N={len(available_subs)} ---\n")

# ── Main pipeline ─────────────────────────────────────────────────────────────
results = []

for sub_id in available_subs:
    try:
        group = df_participants.loc[
            df_participants['participant_id'] == sub_id, 'Group'].values[0]

        data_path = (f"data/derivatives/{sub_id}/eeg/"
                     f"{sub_id}_task-eyesclosed_eeg.set")
        raw = mne.io.read_raw_eeglab(data_path, preload=True, verbose=False)

        for band_name, (fmin, fmax) in PRIMARY_BANDS.items():
            raw_band = raw.copy().filter(fmin, fmax,
                                         fir_design='firwin', verbose=False)
            epochs = mne.make_fixed_length_epochs(
                raw_band, duration=2.0, preload=True, verbose=False)

            con = spectral_connectivity_epochs(
                epochs, method='pli', mode='multitaper',
                fmin=fmin, fmax=fmax, faverage=True, verbose=False)
            matrix = con.get_data(output='dense')[:, :, 0]

            # Save per-band connectivity matrix for Step 9
            np.save(f"{sub_id}_{band_name}_connectivity.npy", matrix)

            threshold = np.percentile(matrix, THRESHOLD_PC)
            adj = (matrix > threshold).astype(int)
            np.fill_diagonal(adj, 0)
            G = nx.from_numpy_array(adj)

            # Core metrics
            global_eff = nx.global_efficiency(G)
            local_eff  = nx.local_efficiency(G)
            clustering = nx.average_clustering(G)
            path_len   = safe_path_length(G)
            mod_Q      = compute_modularity(G)
            centrality = nx.betweenness_centrality(G, normalized=True)

            entry = {
                'ID':           sub_id,
                'Group':        group,
                'Band':         band_name,
                'Global_Eff':   global_eff,
                'Local_Eff':    local_eff,
                'Clustering':   clustering,
                'Path_Length':  path_len,
                'Modularity_Q': mod_Q,
            }
            for i, ch in enumerate(raw.ch_names):
                entry[f'Cent_{ch}'] = centrality.get(i, np.nan)

            results.append(entry)

        print(f"  Done: {sub_id} ({group})")

    except Exception as e:
        print(f"  Error on {sub_id}: {e}")

# ── Save long-format CSV ──────────────────────────────────────────────────────
full_df = pd.DataFrame(results)
full_df.to_csv("final_isef_data_dualband.csv", index=False)

# Wide-format: one row per subject — used by Step 9 and Step 10
wide_rows = []
for sub_id in full_df['ID'].unique():
    sub_data = full_df[full_df['ID'] == sub_id]
    group = sub_data['Group'].values[0]
    row = {'ID': sub_id, 'Group': group}
    for band in PRIMARY_BANDS:
        band_row = sub_data[sub_data['Band'] == band]
        if len(band_row) > 0:
            for col in ['Global_Eff', 'Local_Eff', 'Clustering',
                        'Path_Length', 'Modularity_Q']:
                row[f'{band}_{col}'] = band_row[col].values[0]
    wide_rows.append(row)

wide_df = pd.DataFrame(wide_rows)
wide_df.to_csv("final_isef_data.csv", index=False)

# ── Statistical report ────────────────────────────────────────────────────────
core_metrics = ['Global_Eff', 'Local_Eff', 'Clustering', 'Path_Length', 'Modularity_Q']

n_ad = full_df[full_df['Group'] == 'A']['ID'].nunique()
n_cn = full_df[full_df['Group'] == 'C']['ID'].nunique()

print(f"\n{'='*72}")
print(f"  DUAL-BAND MECHANISTIC COMPARISON  (N_AD={n_ad}, N_CN={n_cn})")
print(f"  Key: ✓ = p<0.05   ~ = p<0.10 trend")
print(f"{'='*72}")

summary_rows = []
for band in PRIMARY_BANDS:
    band_df = full_df[full_df['Band'] == band]
    grp_a   = band_df[band_df['Group'] == 'A']
    grp_c   = band_df[band_df['Group'] == 'C']

    print(f"\n  [ {band} Band ]")
    print(f"  {'Metric':<14} {'AD mean':>9} {'CN mean':>9} "
          f"{'p(MWU)':>9} {'Cohen d':>9}  Sig?")
    print(f"  {'-'*60}")

    for m in core_metrics:
        a_vals = grp_a[m].dropna().values
        c_vals = grp_c[m].dropna().values
        if len(a_vals) < 3 or len(c_vals) < 3:
            continue
        _, p = stats.mannwhitneyu(a_vals, c_vals, alternative='two-sided')
        d    = cohens_d(a_vals, c_vals)
        flag = '✓' if p < 0.05 else ('~' if p < 0.10 else '')
        print(f"  {m:<14} {np.mean(a_vals):>9.4f} {np.mean(c_vals):>9.4f} "
              f"{p:>9.4f} {d:>9.3f}  {flag}")
        summary_rows.append({'Band': band, 'Metric': m,
                              'AD_mean': np.mean(a_vals),
                              'CN_mean': np.mean(c_vals),
                              'p_MWU': p, 'cohens_d': d})

# ── Nodal hub disruption ──────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  NODAL HUB DISRUPTION (p < 0.05, either band)")
print(f"{'='*72}")

cent_cols = [c for c in full_df.columns if c.startswith('Cent_')]
found_any = False
for band in PRIMARY_BANDS:
    band_df = full_df[full_df['Band'] == band]
    grp_a   = band_df[band_df['Group'] == 'A']
    grp_c   = band_df[band_df['Group'] == 'C']
    for col in cent_cols:
        a_vals = grp_a[col].dropna().values
        c_vals = grp_c[col].dropna().values
        if len(a_vals) < 3 or len(c_vals) < 3:
            continue
        _, p = stats.mannwhitneyu(a_vals, c_vals, alternative='two-sided')
        if p < 0.05:
            d = cohens_d(a_vals, c_vals)
            print(f"  {band:<6} {col.replace('Cent_', ''):<10} "
                  f"p={p:.4f}  d={d:.3f}")
            found_any = True

if not found_any:
    print("  No individual electrodes reached p < 0.05 in either band.")

pd.DataFrame(summary_rows).to_csv("mechanistic_summary_dualband.csv", index=False)
print(f"\nSaved: final_isef_data_dualband.csv")
print(f"Saved: final_isef_data.csv (wide — used by Steps 9 & 10)")
print(f"Saved: mechanistic_summary_dualband.csv")
print(f"Saved: per-subject Alpha and Beta connectivity .npy files")
