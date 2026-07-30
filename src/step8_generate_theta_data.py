"""
STEP 8: THETA-BAND COMPLETE ANALYSIS
Generates ALL Theta-band data needed for figures:
  - 5 core metrics (Global Eff, Local Eff, Clustering, Path Length, Modularity Q)
  - 19 electrode betweenness centrality values
  - Saves to theta_band_data.csv

This script processes the ACTUAL EEG data - NO SYNTHETIC DATA.
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

# Try to import Louvain
try:
    import community as community_louvain
    USE_LOUVAIN = True
except ImportError:
    USE_LOUVAIN = False
    print("[INFO] Installing python-louvain...")
    os.system("pip install python-louvain --break-system-packages")
    try:
        import community as community_louvain
        USE_LOUVAIN = True
    except:
        USE_LOUVAIN = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TARGET_BAND  = (4, 8)      # THETA BAND
BAND_NAME    = "Theta"
THRESHOLD_PC = 80          # Top 20% edges retained
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def safe_path_length(G):
    """Return average shortest path length, using LCC if disconnected."""
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if len(lcc) < 3:
        return np.nan
    return nx.average_shortest_path_length(lcc)

def modularity(G):
    """Modularity Q using Louvain or greedy fallback."""
    if USE_LOUVAIN:
        partition = community_louvain.best_partition(G)
        return community_louvain.modularity(partition, G)
    else:
        communities = nx.community.greedy_modularity_communities(G)
        return nx.community.modularity(G, communities)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

print("="*80)
print(f"THETA-BAND ANALYSIS | GENERATING COMPLETE DATASET")
print("="*80)
print(f"\nProcessing {BAND_NAME} band ({TARGET_BAND[0]}-{TARGET_BAND[1]} Hz)")
print("This will take ~5-10 minutes for all 65 subjects...")
print()

results = []
df_participants = pd.read_csv('data/participants.tsv', sep='\t')
df_participants['Group'] = df_participants['Group'].str.strip()

available_subs = sorted([f for f in os.listdir('data/derivatives') if f.startswith('sub-')])

for idx, sub_id in enumerate(available_subs, 1):
    try:
        group = df_participants.loc[
            df_participants['participant_id'] == sub_id, 'Group'].values[0]

        data_path = f"data/derivatives/{sub_id}/eeg/{sub_id}_task-eyesclosed_eeg.set"
        raw = mne.io.read_raw_eeglab(data_path, preload=True, verbose=False)
        raw.filter(*TARGET_BAND, fir_design='firwin', verbose=False)
        epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True, verbose=False)

        con = spectral_connectivity_epochs(
            epochs, method='pli', mode='multitaper',
            fmin=TARGET_BAND[0], fmax=TARGET_BAND[1],
            faverage=True, verbose=False)
        matrix = con.get_data(output='dense')[:, :, 0]
        
        # Save connectivity matrix
        np.save(f"{sub_id}_Theta_connectivity.npy", matrix)

        threshold = np.percentile(matrix, THRESHOLD_PC)
        adj = (matrix > threshold).astype(int)
        G   = nx.from_numpy_array(adj)

        # ══ Core metrics ═══════════════════════════════════════════════════════
        global_eff  = nx.global_efficiency(G)
        local_eff   = nx.local_efficiency(G)
        clustering  = nx.average_clustering(G)
        path_len    = safe_path_length(G)
        mod_Q       = modularity(G)

        # ══ Betweenness centrality (all electrodes) ════════════════════════════
        centrality_dict = nx.betweenness_centrality(G, normalized=True)

        res_entry = {
            'ID':              sub_id,
            'Group':           group,
            'Band':            'Theta',
            'Global_Eff':      global_eff,
            'Local_Eff':       local_eff,
            'Clustering':      clustering,
            'Path_Length':     path_len,
            'Modularity_Q':    mod_Q,
        }
        
        # Add all electrode centralities
        for i, ch in enumerate(raw.ch_names):
            res_entry[f'Cent_{ch}'] = centrality_dict.get(i, np.nan)

        results.append(res_entry)
        
        print(f"  [{idx}/65] {sub_id} ({group})  |  Eff={global_eff:.3f}  Clust={clustering:.3f}  "
              f"PL={path_len:.3f}  Q={mod_Q:.3f}")

    except Exception as e:
        print(f"  [{idx}/65] ERROR on {sub_id}: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

full_df = pd.DataFrame(results)
full_df.to_csv("theta_band_data.csv", index=False)

print("\n" + "="*80)
print("THETA-BAND ANALYSIS COMPLETE")
print("="*80)
print(f"\nProcessed: {len(results)} subjects")
print(f"Saved: theta_band_data.csv")
print(f"Saved: {len(results)} connectivity .npy files (sub-XXX_Theta_connectivity.npy)")

# ══════════════════════════════════════════════════════════════════════════════
# QUICK STATISTICS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

group_a = full_df[full_df['Group'] == 'A']
group_c = full_df[full_df['Group'] == 'C']

core_metrics = ['Global_Eff', 'Local_Eff', 'Clustering', 'Path_Length', 'Modularity_Q']

print("\n" + "="*80)
print(f"THETA-BAND STATISTICS (N_AD={len(group_a)}, N_CN={len(group_c)})")
print("="*80)
print(f"  {'Metric':<18}  {'AD mean':>9}  {'CN mean':>9}  {'p (MWU)':>9}  Significant?")
print(f"  {'-'*70}")

for m in core_metrics:
    a_vals = group_a[m].dropna().values
    c_vals = group_c[m].dropna().values
    _, p = stats.mannwhitneyu(a_vals, c_vals, alternative='two-sided')
    sig = '✓' if p < 0.05 else ''
    print(f"  {m:<18}  {np.mean(a_vals):>9.4f}  {np.mean(c_vals):>9.4f}  {p:>9.4f}  {sig}")

print("="*80)
print("\n✓ Theta data ready for figure generation!")
print("="*80)
