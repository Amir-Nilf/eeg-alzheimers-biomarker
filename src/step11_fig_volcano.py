"""
VOLCANO PLOT - ALL THREE BANDS (REAL DATA ONLY)
Shows all 57 electrode tests (19 electrodes × 3 bands) with FDR correction.

Color coding:
  - Pink circles = Theta band
  - Green circles = Alpha band  
  - Yellow circles = Beta band
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

print("="*80)
print("VOLCANO PLOT - ALL THREE BANDS (REAL DATA)")
print("="*80)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

data_file = 'hub_centrality_results_ALL_BANDS.csv'

if not os.path.exists(data_file):
    print(f"\n✗ ERROR: {data_file} not found!")
    print("\nYou must first run:")
    print("  python3 analyze_all_bands_centrality.py")
    print("\nThis generates the complete hub centrality results for all 3 bands.")
    sys.exit(1)

print(f"\nLoading: {data_file}")
df = pd.read_csv(data_file)

print(f"  ✓ Loaded {len(df)} tests")
print(f"    • Theta: {len(df[df['Band']=='Theta'])}")
print(f"    • Alpha: {len(df[df['Band']=='Alpha'])}")
print(f"    • Beta: {len(df[df['Band']=='Beta'])}")

# Calculate -log10(p)
df['neg_log10_p'] = -np.log10(df['p_raw'])

# ══════════════════════════════════════════════════════════════════════════════
# CREATE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating volcano plot...")

fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor('white')

# Color mapping
colors = {'Theta': '#FF69B4', 'Alpha': '#2ECC71', 'Beta': '#F4D03F'}

# Plot by significance level
for band in ['Theta', 'Alpha', 'Beta']:
    band_data = df[df['Band'] == band]
    
    # FDR-significant (gold ring, larger)
    fdr_sig = band_data[band_data['sig_fdr']]
    if len(fdr_sig) > 0:
        ax.scatter(fdr_sig['cohens_d'], fdr_sig['neg_log10_p'],
                   s=300, c=colors[band], edgecolors='red', linewidths=4,
                   alpha=0.9, zorder=5, label=f'{band} FDR-sig')
    
    # Nominally significant (black ring)
    nom_sig = band_data[(~band_data['sig_fdr']) & (band_data['p_raw'] < 0.05)]
    if len(nom_sig) > 0:
        ax.scatter(nom_sig['cohens_d'], nom_sig['neg_log10_p'],
                   s=180, c=colors[band], edgecolors='black', linewidths=2,
                   alpha=0.7, zorder=4)
    
    # Not significant (gray ring, smaller)
    not_sig = band_data[band_data['p_raw'] >= 0.05]
    if len(not_sig) > 0:
        ax.scatter(not_sig['cohens_d'], not_sig['neg_log10_p'],
                   s=100, c=colors[band], edgecolors='gray', linewidths=0.5,
                   alpha=0.4, zorder=3)

# Threshold lines
ax.axhline(-np.log10(0.05), color='orange', linestyle='--', linewidth=2.5,
           alpha=0.7, label='p=0.05 (nominal)', zorder=2)
ax.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5, zorder=1)

# Annotate FDR-significant points with BIGGER labels
fdr_points = df[df['sig_fdr']].sort_values('p_raw')
for idx, row in fdr_points.iterrows():
    label = f"{row['Band'][0]}-{row['Electrode']}"
    
    # Offset to avoid overlap
    offset_x = 15 if row['cohens_d'] > 0 else -15
    offset_y = 15
    
    ax.annotate(label, (row['cohens_d'], row['neg_log10_p']),
                xytext=(offset_x, offset_y), textcoords='offset points',
                fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                         edgecolor='black', linewidth=2.5, alpha=0.95),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Labels and title
ax.set_xlabel('Cohen\'s d (Effect Size)\n← CN > AD  |  AD > CN →',
              fontsize=14, fontweight='bold')
ax.set_ylabel('-log₁₀(p-value)\n(Higher = More Significant)',
              fontsize=14, fontweight='bold')
ax.set_title('Volcano Plot: Hub Centrality Across All Electrodes\n'
             '57 tests (19 electrodes × 3 bands) with FDR correction | 100% REAL DATA',
             fontsize=15, fontweight='bold', pad=20)

# Custom legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Patch(facecolor='#FF69B4', edgecolor='black', linewidth=1.5, label='Theta (4-8 Hz)'),
    Patch(facecolor='#2ECC71', edgecolor='black', linewidth=1.5, label='Alpha (8-13 Hz)'),
    Patch(facecolor='#F4D03F', edgecolor='black', linewidth=1.5, label='Beta (13-30 Hz)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=13, markeredgecolor='red', markeredgewidth=4,
           label='FDR-corrected p<0.05 ★', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=11, markeredgecolor='black', markeredgewidth=2,
           label='Nominal p<0.05', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
           markersize=8, markeredgecolor='gray', markeredgewidth=0.5,
           label='Not significant', linestyle='None'),
]

ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
          framealpha=0.95, edgecolor='black', fancybox=True, shadow=True)

# Info box explaining -log10(p)
info_text = (
    "Why -log₁₀(p)?\n\n"
    "Spreads out significance:\n"
    "p=0.05  → 1.3\n"
    "p=0.01  → 2.0\n"
    "p=0.001 → 3.0\n\n"
    "Higher = More significant"
)
ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10, family='monospace',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                 edgecolor='black', linewidth=2, alpha=0.95))

# Grid
ax.grid(alpha=0.3, zorder=0)

# Save
FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'volcano-plot-all.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ Saved: Fig2_Volcano_ALL_BANDS_REAL.png")

# Summary
print("\n" + "="*80)
print("VOLCANO PLOT COMPLETE")
print("="*80)
print(f"\n✓ All {len(df)} tests plotted")
print(f"✓ FDR-significant tests: {df['sig_fdr'].sum()}")
print("✓ 100% REAL DATA")
print("="*80)
