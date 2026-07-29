"""
FIGURE 3 (FIXED): TOPOGRAPHY - ALL THREE BANDS (REAL DATA ONLY)
=================================================================
Shows electrode-level effects on 3 scalp maps (Theta, Alpha, Beta).

IMPROVEMENTS:
  - BIGGER electrode circles (easier to see)
  - Comprehensive legend with size/color key
  - 100% REAL DATA from hub_centrality_results_ALL_BANDS.csv

REQUIRES:
  - hub_centrality_results_ALL_BANDS.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import pandas as pd
import sys
import os

print("="*80)
print("TOPOGRAPHY FIGURE - ALL THREE BANDS (REAL DATA)")
print("="*80)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

data_file = 'hub_centrality_results_ALL_BANDS.csv'

if not os.path.exists(data_file):
    print(f"\n✗ ERROR: {data_file} not found!")
    print("\nRun first: python3 analyze_all_bands_centrality.py")
    sys.exit(1)

print(f"\nLoading: {data_file}")
df = pd.read_csv(data_file)
print(f"  ✓ Loaded {len(df)} tests")

# ══════════════════════════════════════════════════════════════════════════════
# ELECTRODE POSITIONS (10-20 system)
# ══════════════════════════════════════════════════════════════════════════════

electrode_pos = {
    'Fp1': (-0.3, 0.85), 'Fp2': (0.3, 0.85),
    'F7': (-0.7, 0.6), 'F3': (-0.4, 0.5), 'Fz': (0, 0.5), 'F4': (0.4, 0.5), 'F8': (0.7, 0.6),
    'T3': (-0.85, 0), 'C3': (-0.4, 0), 'Cz': (0, 0), 'C4': (0.4, 0), 'T4': (0.85, 0),
    'T5': (-0.7, -0.6), 'P3': (-0.4, -0.5), 'Pz': (0, -0.5), 'P4': (0.4, -0.5), 'T6': (0.7, -0.6),
    'O1': (-0.3, -0.85), 'O2': (0.3, -0.85)
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def draw_head(ax):
    """Draw head outline with nose and ears"""
    # Head
    head = Circle((0, 0), 1.05, fill=False, edgecolor='black', linewidth=4)
    ax.add_patch(head)
    
    # Nose
    ax.plot([0, 0], [1.05, 1.15], 'k-', linewidth=4)
    ax.plot([-0.05, 0], [1.15, 1.2], 'k-', linewidth=4)
    ax.plot([0.05, 0], [1.15, 1.2], 'k-', linewidth=4)
    
    # Ears
    left_ear = Circle((-1.05, 0), 0.08, fill=False, edgecolor='black', linewidth=3)
    right_ear = Circle((1.05, 0), 0.08, fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)

def plot_electrodes(ax, band_results):
    """Plot electrodes with BIGGER circles, sized by significance"""
    for elec, pos in electrode_pos.items():
        # Find this electrode's data
        elec_data = band_results[band_results['Electrode'] == elec]
        
        if len(elec_data) == 0:
            # Electrode not in data - plot as gray
            ax.scatter(pos[0], pos[1], s=250, c='lightgray',
                      edgecolors='gray', linewidths=1.5, alpha=0.5, zorder=3)
            ax.text(pos[0], pos[1], elec, ha='center', va='center',
                   fontsize=9, fontweight='bold', color='black', zorder=4)
            continue
        
        row = elec_data.iloc[0]
        p = row['p_raw']
        d = row['cohens_d']
        fdr = row['sig_fdr']
        
        # SIZE based on significance (BIGGER than before)
        if fdr:
            size = 800  # VERY LARGE for FDR-significant
            edge_width = 5
            edge_color = 'gold'
        elif p < 0.05:
            size = 500  # LARGE for nominal sig
            edge_width = 3
            edge_color = 'black'
        else:
            size = 250  # MEDIUM for not sig
            edge_width = 1.5
            edge_color = 'gray'
        
        # COLOR based on effect direction
        if abs(d) < 0.1:
            color = 'lightgray'  # Negligible
        elif d > 0:
            color = '#E74C3C'  # RED = AD > CN (hub loading)
        else:
            color = '#3498DB'  # BLUE = CN > AD (hub losing)
        
        # Plot
        ax.scatter(pos[0], pos[1], s=size, c=color,
                  edgecolors=edge_color, linewidths=edge_width,
                  alpha=0.85, zorder=3)
        
        # Label (white if colored, black if gray)
        label_color = 'white' if color != 'lightgray' else 'black'
        ax.text(pos[0], pos[1], elec, ha='center', va='center',
               fontsize=10, fontweight='bold', color=label_color, zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# CREATE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating topography figure...")

fig = plt.figure(figsize=(20, 8))
fig.patch.set_facecolor('white')

# Create 3 subplots
gs = fig.add_gridspec(1, 3, wspace=0.3)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

band_info = [
    ('Theta', '#FF69B4', '4-8 Hz'),
    ('Alpha', '#2ECC71', '8-13 Hz'),
    ('Beta', '#F4D03F', '13-30 Hz')
]

for ax, (band_name, color, freq) in zip(axes, band_info):
    draw_head(ax)
    
    band_data = df[df['Band'] == band_name]
    plot_electrodes(ax, band_data)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title with colored background
    title_text = f'{band_name}-Band Hub Disruption\n({freq})'
    ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20,
                bbox=dict(boxstyle='round,pad=0.7', facecolor=color, alpha=0.3,
                         edgecolor='black', linewidth=3))

# Overall title
fig.suptitle('Nodal Hub Disruption Across Frequency Bands | 100% REAL DATA\n'
             'N=65 (AD=36, CN=29) | 57 tests with FDR correction',
             fontsize=15, fontweight='bold', y=0.96)

# COMPREHENSIVE LEGEND
legend_elements = [
    # Size key
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=20, markeredgecolor='gold', markeredgewidth=5,
           label='FDR-corrected sig. (p<0.05) ★', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=16, markeredgecolor='black', markeredgewidth=3,
           label='Nominal sig. (p<0.05)', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
           markersize=12, markeredgecolor='gray', markeredgewidth=1.5,
           label='Not significant', linestyle='None'),
    
    # Spacer
    Line2D([0], [0], marker='', color='w', label='', linestyle='None'),
    
    # Color key
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
           markersize=14, markeredgecolor='black', markeredgewidth=2,
           label='RED = AD > CN (hub loading)', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB',
           markersize=14, markeredgecolor='black', markeredgewidth=2,
           label='BLUE = CN > AD (hub losing)', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
           markersize=14, markeredgecolor='gray', markeredgewidth=2,
           label='GRAY = Negligible effect', linestyle='None'),
]

fig.legend(handles=legend_elements, loc='lower center', ncol=4,
          fontsize=11, frameon=True, fancybox=True, shadow=True,
          edgecolor='black', framealpha=0.95,
          bbox_to_anchor=(0.5, -0.08))

# Save
FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'topop.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ Saved: Fig3_Topography_ALL_BANDS_REAL.png")

# Summary
print("\n" + "="*80)
print("TOPOGRAPHY FIGURE COMPLETE")
print("="*80)
print("✓ Bigger electrode circles")
print("✓ Comprehensive legend (size + color keys)")
print("✓ 100% REAL DATA")
print("="*80)
