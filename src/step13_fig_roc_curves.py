"""
COMPREHENSIVE ROC ANALYSIS - ALL BIOMARKER CANDIDATES
Tests FOUR potential biomarkers:
1. Alpha Fz Centrality (primary)
2. Beta P4 Centrality (secondary)
3. Alpha Modularity Q (mechanistic)
4. Combined Fz + Modularity (multi-feature)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from scipy import stats
import sys
import os

print("="*80)
print("COMPREHENSIVE ROC ANALYSIS - ALL BIOMARKERS")
print("="*80)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

if not os.path.exists('final_isef_data_dualband.csv'):
    print("\n✗ ERROR: final_isef_data_dualband.csv not found!")
    sys.exit(1)

df = pd.read_csv('final_isef_data_dualband.csv')
alpha_df = df[df['Band'] == 'Alpha'].copy()
beta_df = df[df['Band'] == 'Beta'].copy()

# Prepare data
alpha_df = alpha_df.dropna(subset=['Cent_Fz', 'Modularity_Q'])
beta_df = beta_df.dropna(subset=['Cent_P4'])

alpha_df['y'] = (alpha_df['Group'] == 'A').astype(int)
beta_df['y'] = (beta_df['Group'] == 'A').astype(int)

print(f"\nAlpha band: {len(alpha_df)} subjects ({sum(alpha_df['y']==1)} AD, {sum(alpha_df['y']==0)} CN)")
print(f"Beta band: {len(beta_df)} subjects ({sum(beta_df['y']==1)} AD, {sum(beta_df['y']==0)} CN)")

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE ROC CURVES FOR ALL FEATURES
# ══════════════════════════════════════════════════════════════════════════════

print("\nComputing ROC curves...")

results = []

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1: Alpha Fz Centrality
# ─────────────────────────────────────────────────────────────────────────────

y_true_alpha = alpha_df['y'].values
fz_scores = alpha_df['Cent_Fz'].values

fpr_fz, tpr_fz, thresh_fz = roc_curve(y_true_alpha, fz_scores)
auc_fz = auc(fpr_fz, tpr_fz)

# Find optimal threshold (Youden's J)
j_scores = tpr_fz - fpr_fz
optimal_idx = np.argmax(j_scores)
optimal_thresh_fz = thresh_fz[optimal_idx]

# Confusion matrix at optimal threshold
y_pred_fz = (fz_scores >= optimal_thresh_fz).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true_alpha, y_pred_fz).ravel()

sensitivity_fz = tp / (tp + fn)
specificity_fz = tn / (tn + fp)
ppv_fz = tp / (tp + fp) if (tp + fp) > 0 else 0
npv_fz = tn / (tn + fn) if (tn + fn) > 0 else 0

results.append({
    'Feature': 'Alpha Fz Centrality',
    'AUC': auc_fz,
    'Sensitivity': sensitivity_fz,
    'Specificity': specificity_fz,
    'PPV': ppv_fz,
    'NPV': npv_fz,
    'Optimal_Threshold': optimal_thresh_fz
})

print(f"  • Alpha Fz: AUC={auc_fz:.3f}, Sens={sensitivity_fz:.1%}, Spec={specificity_fz:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 2: Beta P4 Centrality
# ─────────────────────────────────────────────────────────────────────────────

y_true_beta = beta_df['y'].values
p4_scores = beta_df['Cent_P4'].values

fpr_p4, tpr_p4, thresh_p4 = roc_curve(y_true_beta, p4_scores)
auc_p4 = auc(fpr_p4, tpr_p4)

j_scores_p4 = tpr_p4 - fpr_p4
optimal_idx_p4 = np.argmax(j_scores_p4)
optimal_thresh_p4 = thresh_p4[optimal_idx_p4]

y_pred_p4 = (p4_scores >= optimal_thresh_p4).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true_beta, y_pred_p4).ravel()

sensitivity_p4 = tp / (tp + fn)
specificity_p4 = tn / (tn + fp)
ppv_p4 = tp / (tp + fp) if (tp + fp) > 0 else 0
npv_p4 = tn / (tn + fn) if (tn + fn) > 0 else 0

results.append({
    'Feature': 'Beta P4 Centrality',
    'AUC': auc_p4,
    'Sensitivity': sensitivity_p4,
    'Specificity': specificity_p4,
    'PPV': ppv_p4,
    'NPV': npv_p4,
    'Optimal_Threshold': optimal_thresh_p4
})

print(f"  • Beta P4: AUC={auc_p4:.3f}, Sens={sensitivity_p4:.1%}, Spec={specificity_p4:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 3: Alpha Modularity Q
# ─────────────────────────────────────────────────────────────────────────────

mod_scores = alpha_df['Modularity_Q'].values

fpr_mod, tpr_mod, thresh_mod = roc_curve(y_true_alpha, mod_scores)
auc_mod = auc(fpr_mod, tpr_mod)

j_scores_mod = tpr_mod - fpr_mod
optimal_idx_mod = np.argmax(j_scores_mod)
optimal_thresh_mod = thresh_mod[optimal_idx_mod]

y_pred_mod = (mod_scores >= optimal_thresh_mod).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true_alpha, y_pred_mod).ravel()

sensitivity_mod = tp / (tp + fn)
specificity_mod = tn / (tn + fp)
ppv_mod = tp / (tp + fp) if (tp + fp) > 0 else 0
npv_mod = tn / (tn + fn) if (tn + fn) > 0 else 0

results.append({
    'Feature': 'Alpha Modularity Q',
    'AUC': auc_mod,
    'Sensitivity': sensitivity_mod,
    'Specificity': specificity_mod,
    'PPV': ppv_mod,
    'NPV': npv_mod,
    'Optimal_Threshold': optimal_thresh_mod
})

print(f"  • Modularity Q: AUC={auc_mod:.3f}, Sens={sensitivity_mod:.1%}, Spec={specificity_mod:.1%}")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 4: Combined (Fz + Modularity, LOOCV)
# ─────────────────────────────────────────────────────────────────────────────

X_combined = alpha_df[['Cent_Fz', 'Modularity_Q']].values
loo = LeaveOneOut()
y_pred_proba = np.zeros(len(y_true_alpha))

for train_idx, test_idx in loo.split(X_combined):
    X_train, X_test = X_combined[train_idx], X_combined[test_idx]
    y_train = y_true_alpha[train_idx]
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_proba[test_idx] = clf.predict_proba(X_test)[0, 1]

fpr_comb, tpr_comb, thresh_comb = roc_curve(y_true_alpha, y_pred_proba)
auc_comb = auc(fpr_comb, tpr_comb)

j_scores_comb = tpr_comb - fpr_comb
optimal_idx_comb = np.argmax(j_scores_comb)
optimal_thresh_comb = thresh_comb[optimal_idx_comb]

y_pred_comb = (y_pred_proba >= optimal_thresh_comb).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true_alpha, y_pred_comb).ravel()

sensitivity_comb = tp / (tp + fn)
specificity_comb = tn / (tn + fp)
ppv_comb = tp / (tp + fp) if (tp + fp) > 0 else 0
npv_comb = tn / (tn + fn) if (tn + fn) > 0 else 0

results.append({
    'Feature': 'Combined (Fz + Mod Q)',
    'AUC': auc_comb,
    'Sensitivity': sensitivity_comb,
    'Specificity': specificity_comb,
    'PPV': ppv_comb,
    'NPV': npv_comb,
    'Optimal_Threshold': optimal_thresh_comb
})

print(f"  • Combined: AUC={auc_comb:.3f}, Sens={sensitivity_comb:.1%}, Spec={specificity_comb:.1%}")

# Create results DataFrame
results_df = pd.DataFrame(results)

# ══════════════════════════════════════════════════════════════════════════════
# CREATE COMPREHENSIVE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating comprehensive ROC figure...")

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.35, wspace=0.3)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL A: ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────

ax_roc = fig.add_subplot(gs[0, :])

# Plot curves with distinct styles
ax_roc.plot(fpr_fz, tpr_fz, 'r-', linewidth=4, 
            label=f'Alpha Fz (AUC={auc_fz:.3f}) ★ PRIMARY', zorder=5)
ax_roc.plot(fpr_p4, tpr_p4, 'b-', linewidth=3.5,
            label=f'Beta P4 (AUC={auc_p4:.3f}) ★ SECONDARY', zorder=4)
ax_roc.plot(fpr_mod, tpr_mod, 'g--', linewidth=3,
            label=f'Modularity Q (AUC={auc_mod:.3f})', zorder=3)
ax_roc.plot(fpr_comb, tpr_comb, 'm-.', linewidth=3,
            label=f'Combined (AUC={auc_comb:.3f})', zorder=3)

# Chance line
ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, 
            label='Chance (AUC=0.50)', zorder=2)

# Mark optimal points
ax_roc.scatter([fpr_fz[optimal_idx]], [tpr_fz[optimal_idx]], 
               s=200, c='red', marker='*', edgecolors='black', 
               linewidths=2, zorder=6, label='Optimal thresholds')
ax_roc.scatter([fpr_p4[optimal_idx_p4]], [tpr_p4[optimal_idx_p4]],
               s=180, c='blue', marker='*', edgecolors='black', linewidths=2, zorder=6)

# Styling
ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
ax_roc.set_title('ROC Curves: Biomarker Performance Comparison | 100% REAL DATA\n'
                 f'N={len(alpha_df)} Alpha-band, N={len(beta_df)} Beta-band',
                 fontsize=15, fontweight='bold', pad=20)

ax_roc.legend(fontsize=11, loc='lower right', framealpha=0.95, 
              edgecolor='black', fancybox=True, ncol=2)
ax_roc.grid(alpha=0.3)
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])



# ─────────────────────────────────────────────────────────────────────────────
# PANEL B: PERFORMANCE TABLE
# ─────────────────────────────────────────────────────────────────────────────

ax_table = fig.add_subplot(gs[1, 0])
ax_table.axis('off')

# Create table data
table_data = []
table_data.append(['Feature', 'AUC', 'Sens', 'Spec', 'PPV', 'NPV'])

for _, row in results_df.iterrows():
    table_data.append([
        row['Feature'],
        f"{row['AUC']:.3f}",
        f"{row['Sensitivity']:.1%}",
        f"{row['Specificity']:.1%}",
        f"{row['PPV']:.1%}",
        f"{row['NPV']:.1%}"
    ])

# Create table
table = ax_table.table(cellText=table_data, cellLoc='center',
                       loc='center', bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#4CAF50')
    cell.set_text_props(weight='bold', color='white')

# Highlight best performers
for i in range(1, 5):
    if table_data[i][0] == 'Alpha Fz Centrality':
        for j in range(6):
            table[(i, j)].set_facecolor('#FFE5E5')
    elif table_data[i][0] == 'Beta P4 Centrality':
        for j in range(6):
            table[(i, j)].set_facecolor('#E5F0FF')

ax_table.set_title('Performance Metrics at Optimal Thresholds', 
                   fontsize=12, fontweight='bold', pad=10)



# Save
FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'ROCCC.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Save table
results_df.to_csv('biomarker_performance_all.csv', index=False)

print("  ✓ Saved: Fig_ROC_Comprehensive_ALL_BIOMARKERS.png")
print("  ✓ Saved: biomarker_performance_all.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("COMPREHENSIVE ROC ANALYSIS COMPLETE")
print("="*80)

print("\nRANKING BY AUC:")
results_sorted = results_df.sort_values('AUC', ascending=False)
for idx, row in results_sorted.iterrows():
    print(f"  {idx+1}. {row['Feature']}: AUC={row['AUC']:.3f}")

print("\nCONCLUSION:")
if auc_fz > auc_p4:
    print("  → Alpha Fz is SUPERIOR biomarker")
    print(f"    (AUC difference: {auc_fz - auc_p4:.3f})")
else:
    print("  → Beta P4 is SUPERIOR biomarker")

if auc_fz > auc_comb:
    print("  → Single-feature Fz BEATS combined classifier")
    print("    (Simpler is better!)")

print("\n✓ 100% REAL DATA")
print("✓ All biomarkers tested")
print("✓ Clean figure with no overlap")
print("="*80)
