"""
STEP 10 (REWRITTEN) — Biomarker Validation: Hub + Modularity Classifier
========================================================================
Primary findings from Steps 8-9:
  1. Alpha Fz Betweenness Centrality  (p<0.0001, d~0.945) — headline
  2. Alpha Modularity Q               (p=0.045,  d=0.502) — fragmentation
  3. Alpha Global Efficiency          (p=0.060,  d=0.520) — trend
  4. Beta C4, P3, P4 centrality       (p<0.05)            — parietal hubs

This step:
  A. Individual ROC curves for each significant feature
  B. Combined logistic regression classifier (Fz + Modularity Q)
     → Tests whether combining features improves classification
  C. 3-panel publication figure:
       Panel A — Fz Centrality boxplot (headline finding)
       Panel B — Alpha Modularity Q boxplot
       Panel C — ROC curves: Fz alone, Modularity alone, combined
  D. Full reporting: AUC, sensitivity, specificity, PPV, NPV, Cohen's d CI
  E. Honest caveat language based on actual AUC values

Note on the combined classifier: with N=65 we use leave-one-out cross-
validation (LOOCV) rather than a train/test split, because any held-out
test set would be too small to estimate performance reliably. LOOCV gives
an unbiased estimate of generalization error at this sample size.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
hub_df  = pd.read_csv("hub_centrality_results.csv")
long_df = pd.read_csv("final_isef_data_dualband.csv")
wide_df = pd.read_csv("final_isef_data.csv")

# Build a subject-level feature table
alpha_df = long_df[long_df['Band'] == 'Alpha'][
    ['ID', 'Group', 'Global_Eff', 'Modularity_Q'] +
    [c for c in long_df.columns if c.startswith('Cent_')]
].copy()
alpha_df.columns = (['ID', 'Group', 'Alpha_GlobalEff', 'Alpha_ModQ'] +
                    [c.replace('Cent_', 'Alpha_') for c in
                     [c for c in long_df.columns if c.startswith('Cent_')]])

beta_df = long_df[long_df['Band'] == 'Beta'][
    ['ID', 'Group'] +
    [c for c in long_df.columns if c.startswith('Cent_')]
].copy()
beta_df.columns = (['ID', 'Group'] +
                   [c.replace('Cent_', 'Beta_') for c in
                    [c for c in long_df.columns if c.startswith('Cent_')]])

feat_df = alpha_df.merge(beta_df.drop(columns='Group'), on='ID')
feat_df = feat_df.dropna()

y = (feat_df['Group'] == 'A').astype(int).values   # 1 = AD, 0 = CN

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
    return (np.percentile(boot, (100-ci)/2),
            np.percentile(boot, 100-(100-ci)/2))


def bootstrap_auc_ci(y_true, y_score, n=2_000, ci=95):
    aucs = []
    for _ in range(n):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], y_score[idx])
        aucs.append(auc(fpr_b, tpr_b))
    return (np.percentile(aucs, (100-ci)/2),
            np.percentile(aucs, 100-(100-ci)/2))


def roc_metrics(y_true, y_score, label):
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    auc_lo, auc_hi = bootstrap_auc_ci(y_true, y_score)
    j_idx   = np.argmax(tpr - fpr)
    opt_sens = tpr[j_idx]
    opt_spec = 1 - fpr[j_idx]
    y_pred   = (y_score >= thresh[j_idx]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return dict(label=label, fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                auc_lo=auc_lo, auc_hi=auc_hi, j_idx=j_idx,
                opt_sens=opt_sens, opt_spec=opt_spec, ppv=ppv, npv=npv)


def auc_caveat(auc_val, ci_lo):
    if ci_lo < 0.5:
        return "⚠ CI crosses 0.50 — not reliably above chance; report as preliminary."
    elif auc_val < 0.70:
        return "⚠ AUC < 0.70: group difference established, not a validated biomarker."
    elif auc_val < 0.80:
        return "⚠ AUC 0.70-0.80: acceptable; independent validation required."
    return "AUC ≥ 0.80: good discrimination; independent replication required."


# ── Feature 1: Alpha Fz Centrality ───────────────────────────────────────────
fz_col = 'Alpha_Fz'
if fz_col not in feat_df.columns:
    # Try to find it regardless of capitalisation
    matches = [c for c in feat_df.columns if 'fz' in c.lower() and 'alpha' in c.lower()]
    fz_col  = matches[0] if matches else None

fz_scores = feat_df[fz_col].values if fz_col else None
fz_a = feat_df[feat_df['Group']=='A'][fz_col].values
fz_c = feat_df[feat_df['Group']=='C'][fz_col].values

# ── Feature 2: Alpha Modularity Q ────────────────────────────────────────────
modq_col    = 'Alpha_ModQ'
modq_scores = feat_df[modq_col].values
modq_a = feat_df[feat_df['Group']=='A'][modq_col].values
modq_c = feat_df[feat_df['Group']=='C'][modq_col].values

# ── ROC for individual features ───────────────────────────────────────────────
roc_fz   = roc_metrics(y, fz_scores,   "Alpha Fz Centrality")
roc_modq = roc_metrics(y, modq_scores, "Alpha Modularity Q")

# ── Combined LOOCV classifier ─────────────────────────────────────────────────
print("Running LOOCV combined classifier (Fz + Modularity Q)...")
X = feat_df[[fz_col, modq_col]].values
loo = LeaveOneOut()
loocv_scores = np.zeros(len(y))

for train_idx, test_idx in loo.split(X):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test  = scaler.transform(X[test_idx])
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y[train_idx])
    loocv_scores[test_idx] = clf.predict_proba(X_test)[:, 1]

roc_combined = roc_metrics(y, loocv_scores, "Combined (Fz + Mod Q, LOOCV)")

# ── Print full report ─────────────────────────────────────────────────────────
n_ad = int(y.sum())
n_cn = len(y) - n_ad

print(f"\n{'='*65}")
print(f"  BIOMARKER VALIDATION REPORT")
print(f"  N={len(y)} (AD={n_ad}, CN={n_cn})")
print(f"{'='*65}")

for feat_name, a_vals, c_vals, roc in [
    ("Alpha Fz Centrality",  fz_a,    fz_c,    roc_fz),
    ("Alpha Modularity Q",   modq_a,  modq_c,  roc_modq),
    ("Combined (LOOCV)",     fz_a,    fz_c,    roc_combined),
]:
    _, p_mw = stats.mannwhitneyu(a_vals, c_vals, alternative='two-sided')
    d       = cohens_d(a_vals, c_vals)
    d_lo, d_hi = bootstrap_ci_d(a_vals, c_vals)

    print(f"\n  ── {feat_name} ────────────────────────────────")
    if feat_name != "Combined (LOOCV)":
        print(f"  AD : {np.mean(a_vals):.4f} ± {np.std(a_vals, ddof=1):.4f}")
        print(f"  CN : {np.mean(c_vals):.4f} ± {np.std(c_vals, ddof=1):.4f}")
        print(f"  MWU p     = {p_mw:.4f}")
        print(f"  Cohen's d = {d:.3f}  (95% CI: [{d_lo:.3f}, {d_hi:.3f}])")
    print(f"  AUC       = {roc['roc_auc']:.3f}  "
          f"(95% CI: [{roc['auc_lo']:.3f}, {roc['auc_hi']:.3f}])")
    print(f"  Sens={roc['opt_sens']:.1%}  Spec={roc['opt_spec']:.1%}  "
          f"PPV={roc['ppv']:.1%}  NPV={roc['npv']:.1%}")
    print(f"  {auc_caveat(roc['roc_auc'], roc['auc_lo'])}")

# ── 3-Panel figure ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
fig = plt.figure(figsize=(16, 6))
gs  = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.38)

LABEL_MAP = {'A': "Alzheimer's", 'C': 'Control'}
COLORS    = {"Alzheimer's": '#E07B54', 'Control': '#5B9BD5'}

def make_boxplot(ax, df, col, panel_letter, title, ylabel):
    df_p = df[['Group', col]].copy()
    df_p['Diagnosis'] = df_p['Group'].map(LABEL_MAP)
    df_p.columns      = ['Group', 'Value', 'Diagnosis']

    sns.boxplot(x='Diagnosis', y='Value', data=df_p,
                palette=COLORS, width=0.45, linewidth=1.5,
                fliersize=0, ax=ax)
    sns.swarmplot(x='Diagnosis', y='Value', data=df_p,
                  color='.2', size=5, alpha=0.70, ax=ax)

    a_v = df_p[df_p['Diagnosis']=="Alzheimer's"]['Value'].values
    c_v = df_p[df_p['Diagnosis']=='Control']['Value'].values
    _, p = stats.mannwhitneyu(a_v, c_v, alternative='two-sided')
    d    = cohens_d(a_v, c_v)
    d_lo, d_hi = bootstrap_ci_d(a_v, c_v, n=2000)

    y_max = df_p['Value'].max()
    y_bar = y_max + abs(y_max) * 0.08
    ax.plot([0, 0, 1, 1],
            [y_max + abs(y_max)*0.04, y_bar,
             y_bar, y_max + abs(y_max)*0.04],
            lw=1.5, c='k')
    p_str = f"p = {p:.4f}" + (" *" if p < 0.05 else (" ~" if p < 0.10 else ""))
    ax.text(0.5, y_bar + abs(y_max)*0.01, p_str,
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(0.97, 0.04,
            f"d = {d:.2f} [{d_lo:.2f}, {d_hi:.2f}]",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.3', fc='white',
                      ec='#aaaaaa', alpha=0.85))
    ax.set_title(f"{panel_letter}   {title}", fontsize=12,
                 loc='left', fontweight='bold')
    ax.set_xlabel("Diagnostic Group", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)


ax1 = fig.add_subplot(gs[0])
make_boxplot(ax1, feat_df, fz_col, 'A',
             'Alpha Fz Hub Centrality',
             'Betweenness Centrality (Fz)')

ax2 = fig.add_subplot(gs[1])
make_boxplot(ax2, feat_df, modq_col, 'B',
             'Alpha Network Modularity',
             'Modularity Q')

# ── Panel C: ROC curves ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
roc_styles = [
    (roc_fz,       '#C0392B', '-',  2.5),
    (roc_modq,     '#2E75B6', '--', 2.0),
    (roc_combined, '#27AE60', '-',  2.5),
]
for roc, color, ls, lw in roc_styles:
    label = (f"{roc['label']}\n"
             f"AUC={roc['roc_auc']:.3f} [{roc['auc_lo']:.3f}–{roc['auc_hi']:.3f}]")
    ax3.plot(roc['fpr'], roc['tpr'], color=color,
             linestyle=ls, lw=lw, label=label)
    ax3.scatter(roc['fpr'][roc['j_idx']], roc['tpr'][roc['j_idx']],
                s=80, color=color, zorder=5)

ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Chance (AUC=0.50)')
ax3.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=10)
ax3.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
ax3.set_title("C   ROC Curves — Topometric Biomarker Features",
              fontsize=12, loc='left', fontweight='bold')
ax3.legend(fontsize=7.5, loc='lower right')
ax3.set_xlim(-0.02, 1.02)
ax3.set_ylim(-0.02, 1.08)

# Auto-caveat
worst_auc = min(r['roc_auc'] for r in [roc_fz, roc_modq, roc_combined])
if worst_auc < 0.80:
    ax3.text(0.5, 0.05,
             "Independent validation cohort required\nbefore clinical biomarker status.",
             ha='center', va='bottom', fontsize=7.5, color='#555555',
             transform=ax3.transAxes, style='italic')

plt.suptitle(
    "Alpha-Band Topometric Signature of Alzheimer's Disease\n"
    f"N={len(y)} (AD={n_ad}, CN={n_cn})  |  "
    "PLI connectivity  |  Proportional threshold 20%  |  "
    "Combined: LOOCV logistic regression",
    fontsize=10.5, y=1.03
)

plt.savefig("Final_Biomarker_Figure.png", dpi=300, bbox_inches='tight')
plt.show()
print("\nSaved: Final_Biomarker_Figure.png")

# ── Save numerical summary ────────────────────────────────────────────────────
summary = []
for feat_name, a_vals, c_vals, roc in [
    ("Alpha_Fz_Centrality",  fz_a,   fz_c,   roc_fz),
    ("Alpha_Modularity_Q",   modq_a, modq_c, roc_modq),
    ("Combined_LOOCV",       fz_a,   fz_c,   roc_combined),
]:
    _, p_mw = stats.mannwhitneyu(a_vals, c_vals, alternative='two-sided')
    d       = cohens_d(a_vals, c_vals)
    d_lo, d_hi = bootstrap_ci_d(a_vals, c_vals, n=2000)
    summary.append({
        'Feature': feat_name,
        'AUC': roc['roc_auc'], 'AUC_CI_lo': roc['auc_lo'],
        'AUC_CI_hi': roc['auc_hi'],
        'Sensitivity': roc['opt_sens'], 'Specificity': roc['opt_spec'],
        'PPV': roc['ppv'], 'NPV': roc['npv'],
        'Cohens_d': d, 'd_CI_lo': d_lo, 'd_CI_hi': d_hi,
        'p_MWU': p_mw,
    })

pd.DataFrame(summary).to_csv("biomarker_validation.csv", index=False)
print("Saved: biomarker_validation.csv")
