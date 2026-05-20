"""
UPI PULSE 2024 — VISUAL ANALYTICS
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLE — White Corporate Theme
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
})

C_PAGE   = '#F7F8FA'   # page / figure background
C_CARD   = '#FFFFFF'   # axes / panel background
C_ACCENT = '#0B2545'   # primary navy
C_STEEL  = '#3B6CA8'   # mid blue
C_GREEN  = '#15803D'   # success green
C_RED    = '#B91C1C'   # fraud / risk crimson
C_ORANGE = '#D97706'   # warning amber
C_PURPLE = '#6B21A8'   # purple category
C_TEXT   = '#1A2A42'   # primary text
C_MUTED  = '#64748B'   # secondary text / labels
C_LIGHT  = '#94A3B8'   # axis ticks
C_BORDER = '#E2E7EF'   # borders / grid lines
C_GRID   = '#EDF0F4'   # gridlines

os.makedirs('visuals', exist_ok=True)
os.makedirs('ml',      exist_ok=True)

def white_fig(w, h):
    return plt.figure(figsize=(w, h), facecolor=C_PAGE)

def white_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_LIGHT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(C_BORDER)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=12,
                     fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=C_MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C_MUTED, fontsize=9)
    ax.grid(True, color=C_GRID, linewidth=0.6,
            linestyle='--', alpha=0.9)
    return ax

def save(fig, name):
    fig.savefig(f'visuals/{name}', dpi=150, bbox_inches='tight',
                facecolor=C_PAGE, edgecolor='none')
    plt.close(fig)
    print(f'  saved → visuals/{name}')

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print('\nLoading data...')
df = pd.read_csv('data/upi_clean.csv')
print(f'  {len(df):,} rows  |  {df.shape[1]} columns')

if 'amount_inr' in df.columns and 'amount' not in df.columns:
    df['amount'] = df['amount_inr']
elif 'amount' not in df.columns:
    df['amount'] = df.filter(like='amount').iloc[:, 0]

if 'is_success' not in df.columns:
    df['is_success'] = (df['transaction_status'] == 'SUCCESS').astype(int)
if 'is_same_bank' not in df.columns:
    df['is_same_bank'] = (df['sender_bank'] == df['receiver_bank']).astype(int)

month_map = {
    'January':1,'February':2,'March':3,'April':4,
    'May':5,'June':6,'July':7,'August':8,
    'September':9,'October':10,'November':11,'December':12
}
if 'month_num' not in df.columns and 'month' in df.columns:
    df['month_num'] = df['month'].map(month_map).fillna(1).astype(int)

age_map = {'18-25':1,'26-35':2,'36-45':3,'46-55':4,'56+':5}
if 'sender_age_order' not in df.columns:
    df['sender_age_order'] = df['sender_age_group'].map(age_map)

month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
day_order   = ['Monday','Tuesday','Wednesday','Thursday',
               'Friday','Saturday','Sunday']
age_order   = ['18-25','26-35','36-45','46-55','56+']

# ─────────────────────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────────────────────
print('\nBuilding ML model...')

cat_cols = ['transaction_type','merchant_category','sender_bank',
            'receiver_bank','sender_age_group','receiver_age_group',
            'sender_state','device_type','network_type','day_of_week']
existing_cats = [c for c in cat_cols if c in df.columns]

le    = LabelEncoder()
df_ml = df.copy()
for col in existing_cats:
    df_ml[col+'_enc'] = le.fit_transform(df_ml[col].astype(str))

num_feats = ['amount','hour_of_day','is_weekend',
             'is_same_bank','month_num','sender_age_order']
num_feats = [f for f in num_feats if f in df_ml.columns]

feature_cols   = [c+'_enc' for c in existing_cats] + num_feats
feature_labels = [c.replace('_enc','').replace('_',' ').title()
                  for c in feature_cols]

X = df_ml[feature_cols]
y = df_ml['fraud_flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f'  SMOTE → {(y_res==0).sum():,} legit | {(y_res==1).sum():,} fraud')

model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss', verbosity=0)
model.fit(X_res, y_res)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.3).astype(int)

auc  = roc_auc_score(y_test, y_prob)
ap   = average_precision_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

feat_imp = pd.DataFrame({
    'Feature':    feature_labels,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

feat_imp.to_csv('ml/feature_importance.csv', index=False)
print(f'  AUC={auc:.4f} | Fraud caught={tp} | Missed={fn}')
print('\nGenerating charts...\n')

# ═════════════════════════════════════════════════════════════
# CHART 1 — PLATFORM KPI SCORECARDS
# ═════════════════════════════════════════════════════════════
print('1/10  Platform KPIs...')
fig = white_fig(22, 4)
fig.suptitle('UPI PULSE 2024  —  Platform Intelligence Dashboard',
             color=C_TEXT, fontsize=20, fontweight='bold', y=1.05)

kpis = [
    ('250,000',    'Total Transactions',  C_ACCENT,  '💳'),
    ('₹32.79 Cr',  'Total Value',         C_GREEN,   '💰'),
    ('₹1,312',     'Avg Transaction',     C_STEEL,   '📊'),
    ('95.05%',     'Success Rate',        C_GREEN,   '✅'),
    ('0.192%',     'Fraud Rate',          C_RED,     '🚨'),
    ('10',         'States Covered',      C_ORANGE,  '🗺️'),
    ('8',          'Banks Active',        C_ACCENT,  '🏦'),
    (f'{auc:.4f}', 'Model AUC Score',     C_PURPLE,  '🤖'),
]

axes = fig.subplots(1, 8)
for ax, (val, label, color, icon) in zip(axes, kpis):
    ax.set_facecolor(C_CARD)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_color(C_BORDER)
        sp.set_linewidth(1.2)
    ax.spines['left'].set_color(color)
    ax.spines['left'].set_linewidth(4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.72, val, ha='center', va='center',
            fontsize=19, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.32, label, ha='center', va='center',
            fontsize=8.5, color=C_MUTED,
            transform=ax.transAxes)
    ax.text(0.5, 0.88, icon, ha='center', va='center',
            fontsize=14, transform=ax.transAxes)

plt.tight_layout()
save(fig, '01_platform_kpis.png')

# ═════════════════════════════════════════════════════════════
# CHART 2 — MONTHLY TREND (volume + value + fraud)
# ═════════════════════════════════════════════════════════════
print('2/10  Monthly trend...')
if 'month' in df.columns:
    monthly = (df.groupby('month')
                 .agg(count=('fraud_flag','count'),
                      value=('amount','sum'),
                      fraud=('fraud_flag','sum'))
                 .reindex(month_order).reset_index())
    monthly['value_lakhs'] = monthly['value'] / 1e5
    monthly['month_short'] = [m[:3] for m in monthly['month']]

    fig = white_fig(18, 6)
    ax1 = fig.add_subplot(111)
    white_ax(ax1, 'Monthly Transaction Volume & Value — 2024',
             'Month', 'Transaction Count')

    x = np.arange(len(monthly))
    bars = ax1.bar(x, monthly['count'], color=C_ACCENT,
                   alpha=0.80, width=0.6, zorder=3, edgecolor='none')

    ax2 = ax1.twinx()
    ax2.plot(x, monthly['value_lakhs'], color=C_ORANGE,
             lw=2.5, marker='o', ms=7,
             markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=C_ORANGE,
             label='Value (₹ Lakhs)', zorder=4)
    ax2.fill_between(x, monthly['value_lakhs'],
                     alpha=0.08, color=C_ORANGE)
    ax2.set_ylabel('Value (₹ Lakhs)', color=C_MUTED, fontsize=9)
    ax2.tick_params(colors=C_LIGHT)
    for sp in ax2.spines.values():
        sp.set_color(C_BORDER)

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.09))
    ax3.scatter(x, monthly['fraud'], color=C_RED,
                s=70, zorder=5, label='Fraud Cases',
                edgecolors='white', linewidths=1.2)
    ax3.set_ylabel('Fraud Cases', color=C_RED, fontsize=9)
    ax3.tick_params(colors=C_RED)
    ax3.set_ylim(0, monthly['fraud'].max() * 3.5)
    for sp in ax3.spines.values():
        sp.set_color(C_BORDER)

    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly['month_short'], color=C_MUTED)
    ax1.set_ylim(0, monthly['count'].max() * 1.25)

    legend_items = [
        mpatches.Patch(color=C_ACCENT, label='Transaction Count'),
        mpatches.Patch(color=C_ORANGE, label='Value (₹ Lakhs)'),
        mpatches.Patch(color=C_RED,    label='Fraud Cases'),
    ]
    ax1.legend(handles=legend_items, loc='upper left',
               facecolor=C_CARD, edgecolor=C_BORDER,
               labelcolor=C_TEXT, fontsize=9)

    plt.tight_layout()
    save(fig, '02_monthly_trend.png')

# ═════════════════════════════════════════════════════════════
# CHART 3 — TIME HEATMAP (day × hour) + HOURLY LINE
# ═════════════════════════════════════════════════════════════
print('3/10  Time heatmap...')
if 'hour_of_day' in df.columns and 'day_of_week' in df.columns:
    pivot = (df.groupby(['day_of_week','hour_of_day'])
               .size().reset_index(name='count')
               .pivot(index='day_of_week',
                      columns='hour_of_day', values='count')
               .reindex(day_order))

    fig = white_fig(20, 7)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.2,1], wspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    cmap_blue = LinearSegmentedColormap.from_list(
        'upi', ['#EEF4FF','#DDEAFF','#3B6CA8','#0B2545'], N=256)
    sns.heatmap(pivot, ax=ax1, cmap=cmap_blue,
                linewidths=0.25, linecolor='white',
                cbar_kws={'label':'Transactions'})
    ax1.set_title('Transaction Heatmap — Day of Week × Hour',
                  color=C_TEXT, fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel('Hour of Day', color=C_MUTED, fontsize=9)
    ax1.set_ylabel('', color=C_MUTED)
    ax1.tick_params(colors=C_MUTED, labelsize=9)
    cbar = ax1.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(C_MUTED)
    cbar.ax.tick_params(colors=C_MUTED)

    hourly = df.groupby('hour_of_day').size().reset_index(name='count')
    white_ax(ax2, 'Transactions by Hour of Day', 'Hour', 'Count')
    ax2.fill_between(hourly['hour_of_day'], hourly['count'],
                     alpha=0.12, color=C_ACCENT)
    ax2.plot(hourly['hour_of_day'], hourly['count'],
             color=C_ACCENT, lw=2.5, marker='o', ms=4)
    peak_h = hourly.loc[hourly['count'].idxmax()]
    ax2.annotate(
        f"Peak: {int(peak_h['hour_of_day'])}:00\n{int(peak_h['count']):,} txns",
        xy=(peak_h['hour_of_day'], peak_h['count']),
        xytext=(peak_h['hour_of_day']-6, peak_h['count']*0.88),
        color=C_ORANGE, fontsize=8,
        arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=1.5))

    save(fig, '03_time_heatmap.png')

# ═════════════════════════════════════════════════════════════
# CHART 4 — TRANSACTION TYPE + MERCHANT CATEGORY
# ═════════════════════════════════════════════════════════════
print('4/10  Transaction type & category...')
fig = white_fig(20, 7)
gs  = gridspec.GridSpec(1, 2, wspace=0.4)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

if 'transaction_type' in df.columns:
    txn = df['transaction_type'].value_counts()
    pie_colors = [C_ACCENT, C_STEEL, C_ORANGE, C_PURPLE]
    wedges, texts, autotexts = ax1.pie(
        txn.values, labels=txn.index,
        autopct='%1.1f%%', colors=pie_colors,
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.55,
                        edgecolor='white', linewidth=3))
    for t in texts:
        t.set_color(C_TEXT); t.set_fontsize(11)
    for a in autotexts:
        a.set_color('white'); a.set_fontsize(9); a.set_fontweight('bold')
    ax1.set_facecolor(C_PAGE)
    ax1.set_title('Transaction Type Distribution',
                  color=C_TEXT, fontsize=13, fontweight='bold')
    ax1.text(0, 0, f'{len(df):,}\nTxns', ha='center', va='center',
             color=C_TEXT, fontsize=12, fontweight='bold')

if 'merchant_category' in df.columns:
    cat = (df.groupby('merchant_category')['amount']
             .sum().sort_values(ascending=True) / 1e5)
    bar_colors = [C_RED    if v == cat.max()
                  else C_ORANGE if v >= cat.quantile(0.75)
                  else C_ACCENT
                  for v in cat.values]
    bars = ax2.barh(cat.index, cat.values, color=bar_colors,
                    height=0.6, edgecolor='none')
    white_ax(ax2, 'Merchant Category — Total Value (₹ Lakhs)',
             'Value (₹ Lakhs)', '')
    ax2.tick_params(axis='y', colors=C_TEXT)
    for bar, val in zip(bars, cat.values):
        ax2.text(val + 2, bar.get_y() + bar.get_height()/2,
                 f'₹{val:.0f}L', va='center',
                 color=C_MUTED, fontsize=8)

save(fig, '04_txn_type_category.png')

# ═════════════════════════════════════════════════════════════
# CHART 5 — BANK ANALYSIS
# ═════════════════════════════════════════════════════════════
print('5/10  Bank analysis...')
if 'sender_bank' in df.columns:
    bank = (df.groupby('sender_bank')
              .agg(count=('fraud_flag','count'),
                   failed=('is_success', lambda x: (x==0).sum()),
                   fraud=('fraud_flag','sum'))
              .reset_index())
    bank['failure_pct']  = bank['failed'] / bank['count'] * 100
    bank['market_share'] = bank['count'] / len(df) * 100
    bank_ms   = bank.sort_values('count', ascending=False)
    bank_fail = bank.sort_values('failure_pct', ascending=True)

    fig = white_fig(20, 7)
    gs  = gridspec.GridSpec(1, 2, wspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ms_colors = [C_RED   if r == bank_ms['failure_pct'].max()
                 else C_GREEN if r == bank_ms['failure_pct'].min()
                 else C_ACCENT
                 for r in bank_ms['failure_pct']]
    bars = ax1.bar(np.arange(len(bank_ms)), bank_ms['market_share'],
                   color=ms_colors, width=0.6,
                   alpha=0.85, edgecolor='none')
    white_ax(ax1, 'Bank Market Share\n(Red=highest failure | Green=most reliable)',
             'Bank', 'Market Share %')
    ax1.set_xticks(np.arange(len(bank_ms)))
    ax1.set_xticklabels(bank_ms['sender_bank'], color=C_TEXT, fontsize=9)
    for bar, row in zip(bars, bank_ms.itertuples()):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.15,
                 f'{row.market_share:.1f}%',
                 ha='center', color=C_MUTED, fontsize=8)

    fail_colors = [C_RED   if v == bank_fail['failure_pct'].max()
                   else C_GREEN if v == bank_fail['failure_pct'].min()
                   else C_ACCENT
                   for v in bank_fail['failure_pct']]
    ax2.barh(bank_fail['sender_bank'], bank_fail['failure_pct'],
             color=fail_colors, height=0.6, edgecolor='none')
    white_ax(ax2, 'Bank Failure Rate Comparison (%)',
             'Failure Rate %', '')
    ax2.tick_params(axis='y', colors=C_TEXT)
    for i, val in enumerate(bank_fail['failure_pct']):
        ax2.text(val + 0.02, i, f'{val:.2f}%',
                 va='center', color=C_MUTED, fontsize=9)
    avg_fail = bank_fail['failure_pct'].mean()
    ax2.axvline(avg_fail, color=C_ORANGE, linestyle='--',
                lw=1.5, label=f'Platform avg: {avg_fail:.2f}%')
    ax2.legend(facecolor=C_CARD, edgecolor=C_BORDER,
               labelcolor=C_TEXT, fontsize=9)

    save(fig, '05_bank_analysis.png')

# ═════════════════════════════════════════════════════════════
# CHART 6 — STATE + AGE GROUP
# ═════════════════════════════════════════════════════════════
print('6/10  State & age group...')
fig = white_fig(20, 7)
gs  = gridspec.GridSpec(1, 2, wspace=0.5)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

if 'sender_state' in df.columns:
    state = (df.groupby('sender_state')
               .agg(count=('fraud_flag','count'),
                    value=('amount','sum'),
                    fraud=('fraud_flag','sum'))
               .reset_index())
    state['value_lakhs'] = state['value'] / 1e5
    state['fraud_rate']  = state['fraud'] / state['count'] * 100
    state = state.sort_values('value_lakhs', ascending=True)

    s_colors = [C_RED   if r == state['fraud_rate'].max()
                else C_GREEN if r == state['fraud_rate'].min()
                else C_ACCENT
                for r in state['fraud_rate']]
    ax1.barh(state['sender_state'], state['value_lakhs'],
             color=s_colors, height=0.65, edgecolor='none')
    white_ax(ax1, 'State-wise Value  (Red=highest fraud | Green=lowest)',
             'Total Value (₹ Lakhs)', '')
    ax1.tick_params(axis='y', colors=C_TEXT)
    for i, (val, fr) in enumerate(zip(state['value_lakhs'],
                                       state['fraud_rate'])):
        ax1.text(val + 1, i,
                 f'₹{val:.0f}L  fraud:{fr:.3f}%',
                 va='center', color=C_MUTED, fontsize=7.5)

if 'sender_age_group' in df.columns:
    age = (df.groupby('sender_age_group')
             .agg(avg_spend=('amount','mean'),
                  count=('fraud_flag','count'),
                  fraud=('fraud_flag','sum'))
             .reindex(age_order).reset_index())
    age['fraud_rate'] = age['fraud'] / age['count'] * 100

    x = np.arange(len(age))
    bars_a = ax2.bar(x, age['avg_spend'], color=C_STEEL,
                     alpha=0.85, width=0.5, edgecolor='none',
                     label='Avg Spend')
    ax2r = ax2.twinx()
    ax2r.plot(x, age['fraud_rate'], color=C_RED,
              lw=2.5, marker='D', ms=8,
              markerfacecolor='white',
              markeredgewidth=2, markeredgecolor=C_RED,
              label='Fraud Rate %')
    ax2r.set_ylabel('Fraud Rate %', color=C_RED, fontsize=9)
    ax2r.tick_params(colors=C_RED)
    for sp in ax2r.spines.values():
        sp.set_color(C_BORDER)
    white_ax(ax2, 'Age Group — Avg Spend vs Fraud Rate',
             'Age Group', 'Avg Spend (₹)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(age_order, color=C_TEXT)
    for bar, val in zip(bars_a, age['avg_spend']):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 5,
                 f'₹{val:.0f}', ha='center',
                 color=C_MUTED, fontsize=8)
    legend_items = [
        mpatches.Patch(color=C_STEEL, label='Avg Spend ₹'),
        mpatches.Patch(color=C_RED,   label='Fraud Rate %'),
    ]
    ax2.legend(handles=legend_items, facecolor=C_CARD,
               edgecolor=C_BORDER, labelcolor=C_TEXT)

save(fig, '06_state_age_analysis.png')

# ═════════════════════════════════════════════════════════════
# CHART 7 — FRAUD DEEP DIVE (5 panels)
# ═════════════════════════════════════════════════════════════
print('7/10  Fraud deep dive...')
fraud_df = df[df['fraud_flag'] == 1]
legit_df = df[df['fraud_flag'] == 0]

fig = white_fig(22, 11)
gs  = gridspec.GridSpec(2, 3, hspace=0.5, wspace=0.4)

# A — Fraud by state bubble
ax_a = fig.add_subplot(gs[0, 0])
if 'sender_state' in df.columns:
    fd_s = (df.groupby('sender_state')
              .agg(count=('fraud_flag','count'),
                   fraud=('fraud_flag','sum'),
                   avg_amt=('amount','mean'))
              .reset_index())
    fd_s['fraud_rate'] = fd_s['fraud'] / fd_s['count'] * 100
    sc = ax_a.scatter(fd_s['count'], fd_s['fraud_rate'],
                      s=fd_s['avg_amt']/4,
                      c=fd_s['fraud_rate'],
                      cmap='RdYlGn_r', alpha=0.85,
                      edgecolors='white', linewidth=1.5)
    white_ax(ax_a, 'State: Volume vs Fraud Rate\n(bubble=avg amount)',
             'Transaction Count', 'Fraud Rate %')
    for _, row in fd_s.iterrows():
        ax_a.annotate(row['sender_state'][:4],
                      (row['count'], row['fraud_rate']),
                      fontsize=7, color=C_MUTED)
    plt.colorbar(sc, ax=ax_a, label='Fraud Rate %')

# B — Fraud by hour
ax_b = fig.add_subplot(gs[0, 1])
if 'hour_of_day' in df.columns:
    fd_h = fraud_df.groupby('hour_of_day').size().reset_index(name='fraud')
    ax_b.fill_between(fd_h['hour_of_day'], fd_h['fraud'],
                      alpha=0.12, color=C_RED)
    ax_b.plot(fd_h['hour_of_day'], fd_h['fraud'],
              color=C_RED, lw=2.5, marker='o', ms=4)
    white_ax(ax_b, 'Fraud Cases by Hour of Day',
             'Hour', 'Fraud Count')
    peak = fd_h.loc[fd_h['fraud'].idxmax()]
    ax_b.annotate(
        f"Peak {int(peak['hour_of_day'])}:00\n{int(peak['fraud'])} cases",
        xy=(peak['hour_of_day'], peak['fraud']),
        xytext=(peak['hour_of_day']+2, peak['fraud']*0.88),
        color=C_ORANGE, fontsize=8,
        arrowprops=dict(arrowstyle='->', color=C_ORANGE))

# C — Device x Network fraud heatmap
ax_c = fig.add_subplot(gs[0, 2])
if 'device_type' in df.columns and 'network_type' in df.columns:
    fd_dev = (df.groupby(['device_type','network_type'])
                .agg(fraud=('fraud_flag','sum'),
                     count=('fraud_flag','count'))
                .reset_index())
    fd_dev['rate'] = fd_dev['fraud'] / fd_dev['count'] * 100
    piv_dev = fd_dev.pivot(index='device_type',
                           columns='network_type', values='rate')
    cmap_r = LinearSegmentedColormap.from_list(
        'rd', ['#FEF2F2','#EF9595','#B91C1C'], N=128)
    sns.heatmap(piv_dev, ax=ax_c, cmap=cmap_r,
                annot=True, fmt='.3f',
                linewidths=0.5, linecolor='white',
                annot_kws={'size':10, 'color':C_TEXT},
                cbar_kws={'label':'Fraud Rate %'})
    ax_c.set_title('Fraud Rate: Device x Network (%)',
                   color=C_TEXT, fontsize=12, fontweight='bold')
    ax_c.tick_params(colors=C_TEXT, labelsize=9)
    ax_c.set_xlabel('Network Type', color=C_MUTED)
    ax_c.set_ylabel('Device Type',  color=C_MUTED)

# D — Fraud vs Legit amount distribution
ax_d = fig.add_subplot(gs[1, 0:2])
ax_d.hist(legit_df['amount'].clip(0, 15000), bins=70,
          alpha=0.50, color=C_ACCENT, label='Legit',
          density=True, edgecolor='none')
ax_d.hist(fraud_df['amount'].clip(0, 15000), bins=70,
          alpha=0.70, color=C_RED, label='Fraud',
          density=True, edgecolor='none')
white_ax(ax_d, 'Amount Distribution: Fraud vs Legit Transactions (clipped at 15K)',
         'Transaction Amount ()', 'Density')
ax_d.axvline(legit_df['amount'].mean(), color=C_ACCENT,
             linestyle='--', lw=2,
             label=f'Legit mean {legit_df["amount"].mean():.0f}')
ax_d.axvline(fraud_df['amount'].mean(), color=C_RED,
             linestyle='--', lw=2,
             label=f'Fraud mean {fraud_df["amount"].mean():.0f}')
ax_d.legend(facecolor=C_CARD, edgecolor=C_BORDER,
            labelcolor=C_TEXT, fontsize=9)

# E — High value risk
ax_e = fig.add_subplot(gs[1, 2])
normal  = df[df['amount'] <= 10000]
highval = df[df['amount'] >  10000]
segs    = ['Normal\n(<=10K)', 'High Value\n(>10K)']
rates   = [normal['fraud_flag'].mean()*100,
           highval['fraud_flag'].mean()*100]
counts  = [len(normal), len(highval)]
seg_col = [C_ACCENT, C_RED]
bars_e  = ax_e.bar(segs, rates, color=seg_col,
                   width=0.5, edgecolor='none', alpha=0.85)
white_ax(ax_e, 'Fraud Rate: Normal vs High-Value',
         'Segment', 'Fraud Rate %')
for bar, rate, cnt in zip(bars_e, rates, counts):
    ax_e.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 0.003,
              f'{rate:.3f}%\n({cnt:,} txns)',
              ha='center', color=C_MUTED, fontsize=9)
ax_e.tick_params(axis='x', colors=C_TEXT)

save(fig, '07_fraud_deep_dive.png')

# ═════════════════════════════════════════════════════════════
# CHART 8 — ML MODEL: Feature Importance + ROC + Confusion Matrix
# ═════════════════════════════════════════════════════════════
print('8/10  ML model results...')
fig = white_fig(22, 8)
gs  = gridspec.GridSpec(1, 3, wspace=0.42)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

fig.suptitle('XGBoost + SMOTE — Fraud Detection Model Analysis',
             color=C_TEXT, fontsize=16, fontweight='bold', y=1.02)

# Feature importance bars
top = feat_imp.head(14)
fi_colors = [C_RED    if i < 3
             else C_ORANGE if i < 6
             else C_ACCENT if i < 10
             else C_MUTED
             for i in range(len(top))]
ax1.barh(top['Feature'][::-1], top['Importance'][::-1],
         color=fi_colors[::-1], height=0.65, edgecolor='none')
white_ax(ax1, 'Top Feature Importances\nRed=Critical | Amber=High | Navy=Moderate',
         'Importance Score', '')
ax1.tick_params(axis='y', colors=C_TEXT, labelsize=8)
for i, val in enumerate(top['Importance'][::-1]):
    ax1.text(val + 0.0005, i, f'{val:.4f}',
             va='center', color=C_MUTED, fontsize=7)

# ROC Curve
fpr_c, tpr_c, _ = roc_curve(y_test, y_prob)
ax2.fill_between(fpr_c, tpr_c, alpha=0.08, color=C_ACCENT)
ax2.plot(fpr_c, tpr_c, color=C_ACCENT, lw=2.5,
         label=f'XGBoost  AUC={auc:.4f}')
ax2.plot([0,1],[0,1], '--', color=C_LIGHT, lw=1.5, alpha=0.8,
         label='Random  AUC=0.500')
white_ax(ax2, f'ROC Curve   AUC = {auc:.4f}',
         'False Positive Rate', 'True Positive Rate')
ax2.legend(facecolor=C_CARD, edgecolor=C_BORDER,
           labelcolor=C_TEXT, fontsize=9)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)

# Precision-Recall inset
prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
inset = ax2.inset_axes([0.45, 0.08, 0.5, 0.38])
inset.fill_between(rec_c, prec_c, alpha=0.12, color=C_GREEN)
inset.plot(rec_c, prec_c, color=C_GREEN, lw=1.5)
inset.set_facecolor(C_CARD)
inset.set_title(f'PR Curve  AP={ap:.3f}',
                color=C_TEXT, fontsize=7)
inset.tick_params(colors=C_LIGHT, labelsize=6)
for sp in inset.spines.values():
    sp.set_color(C_BORDER)
inset.set_xlabel('Recall',    color=C_MUTED, fontsize=6)
inset.set_ylabel('Precision', color=C_MUTED, fontsize=6)

# Confusion Matrix
cmap_cm = LinearSegmentedColormap.from_list(
    'cm', ['#EEF4FF','#DDEAFF','#3B6CA8'], N=128)
ax3.imshow([[0.05, 0.6],[0.9, 0.05]], cmap=cmap_cm, aspect='auto')
ax3.set_xticks([0,1])
ax3.set_yticks([0,1])
ax3.set_xticklabels(['Predicted Legit','Predicted Fraud'],
                    color=C_TEXT, fontsize=10)
ax3.set_yticklabels(['Actual Legit','Actual Fraud'],
                    color=C_TEXT, fontsize=10)
ax3.set_title('Confusion Matrix', color=C_TEXT,
              fontsize=13, fontweight='bold', pad=12)
labels_cm = [
    [f'TRUE NEGATIVE\n{tn:,}\nLegit correctly cleared',
     f'FALSE POSITIVE\n{fp:,}\nLegit wrongly flagged'],
    [f'FALSE NEGATIVE\n{fn}\nFraud missed',
     f'TRUE POSITIVE\n{tp}\nFraud caught']
]
text_colors = [[C_TEXT, C_ORANGE],[C_RED, C_GREEN]]
for i in range(2):
    for j in range(2):
        ax3.text(j, i, labels_cm[i][j],
                 ha='center', va='center',
                 color=text_colors[i][j],
                 fontsize=9, fontweight='bold')
for sp in ax3.spines.values():
    sp.set_color(C_BORDER)

save(fig, '08_ml_model_results.png')

# ═════════════════════════════════════════════════════════════
# CHART 9 — AGE x CATEGORY HEATMAP + DEVICE NETWORK BUBBLE
# ═════════════════════════════════════════════════════════════
print('9/10  Segmentation heatmap...')
fig = white_fig(22, 8)
gs  = gridspec.GridSpec(1, 2, wspace=0.42)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

if 'sender_age_group' in df.columns and 'merchant_category' in df.columns:
    age_cat = (df.groupby(['sender_age_group','merchant_category'])
                 ['amount'].mean().reset_index())
    piv_ac = age_cat.pivot(index='sender_age_group',
                           columns='merchant_category',
                           values='amount').reindex(age_order)
    cmap_g = LinearSegmentedColormap.from_list(
        'spend', ['#EEF4FF','#DDEAFF','#3B6CA8','#0B2545'], N=256)
    sns.heatmap(piv_ac, ax=ax1, cmap=cmap_g,
                annot=True, fmt='.0f',
                linewidths=0.5, linecolor='white',
                annot_kws={'size':9, 'color':'white'},
                cbar_kws={'label':'Avg Spend ()'})
    ax1.set_title('Avg Spend Heatmap — Age Group x Merchant Category',
                  color=C_TEXT, fontsize=12, fontweight='bold')
    ax1.tick_params(colors=C_TEXT, labelsize=9)
    ax1.set_xlabel('Merchant Category', color=C_MUTED)
    ax1.set_ylabel('Age Group',         color=C_MUTED)
    cbar = ax1.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(C_MUTED)
    cbar.ax.tick_params(colors=C_MUTED)

if 'device_type' in df.columns and 'network_type' in df.columns:
    dev_net = (df.groupby(['device_type','network_type'])
                 .agg(count=('fraud_flag','count'),
                      fraud=('fraud_flag','mean'))
                 .reset_index())
    dev_net['fraud_pct'] = dev_net['fraud'] * 100
    x_pos = {'3G':0,'4G':1,'5G':2,'WiFi':3}
    y_pos = {'Android':0,'iOS':1,'Web':2}
    white_ax(ax2, 'Device x Network: Volume (size) & Fraud Rate (color)',
             'Network Type', 'Device Type')
    for _, row in dev_net.iterrows():
        xv = x_pos.get(row['network_type'], 0)
        yv = y_pos.get(row['device_type'],  0)
        sc = ax2.scatter(xv, yv,
                         s=row['count']/500,
                         c=[row['fraud_pct']],
                         cmap='RdYlGn_r', vmin=0.1, vmax=0.3,
                         alpha=0.85,
                         edgecolors='white', linewidth=1.5)
        ax2.text(xv, yv+0.2,
                 f"{row['count']:,}\n{row['fraud_pct']:.3f}%",
                 ha='center', fontsize=7.5, color=C_MUTED)
    ax2.set_xticks(list(x_pos.values()))
    ax2.set_xticklabels(list(x_pos.keys()), color=C_TEXT)
    ax2.set_yticks(list(y_pos.values()))
    ax2.set_yticklabels(list(y_pos.keys()), color=C_TEXT)
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-0.5, 2.7)
    plt.colorbar(sc, ax=ax2, label='Fraud Rate %')

save(fig, '09_segmentation_heatmap.png')

# ═════════════════════════════════════════════════════════════
# CHART 10 — SMOTE BALANCE + FULL FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════
print('10/10  SMOTE & full feature importance...')
fig = white_fig(22, 9)
gs  = gridspec.GridSpec(1, 3, wspace=0.45)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1:])

fig.suptitle('Model Training Pipeline — SMOTE Balancing & Feature Ranking',
             color=C_TEXT, fontsize=15, fontweight='bold', y=1.02)

# SMOTE before/after
labels_s = ['Fraud\nBefore','Legit\nBefore',
            'Fraud\nAfter SMOTE','Legit\nAfter SMOTE']
values_s  = [y_train.sum(),  (y_train==0).sum(),
             (y_res==1).sum(), (y_res==0).sum()]
colors_s  = [C_RED, C_LIGHT, C_GREEN, C_ACCENT]
bars_s = ax1.bar(range(4), values_s, color=colors_s,
                 width=0.6, edgecolor='none', alpha=0.85)
white_ax(ax1, 'Class Balance: Before vs After SMOTE', '', 'Samples')
ax1.set_xticks(range(4))
ax1.set_xticklabels(labels_s, color=C_TEXT, fontsize=8.5)
for bar, val in zip(bars_s, values_s):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 300,
             f'{val:,}', ha='center', color=C_MUTED, fontsize=8)
ax1.axvline(1.5, color=C_BORDER, linestyle='--', alpha=0.8)
y_top = max(values_s) * 1.12
ax1.text(0.5,  y_top, 'BEFORE', color=C_MUTED,  fontsize=9, ha='center')
ax1.text(2.5,  y_top, 'AFTER',  color=C_GREEN,  fontsize=9, ha='center')
ax1.set_ylim(0, y_top * 1.08)

# Full feature importance
all_colors = [C_RED    if i < 3
              else C_ORANGE if i < 6
              else C_ACCENT if i < 10
              else C_MUTED
              for i in range(len(feat_imp))]
ax2.barh(feat_imp['Feature'][::-1],
         feat_imp['Importance'][::-1],
         color=all_colors[::-1], height=0.7, edgecolor='none')
white_ax(ax2, 'Complete Feature Importance Ranking — XGBoost',
         'Importance Score', '')
ax2.tick_params(axis='y', colors=C_TEXT, labelsize=9)
for i, val in enumerate(feat_imp['Importance'][::-1]):
    ax2.text(val + 0.0005, i, f'{val:.4f}',
             va='center', color=C_MUTED, fontsize=7.5)

legend_items = [
    mpatches.Patch(color=C_RED,    label='Top 3 — Critical predictors'),
    mpatches.Patch(color=C_ORANGE, label='Ranks 4-6 — High importance'),
    mpatches.Patch(color=C_ACCENT, label='Ranks 7-10 — Moderate'),
    mpatches.Patch(color=C_MUTED,  label='Remaining features'),
]
ax2.legend(handles=legend_items, loc='lower right',
           facecolor=C_CARD, edgecolor=C_BORDER,
           labelcolor=C_TEXT, fontsize=8)

save(fig, '10_smote_feature_importance.png')

# ═════════════════════════════════════════════════════════════
# DONE
# ═════════════════════════════════════════════════════════════
print(f'''
{"="*55}
  ALL 10 CHARTS SAVED TO visuals/
{"="*55}

  01_platform_kpis.png
  02_monthly_trend.png
  03_time_heatmap.png
  04_txn_type_category.png
  05_bank_analysis.png
  06_state_age_analysis.png
  07_fraud_deep_dive.png
  08_ml_model_results.png
  09_segmentation_heatmap.png
  10_smote_feature_importance.png

  Model: XGBoost + SMOTE
  AUC  : {auc:.4f}
  Fraud caught : {tp}
  Fraud missed : {fn}
{"="*55}
''')