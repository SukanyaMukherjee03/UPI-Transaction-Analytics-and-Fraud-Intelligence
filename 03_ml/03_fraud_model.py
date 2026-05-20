"""
UPI PULSE 2024 — FRAUD DETECTION MODEL
Algorithm : XGBoost
Balancing : SMOTE
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

os.makedirs('visuals', exist_ok=True)
os.makedirs('ml',      exist_ok=True)

print("="*60)
print("  UPI FRAUD DETECTION — XGBoost + SMOTE")
print("="*60)

# ── 1. Load clean data ────────────────────────────────────────
print("\nLoading data...")
df = pd.read_csv("data/upi_clean.csv")
print(f"Total Rows  : {len(df):,}")
print(f"Fraud Cases : {df['fraud_flag'].sum()}")
print(f"Fraud Rate  : {df['fraud_flag'].mean()*100:.3f}%")

# ── 2. Label encode categorical columns ───────────────────────
# ML models only understand numbers.
# LabelEncoder converts text → numbers
# Example: SBI=0, HDFC=1, ICICI=2 etc.
cat_cols = [
    'transaction_type',
    'merchant_category',
    'sender_bank',
    'receiver_bank',
    'sender_age_group',
    'receiver_age_group',
    'sender_state',
    'device_type',
    'network_type',
    'day_of_week'
]

le    = LabelEncoder()
df_ml = df.copy()

for col in cat_cols:
    df_ml[col + '_enc'] = le.fit_transform(df_ml[col].astype(str))

print(f"\nEncoded {len(cat_cols)} categorical columns.")

# ── 3. Define features ────────────────────────────────────────
# FIX: removed 'hour' (duplicate of hour_of_day)
# FIX: using amount_inr (correct column name from your clean CSV)
# FIX: added is_same_bank and sender_age_order (created in clean script)
feature_cols = [c + '_enc' for c in cat_cols] + [
    'amount_inr',         # transaction amount
    'hour_of_day',        # hour from raw data
    'is_weekend',         # 1=weekend, 0=weekday
    'is_success',         # 1=success, 0=failed
    'is_same_bank',       # 1=same bank transfer
    'month_num',          # month as number (1-12)
    'sender_age_order'    # age group as number (1-5)
]

# Safety check — remove any feature not in dataframe
feature_cols = [f for f in feature_cols if f in df_ml.columns]
print(f"Features used: {len(feature_cols)}")
print(f"Feature list : {feature_cols}")

X = df_ml[feature_cols]
y = df_ml['fraud_flag']

# ── 4. Train test split ───────────────────────────────────────
# 80% training, 20% testing
# stratify=y ensures same fraud % in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining Rows : {len(X_train):,}")
print(f"Testing Rows  : {len(X_test):,}")
print(f"Fraud in train: {y_train.sum()}")

# ── 5. SMOTE — balance the training data ──────────────────────
# Problem: only ~384 fraud out of 200,000 training rows
# Model would just predict everything as legit and get 99.8% accuracy
# SMOTE creates synthetic fraud examples so model learns properly
# RULE: NEVER apply SMOTE to test data — test must reflect reality
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"After SMOTE:")
print(f"  Legit : {(y_train_bal==0).sum():,}")
print(f"  Fraud : {(y_train_bal==1).sum():,}")

# ── 6. Train XGBoost ──────────────────────────────────────────
print("\nTraining XGBoost model...")
print("(takes ~30-60 seconds)")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    verbosity=0           # suppresses training spam
)

model.fit(X_train_bal, y_train_bal)
print("Model trained ✓")

# ── 7. Predictions ────────────────────────────────────────────
y_pred_prob = model.predict_proba(X_test)[:, 1]
# predict_proba gives probability of fraud (0.0 to 1.0)
# [:, 1] = take the fraud probability column

# Threshold = 0.3 instead of default 0.5
# Lower threshold = catch more fraud (fewer missed frauds)
# Trade-off: more false alarms, but missing fraud is worse
threshold = 0.3
y_pred = (y_pred_prob >= threshold).astype(int)

# ── 8. Evaluation ─────────────────────────────────────────────
print("\n" + "="*60)
print("  MODEL EVALUATION")
print("="*60)

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Legit', 'Fraud']))

auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score : {auc:.4f}")
print("(0.5 = random guessing | 1.0 = perfect)")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"  Fraud caught correctly (True Positives)  : {tp}")
print(f"  Fraud missed (False Negatives)           : {fn}")
print(f"  Legit wrongly flagged (False Positives)  : {fp:,}")
print(f"  Legit correctly cleared (True Negatives) : {tn:,}")

# ── 9. Feature importance ─────────────────────────────────────
feature_importance = pd.DataFrame({
    'Feature':    feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("\nTop 10 Features:")
print(feature_importance.head(10).to_string(index=False))

# ── 10. Charts ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('UPI Fraud Detection — XGBoost + SMOTE',
             fontsize=15, fontweight='bold')

# Chart 1: Feature importance
top_feats = feature_importance.head(10)
axes[0].barh(
    top_feats['Feature'][::-1],
    top_feats['Importance'][::-1],
    color='steelblue'
)
axes[0].set_title('Top 10 Feature Importances', fontweight='bold')
axes[0].set_xlabel('Importance Score')

# Chart 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='darkorange', lw=2.5,
             label=f'XGBoost (AUC = {auc:.3f})')
axes[1].plot([0,1],[0,1], '--', color='grey', lw=1.5,
             label='Random (0.500)')
axes[1].fill_between(fpr, tpr, alpha=0.15, color='darkorange')
axes[1].set_title(f'ROC Curve  (AUC = {auc:.3f})', fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Chart 3: Confusion matrix
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
    xticklabels=['Predicted Legit', 'Predicted Fraud'],
    yticklabels=['Actual Legit',    'Actual Fraud'],
    annot_kws={'size': 14}
)
axes[2].set_title(f'Confusion Matrix\nAUC = {auc:.3f}',
                  fontweight='bold')

plt.tight_layout()
plt.savefig('visuals/fraud_model_results.png', dpi=150,
            bbox_inches='tight')
print("\nChart saved: visuals/fraud_model_results.png")

# ── 11. Save feature importance CSV ───────────────────────────
feature_importance.to_csv('ml/feature_importance.csv', index=False)
print("CSV saved:   ml/feature_importance.csv")

print("\n✅ ML MODEL COMPLETE")
print(f"   AUC     : {auc:.4f}")
print(f"   Fraud caught : {tp}")
print(f"   Fraud missed : {fn}")