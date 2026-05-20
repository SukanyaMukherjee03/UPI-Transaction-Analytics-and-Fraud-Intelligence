import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

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

import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("UPI FRAUD DETECTION — XGBoost + SMOTE")
print("="*60)

# LOAD DATA
df = pd.read_csv("data/upi_clean.csv")

print("\nTotal Rows:", len(df))
print("Fraud Cases:", df['fraud_flag'].sum())

# CATEGORICAL COLUMNS
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

# LABEL ENCODING
le = LabelEncoder()

df_ml = df.copy()

for col in cat_cols:
    df_ml[col + '_enc'] = le.fit_transform(df_ml[col].astype(str))

# FEATURES
feature_cols = [c + '_enc' for c in cat_cols] + [
    'amount_inr',
    'hour_of_day',
    'is_weekend',
    'hour',
    'is_success'
]
X = df_ml[feature_cols]
y = df_ml['fraud_flag']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Rows:", len(X_train))
print("Testing Rows:", len(X_test))

# SMOTE
print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_bal, y_train_bal = smote.fit_resample(
    X_train,
    y_train
)

print("Balanced Fraud Cases:", sum(y_train_bal))

# MODEL
print("\nTraining XGBoost Model...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_bal, y_train_bal)

print("MODEL TRAINED")

# PREDICTIONS
y_pred_prob = model.predict_proba(X_test)[:,1]

threshold = 0.3

y_pred = (y_pred_prob >= threshold).astype(int)

# EVALUATION
print("\nMODEL EVALUATION")
print("="*60)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

auc = roc_auc_score(y_test, y_pred_prob)

print("ROC AUC:", round(auc,4))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\nTrue Positives:", tp)
print("False Negatives:", fn)
print("False Positives:", fp)
print("True Negatives:", tn)

# FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    'Importance',
    ascending=False
)

print("\nTOP FEATURES:\n")

print(feature_importance.head(10))

# CHARTS
fig, axes = plt.subplots(1,3, figsize=(20,6))

# FEATURE IMPORTANCE
top_features = feature_importance.head(10)

axes[0].barh(
    top_features['Feature'][::-1],
    top_features['Importance'][::-1]
)

axes[0].set_title("Feature Importance")

# ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

axes[1].plot(fpr, tpr)
axes[1].plot([0,1],[0,1],'--')

axes[1].set_title(f"ROC Curve AUC={auc:.3f}")

# CONFUSION MATRIX
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=axes[2]
)

axes[2].set_title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "visuals/fraud_model_results.png",
    dpi=150
)

print("\nChart Saved:")
print("visuals/fraud_model_results.png")

# SAVE FEATURE IMPORTANCE CSV
feature_importance.to_csv(
    "ml/feature_importance.csv",
    index=False
)

print("\nFeature importance CSV saved.")

print("\nML MODEL COMPLETE")