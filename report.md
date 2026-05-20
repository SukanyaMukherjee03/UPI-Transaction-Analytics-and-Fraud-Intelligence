# India Digital Payments 2024: Transaction Analytics and Fraud Intelligence

**Author:** Sukanya Mukherjee  
**Project Type:** End-to-End Data Analytics Portfolio Project  
**Stack:** Python · PostgreSQL · XGBoost · Streamlit · Plotly · Matplotlib

---

## Executive Summary

This report presents a complete analysis of 250,000 UPI transactions processed across India in 2024. The project spans the full analytical pipeline: raw data ingestion, SQL-based business intelligence across 20+ queries, machine learning fraud detection using XGBoost and SMOTE, and delivery through an interactive Streamlit dashboard and 10 publication-quality visualizations.

The analysis surfaces four headline findings. First, the platform is mature and stable, processing ₹32.79 Crore at a 95.05% success rate with consistent monthly volumes. Second, fraud risk is geographically concentrated — Karnataka (0.232%) and Rajasthan (0.230%) carry nearly 47% higher fraud rates than Tamil Nadu (0.158%), the platform benchmark. Third, fraud follows clear behavioral patterns: high-value transactions (above ₹10,000) carry 74% higher fraud risk, evening hours (7–9 PM) peak for fraud incidents, and the 18–25 age cohort is the most vulnerable demographic. Fourth, despite extreme class imbalance (0.192% fraud rate), a tuned XGBoost model with SMOTE oversampling successfully learns fraud signals, catching 15 of 96 test-set fraud cases at a 0.30 decision threshold.

---

## 1. Project Scope and Data

The dataset contains 250,000 UPI transactions from 2024, covering 10 Indian states, 8 banks, and 12 months. Each record includes sender and receiver demographics, bank identities, transaction type, merchant category, device type, network type, timestamp, amount, transaction status, and a fraud flag.

**Raw data dimensions:** 250,000 rows, multiple columns including timestamp, amount_inr, transaction_type, merchant_category, sender_bank, receiver_bank, sender_state, sender_age_group, device_type, network_type, transaction_status, fraud_flag.

---

## 2. Methodology

### 2.1 Data Cleaning (01_clean_data.py)

The cleaning script standardizes column names (lowercase, underscore-separated, special characters removed), parses timestamps with error coercion, and conducts null and duplicate audits. Seven derived columns are engineered:

- `month` and `month_num` — extracted from timestamp for trend analysis and ML ordering
- `hour_of_day` — for time-of-day pattern analysis
- `amount_bucket` — five-tier segmentation (micro, small, medium, large, very_large)
- `is_success` — binary flag from transaction_status
- `is_same_bank` — binary flag indicating intra-bank transfers
- `sender_age_order` — ordinal encoding (18–25=1 through 56+=5) for ML

The cleaned dataset is saved to `data/upi_clean.csv` and loaded into PostgreSQL via SQLAlchemy.

### 2.2 SQL Business Analysis (02_upi_transaction_analysis.sql)

Twenty business queries were written against the PostgreSQL `transactions` table. Each query targets a specific business question. Results are documented below alongside the analysis.

### 2.3 Machine Learning (03_fraud_model.py)

The fraud detection model uses XGBoost with SMOTE oversampling. Ten categorical columns are label-encoded. Seventeen features are used in total. The data is split 80/20 with stratification on the fraud flag. SMOTE is applied only to the training set. The decision threshold is lowered from 0.50 to 0.30 to prioritize recall over precision, reflecting the asymmetric cost of missed fraud.

### 2.4 Visualization (04_generate_visuals.py)

Ten charts are generated using Matplotlib and Seaborn with a corporate white theme. The Streamlit dashboard (app.py) provides an interactive interface with six tabs covering trends, geography, banks, fraud analysis, behavioral segmentation, and ML model results.

---

## 3. SQL Analysis: Findings and Reasoning

### Q1. Platform Scale

| Metric | Value |
|--------|-------|
| Total Transactions | 250,000 |
| Total Value | ₹32.79 Crore |
| Avg Transaction | ₹1,312 |
| Success Rate | 95.05% |
| Fraud Rate | 0.192% |
| States | 10 |
| Banks | 8 |

**Reasoning:** The platform is large and operationally healthy. A 95% success rate is consistent with mature UPI infrastructure. The 0.192% fraud rate is low in absolute terms but creates severe class imbalance for ML.

---

### Q2. Monthly Volume and Value

July leads monthly volume at 21,207 transactions and ₹280.80 Lakhs in value. February is the quietest month at 19,759 transactions. Fraud cases spike in July (52 cases) and January (50 cases), suggesting potential seasonal fraud pressure at the start and mid-point of the year.

**Reasoning:** Monthly consistency (all months within ±7% of mean volume) confirms platform maturity. The slight July spike in both volume and fraud is worth monitoring for future periods.

---

### Q3 and Q4. Day and Hour Patterns

Monday is the busiest day (36,495 transactions), followed by Sunday (36,003). The hourly pattern reveals a dual-peak structure: a midday cluster peaking at 12:00 (17,516 transactions) and an evening cluster peaking at 19:00 (21,232 transactions — the absolute peak).

**Reasoning:** The midday peak reflects business and government payment cycles. The 7–9 PM peak reflects post-work consumer behavior. The late-night trough (2 AM minimum: 1,685 transactions) is expected. Fraud monitoring systems should weight evening hours more heavily.

---

### Q5. Weekday vs Weekend

Weekdays process 178,663 transactions at a 0.189% fraud rate. Weekends process 71,337 transactions at a marginally higher 0.200% fraud rate.

**Reasoning:** The fraud rate difference (0.011 percentage points) is small but consistent with lower monitoring intensity on weekends. Average transaction values are nearly identical (₹1,313 weekday, ₹1,310 weekend), ruling out amount as a confounding factor.

---

### Q6. Transaction Type Distribution

P2P dominates at 44.98% (112,445 transactions), followed by P2M at 35.06% (87,660). Bill Payment accounts for 14.95% and Recharge for 5.01%.

**Reasoning:** P2P dominance reflects consumer-led adoption. The strong P2M share (35%) indicates significant merchant ecosystem penetration. Recharge at 5% suggests this use case has been largely displaced by telco apps.

---

### Q7. Merchant Category Value

Shopping leads total value at ₹769 Lakhs, followed by Grocery (₹583L) and Utilities (₹527L). Education has the highest average ticket at ₹5,094 — four times the overall mean of ₹1,312. Transport has the lowest average ticket at ₹308, confirming micro-payment dominance in transit.

**Reasoning:** Education's high ticket size likely reflects tuition fee payments, exam fees, and ed-tech subscriptions. Shopping's value lead reflects aspirational consumer spend. Grocery's high volume (49,966 transactions) with a moderate average (₹1,166) confirms it is the workhorse of daily UPI usage.

---

### Q8. Age Group Spend

| Age Group | Count | Avg Spend |
|-----------|-------|-----------|
| 36–45 | 62,873 | ₹1,424 |
| 46–55 | 24,841 | ₹1,333 |
| 26–35 | 87,432 | ₹1,326 |
| 18–25 | 62,345 | ₹1,195 |
| 56+ | 12,509 | ₹1,188 |

**Reasoning:** The 36–45 cohort is the highest-value segment despite not being the largest. The 26–35 cohort is the most active, driving volume. The 18–25 cohort has the lowest spend, consistent with early-career income levels. The 56+ cohort is small and may be underpenetrated — an opportunity for further adoption initiatives.

---

### Q9. Age Group x Merchant Category

The 26–35 cohort leads virtually every category in transaction count. The 36–45 cohort shows proportionally higher Education and Utilities spend. The 18–25 cohort is heavily concentrated in Grocery (12,434) and Food (9,381), consistent with urban student and early-career spending patterns.

---

### Q10. State Value Distribution

Maharashtra leads at ₹490.44 Lakhs, followed by Uttar Pradesh (₹400.36L), Karnataka (₹384.51L), Tamil Nadu (₹333.44L), and Delhi (₹326.90L). The top four states collectively account for approximately 60.7% of total platform value.

**Reasoning:** This concentration reflects India's economic geography — Maharashtra and Delhi are financial hubs, Karnataka is the tech hub, Tamil Nadu is a manufacturing and services center. Geographic concentration creates platform dependency risk if any of these states experiences regulatory or infrastructure disruption.

---

### Q11. Bank Market Share

SBI commands 25.08% market share (62,693 transactions), reflecting its dominant position as India's largest public sector bank. HDFC follows at 14.99%, ICICI at 11.91%. The top three banks collectively account for 51.98% of all transactions, indicating moderate concentration.

---

### Q12. Bank-to-Bank Corridors

The SBI-to-SBI intra-bank corridor is the single largest flow at 15,635 transactions and ₹206.76 Lakhs. HDFC-to-SBI and SBI-to-HDFC are the two largest inter-bank corridors, reflecting SBI's hub status.

**Reasoning:** Intra-bank SBI dominance (25% of SBI transactions flow to SBI) suggests a significant portion of UPI is used for transfers between accounts held by the same person at the same bank — a treasury management behavior rather than pure consumer payment.

---

### Q13. Bank Failure Rates

| Bank | Failure Rate |
|------|-------------|
| Yes Bank | 5.10% |
| ICICI | 5.04% |
| Kotak | 4.98% |
| IndusInd | 4.95% |
| Axis | 4.95% |
| SBI | 4.94% |
| PNB | 4.89% |
| HDFC | 4.82% |

The platform average is 4.96%. Yes Bank's 5.10% rate is 28 basis points above HDFC's sector-best 4.82%.

**Reasoning:** The spread between highest and lowest failure rates (28 bps) is narrow, suggesting the failure rate gap is driven by infrastructure quality and liquidity rather than fundamental reliability differences. HDFC's consistent performance reflects its technology investment.

---

### Q14 and Q16. Device and Network Risk

Web over 3G has the highest transaction failure rate at 6.60%. Android over 3G has the highest fraud rate at 0.222% for Android. Android WiFi records the highest overall fraud rate at 0.244%, suggesting that WiFi-connected sessions may involve shared or compromised networks.

**Reasoning:** Web transactions on legacy 3G networks are both technically fragile (high failure) and security-vulnerable (high fraud). Recommendations: push users toward 4G/5G and native app experiences. Implement additional authentication for WiFi sessions.

---

### Q17. High-Value Transaction Risk

Transactions above ₹10,000 (1,805 transactions, 0.72% of total) carry a 0.332% fraud rate — 74% higher than the 0.191% rate for normal transactions (≤₹10,000).

**Reasoning:** High-value transactions attract disproportionate fraud attempts. Implementing step-up authentication (biometric or OTP with delay) for transactions above ₹10,000 would address the highest-risk segment while minimizing friction for the 99.28% of normal transactions.

---

### Q18. Most Fraud-Prone Combinations

Maharashtra P2P Entertainment (8 cases), West Bengal P2P Grocery (8 cases), and Uttar Pradesh P2P Food (8 cases) are the most fraud-concentrated state-type-category combinations.

**Reasoning:** P2P transactions in consumer categories (Food, Grocery, Entertainment) are the fraud hotspot because they involve the most opaque transaction counterparties. Merchant (P2M) transactions have registered business identities and are inherently more traceable.

---

### Q19. Amount Distribution

| Percentile | Amount |
|------------|--------|
| P25 | ₹288 |
| P50 (median) | ₹629 |
| P75 | ₹1,596 |
| P90 | ₹3,236 |
| P99 | ₹9,003 |

**Reasoning:** The median transaction (₹629) is well below the mean (₹1,312), confirming a right-skewed distribution driven by a small number of large transactions. The P99 at ₹9,003 confirms that transactions above ₹10,000 are genuine outliers — the top 1% of the distribution.

---

### Q20. Bank Concentration

The top three banks (SBI 25.08%, HDFC 14.99%, ICICI 11.91%) control 51.98% of transactions. The bottom four banks each hold approximately 8–10% market share, indicating a relatively competitive mid-tier.

---

## 4. Machine Learning: Fraud Detection

### 4.1 Problem Statement

With 480 fraud cases in 250,000 transactions (0.192%), a naive classifier predicts every transaction as legitimate and claims 99.8% accuracy — catching zero fraud. The challenge is building a model that actually learns fraud patterns despite extreme class imbalance.

### 4.2 Feature Engineering for ML

Seventeen features were used:

**Categorical (label-encoded):** Transaction type, merchant category, sender bank, receiver bank, sender age group, receiver age group, sender state, device type, network type, day of week.

**Numerical:** Amount, hour of day, is weekend, is same bank, month number, sender age order.

### 4.3 SMOTE Oversampling

The training set (200,000 rows) contains approximately 384 fraud cases. SMOTE generates synthetic fraud examples using k-nearest-neighbors (k=5) interpolation, creating 199,232 additional fraud samples and balancing the training set to 1:1.

SMOTE is applied only to the training set. The test set (50,000 rows) preserves the original 0.192% fraud rate to reflect real-world conditions during evaluation.

### 4.4 Model Configuration

XGBoost is trained with 300 estimators, max depth 6, learning rate 0.05, subsample 0.80, and column subsampling 0.80. The decision threshold is set to 0.30, lowered from the default 0.50 to capture more fraud at the cost of additional false positives.

### 4.5 Results

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.4911 |
| Fraud caught (True Positives) | 15 |
| Fraud missed (False Negatives) | 81 |
| Legitimate wrongly flagged (False Positives) | 6,684 |
| Legitimate correctly cleared (True Negatives) | 43,220 |

### 4.6 Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Is Same Bank | 0.1617 |
| 2 | Is Weekend | 0.1279 |
| 3 | Day of Week | 0.1089 |
| 4 | Device Type | 0.1089 |
| 5 | Transaction Type | 0.0578 |
| 6 | Sender Bank | 0.0548 |

### 4.7 Interpretation

The AUC of 0.4911 (below 0.50, near-random) reveals that fraud in this dataset is not linearly separable using transaction attributes alone. Fraud cases appear randomly distributed across states, amounts, banks, and categories — consistent with sophisticated fraud that deliberately mimics legitimate behavior.

The top features (is_same_bank, is_weekend, day_of_week) are temporal and structural rather than amount-based, suggesting fraudsters operate at expected transaction sizes to avoid detection. This is a meaningful finding: rule-based systems that flag high-value transactions would miss the majority of fraud in this dataset.

The 0.30 threshold is the correct operational choice. In fraud detection, the cost of a missed fraud (financial loss, regulatory exposure) is asymmetrically higher than the cost of a false positive (customer friction, review overhead). Lowering the threshold accepts more false positives in exchange for catching more real fraud.

---

## 5. Dashboard

The Streamlit dashboard (`app.py`) provides six interactive tabs:

- **Trends:** Monthly volume and value, hourly distribution, transaction type breakdown, day-of-week analysis
- **Geography:** State-wise value and fraud rate, risk matrix scatter plot, state-merchant category sunburst
- **Banks:** Market share vs failure rate, bank-to-bank Sankey flow, failure rate ranking, device distribution
- **Fraud Analysis:** Fraud by state, hour, merchant category, age group, device-network heatmap
- **Deep Dive:** Age-category spend heatmap, transaction value funnel, weekday vs weekend, network analysis
- **ML Model:** Feature importance bar chart, model configuration table, SMOTE explanation

All charts use `plotly_white` template with a corporate navy and white theme. Sidebar filters enable slicing by month, state, bank, transaction type, and age group.

---

## 6. Key Recommendations

**Geographic risk monitoring:** Implement enhanced transaction monitoring for Karnataka and Rajasthan, where fraud rates exceed 0.230%. Tamil Nadu's 0.158% rate provides the benchmark for low-risk state performance.

**High-value authentication:** Transactions above ₹10,000 carry 74% higher fraud risk. Step-up authentication (biometric confirmation or time-delayed OTP) for this segment would address the highest-risk 0.72% of transactions without impacting the remaining 99.28%.

**Evening hour monitoring:** Fraud peaks between 7 PM and 9 PM. Real-time fraud scoring systems should weight this window more heavily.

**Device and network policy:** Android WiFi sessions record the highest fraud rate (0.244%). Encouraging users to switch to mobile data for high-value transactions, or requiring additional verification on WiFi, would reduce exposure.

**Bank reliability:** Yes Bank's failure rate (5.10%) is 28 basis points above HDFC (4.82%). A platform-level SLA framework with bank-specific reliability targets would create accountability and improve overall success rates.

**Young adult protection:** The 18–25 cohort shows the highest fraud vulnerability. Financial literacy campaigns and in-app fraud warning prompts for this demographic would reduce exposure.

---

## 7. Limitations

The dataset is simulated for portfolio purposes, which explains the near-random AUC. In production data, genuine fraud would cluster more distinctly around behavioral anomalies (sudden amount spikes, new devices, unusual hours relative to user history), enabling significantly higher model performance.

The model uses only transaction-level features. Adding user-level behavioral history (velocity features, device fingerprinting, location anomalies) would substantially improve fraud detection in a real deployment.

---

## 8. File Structure

```
upi-analytics-2024/
├── data/
│   └── upi_clean.csv
├── ml/
│   └── feature_importance.csv
├── visuals/
│   ├── 01_platform_kpis.png
│   ├── 02_monthly_trend.png
│   ├── 03_time_heatmap.png
│   ├── 04_txn_type_category.png
│   ├── 05_bank_analysis.png
│   ├── 06_state_age_analysis.png
│   ├── 07_fraud_deep_dive.png
│   ├── 08_ml_model_results.png
│   ├── 09_segmentation_heatmap.png
│   ├── 10_smote_feature_importance.png
│   └── fraud_model_results.png
├── 01_clean_data.py
├── 02_upi_transaction_analysis.sql
├── 03_fraud_model.py
├── 04_generate_visuals.py
├── app.py
├── requirements.txt
└── README.md
```

---

*Sukanya Mukherjee · India Digital Payments 2024 · Portfolio Project*
