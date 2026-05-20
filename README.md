# UPI Transaction Analytics and Fraud Intelligence

End-to-end fintech analytics project analyzing 250,000 UPI transactions to evaluate fraud exposure, banking reliability, transaction behavior, merchant dynamics, and payment ecosystem concentration across India.

---

## Project Overview

Built a complete analytics workflow covering data cleaning, SQL business analysis, fraud-risk modeling, dashboard development, and executive reporting. The project analyzed transaction behavior across states, banks, merchant categories, devices, demographics, and payment corridors to generate operational and risk insights for the UPI ecosystem.

---

## Skills & Concepts Applied

### Data Analytics
- Exploratory Data Analysis (EDA)
- Business Intelligence Analysis
- Transaction and Behavioral Analytics
- Fraud and Risk Analysis
- Banking Performance Benchmarking
- Customer and Merchant Segmentation

### SQL & Data Engineering
- PostgreSQL Querying
- Aggregations and Window Analysis
- Multi-dimensional Business Queries
- Data Cleaning and Transformation
- Feature Engineering

### Machine Learning
- Fraud Classification Modeling
- XGBoost
- SMOTE Oversampling
- Class Imbalance Handling
- Threshold Optimization
- Feature Importance Analysis

### Dashboarding & Visualization
- Streamlit Dashboard Development
- Interactive Filtering and Drilldowns
- Plotly Visualizations
- Sankey Flow Diagrams
- KPI and Risk Visualization
- Executive Reporting Design

---

## Tech Stack

Python • PostgreSQL • Pandas • NumPy • XGBoost • Scikit-learn • Streamlit • Plotly • Matplotlib

---

## Key Insights

- Identified Karnataka’s fraud rate at 0.232%, 47% above Tamil Nadu
- Detected 74% higher fraud incidence in transactions above ₹10,000
- Benchmarked Yes Bank’s 5.10% failure rate against HDFC’s sector-best 4.82%
- Found Education payments averaging ₹5,094, 4× platform transaction average
- Identified SBI controlling 25% of transaction volume across the platform
- Detected peak UPI throughput at 19:00 with 21,232 transactions/hour

---

## SQL Business Analysis

Conducted 20+ PostgreSQL business investigations across:

- Platform scale and transaction KPIs
- Monthly and hourly transaction trends
- Banking reliability benchmarking
- Fraud concentration analysis
- Merchant-category segmentation
- Demographic and behavioral analysis
- State-level transaction dynamics
- Interbank payment corridor analysis

---

## Fraud Detection Model

- Built XGBoost fraud classification model on highly imbalanced transaction data
- Applied SMOTE oversampling for minority-class balancing
- Tuned decision threshold for improved fraud recall
- Evaluated feature importance and fraud prediction behavior

---

## Dashboard Overview

Developed an interactive Streamlit dashboard featuring:

- Platform KPI monitoring
- Fraud and geographic risk analysis
- Bank reliability benchmarking
- Transaction trend analysis
- Demographic segmentation
- Merchant-category insights
- Sankey payment-flow visualization
- Interactive filters across states, banks, and transaction types

---

## Dashboard Preview

### Platform KPIs
![Platform KPIs](visuals/1_platform_kpis.png)

### Monthly Transaction Trends
![Monthly Trends](visuals/2_monthly_trend.png)

### Fraud Analysis
![Fraud Analysis](visuals/7_fraud_deep_dive.png)

### ML Feature Importance
![Feature Importance](visuals/10_smote_feature_importance.png)

---

## Results

| Metric & Value |
| Transactions Analyzed: 250,000 |
| Total Transaction Value: ₹32.79 Crore |
| Platform Success Rate: 95.05% |
| Fraud Rate: 0.192% |
| States Covered: 10 |
| Banks Analyzed: 8 |

---

## Repository Structure

```bash
upi-transaction-analytics-and-fraud-intelligence/
│
├── 01_data/
├── 02_upi_transaction_analysis.sql
├── 03_ml/
├── 04_visuals/
├── 05_streamlit_app.py
├── report.md
├── UPI_Project_Presentation.pptx
└── README.md
```

---

## Author

**Sukanya Mukherjee**
