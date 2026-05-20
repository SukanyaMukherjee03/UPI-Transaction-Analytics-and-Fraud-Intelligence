"""
UPI PULSE 2024 — Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Pulse 2024",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F7F8FA; color: #1A2A42; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0B2545;
        border-right: none;
    }
    [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] strong { color: #FFFFFF !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: #E2E8F0 !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E7EF;
        border-left: 4px solid #0B2545;
        border-radius: 0 8px 8px 0;
        padding: 15px;
        box-shadow: 0 1px 4px rgba(11,37,69,0.07);
    }
    [data-testid="metric-container"] label {
        color: #64748B !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0B2545 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #15803D !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #E2E7EF;
        border-radius: 8px 8px 0 0;
        padding: 0 8px;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748B;
        background-color: transparent;
        border-bottom: 3px solid transparent;
        border-radius: 0;
        margin-bottom: -2px;
        padding: 10px 18px;
        font-size: 0.83rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #0B2545 !important;
        border-bottom: 3px solid #0B2545 !important;
        font-weight: 600 !important;
    }

    /* Headers */
    h1, h2, h3 { color: #0B2545 !important; }

    /* Divider */
    hr { border-color: #E2E7EF; }

    /* Insight boxes */
    .insight {
        background-color: #FFFBEB;
        border-left: 4px solid #D97706;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 10px 0;
        color: #78350F;
        font-size: 0.88rem;
        line-height: 1.65;
    }
    .fraud-box {
        background-color: #FEF2F2;
        border-left: 4px solid #B91C1C;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 10px 0;
        color: #7F1D1D;
        font-size: 0.88rem;
        line-height: 1.65;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Try both possible paths
    for path in ['data/upi_clean.csv', 'upi_clean.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # normalize amount column name
            if 'amount_inr' in df.columns and 'amount' not in df.columns:
                df['amount'] = df['amount_inr']
            # safety columns
            if 'is_success' not in df.columns:
                df['is_success'] = (df['transaction_status'] == 'SUCCESS').astype(int)
            if 'is_same_bank' not in df.columns:
                df['is_same_bank'] = (df['sender_bank'] == df['receiver_bank']).astype(int)
            return df
    st.error("❌ Could not find upi_clean.csv. Run data/01_clean_data.py first.")
    st.stop()

df = load_data()

# Month order for sorting
month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
day_order   = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
age_order   = ['18-25','26-35','36-45','46-55','56+']

# Plotly light theme base
TEMPLATE   = 'plotly_white'
C_ACCENT   = '#0B2545'
C_GREEN    = '#15803D'
C_RED      = '#B91C1C'
C_ORANGE   = '#D97706'
C_PURPLE   = '#6B21A8'
C_BG       = '#F7F8FA'
C_CARD     = '#FFFFFF'

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("---")

all_months = month_order
sel_months = st.sidebar.multiselect(
    "📅 Month",
    options=all_months,
    default=all_months
)

sel_state = st.sidebar.selectbox(
    "🗺️ State",
    ['All States'] + sorted(df['sender_state'].unique().tolist())
)

sel_bank = st.sidebar.selectbox(
    "🏦 Sender Bank",
    ['All Banks'] + sorted(df['sender_bank'].unique().tolist())
)

sel_type = st.sidebar.selectbox(
    "💳 Transaction Type",
    ['All Types'] + sorted(df['transaction_type'].unique().tolist())
)

sel_age = st.sidebar.selectbox(
    "👤 Age Group",
    ['All Ages'] + age_order
)

st.sidebar.markdown("---")
st.sidebar.markdown("**About this Dashboard**")
st.sidebar.markdown(
    "Analyzing 250,000 UPI transactions "
    "across India in 2024. Built with "
    "Python · PostgreSQL · XGBoost · Streamlit"
)

# ── Apply filters ──────────────────────────────────────────────
dff = df[df['month'].isin(sel_months)] if sel_months else df.copy()
if sel_state != 'All States': dff = dff[dff['sender_state']  == sel_state]
if sel_bank  != 'All Banks':  dff = dff[dff['sender_bank']   == sel_bank]
if sel_type  != 'All Types':  dff = dff[dff['transaction_type'] == sel_type]
if sel_age   != 'All Ages':   dff = dff[dff['sender_age_group'] == sel_age]

st.sidebar.markdown(f"**Showing:** `{len(dff):,}` transactions")

# ── Header ─────────────────────────────────────────────────────
st.markdown("# 💳 UPI Pulse 2024")
st.markdown(
    "India's Digital Payment Intelligence Dashboard &nbsp;·&nbsp; "
    "250,000 Transactions &nbsp;·&nbsp; "
    "Python · PostgreSQL · XGBoost · Streamlit",
    unsafe_allow_html=True
)
st.markdown("---")

# ── KPI Row ────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)

total_val  = dff['amount'].sum() / 1e7
avg_val    = dff['amount'].mean()
success_rt = dff['is_success'].mean() * 100
fraud_rt   = dff['fraud_flag'].mean() * 100
fraud_cnt  = int(dff['fraud_flag'].sum())

k1.metric("💳 Transactions",   f"{len(dff):,}")
k2.metric("💰 Total Value",    f"₹{total_val:.2f} Cr")
k3.metric("📊 Avg Transaction",f"₹{avg_val:,.0f}")
k4.metric("✅ Success Rate",   f"{success_rt:.2f}%")
k5.metric("🚨 Fraud Cases",    f"{fraud_cnt}")
k6.metric("⚠️ Fraud Rate",    f"{fraud_rt:.3f}%")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Trends",
    "🗺️ Geography",
    "🏦 Banks",
    "🚨 Fraud Analysis",
    "🔬 Deep Dive",
    "🤖 ML Model"
])

# ══ TAB 1: TRENDS ═════════════════════════════════════════════
with tab1:
    st.subheader("Transaction Trends Over Time")

    col1, col2 = st.columns(2)

    with col1:
        # Monthly dual axis
        if 'month' in dff.columns:
            monthly = (dff.groupby('month')
                         .agg(count=('fraud_flag','count'),
                              value=('amount','sum'),
                              fraud=('fraud_flag','sum'))
                         .reindex(month_order).dropna(subset=['count'])
                         .reset_index())
            monthly['value_lakhs'] = monthly['value'] / 1e5

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=monthly['month'], y=monthly['count'],
                name='Transactions', marker_color=C_ACCENT,
                opacity=0.85, marker_line_width=0), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=monthly['month'], y=monthly['value_lakhs'],
                name='Value (₹L)', mode='lines+markers',
                line=dict(color=C_ORANGE, width=2.5),
                marker=dict(size=7, color=C_ORANGE,
                            line=dict(color='white', width=1.5))), secondary_y=True)
            fig.update_layout(
                title='Monthly Volume & Value',
                template=TEMPLATE,
                plot_bgcolor=C_CARD, paper_bgcolor=C_CARD,
                height=380, hovermode='x unified',
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                font=dict(color='#64748B', size=11),
                xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
                yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
            fig.update_yaxes(title_text="Transactions", secondary_y=False,
                             title_font=dict(color='#64748B', size=10))
            fig.update_yaxes(title_text="Value (₹ Lakhs)", secondary_y=True,
                             title_font=dict(color='#64748B', size=10), showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Hourly bar
        if 'hour_of_day' in dff.columns:
            hourly = (dff.groupby('hour_of_day')
                        .agg(count=('fraud_flag','count'),
                             avg=('amount','mean'))
                        .reset_index())
            fig2 = px.bar(
                hourly, x='hour_of_day', y='count',
                color='avg',
                color_continuous_scale=[[0,'#DDEAFF'],[0.5,'#3B6CA8'],[1.0,'#0B2545']],
                title='Transactions by Hour of Day',
                labels={'hour_of_day':'Hour','count':'Transactions','avg':'Avg ₹'},
                template=TEMPLATE)
            fig2.update_traces(marker_line_width=0)
            fig2.update_layout(
                plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=380,
                font=dict(color='#64748B', size=11),
                xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
                yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Transaction type donut
        txn = dff['transaction_type'].value_counts().reset_index()
        txn.columns = ['type','count']
        fig3 = px.pie(
            txn, names='type', values='count',
            title='Transaction Type Split',
            hole=0.45,
            color_discrete_sequence=[C_ACCENT,'#3B6CA8',C_ORANGE,C_PURPLE],
            template=TEMPLATE)
        fig3.update_traces(
            textfont=dict(color='white', size=11),
            marker=dict(line=dict(color='white', width=2.5)))
        fig3.update_layout(
            plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=360,
            font=dict(color='#64748B', size=11))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Day of week bar
        if 'day_of_week' in dff.columns:
            day_data = (dff.groupby('day_of_week')
                          .size().reset_index(name='count'))
            day_data['day_of_week'] = pd.Categorical(
                day_data['day_of_week'],
                categories=day_order, ordered=True)
            day_data = day_data.sort_values('day_of_week')
            fig4 = px.bar(
                day_data, x='day_of_week', y='count',
                title='Transactions by Day of Week',
                color='count',
                color_continuous_scale=[[0,'#DDEAFF'],[1.0,'#0B2545']],
                template=TEMPLATE)
            fig4.update_traces(marker_line_width=0)
            fig4.update_layout(
                plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=360,
                font=dict(color='#64748B', size=11),
                xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
                yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        '<div class="insight">💡 <b>Insight:</b> '
        'Peak transaction hours are 10 AM–12 PM and 7 PM–9 PM. '
        'Monday is the busiest day. July leads monthly volume. '
        'P2P dominates at 44.98% of all transactions.</div>',
        unsafe_allow_html=True)

# ══ TAB 2: GEOGRAPHY ══════════════════════════════════════════
with tab2:
    st.subheader("State-wise Payment Analysis")

    state_data = (dff.groupby('sender_state')
                    .agg(count=('fraud_flag','count'),
                         value=('amount','sum'),
                         avg=('amount','mean'),
                         fraud=('fraud_flag','sum'))
                    .reset_index())
    state_data['value_lakhs'] = state_data['value'] / 1e5
    state_data['fraud_rate']  = state_data['fraud'] / state_data['count'] * 100

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            state_data.sort_values('value_lakhs', ascending=True),
            x='value_lakhs', y='sender_state',
            orientation='h',
            color='fraud_rate',
            color_continuous_scale=[[0,'#DCFCE7'],[0.5,'#FEF3C7'],[1.0,'#FEE2E2']],
            title='State-wise Value (colour = fraud rate)',
            labels={'value_lakhs':'Value (₹L)',
                    'sender_state':'State',
                    'fraud_rate':'Fraud Rate %'},
            template=TEMPLATE)
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=420,
            font=dict(color='#64748B', size=11),
            xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
            yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(
            state_data,
            x='avg', y='count',
            size='value_lakhs', color='fraud_rate',
            hover_name='sender_state',
            color_continuous_scale=[[0,'#15803D'],[0.5,'#D97706'],[1.0,'#B91C1C']],
            title='State Risk Matrix (bubble = total value)',
            labels={'avg':'Avg Transaction ₹',
                    'count':'No. Transactions',
                    'fraud_rate':'Fraud Rate %'},
            template=TEMPLATE)
        fig2.update_traces(marker=dict(line=dict(color='white', width=1)))
        fig2.update_layout(
            plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=420,
            font=dict(color='#64748B', size=11),
            xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
            yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
        st.plotly_chart(fig2, use_container_width=True)

    # Sunburst
    cat_state = (dff.groupby(['sender_state','merchant_category'])
                   .size().reset_index(name='count'))
    fig3 = px.sunburst(
        cat_state,
        path=['sender_state','merchant_category'],
        values='count',
        title='State → Merchant Category Breakdown',
        color_discrete_sequence=['#0B2545','#3B6CA8','#5B8DB8','#8AAFD0',
                                  '#D97706','#15803D','#B91C1C'],
        template=TEMPLATE)
    fig3.update_layout(
        plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=500,
        font=dict(color='#64748B', size=11))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        '<div class="insight">💡 <b>Insight:</b> '
        'Maharashtra leads with ₹490L total value. '
        'Karnataka has the highest fraud rate (0.232%). '
        'Tamil Nadu users have the lowest fraud rate (0.158%).</div>',
        unsafe_allow_html=True)

# ══ TAB 3: BANKS ══════════════════════════════════════════════
with tab3:
    st.subheader("Bank Performance Analysis")

    bank_data = (dff.groupby('sender_bank')
                   .agg(count=('fraud_flag','count'),
                        value=('amount','sum'),
                        failed=('is_success', lambda x: (x==0).sum()),
                        fraud=('fraud_flag','sum'))
                   .reset_index())
    bank_data['failure_rate'] = bank_data['failed'] / bank_data['count'] * 100
    bank_data['value_lakhs']  = bank_data['value'] / 1e5
    bank_data['market_share'] = bank_data['count'] / len(dff) * 100

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            bank_data.sort_values('count', ascending=False),
            x='sender_bank', y='count',
            color='failure_rate',
            color_continuous_scale=[[0,'#DCFCE7'],[0.5,'#FEF3C7'],[1.0,'#FEE2E2']],
            title='Bank Market Share vs Failure Rate',
            labels={'count':'Transactions',
                    'failure_rate':'Failure Rate %'},
            template=TEMPLATE)
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=380,
            font=dict(color='#64748B', size=11),
            xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
            yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sankey bank to bank flow
        flow = (dff.groupby(['sender_bank','receiver_bank'])
                  .size().reset_index(name='count')
                  .sort_values('count', ascending=False)
                  .head(20))
        all_banks = list(set(
            flow['sender_bank'].tolist() + flow['receiver_bank'].tolist()))
        bank_idx = {b: i for i, b in enumerate(all_banks)}

        fig2 = go.Figure(data=[go.Sankey(
            node=dict(
                label=all_banks,
                color=C_ACCENT,
                pad=15, thickness=20,
                line=dict(color='white', width=0.5)),
            link=dict(
                source=[bank_idx[b] for b in flow['sender_bank']],
                target=[bank_idx[b] for b in flow['receiver_bank']],
                value=flow['count'].tolist(),
                color='rgba(11,37,69,0.12)')
        )])
        fig2.update_layout(
            title='Bank-to-Bank Flow (Sankey)',
            template=TEMPLATE,
            plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=380,
            font=dict(color='#64748B', size=11))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.bar(
            bank_data.sort_values('failure_rate', ascending=False),
            x='sender_bank', y='failure_rate',
            color='failure_rate',
            color_continuous_scale=[[0,'#FEF2F2'],[0.5,'#EF9595'],[1.0,'#B91C1C']],
            title='Failure Rate by Bank (%)',
            labels={'failure_rate':'Failure Rate %'},
            template=TEMPLATE)
        fig3.update_traces(marker_line_width=0)
        fig3.update_layout(
            plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=360,
            font=dict(color='#64748B', size=11),
            xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
            yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Device type by bank
        if 'device_type' in dff.columns:
            dev_bank = (dff.groupby(['sender_bank','device_type'])
                          .size().reset_index(name='count'))
            fig4 = px.bar(
                dev_bank,
                x='sender_bank', y='count',
                color='device_type', barmode='stack',
                title='Device Type Distribution by Bank',
                color_discrete_sequence=[C_ACCENT,'#3B6CA8','#8AAFD0'],
                template=TEMPLATE)
            fig4.update_traces(marker_line_width=0)
            fig4.update_layout(
                plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=360,
                font=dict(color='#64748B', size=11),
                xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
                yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        '<div class="insight">💡 <b>Insight:</b> '
        'SBI leads with 25.08% market share. '
        'Yes Bank has the highest failure rate (5.10%). '
        'HDFC is most reliable (4.82%). '
        'SBI→SBI is the biggest single payment corridor (15,635 txns).</div>',
        unsafe_allow_html=True)

# ══ TAB 4: FRAUD ══════════════════════════════════════════════
with tab4:
    st.subheader("🚨 Fraud Detection & Risk Analysis")
    st.markdown(
        '<div class="fraud-box">⚠️ <b>Note:</b> '
        '480 fraud cases out of 250,000 transactions (0.192%). '
        'ML model uses XGBoost + SMOTE to handle extreme class imbalance. '
        'Threshold set to 0.3 to prioritise catching fraud over false alarms.</div>',
        unsafe_allow_html=True)

    fraud_df = dff[dff['fraud_flag'] == 1]

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Total Fraud Cases",  f"{len(fraud_df)}")
    f2.metric("Fraud Rate",         f"{len(fraud_df)/max(len(dff),1)*100:.3f}%")
    f3.metric("Avg Fraud Amount",
              f"₹{fraud_df['amount'].mean():.0f}" if len(fraud_df) > 0 else "N/A")
    f4.metric("States Affected",    f"{fraud_df['sender_state'].nunique()}")

    col1, col2 = st.columns(2)

    with col1:
        fraud_state = (fraud_df.groupby('sender_state')
                               .size().reset_index(name='count'))
        fig = px.bar(
            fraud_state.sort_values('count', ascending=False),
            x='sender_state', y='count',
            color='count',
            color_continuous_scale=[[0,'#FEF2F2'],[0.5,'#EF9595'],[1.0,'#B91C1C']],
            title='Fraud Cases by State',
            template=TEMPLATE)
        fig.update_traces(marker_line_width=0)
        fig.update_layout(
            plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=360,
            font=dict(color='#64748B', size=11),
            xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
            yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'hour_of_day' in dff.columns:
            fraud_hour = (fraud_df.groupby('hour_of_day')
                                  .size().reset_index(name='count'))
            fig2 = px.area(
                fraud_hour, x='hour_of_day', y='count',
                title='Fraud Cases by Hour of Day',
                color_discrete_sequence=[C_RED],
                template=TEMPLATE)
            fig2.update_traces(
                fillcolor='rgba(185,28,28,0.08)',
                line_color=C_RED, line_width=2)
            fig2.update_layout(
                plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=360,
                font=dict(color='#64748B', size=11),
                xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
                yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fraud_cat = (fraud_df.groupby('merchant_category')
                             .size().reset_index(name='count'))
        fig3 = px.pie(
            fraud_cat, names='merchant_category', values='count',
            title='Fraud by Merchant Category', hole=0.35,
            color_discrete_sequence=[C_RED,C_ORANGE,C_ACCENT,'#3B6CA8',C_GREEN],
            template=TEMPLATE)
        fig3.update_traces(
            textfont=dict(color='white', size=11),
            marker=dict(line=dict(color='white', width=2.5)))
        fig3.update_layout(
            plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=360,
            font=dict(color='#64748B', size=11))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fraud_age = (fraud_df.groupby('sender_age_group')
                             .size().reset_index(name='fraud'))
        all_age   = (dff.groupby('sender_age_group')
                       .size().reset_index(name='total'))
        merged = fraud_age.merge(all_age, on='sender_age_group')
        merged['fraud_rate'] = merged['fraud'] / merged['total'] * 100
        merged['sender_age_group'] = pd.Categorical(
            merged['sender_age_group'],
            categories=age_order, ordered=True)
        merged = merged.sort_values('sender_age_group')

        fig4 = px.bar(
            merged, x='sender_age_group', y='fraud_rate',
            color='fraud_rate',
            color_continuous_scale=[[0,'#FEF2F2'],[0.5,'#FEF3C7'],[1.0,'#B91C1C']],
            title='Fraud Rate by Age Group (%)',
            labels={'fraud_rate':'Fraud Rate %'},
            template=TEMPLATE)
        fig4.update_traces(marker_line_width=0)
        fig4.update_layout(
            plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=360,
            font=dict(color='#64748B', size=11),
            xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
            yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
        st.plotly_chart(fig4, use_container_width=True)

    # Fraud heatmap device x network
    if 'device_type' in dff.columns and 'network_type' in dff.columns:
        fd_dev = (dff.groupby(['device_type','network_type'])
                    .agg(fraud=('fraud_flag','sum'),
                         count=('fraud_flag','count'))
                    .reset_index())
        fd_dev['fraud_rate'] = fd_dev['fraud'] / fd_dev['count'] * 100
        fig5 = px.density_heatmap(
            fd_dev,
            x='network_type', y='device_type',
            z='fraud_rate',
            color_continuous_scale=[[0,'#FEF9F0'],[0.5,'#FEF3C7'],[1.0,'#B91C1C']],
            title='Fraud Rate: Device × Network (%)',
            template=TEMPLATE)
        fig5.update_layout(
            plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=300,
            font=dict(color='#64748B', size=11))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        '<div class="fraud-box">🔍 <b>Key Finding:</b> '
        'Karnataka has highest fraud rate (0.232%). '
        '18-25 age group is most vulnerable. '
        'Web + 3G is the riskiest device-network combination (6.6% failure). '
        'High-value transactions (>₹10K) carry higher fraud risk.</div>',
        unsafe_allow_html=True)

# ══ TAB 5: DEEP DIVE ══════════════════════════════════════════
with tab5:
    st.subheader("🔬 Segmentation & Behavioral Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Age x Category heatmap
        age_cat = (dff.groupby(['sender_age_group','merchant_category'])
                     ['amount'].mean().reset_index())
        pivot = (age_cat.pivot(
                    index='sender_age_group',
                    columns='merchant_category',
                    values='amount')
                 .reindex(age_order))

        fig = px.imshow(
            pivot,
            color_continuous_scale=[[0,'#E8EEF7'],[0.5,'#3B6CA8'],[1.0,'#0B2545']],
            title='Avg Spend Heatmap: Age Group × Merchant Category',
            labels=dict(color='Avg ₹'),
            template=TEMPLATE)
        fig.update_layout(
            plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=420,
            font=dict(color='#64748B', size=11))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Amount bucket funnel
        if 'amount_bucket' in dff.columns:
            bucket = (dff['amount_bucket']
                        .value_counts()
                        .reset_index())
            bucket.columns = ['bucket','count']
            fig2 = px.funnel(
                bucket, x='count', y='bucket',
                title='Transaction Value Funnel',
                color_discrete_sequence=[C_ACCENT],
                template=TEMPLATE)
            fig2.update_layout(
                plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=420,
                font=dict(color='#64748B', size=11))
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Weekend vs Weekday
        if 'is_weekend' in dff.columns:
            wknd = (dff.groupby('is_weekend')
                      .agg(count=('fraud_flag','count'),
                           avg=('amount','mean'),
                           fraud=('fraud_flag','mean'))
                      .reset_index())
            wknd['label']      = wknd['is_weekend'].map({0:'Weekday',1:'Weekend'})
            wknd['fraud_rate'] = wknd['fraud'] * 100

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                name='Transactions',
                x=wknd['label'], y=wknd['count'],
                marker_color=C_ACCENT, opacity=0.85, marker_line_width=0))
            fig3.add_trace(go.Scatter(
                name='Avg Amount ₹',
                x=wknd['label'], y=wknd['avg'],
                mode='markers+text',
                marker=dict(color=C_ORANGE, size=20,
                            line=dict(color='white', width=2)),
                text=[f"₹{v:.0f}" for v in wknd['avg']],
                textposition='top center',
                textfont=dict(color='#1A2A42', size=11),
                yaxis='y2'))
            fig3.update_layout(
                title='Weekday vs Weekend',
                template=TEMPLATE,
                plot_bgcolor=C_CARD, paper_bgcolor=C_CARD,
                height=360,
                font=dict(color='#64748B', size=11),
                xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
                yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
                yaxis2=dict(overlaying='y', side='right',
                            tickfont=dict(color='#94A3B8', size=10),
                            showgrid=False))
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Network type analysis
        if 'network_type' in dff.columns:
            net = (dff.groupby('network_type')
                     .agg(count=('fraud_flag','count'),
                          avg=('amount','mean'),
                          fraud=('fraud_flag','mean'))
                     .reset_index())
            net['fraud_rate'] = net['fraud'] * 100
            fig4 = px.scatter(
                net,
                x='count', y='avg',
                size='fraud_rate', color='network_type',
                hover_name='network_type',
                title='Network Type: Volume vs Avg Amount\n(size = fraud rate)',
                labels={'count':'Transactions','avg':'Avg Amount ₹'},
                color_discrete_sequence=[C_ACCENT,'#3B6CA8',C_ORANGE,C_GREEN],
                template=TEMPLATE)
            fig4.update_traces(
                marker=dict(line=dict(color='white', width=1.5)))
            fig4.update_layout(
                plot_bgcolor=C_CARD, paper_bgcolor=C_CARD, height=360,
                font=dict(color='#64748B', size=11),
                xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
                yaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)))
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        '<div class="insight">💡 <b>Deep Insight:</b> '
        '36-45 spends most on average (₹1,424). '
        'Education has highest avg ticket size (₹5,094). '
        'Weekday fraud rate (0.189%) is slightly lower than weekends (0.200%). '
        'Grocery is the most transacted category (49,966 txns).</div>',
        unsafe_allow_html=True)

# ══ TAB 6: ML MODEL ═══════════════════════════════════════════
with tab6:
    st.subheader("🤖 XGBoost Fraud Detection — Model Results")

    st.markdown(
        '<div class="fraud-box">'
        '<b>Model:</b> XGBoost + SMOTE &nbsp;|&nbsp; '
        '<b>Threshold:</b> 0.30 &nbsp;|&nbsp; '
        '<b>Class Imbalance Fix:</b> SMOTE (1:1 synthetic oversampling) &nbsp;|&nbsp; '
        '<b>Features:</b> 17'
        '</div>',
        unsafe_allow_html=True)

    # Load feature importance if it exists
    fi_path = None
    for p in ['ml/feature_importance.csv', 'feature_importance.csv']:
        if os.path.exists(p):
            fi_path = p
            break

    if fi_path:
        feat_imp = pd.read_csv(fi_path)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                feat_imp.head(15),
                x='Importance', y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale=[[0,'#E8EEF7'],[0.5,'#3B6CA8'],[1.0,'#0B2545']],
                title='Top 15 Feature Importances',
                template=TEMPLATE)
            fig.update_traces(marker_line_width=0)
            fig.update_layout(
                plot_bgcolor=C_CARD, paper_bgcolor=C_CARD,
                height=500,
                font=dict(color='#64748B', size=11),
                xaxis=dict(gridcolor='#EDF0F4', tickfont=dict(color='#94A3B8', size=10)),
                yaxis=dict(autorange='reversed',
                           tickfont=dict(color='#64748B', size=10)))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Show model metrics
            st.markdown("### Model Performance Summary")
            st.markdown("---")

            metrics_data = {
                'Metric': [
                    'Algorithm',
                    'Class Balancing',
                    'Train Samples (after SMOTE)',
                    'Test Samples',
                    'Decision Threshold',
                    'Top Predictor',
                    'Fraud in Dataset',
                    'Fraud Rate'
                ],
                'Value': [
                    'XGBoost (n=200, depth=6)',
                    'SMOTE (k_neighbors=5)',
                    '~399,232 (balanced)',
                    '50,000',
                    '0.30',
                    feat_imp.iloc[0]['Feature'],
                    '480 cases',
                    '0.192%'
                ]
            }
            st.dataframe(
                pd.DataFrame(metrics_data),
                use_container_width=True,
                hide_index=True
            )

            st.markdown("### Why SMOTE?")
            st.markdown(
                "Without SMOTE, the model would predict **everything as legit** "
                "and claim 99.8% accuracy — completely useless. SMOTE creates "
                "synthetic fraud examples so the model actually **learns fraud patterns**."
            )

            st.markdown("### Why threshold = 0.30?")
            st.markdown(
                "Default threshold is 0.50. We lowered it to 0.30 because "
                "**missing real fraud is much worse** than a false alarm. "
                "A bank would rather flag a legitimate transaction for review "
                "than let actual fraud through undetected."
            )

        # Show saved chart if it exists
        chart_path = None
        for p in ['visuals/fraud_model_results.png',
                  'visuals/08_ml_model_results.png']:
            if os.path.exists(p):
                chart_path = p
                break

        if chart_path:
            st.markdown("### Model Charts")
            st.image(chart_path, use_column_width=True)
        else:
            st.info(
                "Run `python ml/03_fraud_model.py` first to generate model charts, "
                "then refresh this page."
            )
    else:
        st.warning(
            "Feature importance file not found. "
            "Run `python ml/03_fraud_model.py` first, then refresh."
        )

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94A3B8; font-size:0.85rem'>"
    "UPI Pulse 2024 &nbsp;·&nbsp; "
    "Built with Python, PostgreSQL, XGBoost & Streamlit &nbsp;·&nbsp; "
    "250,000 Transactions Analyzed &nbsp;·&nbsp; "
    "Portfolio Project"
    "</div>",
    unsafe_allow_html=True
)