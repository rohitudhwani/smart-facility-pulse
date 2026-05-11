"""
Smart Facility Pulse — Main App Entry
Page 1: Building Overview (Executive landing page)
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

from utils.data_loader import (
    load_comfort_sensors, load_zones_master, load_outdoor_weather,
    load_pool_sensors, load_pools_master, load_scored_test_sample
)
from utils.alert_engine import (
    classify_comfort, get_comfort_issues,
    classify_pool, get_pool_issues
)
from utils.theme import (
    get_global_css, PLOTLY_TEMPLATE,
    ACCENT_PRIMARY, STATUS_HEALTHY, STATUS_WATCH, STATUS_ALERT,
    BG_SECONDARY, BORDER, TEXT_PRIMARY, TEXT_SECONDARY, VIZ_TEAL,
)


# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Smart Facility Pulse — Sobha Pilot",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS
from utils.sidebar import render_sidebar_alerts, compute_active_alerts  # noqa: E402
st.markdown(get_global_css(), unsafe_allow_html=True)
render_sidebar_alerts()


# ============================================================
# LOAD ALL DATA (cached after first call)
# ============================================================
comfort_df = load_comfort_sensors()
zones_df = load_zones_master()
weather_df = load_outdoor_weather()
pool_df = load_pool_sensors()
pools_master = load_pools_master()
scored_fcus = load_scored_test_sample()


# ============================================================
# COMPUTE CURRENT BUILDING STATE
# ============================================================
# Latest reading per zone
latest_comfort = comfort_df.sort_values('timestamp').groupby('zone_id').tail(1).reset_index(drop=True)
latest_comfort['status'] = latest_comfort.apply(classify_comfort, axis=1)

# Latest reading per pool
latest_pool = pool_df.sort_values('timestamp').groupby('pool_id').tail(1).reset_index(drop=True)
latest_pool['status'] = latest_pool.apply(classify_pool, axis=1)

# Latest scored FCUs (one per fault scenario — pick most recent reading per FCU)
latest_fcus = scored_fcus.sort_values('Datetime').groupby(['fault_type', 'severity']).tail(1).reset_index(drop=True)


# ============================================================
# MAIN PAGE — BRAND BANNER
# ============================================================
st.markdown(
    """
    <div class='brand-banner'>
        <h1>Smart Facility Pulse</h1>
        <p>Sobha — Pilot FM Building · 20 Floors · 80 Zones · 1 Gymnasium · 2 Pools</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI CARDS (6 across)
# ============================================================
n_zones = len(latest_comfort)
n_alert = (latest_comfort['status'] == 'Alert').sum()
n_watch = (latest_comfort['status'] == 'Watch').sum()
n_normal = (latest_comfort['status'] == 'Normal').sum()
comfort_pct = (n_normal / n_zones * 100) if n_zones else 0

n_pools = len(latest_pool)
n_pool_alerts = (latest_pool['status'] != 'Normal').sum()

n_fcus = len(latest_fcus)
n_high_risk = (latest_fcus['fault_proba'] > 0.85).sum()

# Composite health score (weighted)
# Comfort % weighted 40%, pool health 30%, HVAC health 30%
pool_health = ((n_pools - n_pool_alerts) / n_pools * 100) if n_pools else 100
hvac_health = ((n_fcus - n_high_risk) / n_fcus * 100) if n_fcus else 100
health_score = int(0.4 * comfort_pct + 0.3 * pool_health + 0.3 * hvac_health)

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("Health Score", f"{health_score}/100",
              delta=f"{health_score - 90:+d} vs target", delta_color="normal")
with col2:
    st.metric("Comfortable Zones", f"{comfort_pct:.0f}%",
              delta=f"{n_normal}/{n_zones} zones")
with col3:
    st.metric("Zones in Alert", n_alert,
              delta=f"{n_watch} on watch", delta_color="inverse")
with col4:
    st.metric("Pool Status", f"{n_pools - n_pool_alerts}/{n_pools} OK",
              delta="action needed" if n_pool_alerts else "all clear",
              delta_color="inverse" if n_pool_alerts else "normal")
with col5:
    st.metric("HVAC at Risk", n_high_risk,
              delta=f"of {n_fcus} units", delta_color="inverse")
with col6:
    _alerts_for_count, _ = compute_active_alerts()
    st.metric("Active Alerts", len(_alerts_for_count),
              delta="live", delta_color="off")


# ============================================================
# OUTDOOR vs INDOOR TEMPERATURE CHART (last 7 days)
# ============================================================
st.markdown("### Climate Context — Outdoor vs Building Average (Last 7 Days)")

last_7d_cutoff = comfort_df['timestamp'].max() - timedelta(days=7)
recent_comfort = comfort_df[comfort_df['timestamp'] >= last_7d_cutoff]
indoor_avg = recent_comfort.groupby('timestamp')['indoor_temp'].mean().reset_index()
recent_weather = weather_df[weather_df['timestamp'] >= last_7d_cutoff]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recent_weather['timestamp'], y=recent_weather['outdoor_temp'],
    name='Outdoor', mode='lines',
    line=dict(color=ACCENT_PRIMARY, width=2),
    hovertemplate='%{x|%b %d %H:%M}<br>Outdoor: %{y:.1f}°C<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=indoor_avg['timestamp'], y=indoor_avg['indoor_temp'],
    name='Indoor (avg)', mode='lines',
    line=dict(color=VIZ_TEAL, width=2),
    hovertemplate='%{x|%b %d %H:%M}<br>Indoor avg: %{y:.1f}°C<extra></extra>'
))
fig.update_layout(
    template=PLOTLY_TEMPLATE,
    height=320,
    margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(orientation='h', y=1.05, x=0),
    yaxis_title='°C',
    xaxis_title=None,
)
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# ACTIVE ALERTS TABLE (top 10 across all modules)
# ============================================================
st.markdown("### Active Alerts — Recommended Actions")

alert_rows = []

# Comfort
for _, row in latest_comfort[latest_comfort['status'].isin(['Alert', 'Watch'])].iterrows():
    issues = get_comfort_issues(row)
    if issues:
        alert_rows.append({
            'Severity': row['status'],
            'Module': 'Comfort',
            'Asset': row['zone_id'],
            'Issue': issues[0],
            'Action': 'Inspect zone HVAC + ventilation' if row['status'] == 'Alert' else 'Monitor trend',
        })

# Pool
for _, row in latest_pool[latest_pool['status'].isin(['Alert', 'Watch'])].iterrows():
    issues = get_pool_issues(row)
    if issues:
        alert_rows.append({
            'Severity': row['status'],
            'Module': 'Pool',
            'Asset': row['pool_name'],
            'Issue': '; '.join(issues[:2]),
            'Action': 'Service pool — adjust chemistry' if row['status'] == 'Alert' else 'Schedule check',
        })

# HVAC
for _, row in latest_fcus[latest_fcus['fault_proba'] > 0.6].iterrows():
    severity = 'Alert' if row['fault_proba'] > 0.85 else 'Watch'
    alert_rows.append({
        'Severity': severity,
        'Module': 'HVAC',
        'Asset': f"{row['fault_type']} ({row['severity']})",
        'Issue': f"Fault probability {row['fault_proba']*100:.0f}%",
        'Action': 'Dispatch technician' if severity == 'Alert' else 'Schedule inspection',
    })

if alert_rows:
    alerts_table = pd.DataFrame(alert_rows)
    # Sort: Alert first, then Watch
    alerts_table['_sort'] = alerts_table['Severity'].map({'Alert': 0, 'Watch': 1})
    alerts_table = alerts_table.sort_values(['_sort', 'Module']).drop(columns='_sort')
    st.dataframe(
        alerts_table.head(15),
        use_container_width=True, hide_index=True,
        column_config={
            'Severity': st.column_config.TextColumn(width='small'),
            'Module': st.column_config.TextColumn(width='small'),
            'Asset': st.column_config.TextColumn(width='medium'),
        }
    )
else:
    st.success("No active alerts. All systems within normal parameters.")


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    f"<small style='color:{TEXT_SECONDARY}'>"
    f"Smart Facility Pulse · Pilot proof-of-concept for Latinem Facilities Management. "
    f"Comfort & pool data: synthetic (calibrated). HVAC fault model: trained on LBNL public FCU dataset."
    f"</small>",
    unsafe_allow_html=True
)
