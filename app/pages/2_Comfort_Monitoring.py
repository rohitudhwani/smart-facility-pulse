"""
Page 2: Comfort & IAQ Monitoring (Module A)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

from utils.data_loader import (
    load_comfort_sensors, load_zones_master, load_outdoor_weather, load_anomaly_log
)
from utils.alert_engine import (
    classify_comfort, get_comfort_issues, COMFORT_THRESHOLDS, detect_stuck_sensor
)
from utils.theme import (
    get_global_css, PLOTLY_TEMPLATE,
    ACCENT_PRIMARY, STATUS_HEALTHY, STATUS_WATCH, STATUS_ALERT, STATUS_OFFLINE,
    BG_SECONDARY, BG_ELEVATED, BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
    VIZ_TEAL, VIZ_BRONZE, VIZ_PURPLE, VIZ_PINK,
    HEATMAP_NORMAL, HEATMAP_WATCH, HEATMAP_ALERT, HEATMAP_OFFLINE,
)

st.set_page_config(page_title="Comfort Monitoring", layout="wide")
from utils.sidebar import render_sidebar_alerts  # noqa: E402
st.markdown(get_global_css(), unsafe_allow_html=True)
render_sidebar_alerts()

comfort_df = load_comfort_sensors()
zones_df = load_zones_master()
weather_df = load_outdoor_weather()
anomaly_log = load_anomaly_log()

latest = comfort_df.sort_values('timestamp').groupby('zone_id').tail(1).reset_index(drop=True)
latest['status'] = latest.apply(classify_comfort, axis=1)

def is_stuck(zone_id):
    zone_data = comfort_df[comfort_df['zone_id'] == zone_id]
    return detect_stuck_sensor(zone_data, hours=24)

latest['stuck'] = latest['zone_id'].apply(is_stuck)
latest.loc[latest['stuck'], 'status'] = 'Offline'

st.markdown(
    """
    <div class='brand-banner'>
        <h1>Comfort & Indoor Air Quality Monitoring</h1>
        <p>Module A · 80 corridor zones across 20 floors + Gymnasium · Real-time status</p>
    </div>
    """,
    unsafe_allow_html=True
)

status_counts = latest['status'].value_counts()
n_normal = status_counts.get('Normal', 0)
n_watch = status_counts.get('Watch', 0)
n_alert = status_counts.get('Alert', 0)
n_offline = status_counts.get('Offline', 0)

c1, c2, c3, c4 = st.columns(4)
for col, label, count, color in [
    (c1, "Normal", n_normal, STATUS_HEALTHY),
    (c2, "Watch", n_watch, STATUS_WATCH),
    (c3, "Alert", n_alert, STATUS_ALERT),
    (c4, "Sensor Offline", n_offline, STATUS_OFFLINE),
]:
    with col:
        st.markdown(
            f"<div style='background:{BG_SECONDARY}; border-left:4px solid {color}; "
            f"border-radius:8px; padding:14px 18px;'>"
            f"<div style='color:{TEXT_SECONDARY}; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em;'>{label}</div>"
            f"<div style='color:{color}; font-size:2rem; font-weight:600;'>{count}</div>"
            f"</div>", unsafe_allow_html=True
        )

st.markdown("### Floor-Zone Status Map")
st.caption("Hover for details · Use the dropdown below to drill into a specific zone.")

corridor = latest[latest['zone_type'] == 'corridor'].copy()
gym = latest[latest['zone_type'] == 'gymnasium'].copy()

status_num_map = {'Normal': 0, 'Watch': 1, 'Alert': 2, 'Offline': 3}
corridor['status_num'] = corridor['status'].map(status_num_map)

pivot_status = corridor.pivot(index='floor', columns='quadrant', values='status_num').reindex(columns=['NW', 'NE', 'SW', 'SE']).sort_index(ascending=False)
pivot_temp = corridor.pivot(index='floor', columns='quadrant', values='indoor_temp').reindex(columns=['NW', 'NE', 'SW', 'SE']).sort_index(ascending=False)
pivot_co2 = corridor.pivot(index='floor', columns='quadrant', values='co2').reindex(columns=['NW', 'NE', 'SW', 'SE']).sort_index(ascending=False)
pivot_pm = corridor.pivot(index='floor', columns='quadrant', values='pm25').reindex(columns=['NW', 'NE', 'SW', 'SE']).sort_index(ascending=False)
pivot_zone = corridor.pivot(index='floor', columns='quadrant', values='zone_id').reindex(columns=['NW', 'NE', 'SW', 'SE']).sort_index(ascending=False)

hover_text = []
for f in pivot_status.index:
    row_text = []
    for q in pivot_status.columns:
        zid = pivot_zone.loc[f, q]
        t = pivot_temp.loc[f, q]
        c = pivot_co2.loc[f, q]
        p = pivot_pm.loc[f, q]
        s_num = pivot_status.loc[f, q]
        s_label = ['Normal', 'Watch', 'Alert', 'Offline'][int(s_num)] if pd.notna(s_num) else 'No data'
        row_text.append(
            f"<b>{zid}</b><br>Status: {s_label}<br>"
            f"Temp: {t:.1f}°C · CO₂: {c:.0f} ppm · PM2.5: {p:.1f}"
        )
    hover_text.append(row_text)

heatmap_colors = [
    [0.0, HEATMAP_NORMAL],   [0.249, HEATMAP_NORMAL],
    [0.25, HEATMAP_WATCH],   [0.499, HEATMAP_WATCH],
    [0.5, HEATMAP_ALERT],    [0.749, HEATMAP_ALERT],
    [0.75, HEATMAP_OFFLINE], [1.0, HEATMAP_OFFLINE],
]

fig_heat = go.Figure(data=go.Heatmap(
    z=pivot_status.values, x=pivot_status.columns,
    y=[f"Floor {f}" for f in pivot_status.index],
    colorscale=heatmap_colors, zmin=0, zmax=3, showscale=False,
    hoverinfo='text', text=hover_text, xgap=4, ygap=2,
))
fig_heat.update_layout(
    template=PLOTLY_TEMPLATE, height=620,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(side='top', tickfont=dict(size=12, color=TEXT_PRIMARY)),
    yaxis=dict(tickfont=dict(size=11, color=TEXT_SECONDARY)),
)

col_heat, col_gym = st.columns([3, 1])
with col_heat:
    st.plotly_chart(fig_heat, use_container_width=True)
with col_gym:
    st.markdown("**Legend**")
    legend_html = f"""
    <div style='font-size:0.85rem;'>
        <div style='margin-bottom:6px;'><span style='display:inline-block; width:14px; height:14px; background:{HEATMAP_NORMAL}; border-radius:3px; margin-right:8px; vertical-align:middle;'></span>Normal</div>
        <div style='margin-bottom:6px;'><span style='display:inline-block; width:14px; height:14px; background:{HEATMAP_WATCH}; border-radius:3px; margin-right:8px; vertical-align:middle;'></span>Watch</div>
        <div style='margin-bottom:6px;'><span style='display:inline-block; width:14px; height:14px; background:{HEATMAP_ALERT}; border-radius:3px; margin-right:8px; vertical-align:middle;'></span>Alert</div>
        <div style='margin-bottom:14px;'><span style='display:inline-block; width:14px; height:14px; background:{HEATMAP_OFFLINE}; border-radius:3px; margin-right:8px; vertical-align:middle;'></span>Offline</div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    st.markdown("**Gymnasium**")
    if len(gym) > 0:
        g = gym.iloc[0]
        gym_color = {'Normal': STATUS_HEALTHY, 'Watch': STATUS_WATCH,
                     'Alert': STATUS_ALERT, 'Offline': STATUS_OFFLINE}[g['status']]
        st.markdown(
            f"<div style='background:{BG_SECONDARY}; border-left:4px solid {gym_color}; "
            f"border-radius:8px; padding:12px 14px; margin-top:6px;'>"
            f"<div style='color:{TEXT_PRIMARY}; font-weight:600; font-size:0.95rem;'>{g['zone_id']}</div>"
            f"<div style='color:{gym_color}; font-size:0.85rem; margin-top:4px;'>{g['status']}</div>"
            f"<div style='color:{TEXT_SECONDARY}; font-size:0.75rem; margin-top:6px;'>"
            f"Temp {g['indoor_temp']:.1f}°C<br>CO₂ {g['co2']:.0f} ppm<br>PM2.5 {g['pm25']:.1f}"
            f"</div></div>", unsafe_allow_html=True
        )

st.markdown("### Zone Detail View")

status_priority = {'Alert': 0, 'Offline': 1, 'Watch': 2, 'Normal': 3}
latest['_priority'] = latest['status'].map(status_priority)
sorted_zones = latest.sort_values(['_priority', 'zone_id'])['zone_id'].tolist()
default_zone = next((z for z in sorted_zones if latest[latest['zone_id'] == z]['status'].iloc[0] == 'Alert'), sorted_zones[0])

col_zone, col_range = st.columns([2, 1])
with col_zone:
    selected_zone = st.selectbox(
        "Select a zone:", options=sorted_zones,
        index=sorted_zones.index(default_zone),
        format_func=lambda z: f"{z}  ·  {latest[latest['zone_id']==z]['status'].iloc[0]}",
    )
with col_range:
    date_range = st.select_slider(
        "Time window:",
        options=['Last 24 hours', 'Last 7 days', 'Last 30 days', 'All 90 days'],
        value='Last 7 days',
    )

window_map = {
    'Last 24 hours': timedelta(hours=24), 'Last 7 days': timedelta(days=7),
    'Last 30 days': timedelta(days=30), 'All 90 days': timedelta(days=90),
}

zone_meta = latest[latest['zone_id'] == selected_zone].iloc[0]
zone_history = comfort_df[comfort_df['zone_id'] == selected_zone].sort_values('timestamp')
cutoff = zone_history['timestamp'].max() - window_map[date_range]
zone_window = zone_history[zone_history['timestamp'] >= cutoff]

status_color = {'Normal': STATUS_HEALTHY, 'Watch': STATUS_WATCH,
                'Alert': STATUS_ALERT, 'Offline': STATUS_OFFLINE}[zone_meta['status']]
st.markdown(
    f"<div style='display:flex; align-items:center; gap:14px; margin-bottom:14px;'>"
    f"<span style='font-size:1.3rem; font-weight:600; color:{TEXT_PRIMARY};'>{selected_zone}</span>"
    f"<span class='status-pill status-{zone_meta['status'].lower()}'>{zone_meta['status']}</span>"
    f"<span style='color:{TEXT_SECONDARY}; font-size:0.9rem;'>"
    f"Floor {zone_meta['floor']} · {zone_meta['quadrant']} · "
    f"Linked FCU: <code style='background:{BG_ELEVATED}; padding:2px 6px; border-radius:3px; color:{ACCENT_PRIMARY};'>{zone_meta['fcu_id']}</code>"
    f"</span></div>",
    unsafe_allow_html=True
)

m1, m2, m3, m4, m5 = st.columns(5)
with m1: st.metric("Temperature", f"{zone_meta['indoor_temp']:.1f}°C")
with m2: st.metric("Humidity", f"{zone_meta['humidity']:.0f}%")
with m3: st.metric("CO₂", f"{zone_meta['co2']:.0f} ppm")
with m4: st.metric("PM2.5", f"{zone_meta['pm25']:.1f} µg/m³")
with m5: st.metric("CO", f"{zone_meta['co']:.2f} ppm")

# === LINKED ZOOM FIX (all 4 charts share x-axis) ===
fig_ts = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Temperature (°C)', 'CO₂ (ppm)', 'PM2.5 (µg/m³)', 'Humidity (%)'),
    vertical_spacing=0.18, horizontal_spacing=0.10,
    shared_xaxes='all',  # ← was True before; 'all' shares across rows AND columns
)

fig_ts.add_trace(go.Scatter(x=zone_window['timestamp'], y=zone_window['indoor_temp'],
    line=dict(color=VIZ_BRONZE, width=2), showlegend=False), row=1, col=1)
fig_ts.add_hline(y=27, line=dict(color=STATUS_ALERT, dash='dash', width=1), row=1, col=1, opacity=0.5)

fig_ts.add_trace(go.Scatter(x=zone_window['timestamp'], y=zone_window['co2'],
    line=dict(color=VIZ_TEAL, width=2), showlegend=False), row=1, col=2)
fig_ts.add_hline(y=1000, line=dict(color=STATUS_ALERT, dash='dash', width=1), row=1, col=2, opacity=0.5)

fig_ts.add_trace(go.Scatter(x=zone_window['timestamp'], y=zone_window['pm25'],
    line=dict(color=VIZ_PURPLE, width=2), showlegend=False), row=2, col=1)
fig_ts.add_hline(y=35, line=dict(color=STATUS_ALERT, dash='dash', width=1), row=2, col=1, opacity=0.5)

fig_ts.add_trace(go.Scatter(x=zone_window['timestamp'], y=zone_window['humidity'],
    line=dict(color=VIZ_PINK, width=2), showlegend=False), row=2, col=2)

# === LINKED ZOOM FIX (force all xaxes to match xaxis1) ===
fig_ts.update_xaxes(matches='x')  # explicitly link every x-axis to the first

fig_ts.update_layout(template=PLOTLY_TEMPLATE, height=480,
    margin=dict(l=20, r=20, t=40, b=20), showlegend=False, hovermode='x unified')
fig_ts.update_xaxes(showgrid=False, color=TEXT_SECONDARY)
fig_ts.update_yaxes(gridcolor=BORDER, color=TEXT_SECONDARY)

st.plotly_chart(fig_ts, use_container_width=True)
st.caption("Box-zoom on any chart syncs across all 4 panels. Double-click to reset.")

issues = get_comfort_issues(zone_meta)
if zone_meta['status'] == 'Offline':
    issues = ['Sensor reading frozen for 24+ hours — likely sensor fault or connectivity loss']

col_issues, col_actions = st.columns(2)
with col_issues:
    st.markdown("**Active Issues**")
    if issues:
        for i in issues:
            st.markdown(
                f"<div style='background:{BG_SECONDARY}; border-left:3px solid {STATUS_ALERT}; "
                f"padding:10px 14px; margin-bottom:6px; border-radius:4px;'>"
                f"<span style='color:{TEXT_PRIMARY}; font-size:0.9rem;'>{i}</span></div>",
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            f"<div style='background:{BG_SECONDARY}; border-left:3px solid {STATUS_HEALTHY}; "
            f"padding:10px 14px; border-radius:4px;'>"
            f"<span style='color:{TEXT_PRIMARY}; font-size:0.9rem;'>No active issues</span></div>",
            unsafe_allow_html=True
        )

with col_actions:
    st.markdown("**Recommended Actions**")
    if zone_meta['status'] == 'Offline':
        actions = ['Dispatch technician to verify sensor connectivity', 'Check BMS logs for last successful reading']
    elif zone_meta['indoor_temp'] > 27:
        actions = [f'Inspect FCU {zone_meta["fcu_id"]} — possible cooling degradation',
                   'Check valve operation and chilled water supply',
                   'Cross-reference with HVAC Predictive page for fault probability']
    elif zone_meta['co2'] > 1000:
        actions = ['Check ventilation damper and OA intake', 'Verify FCU running in operate mode',
                   'Inspect mixed-air damper position']
    elif zone_meta['pm25'] > 35:
        actions = ['Replace or clean air filter on linked FCU', 'Check filter pressure differential',
                   'Schedule next preventive filter check']
    elif issues:
        actions = ['Monitor trend over next 4 hours', 'Re-evaluate if condition persists']
    else:
        actions = ['No action required — zone within normal parameters']

    for a in actions:
        st.markdown(
            f"<div style='background:{BG_SECONDARY}; border-left:3px solid {ACCENT_PRIMARY}; "
            f"padding:10px 14px; margin-bottom:6px; border-radius:4px;'>"
            f"<span style='color:{TEXT_PRIMARY}; font-size:0.9rem;'>→ {a}</span></div>",
            unsafe_allow_html=True
        )

matching_anomaly = anomaly_log[anomaly_log['issue'].str.contains(selected_zone, na=False)]
if len(matching_anomaly) > 0:
    st.markdown("---")
    st.caption(f"**Demo note:** {matching_anomaly['issue'].iloc[0]}")
