"""
Page 3: Pool Quality Monitoring (Module B)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

from utils.data_loader import load_pool_sensors, load_pools_master
from utils.alert_engine import (
    classify_pool, get_pool_issues, predict_pool_service_hours, POOL_THRESHOLDS
)
from utils.theme import (
    get_global_css, PLOTLY_TEMPLATE,
    ACCENT_PRIMARY, STATUS_HEALTHY, STATUS_WATCH, STATUS_ALERT,
    BG_SECONDARY, BG_ELEVATED, BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
    VIZ_TEAL, VIZ_BRONZE, VIZ_PURPLE, VIZ_PINK,
)

st.set_page_config(page_title="Pool Quality", layout="wide")
from utils.sidebar import render_sidebar_alerts  # noqa: E402
st.markdown(get_global_css(), unsafe_allow_html=True)
render_sidebar_alerts()


def hex_to_rgba(hex_color, alpha=0.2):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16); g = int(hex_color[2:4], 16); b = int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {alpha})'


pool_df = load_pool_sensors()
pools_master = load_pools_master()

latest_pool = pool_df.sort_values('timestamp').groupby('pool_id').tail(1).reset_index(drop=True)
latest_pool['status'] = latest_pool.apply(classify_pool, axis=1)

st.markdown(
    """
    <div class='brand-banner'>
        <h1>Pool Quality & Service Prediction</h1>
        <p>Module B · 2 swimming pools · CDC-aligned chemistry monitoring · Predictive service alerts</p>
    </div>
    """,
    unsafe_allow_html=True
)


def build_gauge(value, title, low_good, high_good,
                low_alert=None, high_alert=None, suffix='', value_format='.2f'):
    if low_alert is None: low_alert = low_good * 0.85
    if high_alert is None: high_alert = high_good * 1.15
    if value < low_alert or value > high_alert: bar_color = STATUS_ALERT
    elif value < low_good or value > high_good: bar_color = STATUS_WATCH
    else: bar_color = STATUS_HEALTHY

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={'suffix': suffix, 'valueformat': value_format,
                'font': {'color': TEXT_PRIMARY, 'size': 28}},
        title={'text': title, 'font': {'color': TEXT_SECONDARY, 'size': 13}},
        gauge={
            'axis': {'range': [low_alert * 0.8, high_alert * 1.1],
                     'tickcolor': TEXT_SECONDARY, 'tickfont': {'color': TEXT_SECONDARY, 'size': 10}},
            'bar': {'color': bar_color, 'thickness': 0.7},
            'bgcolor': BG_ELEVATED, 'borderwidth': 0,
            'steps': [
                {'range': [low_alert * 0.8, low_alert], 'color': hex_to_rgba(STATUS_ALERT, 0.2)},
                {'range': [low_alert, low_good], 'color': hex_to_rgba(STATUS_WATCH, 0.2)},
                {'range': [low_good, high_good], 'color': hex_to_rgba(STATUS_HEALTHY, 0.2)},
                {'range': [high_good, high_alert], 'color': hex_to_rgba(STATUS_WATCH, 0.2)},
                {'range': [high_alert, high_alert * 1.1], 'color': hex_to_rgba(STATUS_ALERT, 0.2)},
            ],
            'threshold': {'line': {'color': TEXT_PRIMARY, 'width': 2}, 'thickness': 0.8, 'value': value}
        }
    ))
    fig.update_layout(paper_bgcolor=BG_SECONDARY, height=180, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def build_directional_gauge(value, title, target_max, alert_at, suffix=''):
    if value > alert_at: bar_color = STATUS_ALERT
    elif value > target_max: bar_color = STATUS_WATCH
    else: bar_color = STATUS_HEALTHY
    axis_max = max(alert_at * 1.3, value * 1.1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={'suffix': suffix, 'valueformat': '.1f',
                'font': {'color': TEXT_PRIMARY, 'size': 28}},
        title={'text': title, 'font': {'color': TEXT_SECONDARY, 'size': 13}},
        gauge={
            'axis': {'range': [0, axis_max],
                     'tickcolor': TEXT_SECONDARY, 'tickfont': {'color': TEXT_SECONDARY, 'size': 10}},
            'bar': {'color': bar_color, 'thickness': 0.7},
            'bgcolor': BG_ELEVATED, 'borderwidth': 0,
            'steps': [
                {'range': [0, target_max], 'color': hex_to_rgba(STATUS_HEALTHY, 0.2)},
                {'range': [target_max, alert_at], 'color': hex_to_rgba(STATUS_WATCH, 0.2)},
                {'range': [alert_at, axis_max], 'color': hex_to_rgba(STATUS_ALERT, 0.2)},
            ],
        }
    ))
    fig.update_layout(paper_bgcolor=BG_SECONDARY, height=180, margin=dict(l=20, r=20, t=40, b=10))
    return fig


def render_pool_card(pool_id, container):
    pool_meta = pools_master[pools_master['pool_id'] == pool_id].iloc[0]
    pool_now = latest_pool[latest_pool['pool_id'] == pool_id].iloc[0]
    pool_history = pool_df[pool_df['pool_id'] == pool_id]
    hours_until, limiting = predict_pool_service_hours(pool_history)
    issues = get_pool_issues(pool_now)
    status = pool_now['status']

    with container:
        st.markdown(
            f"<div style='display:flex; align-items:center; justify-content:space-between; margin-bottom:14px;'>"
            f"<span style='font-size:1.25rem; font-weight:600; color:{TEXT_PRIMARY};'>{pool_meta['pool_name']}</span>"
            f"<span class='status-pill status-{status.lower()}'>{status}</span>"
            f"</div>", unsafe_allow_html=True
        )

        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(build_gauge(pool_now['water_temp'], 'Water Temp',
                low_good=26, high_good=30, suffix='°C', value_format='.1f'),
                use_container_width=True, key=f"temp_{pool_id}")
        with g2:
            st.plotly_chart(build_gauge(pool_now['pH'], 'pH',
                low_good=7.2, high_good=7.8, low_alert=6.8, high_alert=8.2),
                use_container_width=True, key=f"ph_{pool_id}")
        with g3:
            st.plotly_chart(build_gauge(pool_now['chlorine'], 'Free Chlorine',
                low_good=1.0, high_good=3.0, low_alert=0.5, high_alert=4.0, suffix=' ppm'),
                use_container_width=True, key=f"cl_{pool_id}")

        g4, g5 = st.columns(2)
        with g4:
            st.plotly_chart(build_directional_gauge(pool_now['turbidity'], 'Turbidity',
                target_max=1.0, alert_at=2.0, suffix=' NTU'),
                use_container_width=True, key=f"turb_{pool_id}")
        with g5:
            st.plotly_chart(build_directional_gauge(pool_now['hours_since_service'], 'Hours Since Service',
                target_max=pool_meta['service_interval_hours'],
                alert_at=POOL_THRESHOLDS['service_overdue_hours'], suffix=' h'),
                use_container_width=True, key=f"svc_{pool_id}")

        if hours_until is not None and hours_until < 9999:
            if hours_until < 24:
                pred_color, pred_text = STATUS_ALERT, f"Service required in **~{hours_until} hours** ({limiting} approaching limit)"
            elif hours_until < 72:
                pred_color, pred_text = STATUS_WATCH, f"Service recommended within **~{hours_until} hours** ({limiting} trending)"
            else:
                pred_color, pred_text = STATUS_HEALTHY, f"Next service window: **~{hours_until} hours** ({limiting})"

            st.markdown(
                f"<div style='background:{BG_SECONDARY}; border-left:4px solid {pred_color}; "
                f"padding:12px 16px; border-radius:6px; margin-top:6px;'>"
                f"<div style='color:{TEXT_SECONDARY}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;'>"
                f"Predictive Service Forecast</div>"
                f"<div style='color:{TEXT_PRIMARY}; font-size:0.95rem; margin-top:4px;'>{pred_text}</div>"
                f"</div>", unsafe_allow_html=True
            )

        if issues:
            st.markdown(f"<div style='margin-top:10px;'>", unsafe_allow_html=True)
            st.markdown(f"**Active Issues**")
            for i in issues:
                st.markdown(
                    f"<div style='background:{BG_SECONDARY}; border-left:3px solid {STATUS_ALERT}; "
                    f"padding:8px 12px; margin-bottom:5px; border-radius:4px; font-size:0.85rem; color:{TEXT_PRIMARY};'>"
                    f"{i}</div>", unsafe_allow_html=True
                )


col_main, col_kids = st.columns(2)
render_pool_card('POOL-MAIN', col_main)
render_pool_card('POOL-KIDS', col_kids)

st.markdown("---")
st.markdown("### Detailed Trends")

col_pool_select, col_date = st.columns([1, 2])
with col_pool_select:
    selected_pool = st.selectbox(
        "Pool:", options=['POOL-MAIN', 'POOL-KIDS'],
        format_func=lambda p: pools_master[pools_master['pool_id']==p]['pool_name'].iloc[0]
    )
with col_date:
    date_range = st.select_slider(
        "Time window:",
        options=['Last 24 hours', 'Last 7 days', 'Last 30 days', 'All 90 days'],
        value='Last 7 days',
    )

window_map = {
    'Last 24 hours': timedelta(hours=24), 'Last 7 days': timedelta(days=7),
    'Last 30 days': timedelta(days=30), 'All 90 days': timedelta(days=90),
}
cutoff = pool_df['timestamp'].max() - window_map[date_range]
pool_window = pool_df[(pool_df['pool_id'] == selected_pool) &
                      (pool_df['timestamp'] >= cutoff)].sort_values('timestamp')

# === LINKED ZOOM FIX ===
fig_ts = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Water Temperature (°C)', 'pH', 'Free Chlorine (ppm)', 'Turbidity (NTU)'),
    vertical_spacing=0.18, horizontal_spacing=0.10,
    shared_xaxes='all',  # ← was True
)

fig_ts.add_trace(go.Scatter(x=pool_window['timestamp'], y=pool_window['water_temp'],
    line=dict(color=VIZ_BRONZE, width=2), showlegend=False), row=1, col=1)

fig_ts.add_trace(go.Scatter(x=pool_window['timestamp'], y=pool_window['pH'],
    line=dict(color=VIZ_TEAL, width=2), showlegend=False), row=1, col=2)
fig_ts.add_hrect(y0=7.2, y1=7.8, fillcolor=STATUS_HEALTHY, opacity=0.08, line_width=0, row=1, col=2)
fig_ts.add_hline(y=7.2, line=dict(color=STATUS_HEALTHY, dash='dot', width=1), opacity=0.4, row=1, col=2)
fig_ts.add_hline(y=7.8, line=dict(color=STATUS_HEALTHY, dash='dot', width=1), opacity=0.4, row=1, col=2)

fig_ts.add_trace(go.Scatter(x=pool_window['timestamp'], y=pool_window['chlorine'],
    line=dict(color=VIZ_PURPLE, width=2), showlegend=False), row=2, col=1)
fig_ts.add_hline(y=1.0, line=dict(color=STATUS_ALERT, dash='dash', width=1), row=2, col=1, opacity=0.6,
    annotation_text="CDC min", annotation_position='top right', annotation_font_color=STATUS_ALERT)

fig_ts.add_trace(go.Scatter(x=pool_window['timestamp'], y=pool_window['turbidity'],
    line=dict(color=VIZ_PINK, width=2), showlegend=False), row=2, col=2)
fig_ts.add_hline(y=2.0, line=dict(color=STATUS_ALERT, dash='dash', width=1), row=2, col=2, opacity=0.6,
    annotation_text="Alert", annotation_position='top right', annotation_font_color=STATUS_ALERT)

# === Force all xaxes to share range ===
fig_ts.update_xaxes(matches='x')

fig_ts.update_layout(template=PLOTLY_TEMPLATE, height=520,
    margin=dict(l=20, r=20, t=40, b=20), showlegend=False, hovermode='x unified')
fig_ts.update_xaxes(showgrid=False, color=TEXT_SECONDARY)
fig_ts.update_yaxes(gridcolor=BORDER, color=TEXT_SECONDARY)

st.plotly_chart(fig_ts, use_container_width=True)
st.caption("Box-zoom on any chart syncs across all 4 panels. Double-click to reset.")

st.markdown("### Service Cycle History")
fig_svc = go.Figure()
fig_svc.add_trace(go.Scatter(
    x=pool_window['timestamp'], y=pool_window['hours_since_service'],
    line=dict(color=ACCENT_PRIMARY, width=2),
    fill='tozeroy', fillcolor=hex_to_rgba(ACCENT_PRIMARY, 0.15),
    hovertemplate='%{x|%b %d %H:%M}<br>%{y:.0f} hours<extra></extra>'
))
fig_svc.add_hline(y=POOL_THRESHOLDS['service_overdue_hours'],
    line=dict(color=STATUS_ALERT, dash='dash', width=1),
    annotation_text="Overdue threshold", annotation_position='top right',
    annotation_font_color=STATUS_ALERT)
fig_svc.update_layout(template=PLOTLY_TEMPLATE, height=260,
    margin=dict(l=20, r=20, t=20, b=20), yaxis_title='Hours', showlegend=False)
st.plotly_chart(fig_svc, use_container_width=True)
st.caption("Each downward drop indicates a completed service event. Sustained climb above the dashed line = overdue.")
