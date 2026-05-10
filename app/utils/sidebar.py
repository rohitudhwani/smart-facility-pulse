"""
Shared sidebar alert panel — call render_sidebar_alerts() on every page.
"""
import streamlit as st
import pandas as pd
from utils.data_loader import (
    load_comfort_sensors, load_pool_sensors, load_zones_master,
    load_outdoor_weather,
)
from utils.alert_engine import (
    classify_comfort, get_comfort_issues,
    classify_pool, get_pool_issues,
    map_zone_to_fcu_fault,
)
from utils.theme import (
    BG_SECONDARY, STATUS_ALERT, STATUS_WATCH, STATUS_HEALTHY,
    TEXT_PRIMARY, TEXT_SECONDARY, ACCENT_PRIMARY,
)
from datetime import timedelta


@st.cache_data
def compute_active_alerts():
    """Return the unified active alerts list across all 3 modules."""
    comfort_df = load_comfort_sensors()
    pool_df = load_pool_sensors()
    zones_df = load_zones_master()
    weather_df = load_outdoor_weather()

    alerts = []

    # === Comfort module ===
    latest_comfort = comfort_df.sort_values('timestamp').groupby('zone_id').tail(1).reset_index(drop=True)
    latest_comfort['status'] = latest_comfort.apply(classify_comfort, axis=1)
    for _, row in latest_comfort[latest_comfort['status'] == 'Alert'].iterrows():
        issues = get_comfort_issues(row)
        if issues:
            alerts.append({
                'category': 'Comfort', 'source': row['zone_id'], 'message': issues[0],
            })

    # === Pool module ===
    latest_pool = pool_df.sort_values('timestamp').groupby('pool_id').tail(1).reset_index(drop=True)
    latest_pool['status'] = latest_pool.apply(classify_pool, axis=1)
    for _, row in latest_pool[latest_pool['status'] == 'Alert'].iterrows():
        issues = get_pool_issues(row)
        if issues:
            alerts.append({
                'category': 'Pool', 'source': row['pool_name'], 'message': issues[0],
            })

    # === HVAC module (rule-based fleet scoring) ===
    cutoff_24h = comfort_df['timestamp'].max() - timedelta(hours=24)
    last_24h = comfort_df[comfort_df['timestamp'] >= cutoff_24h]
    outdoor_pm25_24h = weather_df[weather_df['timestamp'] >= cutoff_24h]['outdoor_pm25'].mean()

    latest_with_meta = latest_comfort.merge(zones_df[['zone_id', 'temp_setpoint', 'fcu_id']],
                                             on='zone_id', suffixes=('', '_m'))
    if 'fcu_id_m' in latest_with_meta.columns:
        latest_with_meta['fcu_id'] = latest_with_meta['fcu_id_m']
    if 'temp_setpoint_m' in latest_with_meta.columns:
        latest_with_meta['temp_setpoint'] = latest_with_meta['temp_setpoint_m']

    for _, zrow in latest_with_meta.iterrows():
        zone_history = last_24h[last_24h['zone_id'] == zrow['zone_id']]
        pred = map_zone_to_fcu_fault(zrow, zone_history, outdoor_pm25_24h)
        if pred['fault_proba'] > 0.7:
            alerts.append({
                'category': 'HVAC',
                'source': zrow['fcu_id'],
                'message': f"{pred['fault_type'].replace('_', ' ').title()} · {pred['fault_proba']*100:.0f}% risk",
            })

    return alerts, latest_comfort['timestamp'].max()


def render_sidebar_alerts():
    """Render the alert panel in the Streamlit sidebar — call from every page."""
    alerts, last_update = compute_active_alerts()

    with st.sidebar:
        st.markdown("### Live Alerts")

        if not alerts:
            st.success("All systems normal")
        else:
            st.markdown(f"**{len(alerts)} active alerts**")
            for a in alerts[:12]:
                st.markdown(
                    f"<div style='background:{BG_SECONDARY}; border-left:3px solid {STATUS_ALERT}; "
                    f"padding:8px 12px; margin-bottom:6px; border-radius:4px; font-size:0.82rem;'>"
                    f"<b style='color:{STATUS_ALERT}'>{a['category']}</b> · "
                    f"<span style='color:{TEXT_SECONDARY}'>{a['source']}</span><br>"
                    f"<span style='color:{TEXT_PRIMARY}'>{a['message']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            if len(alerts) > 12:
                st.caption(f"+ {len(alerts) - 12} more")

        st.markdown("---")
        st.markdown(
            f"<small style='color:{TEXT_SECONDARY}'>Building: <b>Sobha Pilot FM</b><br>"
            f"Last update: {last_update.strftime('%Y-%m-%d %H:%M')}</small>",
            unsafe_allow_html=True
        )
