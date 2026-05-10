"""
Page 4: HVAC Predictive Maintenance — Hybrid (Building FCU Fleet)
Maps Module A symptoms to LBNL fault types via rule layer.
The actual XGBoost model is referenced for performance validation only.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

from utils.data_loader import (
    load_comfort_sensors, load_zones_master, load_outdoor_weather, load_anomaly_log,
)
from utils.alert_engine import (
    map_zone_to_fcu_fault, FAULT_TYPE_DESCRIPTIONS, RECOMMENDED_ACTIONS,
)
from utils.theme import (
    get_global_css, PLOTLY_TEMPLATE,
    ACCENT_PRIMARY, STATUS_HEALTHY, STATUS_WATCH, STATUS_ALERT, STATUS_OFFLINE,
    BG_SECONDARY, BG_ELEVATED, BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
    VIZ_TEAL, VIZ_BRONZE, VIZ_PURPLE, VIZ_PINK,
)

st.set_page_config(
    page_title="HVAC Predictive — Building Fleet",
    layout="wide",
)
from utils.sidebar import render_sidebar_alerts  # noqa: E402
st.markdown(get_global_css(), unsafe_allow_html=True)
render_sidebar_alerts()


# ============================================================
# LOAD DATA
# ============================================================
comfort_df = load_comfort_sensors()
zones_df = load_zones_master()
weather_df = load_outdoor_weather()
anomaly_log = load_anomaly_log()


# ============================================================
# SCORE ALL FCUs IN THE BUILDING
# ============================================================
@st.cache_data
def score_all_fcus():
    """Run symptom-to-fault mapping over every zone in the building."""
    # Latest reading per zone
    latest = comfort_df.sort_values('timestamp').groupby('zone_id').tail(1).reset_index(drop=True)

    # Last 24 hours per zone (for stuck-sensor detection)
    cutoff_24h = comfort_df['timestamp'].max() - timedelta(hours=24)
    last_24h = comfort_df[comfort_df['timestamp'] >= cutoff_24h]

    # Outdoor PM2.5 24h average (for filter assessment baseline)
    outdoor_pm25_24h = weather_df[weather_df['timestamp'] >= cutoff_24h]['outdoor_pm25'].mean()

    # Merge zone metadata onto latest readings (for setpoint, fcu_id)
    latest = latest.merge(zones_df[['zone_id', 'temp_setpoint', 'fcu_id']],
                          on='zone_id', suffixes=('', '_meta'))
    if 'fcu_id_meta' in latest.columns:
        latest['fcu_id'] = latest['fcu_id_meta']
        latest = latest.drop(columns=['fcu_id_meta'])
    if 'temp_setpoint_meta' in latest.columns:
        latest['temp_setpoint'] = latest['temp_setpoint_meta']
        latest = latest.drop(columns=['temp_setpoint_meta'])

    results = []
    for _, zrow in latest.iterrows():
        zone_history = last_24h[last_24h['zone_id'] == zrow['zone_id']]
        prediction = map_zone_to_fcu_fault(zrow, zone_history, outdoor_pm25_24h)
        results.append({
            'fcu_id': zrow['fcu_id'],
            'zone_id': zrow['zone_id'],
            'floor': zrow['floor'],
            'quadrant': zrow['quadrant'],
            'zone_type': zrow['zone_type'],
            'indoor_temp': zrow['indoor_temp'],
            'co2': zrow['co2'],
            'pm25': zrow['pm25'],
            'humidity': zrow['humidity'],
            **prediction,
        })

    return pd.DataFrame(results)


fcu_scores = score_all_fcus()


# ============================================================
# BRAND BANNER
# ============================================================
st.markdown(
    """
    <div class='brand-banner'>
        <h1>HVAC Predictive Maintenance — Building Fleet</h1>
        <p>Module C · 81 FCUs scored using symptom-to-fault mapping derived from LBNL trained model</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL CONTEXT STRIP
# ============================================================
n_total = len(fcu_scores)
n_high_risk = (fcu_scores['fault_proba'] > 0.7).sum()
n_medium_risk = ((fcu_scores['fault_proba'] >= 0.4) & (fcu_scores['fault_proba'] <= 0.7)).sum()
n_healthy = (fcu_scores['fault_proba'] < 0.4).sum()

c1, c2, c3, c4 = st.columns(4)
for col, label, count, color in [
    (c1, "FCUs Monitored", n_total, ACCENT_PRIMARY),
    (c2, "High Risk (>70%)", n_high_risk, STATUS_ALERT),
    (c3, "Medium Risk (40–70%)", n_medium_risk, STATUS_WATCH),
    (c4, "Healthy (<40%)", n_healthy, STATUS_HEALTHY),
]:
    with col:
        st.markdown(
            f"<div style='background:{BG_SECONDARY}; border-left:4px solid {color}; "
            f"border-radius:8px; padding:14px 18px;'>"
            f"<div style='color:{TEXT_SECONDARY}; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em;'>{label}</div>"
            f"<div style='color:{color}; font-size:2rem; font-weight:600;'>{count}</div>"
            f"</div>", unsafe_allow_html=True
        )


# ============================================================
# FCU FLEET RANKING
# ============================================================
st.markdown("### FCU Fleet Risk Ranking")
st.caption("All 81 building FCUs sorted by current fault probability. Click a row to drill down.")

# Sort by fault probability (high to low)
fcu_ranked = fcu_scores.sort_values('fault_proba', ascending=False).reset_index(drop=True)

# Filter controls
col_filter1, col_filter2 = st.columns([1, 3])
with col_filter1:
    show_filter = st.selectbox(
        "Show:",
        options=['All FCUs', 'High risk only (>70%)', 'Medium + High (>40%)', 'Healthy only (<40%)'],
    )

if show_filter == 'High risk only (>70%)':
    fcu_display = fcu_ranked[fcu_ranked['fault_proba'] > 0.7]
elif show_filter == 'Medium + High (>40%)':
    fcu_display = fcu_ranked[fcu_ranked['fault_proba'] > 0.4]
elif show_filter == 'Healthy only (<40%)':
    fcu_display = fcu_ranked[fcu_ranked['fault_proba'] < 0.4]
else:
    fcu_display = fcu_ranked


# ============================================================
# HORIZONTAL BAR CHART OF FCU FAULT PROBABILITIES
# ============================================================
top_n = min(25, len(fcu_display))
chart_data = fcu_display.head(top_n).iloc[::-1]  # reverse for top-down display

bar_colors = [
    STATUS_ALERT if p > 0.7 else (STATUS_WATCH if p > 0.4 else STATUS_HEALTHY)
    for p in chart_data['fault_proba']
]

fig_bars = go.Figure()
fig_bars.add_trace(go.Bar(
    y=chart_data['fcu_id'],
    x=chart_data['fault_proba'] * 100,
    orientation='h',
    marker=dict(color=bar_colors),
    text=[f"{p*100:.0f}%" for p in chart_data['fault_proba']],
    textposition='outside',
    textfont=dict(color=TEXT_PRIMARY, size=11),
    hovertemplate=(
        '<b>%{y}</b><br>'
        'Fault Probability: %{x:.0f}%<br>'
        '<extra></extra>'
    ),
))
fig_bars.update_layout(
    template=PLOTLY_TEMPLATE,
    height=max(400, top_n * 22),
    margin=dict(l=20, r=60, t=10, b=20),
    xaxis=dict(title='Fault Probability (%)', range=[0, 110], gridcolor=BORDER),
    yaxis=dict(title=None, tickfont=dict(size=11)),
    showlegend=False,
)
st.plotly_chart(fig_bars, use_container_width=True)


# ============================================================
# DETAILED FCU TABLE
# ============================================================
st.markdown("### Full FCU Details")

table_view = fcu_display[[
    'fcu_id', 'zone_id', 'floor', 'fault_proba', 'fault_type', 'severity',
    'indoor_temp', 'co2', 'pm25'
]].copy()
table_view.columns = ['FCU ID', 'Zone', 'Floor', 'Fault Risk', 'Likely Fault', 'Severity',
                       'Temp (°C)', 'CO₂ (ppm)', 'PM2.5 (µg/m³)']
table_view['Fault Risk'] = (table_view['Fault Risk'] * 100).round(0).astype(int).astype(str) + '%'
table_view['Likely Fault'] = table_view['Likely Fault'].map(
    lambda f: FAULT_TYPE_DESCRIPTIONS.get(f, f).split('—')[0].strip()
)

st.dataframe(table_view, use_container_width=True, hide_index=True, height=320)


# ============================================================
# SELECTED FCU DRILL-DOWN
# ============================================================
st.markdown("---")
st.markdown("### FCU Detail View")

# Default selection: first high-risk FCU if any, else top of list
default_fcu_idx = 0  # already sorted by risk descending

selected_fcu = st.selectbox(
    "Select an FCU to inspect:",
    options=fcu_ranked['fcu_id'].tolist(),
    index=default_fcu_idx,
    format_func=lambda fid: (
        f"{fid}  ·  {fcu_ranked[fcu_ranked['fcu_id']==fid]['fault_proba'].iloc[0]*100:.0f}% risk  "
        f"·  {FAULT_TYPE_DESCRIPTIONS.get(fcu_ranked[fcu_ranked['fcu_id']==fid]['fault_type'].iloc[0], '').split('—')[0].strip()}"
    ),
)

fcu_detail = fcu_ranked[fcu_ranked['fcu_id'] == selected_fcu].iloc[0]
fcu_proba = fcu_detail['fault_proba']
fault_type = fcu_detail['fault_type']
severity = fcu_detail['severity']

risk_color = STATUS_ALERT if fcu_proba > 0.7 else (STATUS_WATCH if fcu_proba > 0.4 else STATUS_HEALTHY)
risk_label = 'High Risk' if fcu_proba > 0.7 else ('Medium Risk' if fcu_proba > 0.4 else 'Healthy')

# Header
st.markdown(
    f"<div style='display:flex; align-items:center; gap:14px; margin-bottom:14px;'>"
    f"<span style='font-size:1.4rem; font-weight:600; color:{TEXT_PRIMARY};'>{selected_fcu}</span>"
    f"<span class='status-pill' style='background:{risk_color}22; color:{risk_color}; "
    f"border:1px solid {risk_color}55;'>{risk_label} · {fcu_proba*100:.0f}%</span>"
    f"<span style='color:{TEXT_SECONDARY}; font-size:0.9rem;'>"
    f"Linked zone: <code style='background:{BG_ELEVATED}; padding:2px 6px; border-radius:3px; "
    f"color:{ACCENT_PRIMARY};'>{fcu_detail['zone_id']}</code> · Floor {fcu_detail['floor']}"
    f"</span></div>",
    unsafe_allow_html=True
)

# Detail layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("**Linked Zone Current Readings**")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Temperature", f"{fcu_detail['indoor_temp']:.1f}°C")
        st.metric("CO₂", f"{fcu_detail['co2']:.0f} ppm")
    with m2:
        st.metric("PM2.5", f"{fcu_detail['pm25']:.1f} µg/m³")
        st.metric("Humidity", f"{fcu_detail['humidity']:.0f}%")

    # Fault probability gauge
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fcu_proba * 100,
        number={'suffix': '%', 'valueformat': '.0f',
                'font': {'color': TEXT_PRIMARY, 'size': 32}},
        title={'text': 'Fault Probability', 'font': {'color': TEXT_SECONDARY, 'size': 13}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'color': TEXT_SECONDARY, 'size': 10}},
            'bar': {'color': risk_color, 'thickness': 0.7},
            'bgcolor': BG_ELEVATED, 'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'},
            ],
        }
    ))
    fig_g.update_layout(paper_bgcolor=BG_SECONDARY, height=200, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig_g, use_container_width=True)


with col_right:
    st.markdown("**Inferred Fault Diagnosis**")

    # Fault type card
    fault_desc = FAULT_TYPE_DESCRIPTIONS.get(fault_type, fault_type)
    severity_color = {
        'severe': STATUS_ALERT, 'moderate': STATUS_WATCH,
        'minor': VIZ_TEAL, 'none': STATUS_HEALTHY
    }.get(severity, TEXT_SECONDARY)

    st.markdown(
        f"<div style='background:{BG_SECONDARY}; border-left:4px solid {severity_color}; "
        f"border-radius:8px; padding:14px 18px; margin-bottom:10px;'>"
        f"<div style='color:{TEXT_SECONDARY}; font-size:0.75rem; text-transform:uppercase; "
        f"letter-spacing:0.05em;'>Most Likely Fault Class</div>"
        f"<div style='color:{TEXT_PRIMARY}; font-size:1.05rem; font-weight:600; margin-top:4px;'>"
        f"{fault_type.replace('_', ' ').title()}</div>"
        f"<div style='color:{TEXT_SECONDARY}; font-size:0.85rem; margin-top:4px;'>{fault_desc}</div>"
        f"<div style='margin-top:8px;'>"
        f"<span style='background:{severity_color}22; color:{severity_color}; "
        f"padding:3px 10px; border-radius:10px; font-size:0.75rem; font-weight:600; "
        f"text-transform:uppercase; letter-spacing:0.05em;'>Severity: {severity}</span>"
        f"</div></div>",
        unsafe_allow_html=True
    )

    # Why was it flagged
    st.markdown("**Why This FCU Was Flagged**")
    st.markdown(
        f"<div style='background:{BG_SECONDARY}; border-left:3px solid {ACCENT_PRIMARY}; "
        f"padding:12px 16px; border-radius:6px; margin-bottom:8px;'>"
        f"<div style='color:{TEXT_PRIMARY}; font-size:0.9rem;'>{fcu_detail['rule_fired']}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Contributing signals
    st.markdown("**Contributing Signals**")
    for sig in fcu_detail['contributing_signals']:
        st.markdown(
            f"<div style='background:{BG_ELEVATED}; padding:6px 12px; "
            f"margin-bottom:4px; border-radius:4px; font-size:0.82rem; "
            f"color:{TEXT_PRIMARY}; font-family:monospace;'>{sig}</div>",
            unsafe_allow_html=True
        )


# Recommended actions (full width)
st.markdown("**Recommended Technician Actions**")
actions = RECOMMENDED_ACTIONS.get(fault_type, [])
for i, action in enumerate(actions, 1):
    st.markdown(
        f"<div style='background:{BG_SECONDARY}; border-left:3px solid {ACCENT_PRIMARY}; "
        f"padding:10px 14px; margin-bottom:5px; border-radius:4px;'>"
        f"<span style='color:{TEXT_SECONDARY}; font-size:0.85rem; margin-right:8px;'>{i}.</span>"
        f"<span style='color:{TEXT_PRIMARY}; font-size:0.9rem;'>{action}</span></div>",
        unsafe_allow_html=True
    )


# ============================================================
# METHODOLOGY FOOTER (the well-done part you asked for)
# ============================================================
st.markdown("---")
st.markdown("## How This Page Works — Methodology & Limitations")

with st.expander("📐 Symptom-to-Fault Mapping Methodology", expanded=True):
    st.markdown(
        f"""
        ### Why a rule layer instead of running the model directly

        The **LBNL XGBoost model** (Notebook 04, Test AUC 0.997) is trained on
        **FCU-internal sensors**: cooling valve position, discharge airflow,
        chilled water flow rate, fan power consumption, mixed-air temperature,
        and ~25 other instrumentation points sampled at 1-minute intervals.

        The **Module A synthetic data** in this pilot only contains **zone-level
        environmental sensors**: room temperature, CO₂, humidity, PM2.5. These
        measure what residents *experience* — not what the FCU is *doing internally*.

        Until a real BMS feed is connected to this dashboard, the building FCU
        fleet cannot be scored directly by the model. Instead, this page uses a
        **symptom-to-fault rule layer** that infers the most likely LBNL fault
        class from observed zone symptoms.

        The rules below were derived from the LBNL model's learned feature
        importance (SHAP analysis in Notebook 04, Cells 10–11) and from
        established FM/HVAC physics.
        """
    )

    st.markdown("### Rule Mapping Table")

    rules_table = pd.DataFrame([
        {
            'Symptom': 'Indoor temp >1°C above setpoint, CO₂ normal',
            'LBNL Fault Class': 'cooling_fouling_airside',
            'Severity Scale': '(temp − setpoint − 1) / 4, capped at 1.0',
            'Physics': 'Room won\'t cool but ventilation OK → coil fouled, heat transfer degraded',
        },
        {
            'Symptom': 'CO₂ rising > 800 ppm, temp normal',
            'LBNL Fault Class': 'oa_inlet_blockage or fan_outlet_blockage',
            'Severity Scale': '(CO₂ − 800) / 600, capped at 1.0',
            'Physics': 'Cooling working but ventilation insufficient → OA damper or fan blocked',
        },
        {
            'Symptom': 'PM2.5 elevated above 30% of outdoor baseline',
            'LBNL Fault Class': 'filter_restriction',
            'Severity Scale': '(PM2.5 − expected) / 25, capped at 1.0',
            'Physics': 'Healthy filter blocks ~70% of outdoor PM2.5; elevated indoor PM2.5 = filter loaded',
        },
        {
            'Symptom': 'Indoor temp climbing AND CO₂ climbing',
            'LBNL Fault Class': 'cooling_valve_stuck',
            'Severity Scale': 'max of temp severity and CO₂ severity',
            'Physics': 'Combined cooling + ventilation failure = full FCU failure mode',
        },
        {
            'Symptom': 'Zone temp std = 0 over last 24 hours',
            'LBNL Fault Class': 'sensor_bias',
            'Severity Scale': 'Always 1.0 (severe)',
            'Physics': 'Sensor not responding = same effect as +2°C bias fault in LBNL',
        },
        {
            'Symptom': 'No symptoms above thresholds',
            'LBNL Fault Class': 'fault_free',
            'Severity Scale': 'Baseline 0.02–0.12 (small per-FCU variation)',
            'Physics': 'No detectable degradation in zone signals',
        },
    ])

    st.dataframe(rules_table, use_container_width=True, hide_index=True)


with st.expander("⚠️ Honest Limitations of This Approach"):
    st.markdown(
        """
        - **The XGBoost model is not running in real time on these 81 FCUs.**
          What you see on this page is a rule layer. The XGBoost model lives on the
          *HVAC Performance Reference* page, where it is run on labelled LBNL test data.

        - **The rule layer cannot detect faults that don't surface as zone-level symptoms.**
          For example, a cooling valve leaking 20% (real LBNL fault) might have minimal
          zone-level impact during mild outdoor conditions. The trained model can detect
          this from internal FCU sensors; the rule layer cannot.

        - **Severity is heuristic, not calibrated.**
          The rule severity scales are reasonable but not derived from a labelled dataset.
          They will need tuning once ground-truth fault labels are available from real BMS.

        - **Once BMS is connected, this rule layer becomes redundant.**
          Every FCU will be scored directly by the trained XGBoost model on its full
          sensor stream — the same way the model achieves 0.997 AUC on LBNL test data.
        """
    )


with st.expander("✅ The Production Path Forward"):
    st.markdown(
        """
        **Phase 1 (current pilot):**
        Symptom-to-fault rule layer using zone sensors only. Demonstrates UI/UX, methodology,
        and provides operational value through environmental monitoring + basic diagnostics.

        **Phase 2 (after BMS integration):**
        - Connect to live BMS feed (Schneider EcoStruxure / Honeywell / Siemens / Nectar)
        - Stream FCU-internal sensors into the data pipeline
        - Run the trained XGBoost model on all 81 FCUs in real time
        - Replace rule-based scoring with actual model predictions + SHAP explanations per unit
        - Calibrate model output threshold using FM's historical work-order data

        **Phase 3 (long-term):**
        - Retrain model on FM-specific fault history
        - Add LLM-generated diagnostic narratives (per technician feedback)
        - Integrate with CAFM / work-order system for closed-loop dispatch
        """
    )

st.markdown(
    f"<div style='color:{TEXT_SECONDARY}; font-size:0.78rem; margin-top:20px; "
    f"padding:14px; background:{BG_SECONDARY}; border-radius:6px;'>"
    f"<b>Data sources:</b> Module A synthetic comfort sensors (calibrated to CU-BEMS ranges) · "
    f"Outdoor climate proxy (Dubai weather pattern). "
    f"<b>Rule derivation:</b> LBNL FCU dataset feature importance (Notebook 04 SHAP analysis) · "
    f"ASHRAE comfort standards · HVAC physics. "
    f"<b>Model reference:</b> XGBoost classifier, Test AUC 0.997 on LBNL holdout — see HVAC Performance Reference page for details."
    f"</div>",
    unsafe_allow_html=True
)
