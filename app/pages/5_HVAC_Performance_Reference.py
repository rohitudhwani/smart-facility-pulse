"""
Page 5: HVAC Performance Reference (LBNL model, real predictions, real SHAP)
The "honest" view — the actual XGBoost model running on labelled LBNL test data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import io

from utils.data_loader import (
    load_hvac_model, load_feature_columns,
    load_representative_scenarios, load_shap_values_representative,
    load_global_feature_importance, load_model_metrics, load_shap_base_value,
)
from utils.alert_engine import FAULT_TYPE_DESCRIPTIONS
from utils.theme import (
    get_global_css, PLOTLY_TEMPLATE,
    ACCENT_PRIMARY, STATUS_HEALTHY, STATUS_WATCH, STATUS_ALERT,
    BG_SECONDARY, BG_ELEVATED, BORDER, TEXT_PRIMARY, TEXT_SECONDARY,
    VIZ_TEAL, VIZ_BRONZE, VIZ_PURPLE, VIZ_PINK,
)

st.set_page_config(
    page_title="HVAC Performance Reference",
    layout="wide",
)
from utils.sidebar import render_sidebar_alerts  # noqa: E402
st.markdown(get_global_css(), unsafe_allow_html=True)
render_sidebar_alerts()


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================
model = load_hvac_model()
feature_cols = load_feature_columns()
scenarios = load_representative_scenarios()
shap_values_all = load_shap_values_representative()
global_importance = load_global_feature_importance()
metrics = load_model_metrics()
base_value = load_shap_base_value()


# ============================================================
# BRAND BANNER
# ============================================================
st.markdown(
    """
    <div class='brand-banner'>
        <h1>HVAC Performance Reference — Trained Model on Labelled Data</h1>
        <p>Module C · The actual XGBoost model running on LBNL FCU test data, with real SHAP explanations</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "**This page is separate from the Building FCU Fleet view.** "
    "The Building Fleet page uses a rule layer for the 81 zones because zone-level data "
    "doesn't contain FCU-internal sensors. This page shows the trained model running on "
    "its own evaluation set — the LBNL public dataset — proving model performance on the "
    "data it was actually designed for. "
    "**No rule mapping. No symptom inference. Just the model and its real predictions.**",
    icon="ℹ️",
)


# ============================================================
# HEADLINE METRICS
# ============================================================
st.markdown("### Model Performance — Holdout Test Set")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Test ROC-AUC", f"{metrics['test_auc']:.4f}",
              delta="vs random 0.500", delta_color="off")
with c2:
    st.metric("Faulty Recall", f"{metrics['recall_faulty']*100:.0f}%",
              help="Of all actual faults, how many the model caught")
with c3:
    st.metric("Faulty Precision", f"{metrics['precision_faulty']*100:.0f}%",
              help="When the model flags a fault, how often it's correct")
with c4:
    st.metric("Test Rows", f"{metrics['test_rows']:,}",
              help=f"From {metrics['test_period']}")


# ============================================================
# PER-FAULT-TYPE DETECTION RATES
# ============================================================
st.markdown("### Detection Rate by Fault Type")
st.caption(
    "How well the model identifies each fault scenario. Rates >90% indicate the model "
    "reliably distinguishes that fault class from healthy operation."
)

# Build per-fault detection table
scenarios_proba = scenarios[['fault_type', 'severity', 'fault_proba', 'actual']].copy()

# Detection rate definition:
# - For faulty scenarios: % of samples predicted as faulty (recall)
# - For fault_free: % of samples correctly predicted as healthy (specificity)
detection = []
for (ftype, sev), grp in scenarios.groupby(['fault_type', 'severity']):
    if ftype == 'fault_free':
        # Specificity: how often we correctly say "not faulty"
        rate = (grp['fault_proba'] < 0.5).mean() * 100
        n = len(grp)
    else:
        # Recall: how often we correctly say "faulty"
        rate = (grp['fault_proba'] >= 0.5).mean() * 100
        n = len(grp)
    detection.append({
        'fault_type': ftype,
        'severity': sev,
        'detection_rate': rate,
        'avg_proba': grp['fault_proba'].mean(),
        'n_samples': n,
    })
detection_df = pd.DataFrame(detection).sort_values('detection_rate')

# Bar chart
fig_det = go.Figure()
bar_colors = [
    STATUS_ALERT if r < 85 else (STATUS_WATCH if r < 95 else STATUS_HEALTHY)
    for r in detection_df['detection_rate']
]
fig_det.add_trace(go.Bar(
    y=[f"{r['fault_type']} ({r['severity']})" for _, r in detection_df.iterrows()],
    x=detection_df['detection_rate'],
    orientation='h',
    marker=dict(color=bar_colors),
    text=[f"{r:.0f}%" for r in detection_df['detection_rate']],
    textposition='outside',
    textfont=dict(color=TEXT_PRIMARY, size=11),
    hovertemplate='<b>%{y}</b><br>Detection Rate: %{x:.1f}%<extra></extra>',
))
fig_det.update_layout(
    template=PLOTLY_TEMPLATE,
    height=max(400, len(detection_df) * 28),
    margin=dict(l=20, r=60, t=10, b=20),
    xaxis=dict(title='Detection Rate (%)', range=[0, 110], gridcolor=BORDER),
    yaxis=dict(title=None, tickfont=dict(size=11)),
    showlegend=False,
)
st.plotly_chart(fig_det, use_container_width=True)


# ============================================================
# FAULT SCENARIO INSPECTOR — CLICK ANY SCENARIO, SEE REAL SHAP
# ============================================================
st.markdown("---")
st.markdown("### Fault Scenario Inspector")
st.caption(
    "Pick any of the 18 LBNL fault scenarios. The model's prediction and the SHAP waterfall "
    "below are the actual model output on a real test data point — not a rule mapping."
)

# Build scenario options
scenarios['display'] = scenarios.apply(
    lambda r: f"{r['fault_type']} ({r['severity']})  ·  predicted {r['fault_proba']*100:.0f}% fault",
    axis=1
)
scenario_options = scenarios['display'].tolist()

# Default: pick a moderate fault as the demo
default_idx = next(
    (i for i, d in enumerate(scenario_options)
     if 'cooling_fouling_airside' in d and 'severe' in d),
    0
)
selected_display = st.selectbox(
    "Select a fault scenario:",
    options=scenario_options,
    index=default_idx,
)

scenario_idx = scenario_options.index(selected_display)
scenario_row = scenarios.iloc[scenario_idx]
shap_values_row = shap_values_all[scenario_idx]


# Two-column layout: stats left, waterfall right
col_stats, col_waterfall = st.columns([1, 1.5])

with col_stats:
    fault_type = scenario_row['fault_type']
    fault_proba = scenario_row['fault_proba']
    actual = scenario_row['actual']
    severity = scenario_row['severity']

    proba_color = STATUS_ALERT if fault_proba > 0.7 else (STATUS_WATCH if fault_proba > 0.4 else STATUS_HEALTHY)

    st.markdown(
        f"<div style='background:{BG_SECONDARY}; border-left:4px solid {proba_color}; "
        f"border-radius:8px; padding:14px 18px; margin-bottom:10px;'>"
        f"<div style='color:{TEXT_SECONDARY}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;'>Model Prediction</div>"
        f"<div style='color:{TEXT_PRIMARY}; font-size:2.2rem; font-weight:600; margin-top:4px;'>"
        f"{fault_proba*100:.1f}%</div>"
        f"<div style='color:{proba_color}; font-size:0.9rem; margin-top:2px;'>"
        f"{'Faulty' if fault_proba >= 0.5 else 'Healthy'}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div style='background:{BG_SECONDARY}; border-radius:8px; padding:14px 18px; margin-bottom:10px;'>"
        f"<div style='color:{TEXT_SECONDARY}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;'>Ground Truth Label</div>"
        f"<div style='color:{TEXT_PRIMARY}; font-size:1.1rem; font-weight:600; margin-top:4px;'>"
        f"{'Faulty' if actual == 1 else 'Healthy'}</div>"
        f"<div style='color:{TEXT_SECONDARY}; font-size:0.85rem; margin-top:4px;'>"
        f"{fault_type.replace('_', ' ').title()}<br>Severity: {severity}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Correct/incorrect indicator
    correct = (actual == 1 and fault_proba >= 0.5) or (actual == 0 and fault_proba < 0.5)
    if correct:
        st.success(f"✓ Model prediction correct")
    else:
        st.error(f"✗ Model prediction incorrect — see SHAP waterfall for which features misled the prediction")

    # Description
    st.markdown(
        f"<div style='background:{BG_ELEVATED}; padding:12px 14px; border-radius:6px; margin-top:8px;'>"
        f"<div style='color:{TEXT_SECONDARY}; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.05em;'>About this fault class</div>"
        f"<div style='color:{TEXT_PRIMARY}; font-size:0.88rem; margin-top:6px;'>"
        f"{FAULT_TYPE_DESCRIPTIONS.get(fault_type, fault_type)}</div>"
        f"</div>",
        unsafe_allow_html=True
    )


with col_waterfall:
    st.markdown("**SHAP Waterfall — How the Model Reached This Prediction**")
    st.caption(
        "Each feature's contribution to pushing the prediction up (red) or down (blue) "
        "from the model's average output. The top features are the strongest signals."
    )

    # Build SHAP waterfall plot
    explanation = shap.Explanation(
        values=shap_values_row,
        base_values=base_value,
        data=scenario_row[feature_cols].values,
        feature_names=feature_cols,
    )

    # Render to PNG buffer (matplotlib → bytes)
    plt.figure(figsize=(8, 6))
    shap.waterfall_plot(explanation, max_display=12, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor(BG_SECONDARY)

    # Tweak axis colors for dark mode
    for ax in fig.axes:
        ax.set_facecolor(BG_SECONDARY)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=TEXT_SECONDARY)
        ax.xaxis.label.set_color(TEXT_SECONDARY)
        ax.yaxis.label.set_color(TEXT_SECONDARY)
        ax.title.set_color(TEXT_PRIMARY)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',
                facecolor=BG_SECONDARY, dpi=110)
    plt.close()
    buf.seek(0)
    st.image(buf, use_container_width=True)


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================
st.markdown("---")
st.markdown("### Global Feature Importance")
st.caption(
    "What the model learned overall — across the entire test set, which sensors "
    "consistently drive the largest prediction shifts."
)

top_features = global_importance.head(15).iloc[::-1]

fig_imp = go.Figure()
fig_imp.add_trace(go.Bar(
    y=top_features['feature'],
    x=top_features['mean_abs_shap'],
    orientation='h',
    marker=dict(color=ACCENT_PRIMARY),
    hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.3f}<extra></extra>',
))
fig_imp.update_layout(
    template=PLOTLY_TEMPLATE,
    height=480,
    margin=dict(l=20, r=20, t=10, b=20),
    xaxis=dict(title='Mean |SHAP value|', gridcolor=BORDER),
    yaxis=dict(title=None, tickfont=dict(size=11)),
    showlegend=False,
)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown(
    f"<div style='color:{TEXT_SECONDARY}; font-size:0.85rem; padding:12px 16px; "
    f"background:{BG_SECONDARY}; border-radius:6px; margin-top:8px;'>"
    f"<b>Interpretation:</b> The top features are dominated by airflow signals "
    f"(<code>FCU_DA_CFM</code> and its derivatives) and the engineered "
    f"<code>fan_power_per_cfm</code> ratio. This matches HVAC physics — airflow is "
    f"the primary observable signature of cooling-coil fouling, filter restriction, "
    f"and damper blockage faults. These same features drive the symptom-to-fault rules "
    f"on the Building FCU Fleet page, which is why the rule layer is a defensible proxy "
    f"in the absence of internal FCU sensors."
    f"</div>",
    unsafe_allow_html=True
)


# ============================================================
# DATASET PROVENANCE FOOTER
# ============================================================
st.markdown("---")
st.markdown("## Dataset & Model Provenance")

with st.expander("📚 Training Data Provenance", expanded=True):
    st.markdown(
        f"""
        ### LBNL Fault Detection and Diagnostics Dataset — Fan Coil Unit

        - **Source:** Lawrence Berkeley National Laboratory, public domain
        - **DOI:** [10.25984/1881324](https://dx.doi.org/10.25984/1881324)
        - **Citation:** Granderson J, Lin G, Chen Y, Casillas A, et al. *LBNL Fault Detection and Diagnostics Datasets.* DOE Open Energy Data Initiative, 2022.
        - **Simulation engine:** {metrics['simulation_engine']}
        - **Weather basis:** {metrics['weather_basis']}
        - **Fault scenarios:** 48 fault types × multiple severities + 1 fault-free baseline
        - **Total dataset size:** ~3.9 GB (49 CSV files, 1-minute interval)
        - **Used in this project:** 18 cooling-relevant fault scenarios for Dubai context
          (see Notebook 03 for selection rationale)

        Why this dataset is the right benchmark:
        It is the **de facto academic standard for HVAC fault detection research**. The data
        is generated from HVACSIM+ — a high-fidelity HVAC simulator developed by NIST —
        which means fault signatures are physically realistic, not synthetic noise.
        """
    )

with st.expander("🔧 Model Architecture & Training"):
    st.markdown(
        f"""
        - **Model type:** {metrics['model_type']}
        - **Number of features:** {metrics['n_features']}
        - **Estimators:** {metrics['n_estimators']}
        - **Max tree depth:** {metrics['max_depth']}
        - **Learning rate:** {metrics['learning_rate']}
        - **Class balance handling:** `scale_pos_weight` adjusted for ~95/5 imbalance
        - **Train/test split:** Time-based — train on Jan–Sep 2018, test on Oct–Dec 2018
        - **Training rows:** ~{metrics['training_rows']:,} (occupied hours only)
        - **Test rows:** {metrics['test_rows']:,}

        ### Feature engineering
        - Raw FCU sensors (~23 columns from the LBNL dataset)
        - Engineered efficiency ratios: `fan_power_per_cfm`, `airflow_per_valve_open`,
          `valve_command_error`, `chw_temp_delta`, `supply_temp_delta`
        - 6-hour rolling means and standard deviations on key signals
        - Lag features and time deltas

        ### Why time-based split, not random
        Using a random train/test split would leak future information into the training
        set (each minute correlates strongly with the next). Time-based split mirrors
        production: train on past, predict future.
        """
    )

with st.expander("⚠️ Known Limitations of This Benchmark"):
    st.markdown(
        """
        - **Test AUC of 0.997 is exceptional but not generalizable.**
          The data is from a simulator. Faults are clean, well-separated, and labelled.
          Real-world BMS data will be noisier, with sensor drift, intermittent outages,
          and unlabelled faults. **Expected real-world performance: AUC 0.85–0.92.**

        - **Climate mismatch.**
          LBNL data uses Des Moines, Iowa weather (cold winters, mild summers) — including
          heating cycles. We filtered to cooling-relevant faults (Notebook 03), but the
          ambient temperature distribution still differs from Dubai. Recalibration on
          Dubai climate data will be needed in Phase 2.

        - **Single-FCU simulation.**
          Each LBNL file represents one FCU running for one year. The model has not
          seen multi-FCU interactions (e.g. one FCU's fault affecting an adjacent zone).

        - **No fault progression.**
          Each fault is held at constant severity for the full year. Real faults usually
          progress (a filter slowly clogs over weeks). Phase 2 enhancement: incorporate
          fault progression into model training using the FM team's actual maintenance logs.
        """
    )

st.markdown(
    f"<div style='color:{TEXT_SECONDARY}; font-size:0.78rem; margin-top:20px; "
    f"padding:14px; background:{BG_SECONDARY}; border-radius:6px;'>"
    f"<b>Reproducibility:</b> All code in this dashboard is open in the project repository. "
    f"The model artifact (<code>hvac_fault_xgb.pkl</code>), feature columns, SHAP values, "
    f"and metrics shown on this page are all generated from Notebook 04 in the project. "
    f"This page does not regenerate the model — it loads the saved artifact and runs "
    f"the same explainability tooling (SHAP) used in research papers on this dataset."
    f"</div>",
    unsafe_allow_html=True
)
