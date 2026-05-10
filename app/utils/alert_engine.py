"""
Threshold-based alert logic for Modules A and B.
Plus symptom-to-fault mapping for Module C hybrid scoring.
Pure functions — no Streamlit dependencies.
"""
import pandas as pd
import numpy as np


# === COMFORT THRESHOLDS (from project spec, ASHRAE-aligned) ===
COMFORT_THRESHOLDS = {
    'indoor_temp': {'good': (22, 25), 'watch': (25, 27), 'alert_above': 27},
    'humidity':    {'good': (40, 60), 'watch_low': 30, 'watch_high': 70},
    'co2':         {'good': 800, 'watch': 1000},
    'pm25':        {'good': 12, 'watch': 35},
    'co':          {'good': 4, 'watch': 9},
}

# === POOL THRESHOLDS (CDC-aligned) ===
POOL_THRESHOLDS = {
    'pH':        {'good': (7.2, 7.8)},
    'chlorine':  {'min': 1.0, 'good_max': 3.0},
    'turbidity': {'good': 0.5, 'watch': 1.0, 'alert': 2.0},
    'water_temp': {'good': (26, 30)},
    'service_overdue_hours': 200,
}


def classify_comfort(row):
    temp = row['indoor_temp']; co2 = row['co2']; pm25 = row['pm25']
    co = row['co']; humidity = row['humidity']
    if (temp > COMFORT_THRESHOLDS['indoor_temp']['alert_above']
            or co2 > COMFORT_THRESHOLDS['co2']['watch']
            or pm25 > COMFORT_THRESHOLDS['pm25']['watch']
            or co > COMFORT_THRESHOLDS['co']['watch']):
        return 'Alert'
    if (temp > COMFORT_THRESHOLDS['indoor_temp']['good'][1]
            or co2 > COMFORT_THRESHOLDS['co2']['good']
            or pm25 > COMFORT_THRESHOLDS['pm25']['good']
            or humidity < COMFORT_THRESHOLDS['humidity']['watch_low']
            or humidity > COMFORT_THRESHOLDS['humidity']['watch_high']):
        return 'Watch'
    return 'Normal'


def get_comfort_issues(row):
    issues = []
    if row['indoor_temp'] > COMFORT_THRESHOLDS['indoor_temp']['alert_above']:
        issues.append(f"Temperature too high ({row['indoor_temp']:.1f}°C)")
    elif row['indoor_temp'] > COMFORT_THRESHOLDS['indoor_temp']['good'][1]:
        issues.append(f"Temperature elevated ({row['indoor_temp']:.1f}°C)")
    if row['co2'] > COMFORT_THRESHOLDS['co2']['watch']:
        issues.append(f"CO₂ high — ventilation issue ({row['co2']:.0f} ppm)")
    elif row['co2'] > COMFORT_THRESHOLDS['co2']['good']:
        issues.append(f"CO₂ elevated ({row['co2']:.0f} ppm)")
    if row['pm25'] > COMFORT_THRESHOLDS['pm25']['watch']:
        issues.append(f"PM2.5 high — possible filter issue ({row['pm25']:.1f} µg/m³)")
    if row['co'] > COMFORT_THRESHOLDS['co']['watch']:
        issues.append(f"CO elevated ({row['co']:.1f} ppm)")
    return issues


def detect_stuck_sensor(zone_df, hours=24):
    recent = zone_df.sort_values('timestamp').tail(hours)
    if len(recent) < hours:
        return False
    return recent['indoor_temp'].std() < 0.01


def classify_pool(row):
    issues = get_pool_issues(row)
    if not issues:
        return 'Normal'
    serious = [i for i in issues if 'low' in i.lower() or 'overdue' in i.lower() or 'out of range' in i.lower()]
    if len(issues) >= 2 or serious:
        return 'Alert'
    return 'Watch'


def get_pool_issues(row):
    issues = []
    pH_min, pH_max = POOL_THRESHOLDS['pH']['good']
    if row['pH'] < pH_min or row['pH'] > pH_max:
        issues.append(f"pH out of range ({row['pH']:.2f})")
    if row['chlorine'] < POOL_THRESHOLDS['chlorine']['min']:
        issues.append(f"Chlorine low ({row['chlorine']:.2f} ppm)")
    if row['turbidity'] > POOL_THRESHOLDS['turbidity']['watch']:
        issues.append(f"Turbidity high ({row['turbidity']:.2f} NTU)")
    if row['hours_since_service'] > POOL_THRESHOLDS['service_overdue_hours']:
        issues.append(f"Service overdue ({int(row['hours_since_service'])}h)")
    return issues


def predict_pool_service_hours(pool_history):
    recent = pool_history.sort_values('timestamp').tail(24)
    if len(recent) < 12:
        return None, None
    chlorine_values = recent['chlorine'].values
    chlorine_slope = np.polyfit(range(len(chlorine_values)), chlorine_values, 1)[0]
    turb_values = recent['turbidity'].values
    turb_slope = np.polyfit(range(len(turb_values)), turb_values, 1)[0]
    current_cl = chlorine_values[-1]
    current_turb = turb_values[-1]
    hours_to_cl_breach = (current_cl - POOL_THRESHOLDS['chlorine']['min']) / abs(chlorine_slope) if chlorine_slope < 0 else 9999
    hours_to_turb_breach = (POOL_THRESHOLDS['turbidity']['alert'] - current_turb) / turb_slope if turb_slope > 0 else 9999
    if hours_to_cl_breach < hours_to_turb_breach:
        return max(0, int(hours_to_cl_breach)), 'chlorine'
    return max(0, int(hours_to_turb_breach)), 'turbidity'


# ============================================================
# === SYMPTOM → FAULT MAPPING (Module C Hybrid Scoring) ===
# ============================================================
# Maps zone-level symptoms (Module A) to LBNL fault types and probability scores.
# Each rule corresponds to a documented fault category in the LBNL FCU dataset.
#
# WHY THIS APPROACH:
# The LBNL XGBoost model is trained on FCU-internal sensors (valve position, airflow,
# fan power, chilled water flow) which are NOT present in zone-level Module A data.
# Until BMS data is connected, this rule layer infers the most likely LBNL fault
# class from observed zone symptoms, with severity scored on a 0–1 scale.
#
# Once the actual BMS feed is available, every FCU will be scored directly by the
# trained XGBoost model and this rule layer becomes redundant.

FAULT_TYPE_DESCRIPTIONS = {
    'fault_free': 'Operating normally',
    'cooling_fouling_airside': 'Cooling coil fouled (air-side) — heat transfer degraded',
    'cooling_fouling_waterside': 'Cooling coil fouled (water-side) — chilled water flow restricted',
    'filter_restriction': 'Air filter clogged — airflow restricted',
    'cooling_valve_stuck': 'Cooling valve stuck — not modulating to demand',
    'oa_inlet_blockage': 'Outside air inlet blocked — ventilation reduced',
    'fan_outlet_blockage': 'Fan outlet blocked — supply airflow restricted',
    'sensor_bias': 'Zone temperature sensor reporting biased value',
    'control_unstable': 'FCU control loop unstable — oscillating',
}

RECOMMENDED_ACTIONS = {
    'cooling_fouling_airside': [
        'Inspect cooling coil surface for dust / biofilm buildup',
        'Check and clean condensate drain pan',
        'Verify airflow path is unobstructed',
        'Schedule chemical coil cleaning if buildup is severe',
    ],
    'cooling_fouling_waterside': [
        'Check chilled water inlet/outlet temperature delta',
        'Inspect for scale buildup on water-side of coil',
        'Verify chilled water pump operation',
        'Test chilled water chemistry and treatment',
    ],
    'filter_restriction': [
        'Inspect air filter — replace if visibly loaded',
        'Check filter pressure differential reading',
        'Document filter replacement date',
        'Review filter replacement schedule for this zone',
    ],
    'cooling_valve_stuck': [
        'Check cooling coil valve position vs command',
        'Cycle valve manually to confirm mechanical operation',
        'Inspect actuator wiring and signal',
        'Replace valve actuator if mechanical fault confirmed',
    ],
    'oa_inlet_blockage': [
        'Inspect outside air intake louvres for debris / nesting',
        'Check OA damper operation through full travel',
        'Verify OA damper actuator response to BMS command',
        'Clear any obstruction at intake',
    ],
    'fan_outlet_blockage': [
        'Inspect fan outlet path for obstruction',
        'Check supply diffuser dampers',
        'Verify fan speed signal and operation',
        'Listen for unusual fan noise or vibration',
    ],
    'sensor_bias': [
        'Compare zone sensor reading to portable reference instrument',
        'Inspect sensor for physical damage or contamination',
        'Verify sensor wiring and BMS configuration',
        'Recalibrate or replace sensor if bias confirmed',
    ],
    'control_unstable': [
        'Review zone PID tuning parameters',
        'Check for conflicting control signals',
        'Inspect for setpoint hunting in trend logs',
    ],
    'fault_free': [
        'No action required — FCU operating within normal parameters',
    ],
}


def map_zone_to_fcu_fault(zone_row, zone_history_24h, outdoor_pm25_recent_avg):
    """
    Map a zone's current state to most likely FCU fault type and probability.

    Returns dict with:
        fault_type:    str (one of FAULT_TYPE_DESCRIPTIONS keys)
        fault_proba:   float 0–1
        severity:      str ('minor'|'moderate'|'severe'|'none')
        rule_fired:    str — human-readable explanation of why
        contributing_signals: list[str] — which symptoms drove the score
    """
    temp = zone_row['indoor_temp']
    co2 = zone_row['co2']
    pm25 = zone_row['pm25']
    setpoint = zone_row.get('temp_setpoint', 24.0)

    # Check for stuck sensor (overrides everything else)
    if zone_history_24h is not None and len(zone_history_24h) >= 24:
        if zone_history_24h['indoor_temp'].std() < 0.01:
            return {
                'fault_type': 'sensor_bias',
                'fault_proba': 0.95,
                'severity': 'severe',
                'rule_fired': 'Zone temperature sensor frozen for 24+ hours',
                'contributing_signals': ['Indoor temp std = 0 over last 24 hours'],
            }

    # Compute symptom severities (each on 0–1 scale)
    temp_excess = max(0, temp - setpoint - 1.0)         # >1°C above setpoint
    temp_severity = min(1.0, temp_excess / 4.0)         # severe at +5°C above setpoint

    co2_excess = max(0, co2 - 800)
    co2_severity = min(1.0, co2_excess / 600)            # severe at 1400 ppm

    expected_indoor_pm25 = 0.3 * outdoor_pm25_recent_avg
    pm_excess = max(0, pm25 - expected_indoor_pm25 - 5)
    pm_severity = min(1.0, pm_excess / 25)                # severe at filter very loaded

    # ---- Combined signals → cooling valve stuck ----
    # If BOTH temp climbing AND CO2 climbing significantly, valve may be stuck closed
    # (FCU can't cool, can't bring in OA — full failure mode)
    if temp_severity > 0.4 and co2_severity > 0.4:
        proba = min(0.95, 0.55 + 0.35 * max(temp_severity, co2_severity))
        return {
            'fault_type': 'cooling_valve_stuck',
            'fault_proba': proba,
            'severity': 'severe' if proba > 0.85 else 'moderate',
            'rule_fired': 'Combined cooling failure + ventilation failure',
            'contributing_signals': [
                f'Temp {temp:.1f}°C ({temp - setpoint:+.1f}°C from setpoint)',
                f'CO₂ {co2:.0f} ppm ({co2 - 800:+.0f} above ventilation threshold)',
            ],
        }

    # ---- Temperature only → cooling coil fouling (air-side) ----
    if temp_severity > 0.3:
        proba = min(0.95, 0.40 + 0.50 * temp_severity)
        severity_label = 'severe' if temp_severity > 0.7 else ('moderate' if temp_severity > 0.5 else 'minor')
        return {
            'fault_type': 'cooling_fouling_airside',
            'fault_proba': proba,
            'severity': severity_label,
            'rule_fired': 'Indoor temperature persistently above setpoint, ventilation normal',
            'contributing_signals': [
                f'Temp {temp:.1f}°C ({temp - setpoint:+.1f}°C from setpoint)',
                f'CO₂ {co2:.0f} ppm (within normal range)',
            ],
        }

    # ---- CO2 only → ventilation issue (OA blockage or fan outlet blockage) ----
    if co2_severity > 0.3:
        proba = min(0.95, 0.40 + 0.50 * co2_severity)
        severity_label = 'severe' if co2_severity > 0.7 else ('moderate' if co2_severity > 0.5 else 'minor')
        # Pick more specific fault — fan outlet vs OA inlet
        # If PM2.5 is also slightly elevated, more likely OA inlet (no fresh air filtering)
        # If PM2.5 normal, more likely fan outlet blockage (less air movement overall)
        fault_type = 'oa_inlet_blockage' if pm_severity > 0.2 else 'fan_outlet_blockage'
        return {
            'fault_type': fault_type,
            'fault_proba': proba,
            'severity': severity_label,
            'rule_fired': 'Elevated CO₂ — insufficient ventilation, temperature controlled',
            'contributing_signals': [
                f'CO₂ {co2:.0f} ppm (elevated)',
                f'Temp {temp:.1f}°C (within normal range)',
            ],
        }

    # ---- PM2.5 only → filter restriction ----
    if pm_severity > 0.3:
        proba = min(0.92, 0.35 + 0.50 * pm_severity)
        severity_label = 'severe' if pm_severity > 0.7 else ('moderate' if pm_severity > 0.5 else 'minor')
        return {
            'fault_type': 'filter_restriction',
            'fault_proba': proba,
            'severity': severity_label,
            'rule_fired': 'Indoor PM2.5 elevated relative to outdoor-attenuated baseline',
            'contributing_signals': [
                f'PM2.5 {pm25:.1f} µg/m³',
                f'Outdoor PM2.5 24h avg: {outdoor_pm25_recent_avg:.1f} µg/m³',
                f'Filter efficiency appears degraded',
            ],
        }

    # ---- No symptoms — healthy ----
    # Small baseline noise so the ranking isn't flat
    rng = np.random.RandomState(hash(zone_row['zone_id']) % (2**31))
    baseline = rng.uniform(0.02, 0.12)
    return {
        'fault_type': 'fault_free',
        'fault_proba': baseline,
        'severity': 'none',
        'rule_fired': 'No symptoms detected — operating within normal parameters',
        'contributing_signals': [
            f'Temp {temp:.1f}°C (normal)',
            f'CO₂ {co2:.0f} ppm (normal)',
            f'PM2.5 {pm25:.1f} µg/m³ (normal)',
        ],
    }
