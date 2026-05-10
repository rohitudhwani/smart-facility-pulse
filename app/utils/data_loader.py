"""
Centralized data loading with caching for the smart-facility-pulse dashboard.
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')


@st.cache_data
def load_comfort_sensors():
    df = pd.read_parquet(os.path.join(DATA_DIR, 'synthetic', 'comfort_sensors.parquet'))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data
def load_zones_master():
    return pd.read_parquet(os.path.join(DATA_DIR, 'synthetic', 'zones_master.parquet'))


@st.cache_data
def load_outdoor_weather():
    df = pd.read_parquet(os.path.join(DATA_DIR, 'synthetic', 'outdoor_weather.parquet'))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data
def load_anomaly_log():
    return pd.read_parquet(os.path.join(DATA_DIR, 'synthetic', 'anomaly_log.parquet'))


@st.cache_data
def load_pool_sensors():
    df = pd.read_parquet(os.path.join(DATA_DIR, 'synthetic', 'pool_sensors.parquet'))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


@st.cache_data
def load_pools_master():
    return pd.read_parquet(os.path.join(DATA_DIR, 'synthetic', 'pools_master.parquet'))


@st.cache_resource
def load_hvac_model():
    return joblib.load(os.path.join(MODEL_DIR, 'hvac_fault_xgb.pkl'))


@st.cache_data
def load_feature_columns():
    return joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))


@st.cache_data
def load_scored_test_sample():
    df = pd.read_parquet(os.path.join(MODEL_DIR, 'scored_test_sample.parquet'))
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df


@st.cache_data
def load_shap_base_value():
    return joblib.load(os.path.join(MODEL_DIR, 'shap_base_value.pkl'))


# === New loaders for Page 5 ===
@st.cache_data
def load_representative_scenarios():
    df = pd.read_parquet(os.path.join(MODEL_DIR, 'representative_scenarios.parquet'))
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df


@st.cache_data
def load_shap_values_representative():
    return np.load(os.path.join(MODEL_DIR, 'shap_values_representative.npy'))


@st.cache_data
def load_global_feature_importance():
    return pd.read_parquet(os.path.join(MODEL_DIR, 'global_feature_importance.parquet'))


@st.cache_data
def load_model_metrics():
    return joblib.load(os.path.join(MODEL_DIR, 'model_metrics.pkl'))
