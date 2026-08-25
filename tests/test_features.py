import pytest
import pandas as pd
import numpy as np
from src.features.processor import FeaturePipeline

@pytest.fixture
def feature_pipeline():
    return FeaturePipeline(db_path="data/weather_alpha.db")

def test_raw_data_loading(feature_pipeline):
    """Verifies that raw database tables are non-empty and retain required schema columns."""
    df_market, df_weather = feature_pipeline._load_raw_data()
    assert not df_market.empty, "Table market_eua_daily is empty!"
    assert not df_weather.empty, "Table weather_ger_hourly is empty!"
    assert 'close' in df_market.columns, "Missing 'close' column in market data."
    assert 'wind_generation' in df_weather.columns, "Missing 'wind_generation' in renewable dataset."

def test_no_nan_values_in_features(feature_pipeline):
    """Ensures engineered feature matrix contains zero missing values (NaN) post rolling window operations."""
    df_features = feature_pipeline.build_features()
    nan_count = df_features.isna().sum().sum()
    assert nan_count == 0, f"Detected {nan_count} NaN values in feature matrix!"

def test_no_lookahead_bias_in_features(feature_pipeline):
    """
    Validates absence of look-ahead bias and data leakage.
    Features constructed at session T must not exhibit deterministic correlation with target return at T+1.
    """
    df_features = feature_pipeline.build_features()
    corr = df_features['target_return_t1'].corr(df_features['eua_return_lag1'])
    assert abs(corr) < 0.90, "Potential look-ahead bias detected in forward target variable!"

def test_target_spike_binary_consistency(feature_pipeline):
    """Validates mathematical consistency of binary price spike classification (> 1.5% threshold)."""
    df_features = feature_pipeline.build_features()
    expected_spikes = (df_features['target_return_t1'] > 0.015).astype(int)
    mismatches = (df_features['target_spike_15bp'] != expected_spikes).sum()
    assert mismatches == 0, "Inconsistency identified in target_spike_15bp threshold evaluation!"