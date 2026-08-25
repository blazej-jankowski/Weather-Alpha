import pytest
import pandas as pd
import numpy as np
from src.features.processor import FeaturePipeline

@pytest.fixture
def feature_pipeline():
    return FeaturePipeline(db_path="data/weather_alpha.db")

def test_raw_data_loading(feature_pipeline):
    """Sprawdza, czy surowe tabele w bazie danych nie są puste i mają kluczowe kolumny."""
    df_market, df_weather = feature_pipeline._load_raw_data()
    assert not df_market.empty, "Tabela market_eua_daily jest pusta!"
    assert not df_weather.empty, "Tabela weather_ger_hourly jest pusta!"
    assert 'close' in df_market.columns, "Brak kolumny 'close' w danych EUA."
    assert 'wind_generation' in df_weather.columns, "Brak kolumny 'wind_generation' w danych OZE."

def test_no_nan_values_in_features(feature_pipeline):
    """Weryfikuje, czy ostateczna macierz cech po rolling window nie zawiera braków danych (NaN)."""
    df_features = feature_pipeline.build_features()
    nan_count = df_features.isna().sum().sum()
    assert nan_count == 0, f"Wykryto {nan_count} braków danych (NaN) w macierzy cech!"

def test_no_lookahead_bias_in_features(feature_pipeline):
    """
    Weryfikuje brak wycieku danych z przyszłości (Look-Ahead Bias).
    Cechy z dnia T nie mogą idealnie korelować ze stopą zwrotu z dnia T+1.
    """
    df_features = feature_pipeline.build_features()
    corr = df_features['target_return_t1'].corr(df_features['eua_return_lag1'])
    assert abs(corr) < 0.90, "Wykryto potencjalny Look-Ahead Bias w zmiennej celu!"

def test_target_spike_binary_consistency(feature_pipeline):
    """Sprawdza spójność matematyczną definicji skoku ceny (próg > 1.5%)."""
    df_features = feature_pipeline.build_features()
    expected_spikes = (df_features['target_return_t1'] > 0.015).astype(int)
    mismatches = (df_features['target_spike_15bp'] != expected_spikes).sum()
    assert mismatches == 0, "Niezgodność w binarnej definicji target_spike_15bp!"