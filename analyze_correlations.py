import sqlite3
import pandas as pd
import numpy as np


def run_correlation_analysis():
    db_path = "data/weather_alpha.db"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM features_daily", conn)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Wybór cech wejściowych (X) oraz targetów (Y)
    feature_cols = [
        'oze_prior_sum',
        'wind_prior_std',
        'oze_lag1',
        'oze_lag2',
        'oze_lag3',
        'oze_zscore_14d',
        'eua_return_lag1',
        'eua_volatility_5d'
    ]

    target_cols = ['target_return_t1', 'target_return_t2', 'target_spike_15bp']

    print("==================================================")
    print("       ANALIZA KORELACJI CECH Z TARGETAMI         ")
    print("==================================================")

    # 1. Korelacja Pearsona
    corr_pearson = df[feature_cols + target_cols].corr(method='pearson')
    print("\n[1] KORELACJA PEARSONA (Liniowa) z target_return_t1:")
    print(corr_pearson.loc[feature_cols, 'target_return_t1'].sort_values())

    # 2. Korelacja Spearmana (Rangi)
    corr_spearman = df[feature_cols + target_cols].corr(method='spearman')
    print("\n[2] KORELACJA SPEARMANA (Monotoniczna) z target_return_t1:")
    print(corr_spearman.loc[feature_cols, 'target_return_t1'].sort_values())

    # 3. Warunkowa średnia stóp zwrotu podczas anomalii pogodowych
    print("\n--------------------------------------------------")
    print("[3] EFEKT ANOMALII POGODOWYCH (Dunkelflaute vs. Nadprodukcja):")

    dunkelflaute_mask = df['oze_zscore_14d'] < -1.0  # Załamanie wiatru/słońca
    high_wind_mask = df['oze_zscore_14d'] > 1.0  # Nadprodukcja OZE

    mean_ret_dunkelflaute = df.loc[dunkelflaute_mask, 'target_return_t1'].mean() * 100
    mean_ret_high_wind = df.loc[high_wind_mask, 'target_return_t1'].mean() * 100
    mean_ret_all = df['target_return_t1'].mean() * 100

    print(f"- Średni zwrot EUA (T+1) ogółem:                     {mean_ret_all:+.3f}%")
    print(f"- Średni zwrot EUA (T+1) po Dunkelflaute (Z < -1):   {mean_ret_dunkelflaute:+.3f}%")
    print(f"- Średni zwrot EUA (T+1) po Nadprodukcji (Z > +1):   {mean_ret_high_wind:+.3f}%")


if __name__ == "__main__":
    run_correlation_analysis()