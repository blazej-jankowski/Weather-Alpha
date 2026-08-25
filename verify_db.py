import sqlite3
import pandas as pd


def audit_database():
    db_path = "data/weather_alpha.db"

    with sqlite3.connect(db_path) as conn:
        df_market = pd.read_sql_query("SELECT * FROM market_eua_daily", conn)
        df_weather = pd.read_sql_query("SELECT * FROM weather_ger_hourly", conn)

    print("==================================================")
    print("           AUDYT BAZY: WEATHER_ALPHA.DB           ")
    print("==================================================")

    # 1. TABELA RYNKOWA (EUA)
    print("\n[1] TABELA: market_eua_daily")
    print(f"Liczba rekordów: {len(df_market)}")
    print(f"Zakres dat: {df_market['date'].min()} -> {df_market['date'].max()}")
    print("Braki danych (NaN):")
    print(df_market.isna().sum())
    print("\nStatystyki opisowe cen zamknięcia (Close):")
    print(df_market['close'].describe())

    # 2. TABELA POGODOWA (OZE)
    print("\n--------------------------------------------------")
    print("[2] TABELA: weather_ger_hourly")
    print(f"Liczba rekordów: {len(df_weather)}")
    print(f"Zakres timestampów: {df_weather['timestamp'].min()} -> {df_weather['timestamp'].max()}")
    print("Braki danych (NaN):")
    print(df_weather.isna().sum())
    print("\nStatystyki opisowe generacji OZE [MW]:")
    print(df_weather[['wind_generation', 'solar_generation']].describe())

    # 3. KONTROLA JAKOŚCI DANYCH (Sanity Checks)
    print("\n--------------------------------------------------")
    print("[3] DIAGNOSTYKA INTEGRALNOŚCI:")

    neg_prices = (df_market['close'] <= 0).sum()
    neg_wind = (df_weather['wind_generation'] < 0).sum()
    neg_solar = (df_weather['solar_generation'] < 0).sum()

    print(f"- Ceny EUA <= 0: {neg_prices} (Oczekiwane: 0)")
    print(f"- Generacja wiatru < 0: {neg_wind} (Oczekiwane: 0)")
    print(f"- Generacja słońca < 0: {neg_solar} (Oczekiwane: 0)")

    if neg_prices == 0 and neg_wind == 0 and neg_solar == 0 and df_market.isna().sum().sum() == 0:
        print("\nSTATUS: Zbiór danych czysty. Gotowy do KROKU 2.")
    else:
        print("\nSTATUS: Wykryto anomalie wymagające wyczyszczenia.")


if __name__ == "__main__":
    audit_database()