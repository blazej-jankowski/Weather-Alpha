import logging
from src.ingestion.db_manager import DatabaseManager
from src.ingestion.market import MarketDataClient
from src.ingestion.entsoe import WeatherDataClient
from src.config import ENTSOE_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info("Rozpoczęcie procesu Data Ingestion...")
    db_manager = DatabaseManager(db_path="data/weather_alpha.db")

    # 1. POBIERANIE DANYCH RYNKOWYCH
    market_client = MarketDataClient(ticker="KE=F")
    market_df = market_client.fetch_historical_data(start_date="2020-01-01", end_date="2024-01-01")
    if market_df is not None:
        db_manager.save_dataframe(market_df, table_name="market_eua_daily")

    # 2. POBIERANIE DANYCH POGODOWYCH (OZE)
    if ENTSOE_API_KEY:
        weather_client = WeatherDataClient(api_key=ENTSOE_API_KEY)
        weather_df = weather_client.fetch_historical_generation(start_date="2020-01-01", end_date="2024-01-01")
        if weather_df is not None:
            db_manager.save_dataframe(weather_df, table_name="weather_ger_hourly")
    else:
        logging.warning("Pominięto ENTSO-E: Brak klucza API w pliku .env.")

    logging.info("Krok 1 zakończony.")


if __name__ == "__main__":
    main()