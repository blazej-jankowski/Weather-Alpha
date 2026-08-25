import sqlite3
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Tworzy tabele, jeśli nie istnieją."""
        query_market = """
        CREATE TABLE IF NOT EXISTS market_eua_daily (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        );
        """
        query_weather = """
        CREATE TABLE IF NOT EXISTS weather_ger_hourly (
            timestamp TEXT PRIMARY KEY,
            wind_generation REAL,
            solar_generation REAL
        );
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query_market)
            cursor.execute(query_weather)
            logging.info("Inicjalizacja bazy danych zakończona.")

    def save_dataframe(self, df: pd.DataFrame, table_name: str):
        """Zapisuje DataFrame do bazy SQLite."""
        if df.empty:
            logging.warning(f"Brak danych do zapisu w tabeli {table_name}.")
            return

        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=True)
            logging.info(f"Zapisano {len(df)} wierszy do tabeli {table_name}.")