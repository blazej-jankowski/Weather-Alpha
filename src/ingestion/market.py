import yfinance as yf
import pandas as pd
import logging
from typing import Optional


class MarketDataClient:
    def __init__(self, ticker: str = "KE=F"):
        self.ticker = ticker

    def fetch_historical_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Pobiera dane rynkowe z Yahoo Finance."""
        logging.info(f"Pobieranie danych dla {self.ticker} od {start_date} do {end_date}...")
        try:
            df = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                logging.error("Otrzymano pusty zbiór danych rynkowych.")
                return None

            # POPRAWKA: Spłaszczenie MultiIndex z najnowszych wersji yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Zostawiamy tylko pierwszy poziom (np. 'Open', usuwamy 'KE=F')
                df.columns = df.columns.get_level_values(0)

            # Teraz możemy bezpiecznie filtrować i zmieniać na małe litery
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = [str(col).lower() for col in df.columns]
            df.index.name = 'date'

            df.index = df.index.tz_localize(None)
            return df

        except Exception as e:
            logging.error(f"Błąd podczas pobierania danych: {e}")
            return None