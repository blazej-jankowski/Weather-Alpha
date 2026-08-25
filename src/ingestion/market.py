import yfinance as yf
import pandas as pd
import logging
from typing import Optional


class MarketDataClient:
    def __init__(self, ticker: str = "KE=F"):
        self.ticker = ticker

    def fetch_historical_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetches historical market price data from Yahoo Finance."""
        logging.info(f"Fetching market data for ticker {self.ticker} from {start_date} to {end_date}...")
        try:
            df = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                logging.error("Received empty market data response.")
                return None

            # Flatten MultiIndex columns introduced in recent yfinance releases
            if isinstance(df.columns, pd.MultiIndex):
                # Retain primary level (e.g. 'Open', dropping 'KE=F')
                df.columns = df.columns.get_level_values(0)

            # Subset core OHLCV columns and standardize to lowercase schema
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = [str(col).lower() for col in df.columns]
            df.index.name = 'date'

            df.index = df.index.tz_localize(None)
            return df

        except Exception as e:
            logging.error(f"Error occurred while fetching market data: {e}")
            return None