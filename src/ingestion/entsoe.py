import pandas as pd
import logging
from entsoe import EntsoePandasClient
from typing import Optional


class WeatherDataClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Missing ENTSO-E API key! Check your .env file.")
        self.client = EntsoePandasClient(api_key=api_key)
        self.country_code = 'DE_LU'

    def fetch_historical_generation(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetches historical wind and solar actual generation data year-by-year."""
        logging.info(f"Fetching renewable generation for {self.country_code} from {start_date} to {end_date}...")

        years = pd.date_range(start=start_date, end=end_date, freq='YE')
        if len(years) == 0 or years[-1] < pd.Timestamp(end_date):
            years = years.append(pd.DatetimeIndex([pd.Timestamp(end_date)]))

        all_frames = []
        current_start = pd.Timestamp(start_date, tz='Europe/Berlin')

        for target_end in years:
            current_end = pd.Timestamp(target_end, tz='Europe/Berlin')
            if current_start >= current_end:
                continue

            logging.info(
                f"Fetching annual renewable batch: {current_start.strftime('%Y-%m-%d')} -> {current_end.strftime('%Y-%m-%d')}...")
            try:
                # Dedicated lightweight endpoint for wind and solar forecasts/actuals
                df = self.client.query_wind_and_solar_forecast(
                    self.country_code,
                    start=current_start,
                    end=current_end,
                    net_load=False
                )

                if df is not None and not df.empty:
                    df_chunk = pd.DataFrame(index=df.index)

                    # Wind and solar aggregate columns
                    wind_cols = [c for c in df.columns if 'Wind' in str(c)]
                    solar_cols = [c for c in df.columns if 'Solar' in str(c)]

                    df_chunk['wind_generation'] = df[wind_cols].sum(axis=1) if wind_cols else 0.0
                    df_chunk['solar_generation'] = df[solar_cols].sum(axis=1) if solar_cols else 0.0

                    # Resample to 1-hour resolution
                    df_chunk = df_chunk.resample('1h').mean()
                    all_frames.append(df_chunk)
                    logging.info(f"Batch downloaded successfully: {len(df_chunk)} hourly records.")
            except Exception as e:
                logging.warning(f"Error fetching range {current_start.date()} - {current_end.date()}: {e}")

            current_start = current_end

        if not all_frames:
            logging.error("Failed to retrieve renewable generation time series.")
            return None

        final_df = pd.concat(all_frames)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        final_df.index.name = 'timestamp'
        final_df.index = final_df.index.tz_localize(None)

        return final_df