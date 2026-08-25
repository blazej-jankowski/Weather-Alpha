import pandas as pd
import logging
from entsoe import EntsoePandasClient
from typing import Optional


class WeatherDataClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Brak klucza API dla ENTSO-E! Sprawdź plik .env")
        self.client = EntsoePandasClient(api_key=api_key)
        self.country_code = 'DE_LU'

    def fetch_historical_generation(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Pobiera wyłącznie dane wiatru i słońca rok po roku."""
        logging.info(f"Pobieranie danych wiatru i słońca dla {self.country_code} od {start_date} do {end_date}...")

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
                f"Pobieranie rocznej paczki OZE: {current_start.strftime('%Y-%m-%d')} -> {current_end.strftime('%Y-%m-%d')}...")
            try:
                # Dedykowany, lekki endpoint tylko dla wiatru i słońca
                df = self.client.query_wind_and_solar_forecast(
                    self.country_code,
                    start=current_start,
                    end=current_end,
                    net_load=False
                )

                if df is not None and not df.empty:
                    df_chunk = pd.DataFrame(index=df.index)

                    # Kolumny wiatru i słońca
                    wind_cols = [c for c in df.columns if 'Wind' in str(c)]
                    solar_cols = [c for c in df.columns if 'Solar' in str(c)]

                    df_chunk['wind_generation'] = df[wind_cols].sum(axis=1) if wind_cols else 0.0
                    df_chunk['solar_generation'] = df[solar_cols].sum(axis=1) if solar_cols else 0.0

                    # Agregacja do 1 godziny
                    df_chunk = df_chunk.resample('1h').mean()
                    all_frames.append(df_chunk)
                    logging.info(f"Paczka pobrana: {len(df_chunk)} rekordów godzinowych.")
            except Exception as e:
                logging.warning(f"Błąd dla zakresu {current_start.date()} - {current_end.date()}: {e}")

            current_start = current_end

        if not all_frames:
            logging.error("Nie udało się pobrać danych OZE.")
            return None

        final_df = pd.concat(all_frames)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        final_df.index.name = 'timestamp'
        final_df.index = final_df.index.tz_localize(None)

        return final_df