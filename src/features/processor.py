import sqlite3
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FeaturePipeline:
    def __init__(self, db_path: str = "data/weather_alpha.db"):
        self.db_path = db_path

    def _load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetches raw market and weather tables from SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            df_market = pd.read_sql_query("SELECT * FROM market_eua_daily", conn)
            df_weather = pd.read_sql_query("SELECT * FROM weather_ger_hourly", conn)

        df_market['date'] = pd.to_datetime(df_market['date'])
        df_market = df_market.sort_values('date').set_index('date')

        df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
        df_weather = df_weather.sort_values('timestamp').set_index('timestamp')

        return df_market, df_weather

    def _process_weather_daily(self, df_weather: pd.DataFrame) -> pd.DataFrame:
        """Aggregates hourly renewable generation to daily resolution with intraday volatility."""
        # 1. Total and average generation within 24h window
        daily_sum = df_weather.resample('D').sum()
        daily_mean = df_weather.resample('D').mean()
        daily_std = df_weather.resample('D').std()

        df_daily_weather = pd.DataFrame(index=daily_sum.index)
        df_daily_weather['wind_daily_sum'] = daily_sum['wind_generation']
        df_daily_weather['solar_daily_sum'] = daily_sum['solar_generation']
        df_daily_weather['oze_total_sum'] = daily_sum['wind_generation'] + daily_sum['solar_generation']

        # Intraday volatility metrics
        df_daily_weather['wind_intraday_std'] = daily_std['wind_generation']
        df_daily_weather['oze_mean_mw'] = daily_mean['wind_generation'] + daily_mean['solar_generation']

        return df_daily_weather

    def _handle_weekend_asymmetry(self, df_daily_weather: pd.DataFrame,
                                  trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Aggregates weekend renewable generation (Saturday + Sunday) and maps it to Monday session.
        For Tuesday-Friday, retains generation from the previous day (T-1).
        """
        aligned_weather = pd.DataFrame(index=trading_dates)

        # Monday flag (dayofweek == 0)
        is_monday = aligned_weather.index.dayofweek == 0

        # Construct weather features prior to market open on day T
        # Computes cumulative generation since previous market close
        oze_prior_session = []
        oze_std_prior_session = []

        for current_date in trading_dates:
            if current_date.dayofweek == 0:  # Monday -> aggregate Friday, Saturday, Sunday
                window_start = current_date - pd.Timedelta(days=3)
            else:  # Tuesday-Friday -> prior day only (T-1)
                window_start = current_date - pd.Timedelta(days=1)

            window_end = current_date - pd.Timedelta(days=1)
            subset = df_daily_weather.loc[window_start:window_end]

            oze_prior_session.append(subset['oze_total_sum'].sum())
            oze_std_prior_session.append(subset['wind_intraday_std'].mean())

        aligned_weather['oze_prior_session_sum'] = oze_prior_session
        aligned_weather['wind_prior_session_std'] = oze_std_prior_session

        return aligned_weather

    def build_features(self) -> pd.DataFrame:
        """Main feature engineering and target variable generation pipeline."""
        logging.info("Loading raw tables from database...")
        df_market, df_weather = self._load_raw_data()

        logging.info("Aggregating hourly weather time series...")
        df_daily_weather = self._process_weather_daily(df_weather)

        logging.info("Mapping weekend generation asymmetry to trading calendar...")
        aligned_weather = self._handle_weekend_asymmetry(df_daily_weather, df_market.index)

        logging.info("Engineering statistical features and anomaly indicators...")
        df_features = pd.DataFrame(index=df_market.index)

        # 1. Historical EUA prices and returns (Market Lags)
        df_features['eua_close'] = df_market['close']
        df_features['eua_return_lag1'] = df_market['close'].pct_change(1)
        df_features['eua_volatility_5d'] = df_features['eua_return_lag1'].rolling(window=5).std()

        # 2. Weather features prior to market session / weekend aggregate
        df_features['oze_prior_sum'] = aligned_weather['oze_prior_session_sum']
        df_features['wind_prior_std'] = aligned_weather['wind_prior_session_std']

        # 3. Renewable lags (T-1, T-2, T-3)
        df_features['oze_lag1'] = df_features['oze_prior_sum']
        df_features['oze_lag2'] = df_features['oze_prior_sum'].shift(1)
        df_features['oze_lag3'] = df_features['oze_prior_sum'].shift(2)

        # 4. Renewable Anomaly Indicator (Rolling 14-day Z-Score for Dunkelflaute regime)
        roll_mean_14 = df_features['oze_prior_sum'].rolling(window=14).mean()
        roll_std_14 = df_features['oze_prior_sum'].rolling(window=14).std()
        df_features['oze_zscore_14d'] = (df_features['oze_prior_sum'] - roll_mean_14) / roll_std_14

        # 5. Target Variables
        # Forward Close return at horizon T+1
        df_features['target_return_t1'] = df_market['close'].pct_change(1).shift(-1)
        # Forward Close return at horizon T+2
        df_features['target_return_t2'] = df_market['close'].pct_change(2).shift(-2)
        # Binary directional spike at T+1 (> +1.5%)
        df_features['target_spike_15bp'] = (df_features['target_return_t1'] > 0.015).astype(int)

        # Drop warm-up NaN rows from rolling windows and forward-looking shifts
        df_features_clean = df_features.dropna().copy()

        logging.info(
            f"Feature pipeline completed. Output matrix: {df_features_clean.shape[0]} rows x {df_features_clean.shape[1]} columns.")
        return df_features_clean

    def save_features_to_db(self, df: pd.DataFrame, table_name: str = "features_daily"):
        """Persists engineered feature matrix to SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=True)
            logging.info(f"Persisted table '{table_name}' to database at {self.db_path}.")