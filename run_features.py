import logging
from src.features.processor import FeaturePipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info("Uruchomienie KROKU 2: Feature Engineering & Signal Generation...")
    pipeline = FeaturePipeline(db_path="data/weather_alpha.db")

    # 1. Budowa macierzy cech
    df_features = pipeline.build_features()

    # 2. Zapis do SQLite
    pipeline.save_features_to_db(df_features, table_name="features_daily")

    # 3. Podgląd wygenerowanych cech
    print("\n--- PODGLĄD MACIERZY CECH (Pierwsze 3 wiersze) ---")
    print(
        df_features[['eua_close', 'oze_prior_sum', 'oze_zscore_14d', 'target_return_t1', 'target_spike_15bp']].head(3))

    print("\n--- ROZKŁAD ZMIENNEJ CELU (target_spike_15bp) ---")
    print(df_features['target_spike_15bp'].value_counts(normalize=True))


if __name__ == "__main__":
    main()