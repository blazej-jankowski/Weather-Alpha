import logging
from src.models.trainer import QuantModelTrainer
from src.backtest.engine import BacktestEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info("Uruchomienie Backtestu Long/Short (Market-Neutral)...")

    # 1. Pobieranie danych i trening modelu
    trainer = QuantModelTrainer(db_path="data/weather_alpha.db", split_date="2023-01-01")
    train_df, test_df = trainer.load_data()
    trainer.train_model(train_df)

    # 2. Generowanie predykcji na zbiorze Out-of-Sample (2023)
    results_df = trainer.evaluate_model(test_df)

    # 3. Uruchomienie symulacji Long/Short
    engine = BacktestEngine(initial_capital=100_000.0, transaction_cost_bps=5.0)

    # Progi: Long >= 0.58 (anomalia braku wiatru), Short <= 0.42 (anomalia nadmiaru wiatru)
    bt_results = engine.run_backtest_long_short(results_df, long_threshold=0.58, short_threshold=0.42)
    metrics = engine.calculate_performance_metrics(bt_results)

    print("\n==================================================")
    print("    WYNIKI STRATEGII LONG/SHORT WEATHER-ALPHA     ")
    print("            OUT-OF-SAMPLE: ROK 2023               ")
    print("==================================================")
    for k, v in metrics.items():
        print(f"{k:<32}: {v}")


if __name__ == "__main__":
    main()