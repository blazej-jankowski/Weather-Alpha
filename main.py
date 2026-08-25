import logging
import pandas as pd
from sklearn.inspection import permutation_importance
from src.models.trainer import QuantModelTrainer
from src.backtest.engine import BacktestEngine
from src.reports.visualizer import PerformanceVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info("Starting systematic strategy execution and tearsheet generation...")

    # 1. Load data partitions and fit predictive model
    trainer = QuantModelTrainer(db_path="data/weather_alpha.db", split_date="2023-01-01")
    train_df, test_df = trainer.load_data()
    trainer.train_model(train_df)
    results_df = trainer.evaluate_model(test_df)

    # 2. Compute permutation feature importance
    X_test = test_df[trainer.feature_cols]
    y_test = test_df[trainer.target_col]
    perm = permutation_importance(trainer.model, X_test, y_test, n_repeats=10, random_state=42)
    importance_df = pd.DataFrame({
        'Feature': trainer.feature_cols,
        'Importance': perm.importances_mean
    })

    # 3. Execute comparative backtesting simulation
    engine = BacktestEngine(initial_capital=100_000.0, transaction_cost_bps=5.0)
    bt_comparison = engine.run_comparison_backtest(results_df, long_threshold=0.60, short_threshold=0.42)
    summary_table = engine.calculate_metrics_summary(bt_comparison)

    # Print comparative performance summary to stdout
    print("\n==========================================================================================")
    print("                 QUANTITATIVE STRATEGY PERFORMANCE COMPARISON (2023 OOS)                   ")
    print("==========================================================================================")
    print(summary_table.to_string())
    print("==========================================================================================\n")

    # 4. Generate and persist comprehensive tearsheet PNG
    visualizer = PerformanceVisualizer(output_dir="reports")
    visualizer.generate_comparison_tearsheet(bt_comparison, importance_df,
                                             output_name="performance_comparison_tearsheet.png")


if __name__ == "__main__":
    main()