import sqlite3
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QuantModelTrainer:
    def __init__(self, db_path: str = "data/weather_alpha.db", split_date: str = "2023-01-01"):
        self.db_path = db_path
        self.split_date = split_date
        self.feature_cols = [
            'oze_prior_sum',
            'wind_prior_std',
            'oze_lag1',
            'oze_lag2',
            'oze_lag3',
            'oze_zscore_14d',
            'eua_return_lag1',
            'eua_volatility_5d'
        ]
        self.target_col = 'target_spike_15bp'
        self.model = None

    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads feature matrix and performs chronological train/test split."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM features_daily", conn)

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')

        train_df = df.loc[df.index < self.split_date].copy()
        test_df = df.loc[df.index >= self.split_date].copy()

        logging.info(
            f"In-Sample training set: {len(train_df)} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
        logging.info(
            f"Out-of-Sample test set: {len(test_df)} rows ({test_df.index.min().date()} to {test_df.index.max().date()})")

        return train_df, test_df

    def train_model(self, train_df: pd.DataFrame) -> HistGradientBoostingClassifier:
        """Trains Gradient Boosting Classifier with regularization against overfitting."""
        X_train = train_df[self.feature_cols]
        y_train = train_df[self.target_col]

        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            learning_rate=0.03,
            class_weight='balanced',
            random_state=42
        )

        self.model.fit(X_train, y_train)
        logging.info("Gradient Boosting model trained successfully.")
        return self.model

    def evaluate_model(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluates predictions on the Out-of-Sample test set (2023)."""
        X_test = test_df[self.feature_cols]
        y_test = test_df[self.target_col]

        # Predicted price spike probabilities
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        print("\n==================================================")
        print("          OUT-OF-SAMPLE RESULTS (YEAR 2023)       ")
        print("==================================================")
        print(f"ROC-AUC Score: {auc:.4f}")

        print("\n--- CLASSIFICATION REPORT ---")
        print(classification_report(y_test, y_pred, digits=4))

        # Permutation Feature Importance (robust evaluation of feature relevance)
        perm_importance = permutation_importance(self.model, X_test, y_test, n_repeats=10, random_state=42)
        importance_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': perm_importance.importances_mean
        }).sort_values('Importance', ascending=False)

        print("\n--- FEATURE IMPORTANCE (Permutation Importance) ---")
        print(importance_df.to_string(index=False))

        # Attach predictions for downstream backtest engine
        results_df = test_df.copy()
        results_df['signal_prob'] = y_prob
        results_df['signal_binary'] = y_pred
        return results_df