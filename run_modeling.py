import logging
from src.models.trainer import QuantModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    trainer = QuantModelTrainer(db_path="data/weather_alpha.db", split_date="2023-01-01")

    # 1. Załadowanie i podział danych
    train_df, test_df = trainer.load_data()

    # 2. Trening
    trainer.train_model(train_df)

    # 3. Ewaluacja na niewidzianym roku 2023
    results_df = trainer.evaluate_model(test_df)


if __name__ == "__main__":
    main()