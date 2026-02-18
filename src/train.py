# src/train.py

import joblib

from .config import (
    MODEL_WITH_YEAR_PATH,
    MODEL_NO_YEAR_PATH,
    BASELINE_SCENARIO_FOR_TRAINING_FUTURE,
)
from .data_loaders import load_barley, load_climate
from .features import build_training_table
from .modeling import train_ridge_model

def main():
    barley = load_barley()
    climate = load_climate()

    df = build_training_table(barley, climate, baseline_scenario_future=BASELINE_SCENARIO_FOR_TRAINING_FUTURE)

    target_col = "yield"
    climate_cols = [c for c in df.columns if "__" in c]

    # Model A: with year
    feature_cols_with_year = ["department", "year"] + climate_cols

    # Model B: no year
    feature_cols_no_year = ["department"] + climate_cols

    artifact_with_year = train_ridge_model(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols_with_year,
        scale_numeric=True,
        alpha=1.0,
        n_splits=5,
        test_size_years=2,
    )

    artifact_no_year = train_ridge_model(
        df=df,
        target_col=target_col,
        feature_cols=feature_cols_no_year,
        scale_numeric=True,
        alpha=1.0,
        n_splits=5,
        test_size_years=2,
    )

    joblib.dump(artifact_with_year, MODEL_WITH_YEAR_PATH)
    joblib.dump(artifact_no_year, MODEL_NO_YEAR_PATH)

    print("Saved:", MODEL_WITH_YEAR_PATH, "RMSE_CV:", artifact_with_year["rmse_cv"])
    print("Saved:", MODEL_NO_YEAR_PATH, "RMSE_CV:", artifact_no_year["rmse_cv"])

if __name__ == "__main__":
    main()