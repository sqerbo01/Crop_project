# src/config.py

BARLEY_CSV_PATH = "data/barley_yield_from_1982.csv"
CLIMATE_PARQUET_PATH = "data/climate_data_from_1982.parquet"

MODEL_WITH_YEAR_PATH = "models/model_with_year.joblib"
MODEL_NO_YEAR_PATH = "models/model_no_year.joblib"

DEFAULT_PRICE_EUR_PER_TONNE = 220.0  # user can edit in the app
DEFAULT_RISK_WINDOW_START = 2036
DEFAULT_RISK_WINDOW_END = 2050

SCENARIOS_FUTURE = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]
BASELINE_SCENARIO_FOR_TRAINING_FUTURE = "ssp2_4_5"
HISTORICAL_SCENARIO = "historical"

# Crop season definition (Sep–Jun)
SEASON_START_MONTH = 9
SEASON_END_MONTH = 6