# src/data_loaders.py

import pandas as pd
from .config import BARLEY_CSV_PATH, CLIMATE_PARQUET_PATH

def load_barley() -> pd.DataFrame:
    barley = pd.read_csv(BARLEY_CSV_PATH, sep=";")
    barley = barley.copy()

    if "Unnamed: 0" in barley.columns:
        barley = barley.drop(columns=["Unnamed: 0"])

    # Fill missing yield from production/area when possible
    mask = (
        barley["yield"].isna()
        & barley["production"].notna()
        & barley["area"].notna()
        & (barley["area"] > 0)
    )
    barley.loc[mask, "yield"] = barley.loc[mask, "production"] / barley.loc[mask, "area"]

    return barley

def load_climate() -> pd.DataFrame:
    climate = pd.read_parquet(CLIMATE_PARQUET_PATH)
    return climate