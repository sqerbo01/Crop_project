# src/risk.py

import numpy as np
import pandas as pd

def compute_historical_mean(barley: pd.DataFrame, year_min: int = 2000, year_max: int = 2018) -> pd.DataFrame:
    mask = (barley["year"] >= year_min) & (barley["year"] <= year_max)
    return (
        barley[mask]
        .groupby("department")["yield"]
        .mean()
        .reset_index()
        .rename(columns={"yield": "historical_mean"})
    )

def compute_area_mean(barley: pd.DataFrame) -> pd.DataFrame:
    return (
        barley.groupby("department")["area"]
        .mean()
        .reset_index()
        .rename(columns={"area": "area_mean"})
    )

def risk_table_from_predictions(
    df_pred: pd.DataFrame,
    scenario: str,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    window = (df_pred["year"] >= year_start) & (df_pred["year"] <= year_end)
    dfw = df_pred[window & (df_pred["scenario"] == scenario)].copy()

    tbl = (
        dfw.groupby("department")["yield_pred"]
        .agg(
            mean="mean",
            std="std",
            p10=lambda x: np.percentile(x, 10),
            worst="min",
        )
        .reset_index()
    )

    # Normalize for risk score
    tbl["p10_norm"] = (tbl["p10"] - tbl["p10"].min()) / (tbl["p10"].max() - tbl["p10"].min())
    tbl["std_norm"] = (tbl["std"] - tbl["std"].min()) / (tbl["std"].max() - tbl["std"].min())

    # Higher risk = low p10 + high volatility
    tbl["risk_score"] = (1 - tbl["p10_norm"]) * 0.7 + tbl["std_norm"] * 0.3
    tbl = tbl.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return tbl

def add_financial_impact(
    risk_tbl: pd.DataFrame,
    historical_mean: pd.DataFrame,
    area_mean: pd.DataFrame,
    price_eur_per_kg: float,
    downside_only: bool,
) -> pd.DataFrame:
    out = risk_tbl.merge(historical_mean, on="department", how="left").merge(area_mean, on="department", how="left")

    out["downside_gap"] = out["historical_mean"] - out["p10"]

    if downside_only:
        out["downside_gap"] = out["downside_gap"].clip(lower=0.0)

    out["loss_kg"] = out["downside_gap"] * out["area_mean"]
    out["loss_eur"] = out["loss_kg"] * price_eur_per_kg
    return out