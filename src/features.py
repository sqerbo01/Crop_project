# src/features.py

import numpy as np
import pandas as pd
from .config import SEASON_START_MONTH, SEASON_END_MONTH

AGG_STATS = ["mean", "std", "min", "max"]

def _add_crop_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["crop_year"] = df["year"] + (df["month"] >= SEASON_START_MONTH).astype(int)
    return df

def _filter_season_sep_jun(df: pd.DataFrame) -> pd.DataFrame:
    # Keep Sep..Dec OR Jan..Jun
    return df[(df["month"] >= SEASON_START_MONTH) | (df["month"] <= SEASON_END_MONTH)].copy()

def build_seasonal_features(
    climate: pd.DataFrame,
    departments: set[str],
    scenarios: list[str],
    crop_year_min: int,
    crop_year_max: int,
    include_precip_sum: bool = True,
) -> pd.DataFrame:
    """
    Output columns:
    - department, year, scenario
    - <metric>__value_<stat> for stat in {mean,std,min,max}
    - precipitation__season_SepJun__sum (optional)
    """
    cl = climate.copy()
    cl = cl[cl["nom_dep"].astype(str).isin(departments)].copy()
    cl = cl[cl["scenario"].isin(scenarios)].copy()

    cl = _add_crop_year(cl)
    cl = _filter_season_sep_jun(cl)

    cl = cl[(cl["crop_year"] >= crop_year_min) & (cl["crop_year"] <= crop_year_max)].copy()

    cl_season = (
        cl.groupby(["nom_dep", "crop_year", "scenario", "metric"], observed=True)["value"]
          .agg(AGG_STATS)
          .reset_index()
    )

    wide = (
        cl_season.pivot_table(
            index=["nom_dep", "crop_year", "scenario"],
            columns="metric",
            values=AGG_STATS,
            aggfunc="first"
        )
    )

    wide.columns = [f"{metric}__value_{stat}" for stat, metric in wide.columns]
    wide = (
        wide.reset_index()
            .rename(columns={"nom_dep": "department", "crop_year": "year"})
    )

    if include_precip_sum:
        precip_sum = (
            cl[cl["metric"] == "precipitation"]
            .groupby(["nom_dep", "crop_year", "scenario"], observed=True)["value"]
            .sum()
            .reset_index()
            .rename(columns={
                "nom_dep": "department",
                "crop_year": "year",
                "value": "precipitation__season_SepJun__sum",
            })
        )
        wide = wide.merge(precip_sum, on=["department", "year", "scenario"], how="left", validate="m:1")

    return wide

def build_training_table(
    barley: pd.DataFrame,
    climate: pd.DataFrame,
    baseline_scenario_future: str,
) -> pd.DataFrame:
    """
    Merge barley (1982..2018) with baseline climate:
    - historical up to 2014
    - baseline_scenario_future from 2015..max_year
    """
    barley_key = barley[["department", "year"]].dropna().drop_duplicates()
    min_year = int(barley_key["year"].min())
    max_year = int(barley_key["year"].max())
    deps = set(barley_key["department"].astype(str).unique())

    # Select only historical + chosen scenario for future years (for consistent training table)
    cl = climate.copy()
    cl = cl[cl["nom_dep"].astype(str).isin(deps)].copy()

    cl["year"] = cl["time"].dt.year
    cl = cl[
        ((cl["scenario"] == "historical") & (cl["year"] <= 2014)) |
        ((cl["scenario"] == baseline_scenario_future) & (cl["year"] >= 2015) & (cl["year"] <= max_year))
    ].copy()

    # Build Sep–Jun seasonal features (scenario is not meaningful for training after selection)
    cl = cl.drop(columns=["scenario"])
    cl = _add_crop_year(cl)
    cl = _filter_season_sep_jun(cl)

    cl_season = (
        cl.groupby(["nom_dep", "crop_year", "metric"], observed=True)["value"]
          .agg(AGG_STATS)
          .reset_index()
    )

    wide = (
        cl_season.pivot_table(
            index=["nom_dep", "crop_year"],
            columns="metric",
            values=AGG_STATS,
            aggfunc="first"
        )
    )

    wide.columns = [f"{metric}__value_{stat}" for stat, metric in wide.columns]
    wide = wide.reset_index().rename(columns={"nom_dep": "department", "crop_year": "year"})

    precip_sum = (
        cl[cl["metric"] == "precipitation"]
        .groupby(["nom_dep", "crop_year"], observed=True)["value"]
        .sum()
        .reset_index()
        .rename(columns={"nom_dep": "department", "crop_year": "year", "value": "precipitation__season_SepJun__sum"})
    )

    wide = wide.merge(precip_sum, on=["department", "year"], how="left", validate="1:1")

    df_merged = barley.merge(wide, on=["department", "year"], how="left", validate="m:1")
    return df_merged