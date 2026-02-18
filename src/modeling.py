# src/modeling.py

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def make_preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

def build_year_folds(df: pd.DataFrame, n_splits: int = 5, test_size_years: int = 2):
    years = np.array(sorted(df["year"].unique()))
    folds = []

    start_k = len(years) - (n_splits * test_size_years)
    for k in range(start_k, len(years), test_size_years):
        train_years = years[:k]
        test_years = years[k:k + test_size_years]
        if len(test_years) < test_size_years:
            continue
        tr_idx = df["year"].isin(train_years).to_numpy().nonzero()[0]
        te_idx = df["year"].isin(test_years).to_numpy().nonzero()[0]
        folds.append((tr_idx, te_idx))

    return folds

def cv_rmse(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    rmses = []
    for tr, te in folds:
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        rmse = np.sqrt(mean_squared_error(y[te], pred))
        rmses.append(rmse)
    return float(np.mean(rmses))

def train_ridge_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    scale_numeric: bool,
    alpha: float = 1.0,
    n_splits: int = 5,
    test_size_years: int = 2,
) -> dict:
    df_use = df.dropna(subset=[target_col] + feature_cols).copy()
    df_use = df_use.sort_values(["year", "department"]).reset_index(drop=True)

    X = df_use[feature_cols]
    y = df_use[target_col].astype(float).to_numpy()

    categorical_features = ["department"]
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    pre = make_preprocessor(numeric_features, categorical_features, scale_numeric=scale_numeric)
    model = Ridge(alpha=alpha)

    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])

    folds = build_year_folds(df_use, n_splits=n_splits, test_size_years=test_size_years)
    rmse_cv = cv_rmse(pipe, X, y, folds)

    # Fit on all data for final artifact
    pipe.fit(X, y)

    return {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "rmse_cv": rmse_cv,
        "n_rows_train": int(df_use.shape[0]),
        "year_min": int(df_use["year"].min()),
        "year_max": int(df_use["year"].max()),
    }