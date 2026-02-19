# app/app.py

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.config import (
    MODEL_WITH_YEAR_PATH,
    MODEL_NO_YEAR_PATH,
    DEFAULT_PRICE_EUR_PER_TONNE,
    DEFAULT_RISK_WINDOW_START,
    DEFAULT_RISK_WINDOW_END,
    SCENARIOS_FUTURE,
    SCENARIO_LABELS,
)
from src.data_loaders import load_barley, load_climate
from src.features import build_seasonal_features
from src.risk import (
    compute_historical_mean,
    compute_area_mean,
    risk_table_from_predictions,
    add_financial_impact,
)

st.set_page_config(page_title="Barley Climate App", layout="wide")


@st.cache_data
def get_data():
    barley = load_barley()
    climate = load_climate()
    return barley, climate


@st.cache_resource
def load_models():
    m_with = joblib.load(MODEL_WITH_YEAR_PATH)
    m_no = joblib.load(MODEL_NO_YEAR_PATH)
    return m_with, m_no


def scenario_id_from_label(label: str) -> str:
    reverse = {v: k for k, v in SCENARIO_LABELS.items()}
    if label not in reverse:
        raise ValueError(f"Unknown scenario label: {label}")
    return reverse[label]


def plot_single_scenario_line(df_plot: pd.DataFrame, title: str):
    fig = plt.figure(figsize=(10, 4.5))
    plt.plot(df_plot["year"], df_plot["yield_pred"])
    plt.title(title)
    plt.xlabel("Year (crop year / harvest)")
    plt.ylabel("Predicted yield (t/ha)")
    plt.tight_layout()
    st.pyplot(fig)


def build_future_predictions(
    climate: pd.DataFrame,
    departments: set[str],
    scenario: str,
    year_min: int,
    year_max: int,
    model_artifact: dict,
) -> pd.DataFrame:
    df_future = build_seasonal_features(
        climate=climate,
        departments=departments,
        scenarios=[scenario],
        crop_year_min=year_min,
        crop_year_max=year_max,
        include_precip_sum=True,
    )

    feature_cols = model_artifact["feature_cols"]
    pipe = model_artifact["pipeline"]

    missing = [c for c in feature_cols if c not in df_future.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in future table: {missing}")

    df_future = df_future.copy()
    df_future["yield_pred"] = pipe.predict(df_future[feature_cols])
    return df_future


def metrics_from_department_view(yield_t_ha: float, area_ha: float, price_eur_t: float):
    tonnes = yield_t_ha * area_ha
    eur = tonnes * price_eur_t

    c1, c2, c3 = st.columns(3)
    c1.metric("Yield (t/ha)", f"{yield_t_ha:.2f}")
    c2.metric("Production (t)", f"{tonnes:,.0f}")
    c3.metric("Value (€)", f"{eur:,.0f}")


def metrics_from_all_departments_view(df_year_dep: pd.DataFrame, area_mean_tbl: pd.DataFrame, price_eur_t: float):
    tmp = df_year_dep.merge(area_mean_tbl, on="department", how="left")
    tmp["production_t"] = tmp["yield_pred"] * tmp["area_mean"]

    total_area = float(tmp["area_mean"].sum())
    total_production = float(tmp["production_t"].sum())
    yield_weighted = total_production / total_area if total_area > 0 else np.nan
    total_value = total_production * price_eur_t

    c1, c2, c3 = st.columns(3)
    c1.metric("Area (ha)", f"{total_area:,.0f}")
    c2.metric("Production (t)", f"{total_production:,.0f}")
    c3.metric("Value (€)", f"{total_value:,.0f}")

    st.caption(f"Area-weighted yield: {yield_weighted:.2f} t/ha")


def add_climate_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds interpretable climate indices using only the available metrics.
    """

    out = df.copy()

    temp_k_col = "near_surface_air_temperature__value_mean"
    precip_col = "precipitation__season_SepJun__sum"

    if temp_k_col in out.columns:
        out["temp_mean_c"] = out[temp_k_col] - 273.15
    else:
        out["temp_mean_c"] = np.nan

    if precip_col in out.columns:
        out["precip_season_mm"] = out[precip_col]
    else:
        out["precip_season_mm"] = np.nan

    # Water stress index: normalized dryness proxy (0..1 within the selected dataset)
    dryness_proxy = (out["temp_mean_c"].clip(lower=-10) + 10.0) / (
        out["precip_season_mm"].clip(lower=0.0) + 1.0
    )

    d = dryness_proxy
    denom = (d.max() - d.min())
    out["water_stress_index"] = (d - d.min()) / denom if denom != 0 else 0.0

    return out


def slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Robust slope estimate for y ~ a*x + b.
    Filters non-finite values and guards against low-variance inputs.
    Returns NaN when the slope is not reliably computable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 5:
        return float("nan")

    # Require variation in x (otherwise slope is undefined / unstable)
    if np.nanstd(x) < 1e-8:
        return float("nan")

    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return float("nan")


def compute_department_sensitivities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sensitivities computed from predicted yields over time within the chosen window:
    - climate_sensitivity_temp: slope of yield_pred vs temp_mean_c
    - climate_sensitivity_precip: slope of yield_pred vs precip_season_mm
    """
    rows = []
    for dep, g in df.groupby("department"):
        g = g.copy()

        # Keep only rows where we can compute indices + sensitivity safely
        g = g[np.isfinite(g["yield_pred"])].copy()

        x_t = g["temp_mean_c"].to_numpy(dtype=float)
        x_p = g["precip_season_mm"].to_numpy(dtype=float)
        y = g["yield_pred"].to_numpy(dtype=float)

        rows.append(
            {
                "department": dep,
                "climate_sensitivity_temp": slope(x_t, y),
                "climate_sensitivity_precip": slope(x_p, y),
                "avg_water_stress_index": float(np.nanmean(g["water_stress_index"].to_numpy(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def main():
    st.title("Barley yield under climate scenarios")

    barley, climate = get_data()
    model_with, model_no = load_models()

    departments = sorted(barley["department"].astype(str).unique())
    department_options = ["All"] + departments
    dept_set = set(departments)

    area_mean_tbl = compute_area_mean(barley)

    scenario_labels = [SCENARIO_LABELS[s] for s in SCENARIOS_FUTURE]

    with st.sidebar:
        st.header("Controls")

        dep = st.selectbox(
            "Department",
            department_options,
            index=department_options.index("Yvelines") if "Yvelines" in department_options else 0,
        )

        scenario_label = st.selectbox(
            "Scenario",
            scenario_labels,
            index=scenario_labels.index(SCENARIO_LABELS["ssp2_4_5"]),
        )
        scenario = scenario_id_from_label(scenario_label)

        year_min = st.slider("Start year", 2019, 2050, 2019)
        year_max = st.slider("End year", 2019, 2050, 2050)

        price = st.number_input(
            "Price (€/tonne)",
            min_value=0.0,
            value=float(DEFAULT_PRICE_EUR_PER_TONNE),
            step=10.0,
        )

        if dep != "All":
            default_area = float(barley[barley["department"] == dep]["area"].mean())
            area = st.number_input("Area (ha)", min_value=0.0, value=default_area, step=100.0)
        else:
            area = None
            st.number_input(
                "Area (ha)",
                min_value=0.0,
                value=float(area_mean_tbl["area_mean"].sum()),
                step=100.0,
                disabled=True,
            )

    # Tab names updated as requested
    tab1, tab2, tab3 = st.tabs(["Forecast", "Climate-only", "Risk"])

    # ---------- TAB 1 ----------
    with tab1:
        st.subheader("Forecast")

        df_future = build_future_predictions(
            climate=climate,
            departments=dept_set,
            scenario=scenario,
            year_min=year_min,
            year_max=year_max,
            model_artifact=model_with,
        )

        if dep == "All":
            st.caption("Portfolio-level view across all departments (area-weighted).")

            df_plot = (
                df_future.merge(area_mean_tbl, on="department", how="left")
                .assign(production_t=lambda d: d["yield_pred"] * d["area_mean"])
                .groupby("year", as_index=False)
                .agg(total_production_t=("production_t", "sum"), total_area=("area_mean", "sum"))
            )
            df_plot["yield_pred"] = df_plot["total_production_t"] / df_plot["total_area"]
            df_plot = df_plot[["year", "yield_pred"]].sort_values("year")

            last_year = int(df_plot["year"].max())
            df_last = df_future[df_future["year"] == last_year][["department", "yield_pred"]].copy()
            metrics_from_all_departments_view(df_last, area_mean_tbl, float(price))

            plot_single_scenario_line(df_plot, f"Yield projection – All departments – {scenario_label}")

        else:
            df_dep = df_future[df_future["department"] == dep].copy()
            if df_dep.empty:
                st.warning("No data for the selected department.")
            else:
                df_plot = df_dep.groupby("year", as_index=False)["yield_pred"].mean().sort_values("year")

                last_year = int(df_plot["year"].max())
                last_yield = float(df_plot[df_plot["year"] == last_year]["yield_pred"].iloc[0])
                metrics_from_department_view(last_yield, float(area), float(price))

                plot_single_scenario_line(df_plot, f"Yield projection – {dep} – {scenario_label}")

    # ---------- TAB 2 ----------
    with tab2:
        st.subheader("Climate-only")

        df_future = build_future_predictions(
            climate=climate,
            departments=dept_set,
            scenario=scenario,
            year_min=year_min,
            year_max=year_max,
            model_artifact=model_no,
        )

        if dep == "All":
            st.caption("Portfolio-level view across all departments (area-weighted).")

            df_plot = (
                df_future.merge(area_mean_tbl, on="department", how="left")
                .assign(production_t=lambda d: d["yield_pred"] * d["area_mean"])
                .groupby("year", as_index=False)
                .agg(total_production_t=("production_t", "sum"), total_area=("area_mean", "sum"))
            )
            df_plot["yield_pred"] = df_plot["total_production_t"] / df_plot["total_area"]
            df_plot = df_plot[["year", "yield_pred"]].sort_values("year")

            last_year = int(df_plot["year"].max())
            df_last = df_future[df_future["year"] == last_year][["department", "yield_pred"]].copy()
            metrics_from_all_departments_view(df_last, area_mean_tbl, float(price))

            plot_single_scenario_line(df_plot, f"Climate-only yield projection – All departments – {scenario_label}")

        else:
            df_dep = df_future[df_future["department"] == dep].copy()
            if df_dep.empty:
                st.warning("No data for the selected department.")
            else:
                df_plot = df_dep.groupby("year", as_index=False)["yield_pred"].mean().sort_values("year")

                last_year = int(df_plot["year"].max())
                last_yield = float(df_plot[df_plot["year"] == last_year]["yield_pred"].iloc[0])
                metrics_from_department_view(last_yield, float(area), float(price))

                plot_single_scenario_line(df_plot, f"Climate-only yield projection – {dep} – {scenario_label}")

    # ---------- TAB 3 ----------
    with tab3:
        st.subheader("Risk ranking (portfolio view)")

        c1, c2, c3 = st.columns(3)
        scen_risk_label = c1.selectbox(
            "Risk scenario",
            scenario_labels,
            index=scenario_labels.index(SCENARIO_LABELS["ssp2_4_5"]),
        )
        scen_risk = scenario_id_from_label(scen_risk_label)

        win_start = c2.number_input("Window start", min_value=2019, max_value=2050, value=int(DEFAULT_RISK_WINDOW_START))
        win_end = c3.number_input("Window end", min_value=2019, max_value=2050, value=int(DEFAULT_RISK_WINDOW_END))

        # IMPORTANT CHANGE:
        # We do NOT clip improvements anymore. Negative loss means projected improvement vs historical mean.
        downside_only = False

        df_future_all = build_future_predictions(
            climate=climate,
            departments=dept_set,
            scenario=scen_risk,
            year_min=int(win_start),
            year_max=int(win_end),
            model_artifact=model_no,
        )

        df_future_all = add_climate_indices(df_future_all)

        df_pred = df_future_all[["department", "year", "scenario", "yield_pred"]].copy()

        risk_tbl = risk_table_from_predictions(
            df_pred,
            scenario=scen_risk,
            year_start=int(win_start),
            year_end=int(win_end),
        )

        hist_mean = compute_historical_mean(barley, year_min=2000, year_max=2018)

        risk_tbl = add_financial_impact(
            risk_tbl=risk_tbl,
            historical_mean=hist_mean,
            area_mean=area_mean_tbl,
            price_eur_per_tonne=float(price),
            downside_only=downside_only,
        )

        # Add climate indices and sensitivities (derived from available data)
        sens = compute_department_sensitivities(df_future_all)
        risk_tbl = risk_tbl.merge(sens, on="department", how="left")

        if dep != "All":
            snap = risk_tbl[risk_tbl["department"] == dep]
            if not snap.empty:
                s = snap.iloc[0]
                st.markdown("#### Selected department snapshot")
                a, b, c, d = st.columns(4)
                a.metric("Severe-year yield (p10, t/ha)", f"{s['p10']:.2f}")
                b.metric("Downside gap (t/ha)", f"{s['downside_gap']:.2f}")
                c.metric("Loss (t)", f"{s['loss_tonnes']:,.0f}")
                d.metric("Loss (€)", f"{s['loss_eur']:,.0f}")

        st.markdown("#### Ranking")

        # Remove risk_score from display, rename p10 to a clearer label,
        # and sort by Loss tonnes (descending = biggest losses first).
        display_tbl = risk_tbl.copy()
        display_tbl = display_tbl.sort_values("loss_tonnes", ascending=False).reset_index(drop=True)

        display_tbl = display_tbl.rename(
            columns={
                "p10": "severe_year_yield_p10",
                "worst": "worst_year_yield",
            }
        )

        show_cols = [
        "department",
        "mean",
        "std",
        "severe_year_yield_p10",
        "worst_year_yield",
        "historical_mean",
        "downside_gap",
        "area_mean",
        "loss_tonnes",
        "loss_eur",
        "avg_water_stress_index",
        "climate_sensitivity_temp",
        "climate_sensitivity_precip",
    ]

        st.dataframe(display_tbl[show_cols], use_container_width=True)

        st.caption(
            "Notes: 'severe_year_yield_p10' is the 10th percentile of projected yields within the window. "
            "If 'downside_gap' or 'loss' are negative, it means projected improvement vs historical mean (opportunity)."
        )


if __name__ == "__main__":
    main()