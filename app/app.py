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

def plot_lines(df_plot: pd.DataFrame, scenarios: list[str], title: str):
    wide = df_plot.pivot(index="year", columns="scenario", values="yield_pred").sort_index()
    ymin, ymax = wide.min(axis=1), wide.max(axis=1)

    fig = plt.figure(figsize=(10, 4.5))
    for scen in scenarios:
        if scen in wide.columns:
            plt.plot(wide.index, wide[scen], label=scen)
    plt.fill_between(wide.index, ymin, ymax, alpha=0.2)
    plt.title(title)
    plt.xlabel("Year (crop year / harvest)")
    plt.ylabel("Predicted yield (t/ha)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

def build_future_predictions(
    climate: pd.DataFrame,
    departments: set[str],
    scenarios: list[str],
    year_min: int,
    year_max: int,
    model_artifact: dict,
) -> pd.DataFrame:
    df_future = build_seasonal_features(
        climate=climate,
        departments=departments,
        scenarios=scenarios,
        crop_year_min=year_min,
        crop_year_max=year_max,
        include_precip_sum=True,
    )

    feature_cols = model_artifact["feature_cols"]
    pipe = model_artifact["pipeline"]

    # Some models include 'year' in feature cols; ensure it's present
    missing = [c for c in feature_cols if c not in df_future.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in future table: {missing}")

    df_future = df_future.copy()
    df_future["yield_pred"] = pipe.predict(df_future[feature_cols])
    return df_future

def metric_row(yield_t_ha: float, area_ha: float, price_eur_t: float):
    tonnes = yield_t_ha * area_ha
    eur = tonnes * price_eur_t
    c1, c2, c3 = st.columns(3)
    c1.metric("Yield (t/ha)", f"{yield_t_ha:.2f}")
    c2.metric("Production (t)", f"{tonnes:,.0f}")
    c3.metric("Value (€)", f"{eur:,.0f}")

def main():
    st.title("Barley yield under climate scenarios")

    barley, climate = get_data()
    model_with, model_no = load_models()

    departments = sorted(barley["department"].astype(str).unique())
    dept_set = set(departments)

    with st.sidebar:
        st.header("Controls")
        dep = st.selectbox("Department", departments, index=departments.index("Yvelines") if "Yvelines" in departments else 0)

        scenarios = st.multiselect("Scenarios", SCENARIOS_FUTURE, default=SCENARIOS_FUTURE)

        year_min = st.slider("Start year", 2019, 2050, 2019)
        year_max = st.slider("End year", 2019, 2050, 2050)

        price = st.number_input("Price (€/tonne)", min_value=0.0, value=float(DEFAULT_PRICE_EUR_PER_TONNE), step=10.0)

        area_mean = float(barley[barley["department"] == dep]["area"].mean())
        area = st.number_input("Area (ha)", min_value=0.0, value=area_mean, step=100.0)

    tab1, tab2, tab3 = st.tabs(["Forecast (with Year)", "Climate-only (no Year)", "Risk"])

    # ---------- TAB 1 ----------
    with tab1:
        st.subheader("Forecast (model includes time trend via 'year')")
        df_future = build_future_predictions(
            climate=climate,
            departments=dept_set,
            scenarios=scenarios,
            year_min=year_min,
            year_max=year_max,
            model_artifact=model_with,
        )

        df_dep = df_future[df_future["department"] == dep].copy()
        if df_dep.empty:
            st.warning("No data for the selected department.")
        else:
            last_year = df_dep["year"].max()
            last_row = df_dep[df_dep["year"] == last_year].groupby("year")["yield_pred"].mean().iloc[0]
            metric_row(last_row, area, price)

            # Plot department lines
            df_plot = df_dep.groupby(["year", "scenario"], as_index=False)["yield_pred"].mean()
            plot_lines(df_plot, scenarios, f"Yield projection – {dep}")

    # ---------- TAB 2 ----------
    with tab2:
        st.subheader("Climate-only (model excludes 'year' to isolate climate signal)")
        df_future = build_future_predictions(
            climate=climate,
            departments=dept_set,
            scenarios=scenarios,
            year_min=year_min,
            year_max=year_max,
            model_artifact=model_no,
        )

        df_dep = df_future[df_future["department"] == dep].copy()
        if df_dep.empty:
            st.warning("No data for the selected department.")
        else:
            last_year = df_dep["year"].max()
            last_row = df_dep[df_dep["year"] == last_year].groupby("year")["yield_pred"].mean().iloc[0]
            metric_row(last_row, area, price)

            df_plot = df_dep.groupby(["year", "scenario"], as_index=False)["yield_pred"].mean()
            plot_lines(df_plot, scenarios, f"Climate-only yield projection – {dep}")

    # ---------- TAB 3 ----------
    with tab3:
        st.subheader("Risk ranking (portfolio view)")

        c1, c2, c3 = st.columns(3)
        scen_risk = c1.selectbox("Risk scenario", SCENARIOS_FUTURE, index=SCENARIOS_FUTURE.index("ssp2_4_5") if "ssp2_4_5" in SCENARIOS_FUTURE else 0)
        win_start = c2.number_input("Window start", min_value=2019, max_value=2050, value=int(DEFAULT_RISK_WINDOW_START))
        win_end = c3.number_input("Window end", min_value=2019, max_value=2050, value=int(DEFAULT_RISK_WINDOW_END))

        downside_only = st.toggle("Downside only (ignore improvements)", value=True)

        # Use climate-only model by default for risk (more defensible for scenarios)
        df_future_all = build_future_predictions(
            climate=climate,
            departments=dept_set,
            scenarios=[scen_risk],
            year_min=int(win_start),
            year_max=int(win_end),
            model_artifact=model_no,
        )

        # Risk table from predictions
        df_pred = df_future_all[["department", "year", "scenario", "yield_pred"]].copy()
        risk_tbl = risk_table_from_predictions(df_pred, scenario=scen_risk, year_start=int(win_start), year_end=int(win_end))

        hist_mean = compute_historical_mean(barley, year_min=2000, year_max=2018)
        area_mean_tbl = compute_area_mean(barley)

        risk_tbl = add_financial_impact(
            risk_tbl=risk_tbl,
            historical_mean=hist_mean,
            area_mean=area_mean_tbl,
            price_eur_per_tonne=float(price),
            downside_only=bool(downside_only),
        )

        # Highlight selected department snapshot
        snap = risk_tbl[risk_tbl["department"] == dep]
        if not snap.empty:
            s = snap.iloc[0]
            st.markdown("#### Selected department snapshot")
            a, b, c, d = st.columns(4)
            a.metric("Risk score", f"{s['risk_score']:.3f}")
            b.metric("p10 (t/ha)", f"{s['p10']:.2f}")
            c.metric("Downside gap (t/ha)", f"{s['downside_gap']:.2f}")
            d.metric("Loss (€)", f"{s['loss_eur']:,.0f}")

        st.markdown("#### Ranking")
        show_cols = ["department", "risk_score", "mean", "std", "p10", "worst", "historical_mean", "downside_gap", "area_mean", "loss_tonnes", "loss_eur"]
        st.dataframe(risk_tbl[show_cols], use_container_width=True)

if __name__ == "__main__":
    main()