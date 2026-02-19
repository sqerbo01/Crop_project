# app/app.py

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import geopandas as gpd
import unicodedata
from matplotlib.colors import TwoSlopeNorm

from src.config import (
    MODEL_WITH_YEAR_PATH,
    MODEL_NO_YEAR_PATH,
    DEFAULT_PRICE_EUR_PER_kg,
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


def norm_name(s):
    if pd.isna(s):
        return s
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = s.replace(" ", "_").replace("-", "_").replace("'", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")

def plot_risk_map(
    deps_gdf,
    risk_tbl,
    metric="loss_kg",
    title=None,
    diverging_center_zero=True,
    highlight_dep: str | None = None,
):
    risk_map = risk_tbl[["department", metric]].copy()
    risk_map["dep_key"] = risk_map["department"].map(norm_name)

    plot_gdf = deps_gdf.merge(risk_map[["dep_key", metric]], on="dep_key", how="left")

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    cmap = "RdYlGn_r"

    norm = None
    if diverging_center_zero:
        vals = plot_gdf[metric].dropna()
        if len(vals) > 0:
            vmin = np.percentile(vals, 5)
            vmax = np.percentile(vals, 95)
            if vmin < 0 < vmax:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plot_gdf.plot(
        column=metric,
        cmap=cmap,
        norm=norm,
        legend=False,
        linewidth=0.3,
        edgecolor="white",
        missing_kwds={"color": "lightgrey"},
        ax=ax,
    )

    # ✅ Highlight selected department
    if highlight_dep and highlight_dep != "All":
        key = norm_name(highlight_dep)
        sel = plot_gdf[plot_gdf["dep_key"] == key]
        if not sel.empty:
            sel.boundary.plot(ax=ax, linewidth=2.5, edgecolor="black")
            # optional: add label in centroid
            try:
                cx = sel.geometry.centroid.x.values[0]
                cy = sel.geometry.centroid.y.values[0]
                ax.text(cx, cy, highlight_dep, fontsize=7, ha="center", va="center")
            except Exception:
                pass

    if title:
        ax.set_title(title, fontsize=10)

    ax.axis("off")
    plt.tight_layout()
    return fig


@st.cache_resource
def load_deps_geo():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"

    gdf = gpd.read_file(url)

    # In this geojson, the department name column is "nom"
    name_col = "nom"
    if name_col not in gdf.columns:
        raise ValueError(f"Expected '{name_col}' in geojson columns. Found: {list(gdf.columns)}")

    gdf["dep_key"] = gdf[name_col].map(norm_name)

    return gdf


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
    plt.ylabel("Predicted yield (kg/ha)")
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

def compute_thresholds(risk_tbl: pd.DataFrame) -> dict:
    # Robust quantile thresholds (change q levels if you want)
    q = lambda col, p: float(risk_tbl[col].dropna().quantile(p)) if col in risk_tbl else np.nan

    return {
        "risk_high": q("risk_score", 0.80),
        "risk_low":  q("risk_score", 0.20),

        "loss_high": q("loss_kg", 0.80),        # big losses
        "loss_low":  q("loss_kg", 0.20),

        "ws_high":   q("avg_water_stress_index", 0.80),
        "ws_low":    q("avg_water_stress_index", 0.20),

        "temp_sens_high": q("climate_sensitivity_temp", 0.80),
        "prec_sens_high": q("climate_sensitivity_precip", 0.80),
    }


def recommendations_portfolio(risk_tbl: pd.DataFrame, th: dict) -> str:
    # sort tables
    most_risky = risk_tbl.sort_values(["risk_score", "loss_kg"], ascending=False)
    safest = risk_tbl.sort_values(["risk_score", "avg_water_stress_index"], ascending=True)

    top10_risk = most_risky.head(10)["department"]
    alt_lowrisk = safest.head(8)["department"]

    high_ws = risk_tbl[risk_tbl["avg_water_stress_index"] >= th["ws_high"]].sort_values(
        "avg_water_stress_index", ascending=False
    )["department"].head(8)

    # optional: "opportunity" = projected improvement vs historical (negative loss)
    opp = risk_tbl[risk_tbl["loss_kg"] < 0].sort_values("loss_kg").head(5)["department"]

    txt = f"""
    **Recommended sourcing shifts (rule-based)**

    **1) Reallocate volumes away from the highest-risk departments (Top 10):**  
    {top_n_list(top10_risk, 10)}  
    → Reduce medium-term contract exposure and prioritize mitigation-only support (no expansion).

    **2) Prioritize contracting in structurally resilient departments (low risk + low water stress):**  
    {top_n_list(alt_lowrisk, 8)}  
    → Use these as “anchor basins” for longer-term sourcing and multi-year contracts.

    **3) Avoid expansion in high water-stress departments (likely higher irrigation costs):**  
    {top_n_list(high_ws, 8)}  
    → Cap incremental volumes; focus on water-efficient practices if sourcing is unavoidable.

    **4) Quick-win opportunities (projected improvement vs historical mean):**  
    {top_n_list(opp, 5)}  
    → Consider selective upside allocation where climate conditions improve.
    """
    return txt

def recommendations_department(risk_tbl: pd.DataFrame, dep: str, th: dict) -> str:
    row = risk_tbl[risk_tbl["department"] == dep]
    if row.empty:
        return "No recommendation available (department not found)."

    s = row.iloc[0]

    risk = float(s.get("risk_score", np.nan))
    loss = float(s.get("loss_kg", np.nan))
    ws   = float(s.get("avg_water_stress_index", np.nan))

    flags = []
    actions = []

    # Risk / loss
    if np.isfinite(risk) and risk >= th["risk_high"]:
        flags.append("**High structural climate risk**")
        actions.append("Avoid increasing volumes; renegotiate contracts with flexibility (volume bands / optionality).")

    if np.isfinite(loss) and loss >= th["loss_high"]:
        flags.append("**High downside exposure (kg)**")
        actions.append("Prioritize mitigation: agronomic support, varietal trials, and contingency sourcing plans.")

    if np.isfinite(loss) and loss < 0:
        flags.append("**Projected opportunity vs historical**")
        actions.append("Consider selective volume increase, but keep monitoring volatility and water stress.")

    # Water stress
    if np.isfinite(ws) and ws >= th["ws_high"]:
        flags.append("**High water stress**")
        actions.append("Limit exposure to irrigation costs: cap expansion and require water-efficiency commitments from growers.")
    elif np.isfinite(ws) and ws <= th["ws_low"]:
        flags.append("**Low water stress**")
        actions.append("Good candidate for longer-term contracting (more stable cost profile).")

    # Sensitivities (optional)
    t_sens = float(s.get("climate_sensitivity_temp", np.nan))
    p_sens = float(s.get("climate_sensitivity_precip", np.nan))

    if np.isfinite(t_sens) and t_sens >= th["temp_sens_high"]:
        actions.append("High temperature sensitivity: prioritize heat-tolerant varieties and adjust sowing/harvest planning.")
    if np.isfinite(p_sens) and p_sens >= th["prec_sens_high"]:
        actions.append("High precipitation sensitivity: improve drainage/soil practices; review flood/excess rain exposure.")

    if not flags:
        flags = ["**Moderate risk profile**"]
    if not actions:
        actions = ["Maintain current sourcing; monitor annually and trigger review if risk moves into top quintile."]

    actions_md = "\n".join([f"- {a}" for a in actions])

    txt = (
        f"**Department recommendation — {dep}**\n\n"
        f"**Risk signals:** {' · '.join(flags)}\n\n"
        f"**Recommended actions:**\n"
        f"{actions_md}"
    )
    return txt

def top_n_list(series: pd.Series, n=5) -> str:
    vals = [v for v in series.dropna().astype(str).tolist()][:n]
    return ", ".join(vals) if vals else "—"


def metrics_from_department_view(yield_t_ha: float, area_ha: float, price_eur_t: float):
    kg = yield_t_ha * area_ha
    eur = kg * price_eur_t

    c1, c2, c3 = st.columns(3)
    c1.metric("Yield (kg/ha)", f"{yield_t_ha:.2f}")
    c2.metric("Production (kg)", f"{kg:,.0f}")
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
    c2.metric("Production (kg)", f"{total_production:,.0f}")
    c3.metric("Value (€)", f"{total_value:,.0f}")

    st.caption(f"Area-weighted yield: {yield_weighted:.2f} kg/ha")


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
            index=department_options.index("All") if "All" in department_options else 0,
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
            "Price (€/kg)",
            min_value=0.0,
            value=float(DEFAULT_PRICE_EUR_PER_kg),
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
    tab1, tab2 = st.tabs(["Forecast", "Risk"])

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
            price_eur_per_kg=float(price),
            downside_only=downside_only,
        )

        # Add climate indices and sensitivities (derived from available data)
        sens = compute_department_sensitivities(df_future_all)
        risk_tbl = risk_tbl.merge(sens, on="department", how="left")

        if dep != "All":
            snap = risk_tbl[risk_tbl["department"] == dep]
            if not snap.empty:
                s = snap.iloc[0]
                st.markdown("#### Key metrics for selected department")
                a, b, c, d = st.columns(4)
                a.metric("Severe-year yield (p10, kg/ha)", f"{s['p10']:.2f}")
                b.metric("Downside gap (kg/ha)", f"{s['downside_gap']:.2f}")
                c.metric("Loss (kg)", f"{s['loss_kg']:,.0f}")
                d.metric("Loss (€)", f"{s['loss_eur']:,.0f}")

        deps_gdf = load_deps_geo()

        st.markdown("#### Risk map")

        map_metric = st.selectbox(
            "Map metric",
            ["risk_score"],
            index=0
        )

        metric_map_lookup = {
            "loss_kg": "loss_kg",
            "risk_score": "risk_score",
        }
        metric_to_plot = metric_map_lookup[map_metric]

        fig = plot_risk_map(
            deps_gdf=deps_gdf,
            risk_tbl=risk_tbl,
            metric=metric_to_plot,
            title=f"{map_metric} — {scen_risk_label} | {win_start}-{win_end}",
            diverging_center_zero=(metric_to_plot in ["loss_kg", "loss_eur", "downside_gap"]),
            highlight_dep=dep,   # ✅ add this
        )

        col1, col2, col3 = st.columns([1,2,1])  # center column bigger
        with col2:
            st.pyplot(fig)

        st.markdown("### Recommendations")

        th = compute_thresholds(risk_tbl)

        if dep == "All":
            st.markdown(recommendations_portfolio(risk_tbl, th))
        else:
            st.markdown(recommendations_department(risk_tbl, dep, th))

        st.markdown("#### Ranking")

        # Remove risk_score from display, rename p10 to a clearer label,
        # and sort by Loss kg (descending = biggest losses first).
        display_tbl = risk_tbl.copy()
        display_tbl = display_tbl.sort_values("loss_kg", ascending=False).reset_index(drop=True)

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
        "loss_kg",
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