from __future__ import annotations

from io import BytesIO, StringIO
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


REQUIRED_COLUMNS = {"date", "contacts"}
ALL_COLUMNS = ["date", "channel", "skill", "contacts", "aht_minutes"]


st.set_page_config(page_title="Lightweight Forecaster", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #171717;
        --paper: #F8F4EC;
        --amber: #F4C542;
        --mint: #BFF2DE;
        --blue: #BFD7FF;
        --clay: #FFB7A8;
        --lavender: #C9B6FF;
    }
    .stApp {
        background:
            linear-gradient(#DED8CC 1px, transparent 1px),
            linear-gradient(90deg, #DED8CC 1px, transparent 1px),
            var(--paper);
        background-size: 80px 80px;
    }
    h1, h2, h3, label, p {
        letter-spacing: 0 !important;
    }
    div[data-testid="stMetric"],
    div[data-testid="stExpander"],
    div[data-testid="stFileUploader"] {
        border: 4px solid var(--ink);
        border-radius: 18px;
        background: white;
        box-shadow: 7px 7px 0 var(--ink);
        padding: 10px;
    }
    .brand-card {
        border: 5px solid var(--ink);
        border-radius: 28px;
        background: white;
        box-shadow: 8px 8px 0 var(--ink);
        padding: 24px 28px;
        margin-bottom: 22px;
    }
    .brand-pill {
        display: inline-block;
        border: 4px solid var(--ink);
        border-radius: 999px;
        background: var(--amber);
        padding: 8px 16px;
        font-weight: 900;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def make_template_workbook() -> bytes:
    template = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6, freq="W-MON"),
            "channel": ["Voice", "Voice", "Voice", "Chat", "Chat", "Chat"],
            "skill": ["Billing", "Billing", "Billing", "Support", "Support", "Support"],
            "contacts": [1200, 1280, 1315, 760, 805, 790],
            "aht_minutes": [10.2, 10.0, 10.4, 6.1, 6.3, 6.2],
        }
    )
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template.to_excel(writer, index=False, sheet_name="forecast_input")
    return output.getvalue()


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Upload a CSV or Excel file.")


def read_pasted_csv(text: str) -> pd.DataFrame:
    if not text.strip():
        raise ValueError("Paste CSV data before running the forecast.")
    pasted = pd.read_csv(StringIO(text.strip()))
    pasted.columns = [str(c).strip().lower() for c in pasted.columns]
    if list(pasted.columns) != ALL_COLUMNS:
        raise ValueError(
            "Pasted CSV must use this exact header: "
            "date,channel,skill,contacts,aht_minutes."
        )
    return pasted


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(c).strip().lower() for c in normalized.columns]

    missing = REQUIRED_COLUMNS - set(normalized.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}.")

    unknown = set(normalized.columns) - set(ALL_COLUMNS)
    if unknown:
        st.warning("Ignoring extra column(s): " + ", ".join(sorted(unknown)) + ".")
        normalized = normalized[[c for c in normalized.columns if c in ALL_COLUMNS]]

    if "channel" not in normalized.columns:
        normalized["channel"] = "All Channels"
    if "skill" not in normalized.columns:
        normalized["skill"] = "All Skills"
    if "aht_minutes" not in normalized.columns:
        normalized["aht_minutes"] = np.nan

    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["contacts"] = pd.to_numeric(normalized["contacts"], errors="coerce")
    normalized["aht_minutes"] = pd.to_numeric(normalized["aht_minutes"], errors="coerce")
    normalized["channel"] = normalized["channel"].fillna("Unknown Channel").astype(str)
    normalized["skill"] = normalized["skill"].fillna("Unknown Skill").astype(str)

    bad_required = normalized["date"].isna() | normalized["contacts"].isna()
    if bad_required.any():
        raise ValueError(
            f"{bad_required.sum()} row(s) have blank or invalid date/contacts values."
        )

    normalized = normalized.dropna(subset=["date", "contacts"]).sort_values("date")
    return normalized


def infer_cadence(dates: pd.Series) -> tuple[str, str]:
    unique_dates = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    if len(unique_dates) < 2:
        return "D", "Daily"

    median_days = unique_dates.diff().dropna().dt.days.median()
    if median_days <= 2:
        return "D", "Daily"
    if median_days <= 10:
        return "W-MON", "Weekly"
    return "MS", "Monthly"


def assign_period(dates: pd.Series, freq: str) -> pd.Series:
    if freq == "D":
        return dates.dt.floor("D")
    if freq.startswith("W"):
        return dates.dt.to_period("W-SUN").dt.start_time
    return dates.dt.to_period("M").dt.start_time


def aggregate_input(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    work = df.copy()
    work["period"] = assign_period(work["date"], freq)
    grouped = (
        work.groupby(["period", "channel", "skill"], as_index=False)
        .agg(
            contacts=("contacts", "sum"),
            weighted_aht_numerator=(
                "aht_minutes",
                lambda x: np.nan,
            ),
        )
    )

    # Calculate AHT separately so multi-row period/channel/skill groups are weighted by contacts.
    work["aht_weighted"] = work["aht_minutes"] * work["contacts"]
    aht = (
        work.groupby(["period", "channel", "skill"], as_index=False)
        .agg(aht_weighted=("aht_weighted", "sum"), aht_weight=("contacts", "sum"))
    )
    grouped = grouped.drop(columns=["weighted_aht_numerator"]).merge(
        aht, on=["period", "channel", "skill"], how="left"
    )
    grouped["aht_minutes"] = grouped["aht_weighted"] / grouped["aht_weight"]
    return grouped.drop(columns=["aht_weighted", "aht_weight"])


def build_series(
    df: pd.DataFrame,
    freq: str,
    selected_channels: list[str],
    selected_skills: list[str],
    metric: str,
) -> pd.Series:
    filtered = df[
        df["channel"].isin(selected_channels) & df["skill"].isin(selected_skills)
    ].copy()
    if filtered.empty:
        return pd.Series(dtype=float)

    if metric == "contacts":
        series = filtered.groupby("period")["contacts"].sum()
    else:
        filtered["aht_weighted"] = filtered["aht_minutes"] * filtered["contacts"]
        agg = filtered.groupby("period").agg(
            aht_weighted=("aht_weighted", "sum"), contacts=("contacts", "sum")
        )
        series = agg["aht_weighted"] / agg["contacts"]

    full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    series = series.reindex(full_index)
    if metric == "contacts":
        return series.fillna(0).astype(float)
    return series.interpolate(limit_direction="both").astype(float)


def future_index(series: pd.Series, freq: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(series.index[-1], periods=periods + 1, freq=freq)[1:]


def constant_forecast(
    series: pd.Series, horizon: int, freq: str, value: float
) -> pd.Series:
    return pd.Series(value, index=future_index(series, freq, horizon))


def forecast_historical_average(series: pd.Series, horizon: int, freq: str) -> pd.Series:
    return constant_forecast(series, horizon, freq, float(series.mean()))


def forecast_weighted_average(series: pd.Series, horizon: int, freq: str) -> pd.Series:
    recent = series.tail(min(6, len(series)))
    weights = np.arange(1, len(recent) + 1)
    value = float(np.average(recent.values, weights=weights))
    return constant_forecast(series, horizon, freq, value)


def forecast_linear_regression(series: pd.Series, horizon: int, freq: str) -> pd.Series:
    x = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    model = LinearRegression()
    model.fit(x, y)
    future_x = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
    return pd.Series(model.predict(future_x), index=future_index(series, freq, horizon))


def forecast_exponential_smoothing(
    series: pd.Series, horizon: int, freq: str
) -> pd.Series:
    if len(series) < 4:
        return forecast_historical_average(series, horizon, freq)
    model = ExponentialSmoothing(series, trend="add", seasonal=None).fit(optimized=True)
    values = model.forecast(horizon)
    values.index = future_index(series, freq, horizon)
    return values


def forecast_arima(series: pd.Series, horizon: int, freq: str) -> pd.Series:
    if len(series) < 6:
        return forecast_historical_average(series, horizon, freq)
    model = ARIMA(series, order=(1, 1, 1)).fit()
    values = model.forecast(horizon)
    values.index = future_index(series, freq, horizon)
    return values


def forecast_prophet(series: pd.Series, horizon: int, freq: str) -> pd.Series:
    try:
        from prophet import Prophet
    except Exception as exc:
        raise RuntimeError("Prophet is not installed or could not be imported.") from exc

    frame = pd.DataFrame({"ds": series.index, "y": series.values})
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=freq == "D",
        yearly_seasonality=len(series) >= 24,
    )
    model.fit(frame)
    future = model.make_future_dataframe(periods=horizon, freq=freq)
    forecast = model.predict(future).tail(horizon)
    return pd.Series(forecast["yhat"].values, index=future_index(series, freq, horizon))


FORECASTERS: dict[str, Callable[[pd.Series, int, str], pd.Series]] = {
    "Historical Average": forecast_historical_average,
    "Weighted Average": forecast_weighted_average,
    "Linear Regression": forecast_linear_regression,
    "Exponential Smoothing": forecast_exponential_smoothing,
    "Prophet": forecast_prophet,
    "ARIMA": forecast_arima,
}


def run_forecasts(series: pd.Series, horizon: int, freq: str) -> tuple[pd.DataFrame, list[str]]:
    forecast_frame = pd.DataFrame(index=future_index(series, freq, horizon))
    warnings: list[str] = []
    for name, forecaster in FORECASTERS.items():
        try:
            values = forecaster(series, horizon, freq).clip(lower=0)
            forecast_frame[name] = values
        except Exception as exc:
            warnings.append(f"{name} could not run: {exc}")
    return forecast_frame, warnings


def make_plot(history: pd.Series, forecast_frame: pd.DataFrame, metric_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines+markers",
            name="Historical",
            line=dict(color="#171717", width=4),
        )
    )
    colors = ["#2F80ED", "#F4C542", "#76E4B8", "#C94C35", "#C9B6FF", "#FF7043"]
    for i, column in enumerate(forecast_frame.columns):
        fig.add_trace(
            go.Scatter(
                x=forecast_frame.index,
                y=forecast_frame[column],
                mode="lines+markers",
                name=column,
                line=dict(color=colors[i % len(colors)], width=3),
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=560,
        margin=dict(l=30, r=30, t=40, b=30),
        title=f"Historical vs Forecast: {metric_label}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#DED8CC")
    fig.update_yaxes(showgrid=True, gridcolor="#DED8CC")
    return fig


def render_template_download() -> None:
    st.download_button(
        "Download Excel Template",
        data=make_template_workbook(),
        file_name="forecast_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


st.markdown(
    """
    <div class="brand-card">
        <div class="brand-pill">FORECAST WORKBENCH</div>
        <h1 style="margin:0;font-size:54px;">Lightweight Contact Forecaster</h1>
        <p style="font-weight:800;opacity:.72;margin-top:10px;">
        Upload a CSV/Excel file or paste CSV data, then compare six forecast methods by channel and skill.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input")
    render_template_download()
    source = st.radio(
        "How would you like to provide data?",
        ["Upload a file", "Paste CSV data"],
    )
    horizon = st.number_input("Forecast periods", min_value=1, max_value=52, value=12)

input_df: pd.DataFrame | None = None
error: str | None = None

if source == "Upload a file":
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            input_df = read_uploaded_file(uploaded)
        except Exception as exc:
            error = str(exc)
else:
    st.info("Paste data in this exact column format: `date,channel,skill,contacts,aht_minutes`")
    pasted = st.text_area(
        "Paste CSV data",
        height=220,
        placeholder=(
            "date,channel,skill,contacts,aht_minutes\n"
            "2026-01-05,Voice,Billing,1200,10.2\n"
            "2026-01-12,Voice,Billing,1280,10.0"
        ),
    )
    if pasted.strip():
        try:
            input_df = read_pasted_csv(pasted)
        except Exception as exc:
            error = str(exc)

if error:
    st.error(error)

if input_df is None:
    st.subheader("Required Format")
    st.write(
        "At minimum, your file must include `date` and `contacts`. "
        "`channel`, `skill`, and `aht_minutes` are optional."
    )
    st.code("date,channel,skill,contacts,aht_minutes", language="text")
    st.stop()

try:
    clean_df = normalize_columns(input_df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

freq, cadence_label = infer_cadence(clean_df["date"])
grouped_df = aggregate_input(clean_df, freq)

st.success(f"Detected cadence: {cadence_label}")

channels = sorted(grouped_df["channel"].unique())
skills = sorted(grouped_df["skill"].unique())

controls = st.columns([1, 1, 1])
with controls[0]:
    selected_channels = st.multiselect("Channel(s)", channels, default=channels)
with controls[1]:
    selected_skills = st.multiselect("Skill(s)", skills, default=skills)
metric_options = ["contacts"]
if grouped_df["aht_minutes"].notna().any():
    metric_options.append("aht_minutes")
with controls[2]:
    metric = st.selectbox(
        "Metric",
        metric_options,
        format_func=lambda x: "Contacts" if x == "contacts" else "AHT Minutes",
    )

if not selected_channels or not selected_skills:
    st.warning("Select at least one channel and one skill.")
    st.stop()

series = build_series(grouped_df, freq, selected_channels, selected_skills, metric)
if series.empty or series.notna().sum() < 3:
    st.warning("At least three historical periods are needed to generate forecasts.")
    st.stop()

forecast_frame, forecast_warnings = run_forecasts(series, int(horizon), freq)
for warning in forecast_warnings:
    st.warning(warning)

metric_label = "Contacts" if metric == "contacts" else "AHT Minutes"

summary = st.columns(4)
summary[0].metric("Historical Periods", f"{len(series):,}")
summary[1].metric("Latest Actual", f"{series.iloc[-1]:,.1f}")
summary[2].metric("Historical Avg", f"{series.mean():,.1f}")
summary[3].metric("Forecast Horizon", f"{int(horizon)} {cadence_label.lower()} periods")

st.plotly_chart(make_plot(series, forecast_frame, metric_label), use_container_width=True)

with st.expander("Prepared Data"):
    st.dataframe(grouped_df, use_container_width=True)

with st.expander("Forecast Values"):
    st.dataframe(forecast_frame.round(2), use_container_width=True)
