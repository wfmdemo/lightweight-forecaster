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
    div[data-testid="stMetric"] * {
        color: var(--ink) !important;
    }
    div[data-testid="stMetric"] label {
        font-weight: 900 !important;
        opacity: .82 !important;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 900 !important;
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

