# Lightweight Contact Forecaster

A Streamlit app for forecasting contact volume and AHT from uploaded CSV/Excel data or pasted CSV input.

## Input Columns

Required:
- `date`
- `contacts`

Optional:
- `channel`
- `skill`
- `aht_minutes`

Pasted data should use this exact header:

```csv
date,channel,skill,contacts,aht_minutes
```

`channel + skill` are used as the grouping identifiers. If either column is missing, the app fills a default grouping.

## Forecast Models

- Historical Average
- Weighted Average
- Linear Regression
- Exponential Smoothing
- Prophet
- ARIMA

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy To Streamlit Community Cloud

1. Push this folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Create a new app from your GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.
