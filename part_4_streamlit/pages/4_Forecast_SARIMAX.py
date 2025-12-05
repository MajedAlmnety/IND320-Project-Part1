# -*- coding: utf-8 -*-
# Streamlit app for energy forecasting using a SARIMAX time series model

import os
import pandas as pd
import numpy as np
import requests
import streamlit as st
from pymongo import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv

import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# Page setup: title and layout
st.set_page_config(page_title="Forecast SARIMAX (Enhanced)", layout="wide")
st.title("Energy Forecast — SARIMAX Model (Enhanced Version)")

# Load environment variables (.env) for MongoDB credentials
load_dotenv()

# MongoDB connection using credentials from environment
try:
    user = os.getenv("MONGO_USER")
    password = quote_plus(os.getenv("MONGO_PASS") or "")
    cluster = os.getenv("MONGO_CLUSTER")
    uri = f"mongodb+srv://{user}:{password}@{cluster}/?retryWrites=true&w=majority"
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    st.success("Connected to MongoDB")
except Exception as e:
    st.error(f"Mongo connection failed: {e}")
    st.stop()

# Database and collection names
DB = os.getenv("MONGO_DB", "elhub_data")
COLL_PROD = os.getenv("COLL_PROD", "production_2022_2024")
COLL_CONS = os.getenv("COLL_CONS", "consumption_2021_2024")

# Sidebar: model and data settings
st.sidebar.header("Model Settings")

# Dataset selection: production vs consumption
dataset = st.sidebar.radio("Dataset", ["Production", "Consumption"], horizontal=True)
collection_name = COLL_PROD if dataset == "Production" else COLL_CONS
coll = client[DB][collection_name]

# Available price areas and energy groups from MongoDB
price_areas = sorted(list({str(x).upper().replace(" ", "") for x in coll.distinct("price_area")}))
energy_groups = sorted(coll.distinct("energy_group"))

# User selections for price area and energy group
price_area = st.sidebar.selectbox("Price Area", price_areas)
energy_group = st.sidebar.selectbox("Energy Group", energy_groups)

# Training period for the SARIMAX model
train_start = st.sidebar.date_input("Training start", pd.Timestamp("2022-01-01"))
train_end   = st.sidebar.date_input("Training end", pd.Timestamp("2024-06-01"))

# Forecast horizon in hours (how far into the future to predict)
horizon = st.sidebar.number_input("Forecast horizon (hours)", value=168, min_value=24, max_value=2000)

# Choose between manual parameters and auto default parameters
mode = st.sidebar.radio("Model Mode", ["Manual SARIMAX", "Auto (recommended)"])

if mode == "Manual SARIMAX":
    st.sidebar.subheader("SARIMAX Params")
    # Non-seasonal orders (p, d, q)
    p = st.sidebar.number_input("p", 0, 10, 1)
    d = st.sidebar.number_input("d", 0, 2, 1)
    q = st.sidebar.number_input("q", 0, 10, 1)

    # Seasonal orders (P, D, Q, S)
    P = st.sidebar.number_input("P", 0, 10, 1)
    D = st.sidebar.number_input("D", 0, 2, 1)
    Q = st.sidebar.number_input("Q", 0, 10, 1)
    S = st.sidebar.number_input("Seasonality (S)", 1, 168, 24)
else:
    # Default parameters for auto mode (simple, stable choice)
    p = d = q = P = D = Q = 1
    S = 24

# Exogenous inputs: optional weather variables (from Open-Meteo)
use_exog = st.sidebar.checkbox("Use weather as exogenous variables")
exog_vars = st.sidebar.multiselect(
    "Weather variables",
    ["temperature_2m", "wind_speed_10m", "precipitation"],
    default=["temperature_2m"] if use_exog else []
)

# Approximate coordinates per price area (used for weather fetch)
PA_COORDS = {
    "NO1": (59.91, 10.75),
    "NO2": (61.00, 10.00),
    "NO3": (63.43, 10.39),
    "NO4": (68.44, 17.43),
    "NO5": (60.39, 5.32),
}
lat, lon = PA_COORDS.get(price_area, (60.0, 10.0))

# Load energy data for the selected area, group and training period
with st.spinner("Loading energy data..."):
    docs = list(coll.find({
        "price_area": {"$regex": f"^{price_area}$", "$options": "i"},
        "energy_group": energy_group,
        "start_time": {"$gte": pd.Timestamp(train_start, tz="UTC").to_pydatetime(),
                       "$lt": pd.Timestamp(train_end, tz="UTC").to_pydatetime()}
    }, {"_id": 0, "start_time": 1, "value": 1}))

df_energy = pd.DataFrame(docs)
if df_energy.empty:
    st.error("No energy data found.")
    st.stop()

# Prepare energy time series: index by time and rename column
df_energy["time"] = pd.to_datetime(df_energy["start_time"])
df_energy = df_energy.set_index("time").sort_index()
df_energy = df_energy.rename(columns={"value": "energy"})

# Ensure hourly frequency and fill missing values
df_energy = df_energy.asfreq("H")
df_energy = df_energy.fillna(method="ffill").fillna(method="bfill")

# Weather fetch function for exogenous data (Open-Meteo ERA5)
def fetch_weather(start, end):
    """
    Fetch hourly temperature, wind speed and precipitation
    for the given period and price area coordinates.
    """
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ["temperature_2m", "wind_speed_10m", "precipitation"],
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=30)
    d = r.json()
    df = pd.DataFrame({
        "time": pd.to_datetime(d["hourly"]["time"]),
        "temperature_2m": d["hourly"]["temperature_2m"],
        "wind_speed_10m": d["hourly"]["wind_speed_10m"],
        "precipitation": d["hourly"]["precipitation"],
    })
    return df.set_index("time")

# Merge exogenous variables with energy series if selected
if use_exog:
    with st.spinner("Fetching weather for exogenous variables..."):
        df_weather = fetch_weather(train_start, train_end)

        # Ensure hourly alignment for weather data
        df_weather = df_weather.asfreq("H")
        df_weather = df_weather.fillna(method="ffill").fillna(method="bfill")

        # Join energy and weather on the time index
        df = df_energy.join(df_weather, how="inner")
else:
    df = df_energy.copy()

# SARIMAX model training
st.subheader("Training SARIMAX Model")

# Exogenous matrix for training (if any)
exog_train = df[exog_vars] if (use_exog and len(exog_vars) > 0) else None

with st.spinner("Fitting model... (enhanced stable mode)"):
    try:
        model = SARIMAX(
            df["energy"],
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
            exog=exog_train,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit = model.fit(
            disp=False,
            maxiter=200,
            method="lbfgs"
        )
        trained = True
    except Exception as e:
        st.error(f"Main SARIMAX failed: {e}")
        trained = False

# Fallback model in case the main model fails
if not trained:
    st.warning("Switching to fallback non-seasonal model...")
    try:
        model = SARIMAX(df["energy"], order=(1, 1, 1))
        fit = model.fit(disp=False)
    except Exception as e:
        st.error(f"Fallback model failed: {e}")
        st.stop()

# Report basic model quality metrics
st.success("Model training completed.")
st.write(f"**AIC:** {fit.aic:.2f} — **BIC:** {fit.bic:.2f}")

# Build forecast index (future hourly timestamps)
future_index = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="H")

# Prepare exogenous variables for the forecast period if enabled
if use_exog:
    df_weather_future = fetch_weather(future_index[0], future_index[-1])
    df_weather_future = df_weather_future.asfreq("H")
    df_weather_future = df_weather_future.fillna(method="ffill").fillna(method="bfill")
    exog_forecast = df_weather_future[exog_vars]
else:
    exog_forecast = None

# Generate forecast with confidence intervals
forecast_res = fit.get_forecast(steps=horizon, exog=exog_forecast)
forecast_mean = forecast_res.predicted_mean
forecast_ci = forecast_res.conf_int(alpha=0.05)

# Plot historical series and forecast with 95% confidence bands
st.subheader("Forecast Plot")

fig = go.Figure()

# Historical energy data
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["energy"],
    name="Historical",
    line=dict(color="blue")
))

# Forecasted values
fig.add_trace(go.Scatter(
    x=forecast_mean.index, y=forecast_mean,
    name="Forecast",
    line=dict(color="orange")
))

# Lower confidence bound
fig.add_trace(go.Scatter(
    x=forecast_ci.index, y=forecast_ci.iloc[:, 0],
    line=dict(color="gray", dash="dash"),
    name="Lower CI",
    showlegend=False
))

# Upper confidence bound (filled to create a band)
fig.add_trace(go.Scatter(
    x=forecast_ci.index, y=forecast_ci.iloc[:, 1],
    line=dict(color="gray", dash="dash"),
    fill="tonexty",
    fillcolor="rgba(200,200,200,0.3)",
    showlegend=False
))

fig.update_layout(
    title="SARIMAX Forecast with 95% CI",
    height=500,
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

st.success("Forecast completed successfully!")
