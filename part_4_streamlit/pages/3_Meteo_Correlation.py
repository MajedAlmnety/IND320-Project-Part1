# -*- coding: utf-8 -*-
# Meteo–energy correlation dashboard using MongoDB and Open-Meteo ERA5

import os
import pandas as pd
import numpy as np
import requests
import streamlit as st
from pymongo import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv
import plotly.graph_objects as go

# Basic page setup (title and layout)
st.set_page_config(page_title="Meteo Correlation", layout="wide")
st.title("Meteo–Energy Correlation")

# Load environment variables from .env (MongoDB credentials etc.)
load_dotenv()

# Connect to MongoDB using credentials from environment variables
try:
    user = os.getenv("MONGO_USER")
    password = quote_plus(os.getenv("MONGO_PASS") or "")
    cluster = os.getenv("MONGO_CLUSTER")
    uri = f"mongodb+srv://{user}:{password}@{cluster}/?retryWrites=true&w=majority"
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    st.success("Connected to MongoDB")
except Exception as e:
    # Stop the app if MongoDB connection fails
    st.error(f"Mongo connection failed: {e}")
    st.stop()

# Database and collection configuration
DB = os.getenv("MONGO_DB", "elhub_data")
COLL_PROD = os.getenv("COLL_PROD", "production_2022_2024")
COLL_CONS = os.getenv("COLL_CONS", "consumption_2021_2024")

# Sidebar UI – user choices for filters and model parameters
st.sidebar.header("Filters")

# Choose which dataset to use (production or consumption)
dataset = st.sidebar.radio("Dataset", ["Production", "Consumption"], horizontal=True)
collection_name = COLL_PROD if dataset == "Production" else COLL_CONS
coll = client[DB][collection_name]

# Get available price areas and energy groups from MongoDB
areas = sorted(list({str(a).upper().replace(" ", "") for a in coll.distinct("price_area")}))
groups = sorted(coll.distinct("energy_group"))

# User selects one price area and one energy group
price_area = st.sidebar.selectbox("Price Area", areas)
energy_group = st.sidebar.selectbox("Energy Group", groups)

# Weather variables to include in the correlation analysis
meteo_vars = st.sidebar.multiselect(
    "Weather variables",
    ["temperature_2m", "wind_speed_10m", "precipitation"],
    default=["temperature_2m"]
)

# Sliding window size for rolling correlation (hours)
window = st.sidebar.number_input("Sliding Window Size (hours)", value=24, min_value=3, max_value=240)

# Lag between energy and weather time series (in hours, can be negative)
lag = st.sidebar.number_input("Lag (hours)", value=0, min_value=-240, max_value=240)

# Date range to filter the time series
start = st.sidebar.date_input("Start date", pd.Timestamp("2022-01-01"))
end   = st.sidebar.date_input("End date", pd.Timestamp("2024-12-31"))

start_dt = pd.Timestamp(start, tz="UTC")
end_dt   = pd.Timestamp(end, tz="UTC")

# Load energy data from MongoDB based on current filters
with st.spinner("Loading energy data from MongoDB..."):
    docs = list(coll.find({
        "price_area": {"$regex": f"^{price_area}$", "$options": "i"},
        "energy_group": energy_group,
        "start_time": {"$gte": start_dt.to_pydatetime(), "$lt": end_dt.to_pydatetime()}
    }, {
        "_id": 0, "start_time": 1, "value": 1
    }))

# Convert energy records to DataFrame and set time index
df_energy = pd.DataFrame(docs)
if df_energy.empty:
    st.error("No energy data found for this selection.")
    st.stop()

df_energy["time"] = pd.to_datetime(df_energy["start_time"])
df_energy = df_energy.set_index("time").sort_index()
df_energy = df_energy.rename(columns={"value": "energy_value"})

# Fetch weather data from Open-Meteo ERA5 for the given coordinate and date range
def fetch_meteo(lat, lon, start, end):
    """
    Download hourly weather data (ERA5) for given coordinates and date range.
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
    data = r.json()
    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature_2m": data["hourly"]["temperature_2m"],
        "wind_speed_10m": data["hourly"]["wind_speed_10m"],
        "precipitation": data["hourly"]["precipitation"],
    })
    return df

# Approximate representative coordinates for each price area (assumed points)
PA_COORDS = {
    "NO1": (59.91, 10.75),
    "NO2": (61.00, 10.00),
    "NO3": (63.43, 10.39),
    "NO4": (68.44, 17.43),
    "NO5": (60.39, 5.32),
}

# Get latitude/longitude for the selected price area (fallback default if not found)
lat, lon = PA_COORDS.get(price_area, (60.0, 10.0))

# Download and prepare weather time series
with st.spinner("Fetching weather data..."):
    df_meteo = fetch_meteo(lat, lon, start_dt, end_dt)
    df_meteo = df_meteo.set_index("time").sort_index()

# Merge energy and meteo data on the time index
df = df_energy.merge(df_meteo, left_index=True, right_index=True, how="inner")

if df.empty:
    st.error("Merged dataframe is empty.")
    st.stop()

# Apply lag to energy series if lag != 0 (shift in hours)
if lag != 0:
    df["energy_lagged"] = df["energy_value"].shift(lag)
    energy_col = "energy_lagged"
else:
    energy_col = "energy_value"

# Compute sliding-window correlation between energy and selected meteo variables
st.subheader("Sliding-Window Correlation")

corr_results = {}

for var in meteo_vars:
    # Rolling correlation over the specified window
    corr_results[var] = df[energy_col].rolling(window).corr(df[var])

df_corr = pd.DataFrame(corr_results)

# Plot time series and corresponding rolling correlation
st.subheader("Time Series and Correlation")

for var in meteo_vars:
    fig = go.Figure()

    # Energy series (main y-axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=df[energy_col],
        name="Energy", line=dict(color="#D6C467")
    ))
    # Weather series (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=df[var],
        name=var, yaxis="y2", line=dict(color="orange")
    ))
    # Sliding correlation (third y-axis)
    fig.add_trace(go.Scatter(
        x=df_corr.index, y=df_corr[var],
        name=f"{var} Corr", yaxis="y3",
        line=dict(color="green", dash="dot")
    ))

    fig.update_layout(
        title=f"Correlation with {var}",
        height=450,
        yaxis=dict(title="Energy"),
        yaxis2=dict(title=var, overlaying="y", side="right"),
        yaxis3=dict(title="Corr", overlaying="y", side="left", position=0.05),
        legend=dict(orientation="h")
    )

    st.plotly_chart(fig, use_container_width=True)

# Build and display a table with the last correlation value for each variable
st.subheader("Correlation Table (last values)")

corr_summary = df_corr.dropna().tail(1).T
corr_summary.columns = ["Last corr"]

st.dataframe(corr_summary.style.format("{:.3f}"))
