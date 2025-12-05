# app/utils_data.py
# ------------------------------------------------------------
# This module provides helper functions for loading weather data
# from the Open-Meteo ERA5 archive and mapping Norwegian price areas
# to approximate geographic coordinates.
# ------------------------------------------------------------

import requests
import pandas as pd
from datetime import datetime
import streamlit as st

# Approximate coordinates for each Norwegian electricity price area
PRICE_AREA_COORDS = {
    "NO1": (59.91, 10.75),  # Oslo
    "NO2": (58.97, 5.73),   # Stavanger
    "NO3": (63.43, 10.39),  # Trondheim
    "NO4": (69.65, 18.95),  # Tromsø
    "NO5": (60.39, 5.32),   # Bergen
}

@st.cache_data(ttl=3600, show_spinner=True)
def download_open_meteo_hourly(price_area: str, year: int = 2021) -> pd.DataFrame:
    """
    Download hourly temperature and precipitation data for a given
    Norwegian price area and year from the Open-Meteo ERA5 archive.
    Returns a pandas DataFrame with UTC timestamps.
    """
    if price_area not in PRICE_AREA_COORDS:
        raise ValueError(f"Unknown price area: {price_area}")

    lat, lon = PRICE_AREA_COORDS[price_area]

    # Define full-year date range
    start = f"{year}-01-01"
    end = f"{year}-12-31"

    # Construct API URL
    url = (
        "https://archive-api.open-meteo.com/v1/era5"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,precipitation"
        "&timezone=UTC"
    )

    # Send request and parse response
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()

    # Normalize JSON into DataFrame
    df = pd.DataFrame({
        "time": js["hourly"]["time"],
        "temperature_2m": js["hourly"]["temperature_2m"],
        "precipitation": js["hourly"]["precipitation"],
    })

    # Convert to datetime format
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df
