# -*- coding: utf-8 -*-
# Snow drift dashboard using Tabler (2003) and Open-Meteo ERA5 data

import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Snow Drift (Tabler 2003)", layout="wide")
st.title("Snow Drift — Annual Drift + Wind Rose (Tabler 2003 Method)")

# Short description of what this page does
st.markdown("""
This page applies the Tabler (2003) model to compute annual snow drift
using weather data from Open-Meteo ERA5 and the last selected point on the map.
""")

# Check that there is at least one clicked point from the map page
if "clicks" not in st.session_state or len(st.session_state["clicks"]) == 0:
    st.error("No map point selected. Go to the 'Map and Selectors' page and click on the map first.")
    st.stop()

# Use the last clicked latitude/longitude from session state
lat, lon = st.session_state["clicks"][-1]
st.success(f"Selected coordinates: {lat:.5f}, {lon:.5f}")

# Model constants (taken from Snow_drift.py)
T = 3000   # Maximum transport distance (m)
F = 30000  # Fetch distance (m)
theta = 0.5  # Relocation coefficient


# Core functions from Snow_drift.py

def compute_Qupot(hourly_wind_speeds, dt=3600):
    """
    Potential wind-driven transport Qupot [kg/m]
    Qupot = sum((u^3.8) * dt) / 233847
    """
    total = sum((u ** 3.8) * dt for u in hourly_wind_speeds) / 233847
    return total


def sector_index(direction):
    """
    Map a wind direction in degrees to one of 16 sectors (0–15).
    """
    return int(((direction + 11.25) % 360) // 22.5)


def compute_sector_transport(hourly_wind_speeds, hourly_wind_dirs, dt=3600):
    """
    Compute cumulative transport for 16 wind sectors.

    Returns: list of 16 values (kg/m).
    """
    sectors = [0.0] * 16
    for u, d in zip(hourly_wind_speeds, hourly_wind_dirs):
        idx = sector_index(d)
        sectors[idx] += ((u ** 3.8) * dt) / 233847
    return sectors


def compute_snow_transport(T, F, theta, Swe, hourly_wind_speeds, dt=3600):
    """
    Compute snow drifting transport components according to Tabler (2003).

    T: maximum transport distance (m)
    F: fetch distance (m)
    theta: relocation coefficient
    Swe: seasonal snowfall water equivalent (mm)
    hourly_wind_speeds: list of hourly wind speeds (m/s)
    """
    Qupot = compute_Qupot(hourly_wind_speeds, dt)
    Qspot = 0.5 * T * Swe       # Snowfall-limited transport [kg/m]
    Srwe = theta * Swe          # Relocated water equivalent [mm]

    # Decide whether transport is limited by snowfall or wind
    if Qupot > Qspot:
        Qinf = 0.5 * T * Srwe
        control = "Snowfall-controlled"
    else:
        Qinf = Qupot
        control = "Wind-controlled"

    # Mean annual snow transport [kg/m]
    Qt = Qinf * (1 - 0.14 ** (F / T))

    return {
        "Qupot (kg/m)": Qupot,
        "Qspot (kg/m)": Qspot,
        "Srwe (mm)": Srwe,
        "Qinf (kg/m)": Qinf,
        "Qt (kg/m)": Qt,
        "Control": control,
    }


def compute_fence_height(Qt, fence_type: str):
    """
    Calculate fence height for storing a given snow drift.

    Qt: transport [kg/m]
    fence_type: "Wyoming", "Slat-and-wire", or "Solid"
    """
    Qt_tonnes = Qt / 1000.0
    ft = fence_type.lower()

    if ft == "wyoming":
        factor = 8.5
    elif ft in ["slat-and-wire", "slat and wire"]:
        factor = 7.7
    elif ft == "solid":
        factor = 2.9
    else:
        raise ValueError("Unsupported fence type. Use 'Wyoming', 'Slat-and-wire', or 'Solid'.")

    # Empirical relation between Qt and fence height
    H = (Qt_tonnes / factor) ** (1 / 2.2)
    return H


# Fetch historical weather data from Open-Meteo ERA5

@st.cache_data(show_spinner=True)
def fetch_meteo(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical weather data from the Open-Meteo ERA5 archive API,
    using the same approach as in Part 3.

    start_date / end_date are "YYYY-MM-DD".
    """

    url = "https://archive-api.open-meteo.com/v1/era5"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m",
        ],
        "timezone": "Europe/Oslo",
    }

    r = requests.get(url, params=params, timeout=30)

    if r.status_code != 200:
        st.error(f"Error from Open-Meteo API: {r.status_code}\nURL: {r.url}")
        raise Exception(f"Open-Meteo error {r.status_code}")

    data = r.json()

    if "hourly" not in data:
        st.error(f"Open-Meteo response does not contain 'hourly':\n{data}")
        raise Exception("No 'hourly' field in Open-Meteo response")

    df_weather = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature_2m (°C)": data["hourly"]["temperature_2m"],
        "precipitation (mm)": data["hourly"]["precipitation"],
        "wind_speed_10m (m/s)": data["hourly"]["wind_speed_10m"],
        "wind_gusts_10m (m/s)": data["hourly"]["wind_gusts_10m"],
        "wind_direction_10m (°)": data["hourly"]["wind_direction_10m"],
    })

    return df_weather


# Select year range (season from July to June)
st.subheader("Select year range (season 1 July → 30 June)")

colA, colB = st.columns(2)
start_year = colA.number_input("Start year (July start)", value=2018, min_value=1980, max_value=2030)
end_year = colB.number_input("End year (June end)", value=2024, min_value=1980, max_value=2035)

if start_year > end_year:
    st.error("Invalid year range: start year is greater than end year.")
    st.stop()

years_list = list(range(start_year, end_year + 1))

# Download weather data for each season
st.subheader("Download weather data from Open-Meteo ERA5")

all_data = []

with st.spinner("Downloading hourly weather data for all seasons..."):
    for y in years_list:
        # Season runs from 1 July (y) to 30 June (y+1)
        season_start = f"{y}-07-01"
        season_end = f"{y+1}-06-30"

        df = fetch_meteo(lat, lon, season_start, season_end)
        df["season"] = y
        all_data.append(df)

        # Small sleep between requests
        time.sleep(0.3)

if not all_data:
    st.error("No weather data downloaded. Check years or internet connection.")
    st.stop()

df_all = pd.concat(all_data, ignore_index=True)

# Compute Qt for each season + sector transport
st.subheader("Compute seasonal snow drift (Tabler 2003)")

results = []
sector_list = []

for y in years_list:
    df_y = df_all[df_all["season"] == y].copy()
    if df_y.empty:
        continue

    # SWE: precipitation when temperature < 1°C
    df_y["Swe_hourly"] = df_y.apply(
        lambda row: row["precipitation (mm)"] if row["temperature_2m (°C)"] < 1 else 0,
        axis=1
    )
    total_Swe = df_y["Swe_hourly"].sum()

    winds = df_y["wind_speed_10m (m/s)"].tolist()
    wdirs = df_y["wind_direction_10m (°)"].tolist()

    # Tabler transport results for this season
    res = compute_snow_transport(T, F, theta, total_Swe, winds)
    res["season"] = f"{y}-{y+1}"
    results.append(res)

    # Directional (sector-based) transport
    sectors = compute_sector_transport(winds, wdirs)
    sector_list.append(sectors)

if not results:
    st.error("No results after processing the selected seasons. Check the year range or data.")
    st.stop()

df_res = pd.DataFrame(results)

# Bonus: monthly Qt over the entire period
st.subheader("Monthly Snow Drift Qt (kg/m)")

# SWE per hour on df_all (computed once)
df_all["Swe_hourly"] = df_all.apply(
    lambda row: row["precipitation (mm)"] if row["temperature_2m (°C)"] < 1 else 0,
    axis=1
)

# Convert timestamps to month (YYYY-MM)
df_all["month"] = df_all["time"].dt.to_period("M").astype(str)

monthly_results = []
for month, df_m in df_all.groupby("month"):
    if df_m.empty:
        continue

    total_Swe_m = df_m["Swe_hourly"].sum()
    winds_m = df_m["wind_speed_10m (m/s)"].tolist()

    res_m = compute_snow_transport(T, F, theta, total_Swe_m, winds_m)
    res_m["month"] = month
    monthly_results.append(res_m)

if monthly_results:
    df_month = pd.DataFrame(monthly_results).sort_values("month")

    df_month_display = df_month[["month", "Qt (kg/m)"]].copy()
    df_month_display["Qt (tonnes/m)"] = df_month_display["Qt (kg/m)"] / 1000.0

    # Show monthly Qt as a table
    st.dataframe(
        df_month_display.set_index("month").style.format({
            "Qt (kg/m)": "{:.0f}",
            "Qt (tonnes/m)": "{:.2f}",
        })
    )

    # Bar chart for monthly Qt in tonnes per meter
    fig_month = go.Figure()
    fig_month.add_trace(go.Bar(
        x=df_month["month"],
        y=df_month["Qt (kg/m)"] / 1000.0,
        name="Qt (tonnes/m)"
    ))
    fig_month.update_layout(
        height=400,
        xaxis_title="Month",
        yaxis_title="Qt (tonnes/m)",
        margin=dict(l=20, r=20, t=40, b=50)
    )
    st.plotly_chart(fig_month, use_container_width=True)

    st.success("Monthly snow drift Qt calculated and visualized.")
else:
    st.info("No monthly results could be computed for the selected years.")

# Compute average sector transport and overall mean Qt
avg_sectors = np.mean(sector_list, axis=0)
overall_Qt = df_res["Qt (kg/m)"].mean()

# Plot annual Qt per season
st.subheader("Annual Snow Drift Qt (kg/m)")

fig_qt = go.Figure()
fig_qt.add_trace(go.Scatter(
    x=df_res["season"],
    y=df_res["Qt (kg/m)"],
    mode="lines+markers",
    line=dict(width=3),
    marker=dict(size=8)
))
fig_qt.add_hline(
    y=overall_Qt,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Mean Qt = {overall_Qt/1000:.2f} tonnes/m",
    annotation_position="top left"
)
fig_qt.update_layout(
    height=400,
    xaxis_title="Season (July–June)",
    yaxis_title="Qt (kg/m)"
)
st.plotly_chart(fig_qt, use_container_width=True)

# Wind rose: average directional transport over all seasons
st.subheader("Average directional snow transport (Wind Rose, 16 sectors)")

angles_deg = np.arange(0, 360, 360 / 16)
rose_vals_tonnes = avg_sectors / 1000.0  # Convert from kg/m to tonnes/m

fig_rose = go.Figure()
fig_rose.add_trace(go.Barpolar(
    r=rose_vals_tonnes,
    theta=angles_deg,
    width=360 / 16,
    marker_color=rose_vals_tonnes,
    marker_colorscale="Viridis",
))
fig_rose.update_layout(
    polar=dict(
            radialaxis=dict(showticklabels=True, ticks="outside"),
            angularaxis=dict(
                tickmode="array",
                tickvals=angles_deg,
                ticktext=['N', 'NNE', 'NE', 'ENE',
                          'E', 'ESE', 'SE', 'SSE',
                          'S', 'SSW', 'SW', 'WSW',
                          'W', 'WNW', 'NW', 'NNW'],
                direction="clockwise",
                rotation=90,
            ),
    ),
    showlegend=False,
    height=500,
    margin=dict(l=20, r=20, t=40, b=20),
)
st.plotly_chart(fig_rose, use_container_width=True)

# Compute fence height for each season and each fence type
st.subheader("Fence height per season")

fence_types = ["Wyoming", "Slat-and-wire", "Solid"]
rows = []

for _, row in df_res.iterrows():
    qt = row["Qt (kg/m)"]
    rec = {
        "Season": row["season"],
        "Qt (tonnes/m)": qt / 1000.0,
    }
    for ftype in fence_types:
        rec[ftype] = compute_fence_height(qt, ftype)
    rows.append(rec)

df_fence = pd.DataFrame(rows)

st.dataframe(
    df_fence.style.format({
        "Qt (tonnes/m)": "{:.2f}",
        "Wyoming": "{:.2f}",
        "Slat-and-wire": "{:.2f}",
        "Solid": "{:.2f}",
    })
)

st.success("Annual snow drift, monthly Qt, wind rose, and fence height per season were computed successfully.")
