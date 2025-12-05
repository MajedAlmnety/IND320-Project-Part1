import streamlit as st
import pandas as pd
from utils_data import download_open_meteo_hourly

# Page title and short source note
st.title("Page 2 – Table + LineChartColumn (First Month Per Variable)")
st.caption("Source: Open-Meteo ERA5 Archive (2021)")

# --------------------------
# Helper: detect time column
# --------------------------
def detect_time_col(df: pd.DataFrame):
    """Try to find a reasonable time-like column (by attempting datetime parsing)."""
    # Check common timestamp names first, then fall back to scanning all columns
    common = ["time", "date", "datetime", "timestamp"]
    for c in common + list(df.columns):
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None  # No parsable datetime column found

# --------------------------
# Load data via Open-Meteo
# --------------------------
areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]
# Use session-selected area if available; otherwise let user select (default NO5)
area = st.session_state.get("selected_area", None) or st.selectbox(
    "Select price area:", areas, index=areas.index("NO5")
)

# Fetch hourly weather data for the chosen area and year
df = download_open_meteo_hourly(area, year=2021)
st.success(f"Loaded hourly weather data for {area} (2021)")
st.dataframe(df.head(), use_container_width=True)  # Quick preview of columns

# --------------------------
# Process first month of data
# --------------------------
time_col = detect_time_col(df)
if time_col is None:
    st.warning("Could not detect a time/datetime column.")
    st.stop()

# Normalize time column and sort chronologically
df = df.copy()
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col]).sort_values(time_col)

# Identify the first available month in the dataset (often 2021-01)
months = df[time_col].dt.to_period("M")
first_month = months.min()
df_first = df.loc[months == first_month]  # Slice for the first month

# --------------------------
# Summarize numeric columns
# --------------------------
# Consider all numeric columns except the time column
num_cols = [
    c for c in df.columns
    if c != time_col and pd.api.types.is_numeric_dtype(df[c])
]
if not num_cols:
    st.warning("No numeric columns found to summarize.")
    st.stop()

# Build per-variable stats and a sparkline-ready list for the first month
rows = []
for col in num_cols:
    series = df_first[col].dropna().tolist()
    rows.append(
        {
            "variable": col,
            "first_month": series,  # sparkline data points
            "count": len(series),
            "mean": float(pd.Series(series).mean()) if len(series) else None,
        }
    )
summary = pd.DataFrame(rows)

# --------------------------
# Display summary with sparklines
# --------------------------
st.subheader(f"Variables summary for the first month ({first_month})")
st.dataframe(
    summary,
    column_config={
        "variable": "Variable",
        "count": "Count (first month)",
        "mean": "Mean (first month)",
        "first_month": st.column_config.LineChartColumn(
            "First month sparkline", width="medium"
        ),
    },
    hide_index=True,
    use_container_width=True,
)

# Optional: show the raw first-month slice for inspection
with st.expander("Show raw first-month slice"):
    st.dataframe(df_first, use_container_width=True)
