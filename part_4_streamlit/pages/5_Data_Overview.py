# -*- coding: utf-8 -*-
# Streamlit dashboard to explore MongoDB-based energy time series

import os
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from urllib.parse import quote_plus
from dotenv import load_dotenv
import plotly.graph_objects as go
import json

# Page setup: title and layout
st.set_page_config(page_title="Data Overview", layout="wide")
st.title("Data Overview — MongoDB Energy Data")

# Load environment variables (.env) such as MongoDB credentials
load_dotenv()

# MongoDB connection using credentials from environment
try:
    user = os.getenv("MONGO_USER")
    password = quote_plus(os.getenv("MONGO_PASS") or "")
    cluster = os.getenv("MONGO_CLUSTER")
    uri = f"mongodb+srv://{user}:{password}@{cluster}/?retryWrites=true&w=majority"
    
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    st.success("Connected to MongoDB")
except Exception as e:
    # Stop if we cannot connect to MongoDB
    st.error(f"Mongo connection failed: {e}")
    st.stop()

# Database and collection names (from environment or defaults)
DB = os.getenv("MONGO_DB", "elhub_data")
COLL_PROD = os.getenv("COLL_PROD", "production_2022_2024")
COLL_CONS = os.getenv("COLL_CONS", "consumption_2021_2024")

# Sidebar filters for dataset and basic options
st.sidebar.header("Filters")

# Choose dataset: production or consumption
dataset = st.sidebar.radio("Dataset", ["Production", "Consumption"], horizontal=True)
collection_name = COLL_PROD if dataset == "Production" else COLL_CONS
coll = client[DB][collection_name]

# Build lists of price areas and energy groups from MongoDB
price_areas = sorted(list({str(x).upper().replace(" ", "") for x in coll.distinct("price_area")}))
groups = sorted(coll.distinct("energy_group"))

# User selection for price area and energy group
price_area = st.sidebar.selectbox("Price Area", price_areas)
energy_group = st.sidebar.selectbox("Energy Group", groups)

# Date range settings
default_min = pd.Timestamp("2022-01-01")
default_max = pd.Timestamp("2024-12-31")

start_date = st.sidebar.date_input("Start date", default_min)
end_date   = st.sidebar.date_input("End date", default_max)

# Convert date inputs to UTC timestamps
start_dt = pd.Timestamp(start_date, tz="UTC")
end_dt   = pd.Timestamp(end_date, tz="UTC")

# Limit the maximum number of rows fetched from MongoDB
limit_rows = st.sidebar.slider("Limit rows", 100, 100000, 5000)

# Fetch data from MongoDB using selected filters
with st.spinner("Loading data from MongoDB..."):
    cursor = coll.find(
        {
            "price_area": {"$regex": f"^{price_area}$", "$options": "i"},
            "energy_group": energy_group,
            "start_time": {"$gte": start_dt.to_pydatetime(),
                           "$lt": end_dt.to_pydatetime()}
        },
        {"_id": 0}
    ).sort("start_time", 1).limit(limit_rows)
    
    data = list(cursor)

df = pd.DataFrame(data)

# Stop if no data returned by the query
if df.empty:
    st.warning("No data found for this selection.")
    st.stop()

# Prepare DataFrame: parse time, sort, and rename columns
df["start_time"] = pd.to_datetime(df["start_time"])
df = df.sort_values("start_time")
df = df.rename(columns={"start_time": "time", "value": "energy"})

# Summary statistics section
st.subheader("Summary Statistics")

col1, col2, col3 = st.columns(3)
col1.metric("Rows loaded", f"{len(df):,}")
col2.metric("Date span", f"{df['time'].min().date()} → {df['time'].max().date()}")
col3.metric("Energy Mean", f"{df['energy'].mean():,.0f}")

col4, col5, col6 = st.columns(3)
col4.metric("Max value", f"{df['energy'].max():,.0f}")
col5.metric("Min value", f"{df['energy'].min():,.0f}")
col6.metric("Std deviation", f"{df['energy'].std():,.0f}")

# Time series plot of the energy values
st.subheader("Time Series Plot")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["time"],
    y=df["energy"],
    mode="lines",
    line=dict(width=1.5, color="#F0C751"),
    name="Energy"
))
fig.update_layout(
    height=400,
    xaxis_title="Time",
    yaxis_title="Energy value",
    margin=dict(l=20, r=20, t=40, b=20),
    paper_bgcolor="#86989C",
    plot_bgcolor="#86989C"     # inner plot background
)
st.plotly_chart(fig, use_container_width=True)

# Data table showing the loaded records
st.subheader("Data Table")
st.dataframe(df)

# Download section for exporting data as CSV or JSON
st.subheader("Download Data")

colA, colB = st.columns(2)

# CSV download (without index)
csv_data = df.to_csv(index=False).encode("utf-8")
colA.download_button(
    "Download CSV",
    csv_data,
    file_name=f"energy_{dataset}_{price_area}_{energy_group}.csv",
    mime="text/csv"
)

# JSON download (records orientation for easy parsing)
json_data = df.to_json(orient="records").encode("utf-8")
colB.download_button(
    "Download JSON",
    json_data,
    file_name=f"energy_{dataset}_{price_area}_{energy_group}.json",
    mime="application/json"
)
