import os
from urllib.parse import quote_plus

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient


# Streamlit App Configuration

st.set_page_config(page_title="Production Data", layout="wide")
st.title("Electricity Production Data — 2021")
st.caption("Data source: Elhub API • Stored in MongoDB • Visualized with Streamlit")

# MongoDB Credentials (from Streamlit secrets)

load_dotenv(find_dotenv())  # Optional local .env fallback for dev

# Retrieve secrets securely from .streamlit/secrets.toml
user = st.secrets["MONGO_USER"]
password_raw = st.secrets["MONGO_PASS"]
cluster = st.secrets["MONGO_CLUSTER"]

# If any secret is missing → stop the app
if not user or not password_raw or not cluster:
    st.error("Missing MongoDB credentials.")
    st.stop()

# Encode password for URL safety
password = quote_plus(password_raw)

# Build MongoDB connection string
uri = f"mongodb+srv://{user}:{password}@{cluster}/?retryWrites=true&w=majority"


# Connect to MongoDB (cached connection)

@st.cache_resource(show_spinner=False)
def get_mongo_client(_uri: str):
    try:
        client = MongoClient(_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")  # test connection
        return client
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        st.stop()

# Create MongoDB client and reference DB/collection
client = get_mongo_client(uri)
db = client["elhub"]
collection = db["production_2021"]

# Load and Normalize Data from MongoDB


# Candidates for the numeric production column (naming varies)
CANDIDATES_VALUE = [
    "value", "quantity_kwh", "quantityKwh", "quantitykwh",
    "quantity", "Quantity", "Value",
]

@st.cache_data(ttl=300, show_spinner=True)
def load_data() -> pd.DataFrame:
    # Limit fields pulled from MongoDB to minimize data size
    projection = {"_id": 0, "start_time": 1, "price_area": 1, "production_group": 1, **{c: 1 for c in CANDIDATES_VALUE}}
    docs = list(collection.find({}, projection))
    return pd.DataFrame(docs)

# Load dataset from MongoDB
data = load_data()

# Automatically detect the actual numeric value column
found_col = None
for cand in CANDIDATES_VALUE:
    if cand in data.columns:
        found_col = cand
        break

# Abort if no matching column found
if not found_col:
    st.error("Could not find production value column.")
    st.stop()

# Rename column to 'value' for consistency in plots
if found_col != "value":
    data.rename(columns={found_col: "value"}, inplace=True)

# Check for data availability
if data.empty:
    st.error("No data loaded from MongoDB.")
    st.stop()


# Extract Filters from Data

# Unique options for selection widgets
areas = sorted(data["price_area"].dropna().unique())
groups = sorted(data["production_group"].dropna().unique())
months = list(range(1, 13))  # January–December


# Page Layout: Two Columns (Pie + Line)

col1, col2 = st.columns(2)

# ---------------- LEFT COLUMN: PIE CHART ----------------
with col1:
    st.subheader("Production Share by Group")

    # User selects price area via radio buttons
    selected_area = st.radio("Select Price Area:", areas, horizontal=True)

    # Filter data for selected area and aggregate production
    pie_df = data[data["price_area"] == selected_area]
    pie_df = pie_df.groupby("production_group", as_index=False)["value"].sum()

    # Plot pie chart showing share of each production group
    fig_pie = px.pie(
        pie_df,
        names="production_group",
        values="value",
        title=f"Total Production in {selected_area}",
        hole=0.3  # makes it a donut chart
    )
    fig_pie.update_traces(textposition="inside")
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------------- RIGHT COLUMN: LINE CHART ----------------
with col2:
    st.subheader("Hourly Production Over Time")

    # Select one or more production groups using pills
    selected_groups = st.pills("Select Production Group(s):", groups, selection_mode="multi", default=groups[:2])

    # Select a month from dropdown (formatted as name)
    selected_month = st.selectbox(
        "Select Month:",
        months,
        format_func=lambda x: pd.Timestamp(2021, x, 1).strftime('%B')
    )

    # Apply all filters to the dataset
    filtered = data[
        (data["price_area"] == selected_area) &
        (data["production_group"].isin(selected_groups)) &
        (pd.to_datetime(data["start_time"]).dt.month == selected_month)
    ]

    # Warn user if no data available
    if filtered.empty:
        st.warning("No data available for selected options.")
    else:
        # Sort by time for proper line chart rendering
        filtered = filtered.sort_values("start_time")

        # Build line chart by group, across time
        fig_line = px.line(
            filtered,
            x="start_time",
            y="value",
            color="production_group",
            title=f"{selected_area} – {pd.Timestamp(2021, selected_month, 1).strftime('%B')}",
            labels={"value": "Production (kWh)", "start_time": "Time"}
        )
        fig_line.update_layout(margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig_line, use_container_width=True)


# Dataset Documentation

with st.expander("Data Source Information"):
    st.markdown("""
**Dataset:** `PRODUCTION_PER_GROUP_MBA_HOUR`  
**Source:** Elhub API (2021)  
**Pipeline:** API → Spark → Cassandra → MongoDB → Streamlit  
    """)
