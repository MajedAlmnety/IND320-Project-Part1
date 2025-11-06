# pages/2_Production_2021.py

# ---------- must be first ----------
import streamlit as st
st.set_page_config(page_title="Production Data", layout="wide")  # Page title and wide layout
# -----------------------------------

import os
from urllib.parse import quote_plus
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient

# App header and short description
st.title("Electricity Production Data — 2021")
st.caption("Data source: Elhub API • Stored in MongoDB • Visualized with Streamlit")

# ---------------- Secrets (.env or secrets.toml) ----------------
load_dotenv(find_dotenv())  # Load environment variables from .env if present

def get_secret(name: str):
    # Prefer environment variable (local), then Streamlit secrets (deployment)
    if os.getenv(name):
        return os.getenv(name)
    try:
        return st.secrets[name]
    except Exception:
        return None  # Missing secret

# Read connection parameters
user = get_secret("MONGO_USER")
password_raw = get_secret("MONGO_PASS")
cluster = get_secret("MONGO_CLUSTER")

# Fail early if required secrets are missing
if not user or not password_raw or not cluster:
    st.error(
        "MongoDB connection details not found.\n"
        "Please add MONGO_USER, MONGO_PASS, and MONGO_CLUSTER "
        "to your .env file (local dev) or .streamlit/secrets.toml (deployment)."
    )
    st.stop()

# Build MongoDB Atlas URI (escape password safely)
password = quote_plus(password_raw)
uri = f"mongodb+srv://{user}:{password}@{cluster}/?retryWrites=true&w=majority"

# ---------------- Mongo client (cached) ----------------
@st.cache_resource(show_spinner=False)
def get_mongo_client(_uri: str):
    # Create client and verify connectivity with a ping
    client = MongoClient(_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client

client = get_mongo_client(uri)
db = client["elhub"]
collection = db["production_2021"]

# ---------------- Load & normalize data ----------------
# Candidate field names for the numeric production value
CANDIDATES_VALUE = [
    "value", "quantity_kwh", "quantityKwh", "quantitykwh",
    "quantity", "Quantity", "Value",
]

@st.cache_data(ttl=300, show_spinner=True)
def load_data() -> pd.DataFrame:
    # Minimize network payload by projecting needed fields only
    projection = {
        "_id": 0,
        "start_time": 1,
        "price_area": 1,
        "production_group": 1,
        **{c: 1 for c in CANDIDATES_VALUE},
    }
    docs = list(collection.find({}, projection))
    df = pd.DataFrame(docs)

    # Detect and standardize the numeric value column to "value"
    value_col = next((c for c in CANDIDATES_VALUE if c in df.columns), None)
    if value_col is None:
        raise ValueError("Could not find production value column.")
    if value_col != "value":
        df = df.rename(columns={value_col: "value"})

    # Parse timestamps once and drop invalid rows
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["start_time"])
    return df

# Load data with user-friendly error surface
try:
    data = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Known Norwegian price areas (static list for 2021)
ALL_AREAS = ["NO1", "NO2", "NO3", "NO4", "NO5"]
available_areas_in_db = set(data["price_area"].dropna().unique().tolist()) if not data.empty else set()

col1, col2 = st.columns(2)

# ---------------- LEFT COLUMN: PIE ----------------
with col1:
    st.subheader("Production Share by Group")

    # Restore last selected area from session state (default to NO5)
    default_area = st.session_state.get("selected_area", "NO5")
    if default_area not in ALL_AREAS:
        default_area = "NO5"

    # Single-choice area selector
    selected_area = st.radio(
        "Select Price Area:",
        ALL_AREAS,
        index=ALL_AREAS.index(default_area),
        horizontal=True,
    )
    st.session_state["selected_area"] = selected_area  # Persist choice

    if selected_area not in available_areas_in_db:
        st.warning(f"No data found for {selected_area} in MongoDB.")
    else:
        # Aggregate total production by group for the selected area
        pie_df = (
            data.loc[data["price_area"] == selected_area]
                .groupby("production_group", as_index=False)["value"]
                .sum()
        )

        if pie_df.empty:
            st.warning(f"No data available to plot for {selected_area}.")
        else:
            # Donut chart for group share
            fig_pie = px.pie(
                pie_df,
                names="production_group",
                values="value",
                title=f"Total Production in {selected_area}",
                hole=0.3,
            )
            fig_pie.update_traces(textposition="inside")
            st.plotly_chart(fig_pie, use_container_width=True)

# ---------------- RIGHT COLUMN: LINE ----------------
with col2:
    st.subheader("Hourly Production Over Time")

    if selected_area not in available_areas_in_db:
        st.info("Select another area that has data to enable the chart.")
    else:
        # Allow multiple production groups; pick first two as a sensible default
        groups = sorted(data["production_group"].dropna().unique().tolist())
        default_groups = groups[:2] if len(groups) >= 2 else groups
        selected_groups = st.multiselect(
            "Select Production Group(s):",
            groups,
            default=default_groups,
        )
        if not selected_groups:
            selected_groups = groups  # Fallback to all groups if none selected

        # Restrict month choices to those available for the chosen area
        area_mask = (data["price_area"] == selected_area)
        months_available = (
            data.loc[area_mask, "start_time"]
                .dt.to_period("M")
                .unique()
                .tolist()
        )
        # Convert periods to month numbers and sort
        months_available = sorted({p.month for p in months_available})
        if not months_available:
            st.warning("No months available for the selected area.")
        else:
            # Month dropdown with friendly month names
            selected_month = st.selectbox(
                "Select Month:",
                months_available,
                format_func=lambda m: pd.Timestamp(2021, m, 1).strftime("%B"),
            )

            # Apply filters and order by time for proper line rendering
            filtered = data[
                area_mask
                & (data["production_group"].isin(selected_groups))
                & (data["start_time"].dt.month == selected_month)
            ].sort_values("start_time")

            if filtered.empty:
                st.warning("No data available for the selected options (try a different month or group).")
            else:
                # Time series of hourly production per group
                fig_line = px.line(
                    filtered,
                    x="start_time",
                    y="value",
                    color="production_group",
                    title=f"{selected_area} – {pd.Timestamp(2021, selected_month, 1).strftime('%B')}",
                    labels={"value": "Production (kWh)", "start_time": "Time"},
                )
                fig_line.update_layout(margin=dict(l=0, r=0, t=60, b=0))
                st.plotly_chart(fig_line, use_container_width=True)

# ---------------- Dataset doc ----------------
with st.expander("Data Source Information"):
    st.markdown(
        """
**Dataset:** `PRODUCTION_PER_GROUP_MBA_HOUR`  
**Source:** Elhub API (2021)  
**Pipeline:** API → Spark → Cassandra → MongoDB → Streamlit
"""
    )
