# app/pages/4_Production_2021.py
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
user = st.secrets["MONGO_USER"]
password_raw = st.secrets["MONGO_PASS"]
cluster = st.secrets["MONGO_CLUSTER"]

if not user or not password_raw or not cluster:
    st.error("Missing MongoDB credentials.")
    st.stop()

password = quote_plus(password_raw)
uri = f"mongodb+srv://{user}:{password}@{cluster}/?retryWrites=true&w=majority"

# Connect to MongoDB (cached connection)
@st.cache_resource(show_spinner=False)
def get_mongo_client(_uri: str):
    try:
        client = MongoClient(_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return client
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        st.stop()

client = get_mongo_client(uri)
db = client["elhub"]
collection = db["production_2021"]

# Candidates for the numeric production column (naming varies)
CANDIDATES_VALUE = [
    "value", "quantity_kwh", "quantityKwh", "quantitykwh",
    "quantity", "Quantity", "Value",
]

@st.cache_data(ttl=300, show_spinner=True)
def load_data() -> pd.DataFrame:
    projection = {
        "_id": 0,
        "start_time": 1,
        "price_area": 1,
        "production_group": 1,
        **{c: 1 for c in CANDIDATES_VALUE},
    }
    docs = list(collection.find({}, projection))
    return pd.DataFrame(docs)

# Load dataset from MongoDB
data = load_data()

# Detect value column and normalize name
found_col = next((c for c in CANDIDATES_VALUE if c in data.columns), None)
if not found_col:
    st.error("Could not find production value column.")
    st.stop()
if found_col != "value":
    data.rename(columns={found_col: "value"}, inplace=True)

# ✅ Normalize time ONCE here
data["start_time"] = pd.to_datetime(data["start_time"], errors="coerce")
data = data.dropna(subset=["start_time"]).sort_values("start_time")

if data.empty:
    st.error("No data loaded from MongoDB.")
    st.stop()

# Filters
areas = sorted(data["price_area"].dropna().unique())
groups = sorted(data["production_group"].dropna().unique())
months = list(range(1, 13))  # January–December

# Layout
col1, col2 = st.columns(2)

# LEFT: Pie
with col1:
    st.subheader("Production Share by Group")
    selected_area = st.radio("Select Price Area:", areas, horizontal=True)

    pie_df = data[data["price_area"] == selected_area].groupby(
        "production_group", as_index=False
    )["value"].sum()

    fig_pie = px.pie(
        pie_df,
        names="production_group",
        values="value",
        title=f"Total Production in {selected_area}",
        hole=0.3,
    )
    fig_pie.update_traces(textposition="inside")
    st.plotly_chart(fig_pie, use_container_width=True)

# RIGHT: Line
with col2:
    st.subheader("Hourly Production Over Time")

    if not groups:
        st.warning("No production groups found.")
    else:
        selected_groups = st.pills(
            "Select Production Group(s):",
            groups,
            selection_mode="multi",
            default=groups[:2] if len(groups) >= 2 else groups
        )

        selected_month = st.selectbox(
            "Select Month:",
            months,
            format_func=lambda x: pd.Timestamp(2021, x, 1).strftime("%B"),
        )

        filtered = data[
            (data["price_area"] == selected_area) &
            (data["production_group"].isin(selected_groups)) &
            (data["start_time"].dt.month == selected_month)
        ]

        if filtered.empty:
            st.warning("No data available for selected options.")
        else:
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

# Dataset notes
with st.expander("Data Source Information"):
    st.markdown("""
**Dataset:** `PRODUCTION_PER_GROUP_MBA_HOUR`  
**Source:** Elhub API (2021)  
**Pipeline:** API → Spark → Cassandra → MongoDB → Streamlit
""")
