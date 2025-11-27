import streamlit as st
from pymongo import MongoClient
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st



load_dotenv()
st.set_page_config(page_title="Elhub Dashboard", layout="wide")

st.title("Elhub Streamlit Dashboard")
st.write("Welcome to the Elhub project app — Part 4 (ETL + Streamlit).")

# Connect to MongoDB
try:
    user = os.getenv("MONGO_USER")
    password = quote_plus(os.getenv("MONGO_PASS") or "")
    cluster = os.getenv("MONGO_CLUSTER")
    uri = f"mongodb+srv://{user}:{password}@{cluster}/?retryWrites=true&w=majority"
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    st.success("Successfully connected to MongoDB.")
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")
    st.stop()

st.markdown("---")
st.subheader("Available Pages:")
st.markdown("""
1. **Map and Selectors** — Display price area map and select data.  
2. **Snow Drift** — Calculate and visualize annual snow drift.  
3. **Meteo Correlation** — Weather–energy correlation analysis.  
4. **Forecast (SARIMAX)** — Energy prediction using SARIMAX.  
5. **Data Overview** — Data overview and download.  
""")

st.info("Use the Streamlit sidebar to navigate between pages.")
