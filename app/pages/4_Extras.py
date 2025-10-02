# Page 4 – Extras with widgets and simple charts using CSV data

import streamlit as st
import pandas as pd
import pathlib

st.title("Page 4 – Extras / Playground")


# Load CSV file (with caching for speed) ---
@st.cache_data
def load_csv(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)

# Find CSV file (it is in the app folder, one level up from 'pages')
csv_path = pathlib.Path(__file__).parent.parent / "open-meteo-subset.csv"

try:
    df = load_csv(csv_path)

    # Buttons
    st.subheader("Buttons demo")
    if st.button("Show dataset head"):
        st.write(df.head())

    if st.button("Show dataset description"):
        st.write(df.describe())

    # Simple charts from CSV
    st.subheader("Charts from dataset")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        choice = st.selectbox("Choose a column to plot:", options=numeric_cols)
        st.line_chart(df[choice])
        st.bar_chart(df[choice].head(30))
    else:
        st.warning("No numeric columns found in the dataset.")

    # Slider example
    st.subheader("Slider for row selection")
    max_rows = len(df)
    slider_val = st.slider("Pick how many rows to show", 5, min(50, max_rows), 10)
    st.dataframe(df.head(slider_val))

except FileNotFoundError:
    st.error(f"CSV file not found: {csv_path}")
except Exception as e:
    st.exception(e)
