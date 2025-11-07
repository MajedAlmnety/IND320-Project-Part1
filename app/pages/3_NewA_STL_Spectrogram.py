import streamlit as st
import pandas as pd
from utils_analysis import stl_decompose_simple, production_spectrogram
from pathlib import Path

# ===== LOAD DATA =====
@st.cache_data
def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV file with caching to improve performance."""
    return pd.read_csv(path)

# Determine project root and CSV file path
# ROOT points to the parent folder of the current script (e.g., 'app/')
ROOT = Path(__file__).resolve().parents[2]
csv_path = ROOT /"notebook"/ "elhub_production.csv" # Validate that the CSV file exists before loading
if not csv_path.exists():
    st.error(f"CSV not found at: {csv_path}")
    st.stop()

# Try to read the CSV file safely
try:
    elhub_df = load_csv(csv_path)
    st.success(f"Loaded CSV from: {csv_path}")
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()


# ===== PAGE TITLE =====
st.title("New A — STL & Spectrogram Analysis (2021)")
st.caption("Visual analysis of Elhub production data using STL decomposition and Spectrograms.")

# ===== AREA & GROUP SELECTION =====
area = st.session_state.get("selected_area", "NO5")
groups = ["hydro", "wind", "solar", "thermal", "other"]
group = st.selectbox("Select Production Group:", groups, index=0)

# ===== TABS =====
tab1, tab2 = st.tabs(["STL Decomposition", "Spectrogram"])

# --- TAB 1: STL ---
with tab1:
    st.subheader("STL Decomposition")
    fig_stl, _ = stl_decompose_simple(elhub_df, area=area, production_group=group)
    if fig_stl is not None:
        st.pyplot(fig_stl)
    else:
        st.warning("No data available for this selection.")

# --- TAB 2: SPECTROGRAM ---
with tab2:
    st.subheader("Spectrogram")
    nperseg = st.slider("Window length (nperseg)", 64, 512, 256, 16)
    noverlap = st.slider("Window overlap (noverlap)", 0, nperseg - 1, 128, 8)
    fig_spec = production_spectrogram(
        elhub_df, area=area, production_group=group,
        nperseg=nperseg, noverlap=noverlap
    )
    if fig_spec is not None:
        st.pyplot(fig_spec)
    else:
        st.warning("No data available for this selection.")
