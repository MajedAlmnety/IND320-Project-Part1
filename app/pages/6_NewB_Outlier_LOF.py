# pages/6_NewB_Outlier_LOF.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for Streamlit
import matplotlib.pyplot as plt
from io import StringIO
from contextlib import redirect_stdout
from sklearn.neighbors import LocalOutlierFactor

from utils_data import download_open_meteo_hourly
from utils_analysis import plot_temperature_outliers

# ---------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------
st.title("New B — Outlier and LOF Analysis")
st.caption("Temperature outliers via DCT + MAD; precipitation anomalies via LOF. Year = 2021.")

# Use a global area from session if available; otherwise offer a selector
areas = ["NO1", "NO2", "NO3", "NO4", "NO5"]
area = st.session_state.get("selected_area") or st.selectbox("Select Price Area:", areas, index=areas.index("NO5"))

# Load hourly weather data for the selected area (2021)
df = download_open_meteo_hourly(area, year=2021).sort_values("time")

# Tabs for the two analyses
tab1, tab2 = st.tabs(["Outlier (DCT + MAD)", "Anomaly (LOF)"])

# ---------------------------------------------------------------------
# Tab 1 — Temperature Outlier Detection (DCT + MAD)
# ---------------------------------------------------------------------
with tab1:
    st.subheader(f"Temperature Outlier Detection — {area} (2021)")

    # Controls for high-pass cutoff and MAD threshold
    cutoff = st.slider("High-pass cutoff", 0.0, 0.2, 0.02, 0.005)
    n_std  = st.slider("MAD threshold multiplier", 1.0, 6.0, 3.5, 0.1)

    # Placeholders for the plot and summary text
    placeholder_plot = st.empty()
    placeholder_text = st.empty()

    # Temporarily suppress plt.show() so the figure can be captured once
    _real_show = plt.show
    plt.show = lambda *args, **kwargs: None

    # Capture printed summary from the helper function
    buf = StringIO()
    with redirect_stdout(buf):
        outliers_df = plot_temperature_outliers(
            df.copy(),
            column="temperature_2m",
            cutoff=cutoff,
            n_std=n_std,
            show_summary=True
        )

    # Render the generated figure
    fig = plt.gcf()
    placeholder_plot.pyplot(fig, clear_figure=True)

    # Show the captured text summary, if any
    summary_text = buf.getvalue().strip()
    if summary_text:
        placeholder_text.text(summary_text)

    # Restore original show function
    plt.show = _real_show

    # Optional: show the DataFrame with detected outliers
    with st.expander("Show Outlier DataFrame"):
        st.dataframe(outliers_df, use_container_width=True)

# ---------------------------------------------------------------------
# Tab 2 — Precipitation Anomaly Detection (Local Outlier Factor)
# ---------------------------------------------------------------------
with tab2:
    st.subheader(f"Precipitation Anomaly Detection (LOF) — {area} (2021)")

    # Controls for LOF contamination rate and neighborhood size
    contamination = st.slider("Expected anomaly fraction", 0.001, 0.05, 0.01, 0.001)
    n_neighbors = st.slider("LOF neighbors", 10, 60, 35, 1)

    # Prepare the single-feature input array for LOF
    X = df[["precipitation"]].astype(float).values

    # Fit LOF and mark anomalies (-1 = outlier)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(X)
    anomalies = (y_pred == -1)

    # Plot the precipitation series and highlight anomalies
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df["time"], df["precipitation"], label="Precipitation")
    ax2.scatter(df["time"][anomalies], df["precipitation"][anomalies],
                color="red", marker="x", label="Anomaly")
    ax2.set_title(f"LOF anomalies: {int(anomalies.sum())} detected (~{contamination*100:.1f}% expected)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Precipitation (mm)")
    ax2.legend(loc="best")
    st.pyplot(fig2)

    # Quick summary and detailed table of flagged points
    st.metric("Anomaly count", int(anomalies.sum()))
    with st.expander("Show Anomaly DataFrame"):
        st.dataframe(df.loc[anomalies], use_container_width=True)
