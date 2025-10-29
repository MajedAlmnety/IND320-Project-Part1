# Page 3 – Plot view with:
# - Single-column plot OR all-columns plot
# - Month range selection
# - Optional aggregation (Raw / Weekly / Monthly)
# - Scaling options when plotting all columns together:
#     * Min-Max (0–1)
#     * Z-Score (σ)
#     * Index (first=100)
#     * None (raw values)
#
# Design choices:
# - Scaling is applied on the filtered and aggregated subset (fair comparison for the visible range)
# - Matplotlib is used; titles/labels/legends are kept clean and readable
# - We avoid hard-coded colors so default Matplotlib palette is used

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib

st.title("Page 3 – Plots, Filters, Aggregation & Scaling")

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """Load CSV using pandas (cached)."""
    return pd.read_csv(path)

def detect_time_col(df: pd.DataFrame):
    """Return the first column that can be parsed as datetime."""
    for c in ["time", "date", "datetime", "timestamp"] + list(df.columns):
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None

def resample_df(df: pd.DataFrame, time_col: str, rule: str | None, agg: str = "mean") -> pd.DataFrame:
    """Resample dataframe by rule ('W', 'M', or None) using the given aggregation."""
    if rule is None:
        return df.copy()
    dfi = df.set_index(time_col)
    if agg == "mean":
        out = dfi.resample(rule).mean(numeric_only=True)
    elif agg == "median":
        out = dfi.resample(rule).median(numeric_only=True)
    elif agg == "sum":
        out = dfi.resample(rule).sum(numeric_only=True)
    else:
        out = dfi.resample(rule).agg(agg, numeric_only=True)
    return out.reset_index()

def scale_series(s: pd.Series, method: str) -> pd.Series:
    """Scale a series to make multi-series plotting fair when scales differ."""
    s = s.astype(float)
    if method == "Min-Max (0–1)":
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng != 0 else s * 0
    elif method == "Z-Score (σ)":
        std = s.std(ddof=0)
        return (s - s.mean()) / std if std != 0 else s * 0
    elif method == "Index (first=100)":
        first = s.iloc[0] if len(s) else np.nan
        return (s / first) * 100 if pd.notna(first) and first != 0 else s * 0
    else:  # "None"
        return s

csv_path = pathlib.Path(__file__).parent.parent / "open-meteo-subset.csv"
st.caption(f"Loading data from: `{csv_path}`")

try:
    # 1) Load and validate
    df = load_csv(csv_path)
    time_col = detect_time_col(df)
    if time_col is None:
        st.warning("Could not detect a time column. Ensure your CSV has a time-like column.")
        st.stop()

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    num_cols = [
        c for c in df.columns
        if c != time_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not num_cols:
        st.warning("No numeric columns to plot.")
        st.stop()

    # 2) Month range selector (default: first month only)
    months_period = df[time_col].dt.to_period("M")
    months_unique = sorted(months_period.unique())
    months_labels = [str(p) for p in months_unique]
    if not months_labels:
        st.warning("No valid months found in the dataset.")
        st.stop()

    st.subheader("Controls")
    selection = st.selectbox(
        "Select one column or all columns:",
        options=["All columns"] + num_cols,
        index=0,
    )

    default_range = (months_labels[0], months_labels[0])  # first → first
    m_start, m_end = st.select_slider(
        "Select a month range:",
        options=months_labels,
        value=default_range,
    )

    agg_choice = st.radio(
        "Aggregation (smoothing):",
        ["Raw", "Weekly (W)", "Monthly (M)"],
        index=0,
        horizontal=True,
    )
    rule = None if agg_choice == "Raw" else ("W" if "Weekly" in agg_choice else "M")
    period_label = "raw" if rule is None else ("weekly" if rule == "W" else "monthly")

    # 3) Filter by month range then aggregate
    mask = (months_period >= pd.Period(m_start)) & (months_period <= pd.Period(m_end))
    df_sel = df.loc[mask]
    if df_sel.empty:
        st.warning("No data in the selected month range.")
        st.stop()

    df_plot = resample_df(df_sel, time_col, rule=rule, agg="mean")

    # 4) Plot
    st.subheader("Plot")
    fig, ax = plt.subplots(figsize=(10, 4))

    if selection == "All columns":
        scale_method = st.radio(
            "Scaling (only when plotting all columns):",
            ["Min-Max (0–1)", "Z-Score (σ)", "Index (first=100)", "None"],
            index=0,
            horizontal=True,
        )
        for col in num_cols:
            y = scale_series(df_plot[col], scale_method)
            ax.plot(df_plot[time_col], y, label=col)

        ax.set_ylabel({
                          "Min-Max (0–1)": "Normalized (0–1)",
                          "Z-Score (σ)": "Z-score (σ)",
                          "Index (first=100)": "Index (first=100)",
                          "None": "Value",
                      }[scale_method])
        ax.set_title(f"All columns — {m_start} → {m_end}  ({period_label})")
        ax.legend(loc="best", ncols=2, fontsize=9)

    else:
        ax.plot(df_plot[time_col], df_plot[selection].astype(float))
        ax.set_ylabel(selection)
        ax.set_title(f"{selection} — {m_start} → {m_end}  ({period_label})")

    ax.set_xlabel(str(time_col))
    ax.grid(True)
    st.pyplot(fig)

except FileNotFoundError:
    st.error(
        "File `open-meteo-subset.csv` not found. "
        "Place the CSV inside the `app/` folder."
    )
except Exception as e:
    st.exception(e)
