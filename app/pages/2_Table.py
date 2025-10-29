# Page 2 – Table view with a row-wise LineChartColumn for the first month of data
# One row per numeric column in the CSV.
# Notes:
# - Tries to auto-detect a time column
# - Builds a summary table: [variable, first_month (sparkline), count, mean]
# - Displays raw first-month slice for transparency

import streamlit as st
import pandas as pd
import pathlib
st.title("Page 2 – Table + LineChartColumn (First Month Per Variable)")

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """Load CSV using pandas (cached for performance)."""
    return pd.read_csv(path)

def detect_time_col(df: pd.DataFrame):
    """Try to find a reasonable time-like column (by attempting datetime parsing)."""
    common = ["time", "date", "datetime", "timestamp"]
    for c in common + list(df.columns):
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None

csv_path = pathlib.Path(__file__).parent.parent / "open-meteo-subset.csv"
st.caption(f"Loading data from: `{csv_path}`")

try:
    # 1) Load and quick glance
    df = load_csv(csv_path)
    st.subheader("Quick preview of raw data")
    st.dataframe(df, use_container_width=True)

    # 2) Detect time column and clean
    time_col = detect_time_col(df)
    if time_col is None:
        st.warning(
            "Could not detect a time/datetime column. "
            "Please ensure your CSV has a column like 'time'."
        )
        st.stop()

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)

    # 3) First month slice
    months = df[time_col].dt.to_period("M")
    first_month = months.min()
    df_first = df.loc[months == first_month]

    # 4) Numeric columns only (each will become one row in summary)
    num_cols = [
        c for c in df.columns
        if c != time_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not num_cols:
        st.warning("No numeric columns found to summarize.")
        st.stop()

    # 5) Build summary table
    rows = []
    for col in num_cols:
        series = df_first[col].dropna().tolist()
        rows.append(
            {
                "variable": col,
                "first_month": series,  # fed to LineChartColumn as a small sparkline
                "count": len(series),
                "mean": float(pd.Series(series).mean()) if len(series) else None,
            }
        )
    summary = pd.DataFrame(rows)

    st.subheader(f"Variables summary for the first month ({first_month})")
    st.dataframe(
        summary,
        column_config={
            "variable": "Variable",
            "count": "Count (first month)",
            "mean": "Mean (first month)",
            "first_month": st.column_config.LineChartColumn(
                "First month sparkline", width="medium", y_min=None, y_max=None
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Show raw first-month slice"):
        st.dataframe(df_first, use_container_width=True)

except FileNotFoundError:
    st.error(
        "File `open-meteo-subset.csv` not found. "
        "Place the CSV inside the `app/` folder."
    )
except Exception as e:
    st.exception(e)
