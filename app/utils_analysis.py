# app/utils_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# Temperature Anomaly Detection using DCT


def plot_temperature_outliers(df, column="temperature_2m (°C)", 
                              cutoff=0.02, n_std=3.5, show_summary=True):
    """
    Detect temperature outliers using a DCT high-pass filter.
    
    Parameters:
        df (DataFrame): Weather data containing 'time' and temperature columns.
        column (str): Name of the temperature column.
        cutoff (float): Fraction of low frequencies to remove (0.0–1.0).
        n_std (float): Number of MAD-based standard deviations for thresholding.
        show_summary (bool): Whether to print a summary of detected outliers.
    
    Returns:
        DataFrame: Subset of outlier records.
        Also displays a plot and summary statistics.
    """

    # Ensure the column exists
   
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    temp = df[column].values
    temp_dct = dct(temp, norm="ortho")

    N = len(temp_dct)
    cutoff_index = int(N * cutoff)

    # --- LOW-PASS TREND ---
    trend_dct = np.copy(temp_dct)
    trend_dct[cutoff_index:] = 0              # keep only low frequencies
    trend = idct(trend_dct, norm="ortho")     # smooth trend estimate

    # --- MAD THRESHOLDS ---
    mad = np.median(np.abs(temp - trend))
    upper_bound = trend + n_std * mad
    lower_bound = trend - n_std * mad

    # --- OUTLIERS ---
    is_outlier = (temp < lower_bound) | (temp > upper_bound)
    outliers = df[is_outlier]

    # --- PLOT ---
    plt.figure(figsize=(14, 6))
    plt.plot(df["time"], temp, label="Temperature (°C)", alpha=0.8)

    plt.plot(df["time"], trend, label="Trend (low-pass)", alpha=0.9,color="lime" )

    plt.plot(df["time"], upper_bound, "--", linewidth=1, color="tomato",
             label="Upper SPC limit")
    plt.plot(df["time"], lower_bound, "--", linewidth=1,
             label="Lower SPC limit", color="tomato")

    plt.scatter(outliers["time"], outliers[column], color="teal", s=12, 
                label="Outliers")

    plt.title("Temperature with Time-Varying SPC Limits (Low-Pass DCT Trend)")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if show_summary:
        print(f"Number of outliers: {len(outliers)}")
        print(f"Percentage of data: {100 * len(outliers) / len(df):.2f}%")
        print(f"MAD: {mad:.3f}")

    return outliers
# Function for Seasonal Trend Analysis using STL

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

def stl_decompose_simple(
    df,
    area="NO1",
    production_group="hydro",
    period_length=24*7,
    seasonal=25,
    trend=601,
    robust=True
):
    """
    Perform STL (Seasonal-Trend decomposition using Loess) on Elhub-style
    electricity production data and return both the figure and result object.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least:
        ['startTime', 'priceArea', 'productionGroup', 'quantityKwh'].
    area : str, default="NO1"
        Electricity price area to filter (e.g., "NO1", "NO2", ...).
    production_group : str, default="hydro"
        Production group to analyze (e.g., "hydro", "wind", "solar").
    period_length : int, default=24*7
        Period (in hours) for the seasonal component — here one week.
    seasonal : int, default=25
        Length of the seasonal smoother.
    trend : int, default=601
        Length of the trend smoother.
    robust : bool, default=True
        Whether to use a robust fitting method to reduce the impact of outliers.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure displaying trend, seasonal, and residual components.
    result : statsmodels.tsa.seasonal.STLResult
        The STL decomposition result object containing component series.
    s"""

    # Convert and clean datetime
    df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce", utc=True)
    df = df.dropna(subset=["startTime"]).set_index("startTime").sort_index()

    # Filter data
    sub = df[(df["priceArea"] == area) & (df["productionGroup"] == production_group)]
    if sub.empty:
        print("No data found for this area or production group.")
        return None, None

    # Prepare numeric values
    y = pd.to_numeric(sub["quantityKwh"], errors="coerce")
    y = y.resample("h").sum().interpolate()

    # Run STL
    stl = STL(y, period=period_length, seasonal=seasonal, trend=trend, robust=robust)
    result = stl.fit()

    # Plot
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"STL Decomposition — {production_group.capitalize()} ({area})", fontsize=14)

    # Close figure to prevent double rendering
    plt.close(fig)

    return fig, result


# Function to Create a Spectrogram

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def production_spectrogram(
    df: pd.DataFrame,
    area: str = "NO1",
    production_group: str = "hydro",
    nperseg: int = 256,                  # window length (samples)
    noverlap: int = 128,                 # window overlap (samples)
    time_col: str = "startTime",
    area_col: str = "priceArea",
    group_col: str = "productionGroup",
    value_col: str = "quantityKwh",
):
    """
    Build a spectrogram for Elhub production data and return a Matplotlib Figure.

    Parameters
    ----------
    df : pd.DataFrame
        Table with at least [time_col, area_col, group_col, value_col].
    area : str
        Electricity price area filter (e.g. "NO1").
    production_group : str
        Production group filter (e.g. "hydro", "wind", ...).
    nperseg : int
        Window length in samples (hourly data ⇒ samples are hours).
    noverlap : int
        Overlap between windows in samples.
    time_col, area_col, group_col, value_col : str
        Column names in `df`.

    Returns
    -------
    fig : matplotlib.figure.Figure | None
        Figure of the spectrogram, or None if no data matches the filters.
    """

    # Copy & parse time
    s = df.copy()
    s[time_col] = pd.to_datetime(s[time_col], errors="coerce", utc=True)
    s = s.dropna(subset=[time_col])

    # Filter by area and production group
    s = s[(s[area_col] == area) & (s[group_col] == production_group)]
    if s.empty:
        return None

    # Build a time-indexed series, hourly and numeric
    ts = pd.DataFrame({
        time_col: s[time_col].values,
        "value": pd.to_numeric(s[value_col], errors="coerce"),
    }).dropna()

    ts = ts.set_index(time_col).sort_index()
    ts = ts.resample("h").sum().interpolate(limit_direction="both")

    x = ts["value"].astype(float).to_numpy()
    if x.size == 0:
        return None

    # Sampling rate: hourly ⇒ 24 samples/day
    fs = 24.0

    # Clamp overlap
    noverlap = max(0, min(noverlap, nperseg - 1))

    # Spectrogram
    f, t, Sxx = spectrogram(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap,
        scaling="density", mode="magnitude"
    )

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(t, f, Sxx, shading="gouraud", cmap="viridis")
    ax.set_title(f"Spectrogram — {production_group.capitalize()} ({area})")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Frequency (cycles/day)")
    fig.colorbar(im, ax=ax, label="Magnitude")
    fig.tight_layout()

    # Prevent double display in notebooks
    plt.close(fig)
    return fig
