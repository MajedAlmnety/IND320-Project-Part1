# Importing API Connection Libraries

import requests  # For sending and receiving HTTP requests
import datetime as dt  # For handling dates and times

# Function to Download Weather Data from Open-Meteo API

def download_weather_data(latitude: float, longitude: float, year: int):
    """
    Download historical weather data from the Open-Meteo API (ERA5 model)
    Includes the same variables used in Part 1 of the project.
    """

    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    url = "https://archive-api.open-meteo.com/v1/era5"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "wind_direction_10m"
        ],
        "timezone": "Europe/Oslo"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Error connecting to the API: {response.status_code}")

    data = response.json()

    # Build a DataFrame similar to the previous file
    df_weather = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temperature_2m (°C)": data["hourly"]["temperature_2m"],
        "precipitation (mm)": data["hourly"]["precipitation"],
        "wind_speed_10m (m/s)": data["hourly"]["wind_speed_10m"],
        "wind_gusts_10m (m/s)": data["hourly"]["wind_gusts_10m"],
        "wind_direction_10m (°)": data["hourly"]["wind_direction_10m"],
    })

    print(f"Weather data for year {year} downloaded successfully ({len(df_weather)} records)")
    return df_weather
