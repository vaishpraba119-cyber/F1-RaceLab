
import requests
import pandas as pd
import numpy as np
import streamlit as st
import time

from datetime import datetime

BASE_URL = "https://api.openf1.org/v1"


# Retrieve and cache data from the OpenF1 API
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(endpoint, params=None):

    if params is None:
        params = {}

    for attempt in range(4):

        try:
            response = requests.get(
                f"{BASE_URL}/{endpoint}",
                params=params,
                timeout=60
            )

            # Handle API rate limits
            if response.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue

            # Handle temporarily unavailable servers
            if response.status_code >= 500:
                time.sleep(2 ** (attempt + 1))
                continue

            response.raise_for_status()

            data = response.json()

            time.sleep(0.5)

            return pd.DataFrame(data)

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError
        ):

            if attempt == 3:
                raise

            time.sleep(2 ** (attempt + 1))

    raise RuntimeError(
        "OpenF1 API is temporarily unavailable."
    )


# Generate available season options
def get_years():

    current_year = datetime.now().year

    return list(
        range(2023, current_year + 1)
    )[::-1]


# Retrieve Grand Prix events for a selected year
def get_meetings(year):

    meetings = get_data(
        "meetings",
        {"year": year}
    )

    if meetings.empty:
        return meetings

    meetings["date_start"] = pd.to_datetime(
        meetings["date_start"],
        utc=True
    )

    # Remove future events
    meetings = meetings[
        meetings["date_start"]
        <= pd.Timestamp.now(tz="UTC")
    ]

    return meetings.sort_values(
        "date_start"
    ).reset_index(drop=True)


# Retrieve sessions for a selected Grand Prix
def get_sessions(meeting_key):

    sessions = get_data(
        "sessions",
        {"meeting_key": meeting_key}
    )

    if sessions.empty:
        return sessions

    sessions["date_start"] = pd.to_datetime(
        sessions["date_start"],
        utc=True
    )

    sessions["date_end"] = pd.to_datetime(
        sessions["date_end"],
        utc=True
    )

    # Only include completed sessions
    sessions = sessions[
        sessions["date_end"]
        <= pd.Timestamp.now(tz="UTC")
    ]

    return sessions.sort_values(
        "date_start"
    ).reset_index(drop=True)


# Retrieve drivers participating in the session
def get_drivers(session_key):

    drivers = get_data(
        "drivers",
        {"session_key": session_key}
    )

    if drivers.empty:
        return drivers

    drivers = drivers.drop_duplicates(
        "driver_number"
    )

    return drivers.sort_values(
        "full_name"
    ).reset_index(drop=True)


# Retrieve completed laps for a driver
def get_laps(session_key, driver_number):

    laps = get_data(
        "laps",
        {
            "session_key": session_key,
            "driver_number": driver_number
        }
    )

    if laps.empty:
        return laps

    laps = laps.dropna(
        subset=["lap_duration", "date_start"]
    )

    laps = laps[
        laps["lap_duration"] > 0
    ]

    return laps.sort_values(
        "lap_number"
    ).reset_index(drop=True)


# Select the fastest recorded non-pit-out lap
def get_fastest_lap(laps):

    if laps.empty:
        raise ValueError("No completed laps available.")

    valid_laps = laps[
        laps["is_pit_out_lap"] == False
    ]

    if valid_laps.empty:
        raise ValueError("No suitable laps available.")

    return valid_laps.loc[
        valid_laps["lap_duration"].idxmin()
    ]


# Select a specific lap number
def get_specific_lap(laps, lap_number):

    selected = laps[
        laps["lap_number"] == lap_number
    ]

    if selected.empty:
        raise ValueError(
            f"Lap {lap_number} is unavailable."
        )

    return selected.iloc[0]


# Calculate lap start and end timestamps
def get_lap_times(lap):

    start = pd.to_datetime(
        lap["date_start"],
        utc=True
    )

    end = start + pd.to_timedelta(
        lap["lap_duration"],
        unit="s"
    )

    return start, end


# Retrieve car telemetry for a selected lap
def get_telemetry(
    session_key,
    driver_number,
    lap
):

    start, end = get_lap_times(lap)

    telemetry = get_data(
        "car_data",
        {
            "session_key": session_key,
            "driver_number": driver_number,
            "date>": start.isoformat(),
            "date<": end.isoformat()
        }
    )

    if telemetry.empty:
        raise ValueError(
            "No telemetry available for this lap."
        )

    telemetry["date"] = pd.to_datetime(
        telemetry["date"],
        utc=True
    )

    telemetry = telemetry.sort_values("date")

    telemetry = telemetry.drop_duplicates("date")

    telemetry = telemetry.dropna(
        subset=["speed", "brake", "throttle"]
    ).reset_index(drop=True)

    if len(telemetry) < 3:
        raise ValueError(
            "Insufficient telemetry samples."
        )

    # Calculate elapsed time
    telemetry["Time"] = (
        telemetry["date"] - start
    ).dt.total_seconds()

    # Convert speed to metres per second
    speed_ms = (
        telemetry["speed"].to_numpy() / 3.6
    )

    # Calculate time intervals
    dt = np.diff(
        telemetry["Time"].to_numpy()
    )

    # Trapezoidal numerical integration
    distance = (
        (speed_ms[1:] + speed_ms[:-1]) / 2
    ) * dt

    telemetry["Distance"] = np.concatenate(
        ([0], np.cumsum(distance))
    )

    return telemetry


# Retrieve car positioning data
def get_position(
    session_key,
    driver_number,
    lap
):

    start, end = get_lap_times(lap)

    position = get_data(
        "location",
        {
            "session_key": session_key,
            "driver_number": driver_number,
            "date>": start.isoformat(),
            "date<": end.isoformat()
        }
    )

    if position.empty:
        return position

    position["date"] = pd.to_datetime(
        position["date"],
        utc=True
    )

    return position.sort_values(
        "date"
    ).drop_duplicates(
        "date"
    ).reset_index(drop=True)


# Align car telemetry with positioning data
def merge_position_telemetry(
    position,
    telemetry
):

    if position.empty:
        return pd.DataFrame()

    columns = [
        "date",
        "speed",
        "throttle",
        "brake",
        "Distance",
        "Time",
        "rpm",
        "n_gear",
        "drs"
    ]

    available_columns = [
        col for col in columns
        if col in telemetry.columns
    ]

    merged = pd.merge_asof(
        position.sort_values("date"),
        telemetry[available_columns].sort_values("date"),
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=0.5)
    )

    return merged.dropna(
        subset=["x", "y", "speed", "Distance"]
    ).reset_index(drop=True)


# Retrieve session weather conditions
def get_weather(session_key):

    return get_data(
        "weather",
        {"session_key": session_key}
    )


# Retrieve tyre stint information
def get_stints(session_key, driver_number):

    return get_data(
        "stints",
        {
            "session_key": session_key,
            "driver_number": driver_number
        }
    )


# Retrieve official session results where available
def get_session_results(session_key):

    return get_data(
        "session_result",
        {"session_key": session_key}
    )
