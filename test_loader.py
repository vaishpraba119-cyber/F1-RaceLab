
from data_loader import (
    get_years,
    get_meetings,
    get_sessions,
    get_drivers,
    get_laps,
    get_fastest_lap,
    get_telemetry,
    get_position,
    merge_position_telemetry
)

print("\nAVAILABLE SEASONS")
print(get_years())

# Retrieve the 2025 season
meetings = get_meetings(2025)

# Exclude pre-season testing
meetings = meetings[
    meetings["meeting_name"].str.contains(
        "Grand Prix",
        case=False,
        na=False
    )
].reset_index(drop=True)

print("\nAVAILABLE GRAND PRIX EVENTS")

print(
    meetings[[
        "meeting_key",
        "meeting_name"
    ]].to_string(index=False)
)

if meetings.empty:
    raise ValueError("No Grand Prix events found.")

# Select the first actual Grand Prix
meeting = meetings.iloc[0]

print("\nSELECTED GRAND PRIX")
print(meeting["meeting_name"])

sessions = get_sessions(
    int(meeting["meeting_key"])
)

print("\nAVAILABLE SESSIONS")

print(
    sessions[[
        "session_key",
        "session_name"
    ]].to_string(index=False)
)

# Select qualifying
qualifying = sessions[
    sessions["session_name"] == "Qualifying"
]

if qualifying.empty:
    raise ValueError(
        "No qualifying session available."
    )

session_key = int(
    qualifying.iloc[0]["session_key"]
)

print("\nSELECTED SESSION")
print("Qualifying")
print("Session key:", session_key)

# Retrieve participating drivers
drivers = get_drivers(session_key)

print("\nAVAILABLE DRIVERS")

print(
    drivers[[
        "driver_number",
        "full_name",
        "team_name"
    ]].to_string(index=False)
)

# Test two different drivers
selected_drivers = [4, 81]

for driver_number in selected_drivers:

    selected = drivers[
        drivers["driver_number"] == driver_number
    ]

    if selected.empty:
        print("Driver unavailable:", driver_number)
        continue

    driver = selected.iloc[0]

    # Retrieve completed laps
    laps = get_laps(
        session_key,
        driver_number
    )

    # Select fastest recorded lap
    fastest = get_fastest_lap(laps)

    # Retrieve telemetry
    telemetry = get_telemetry(
        session_key,
        driver_number,
        fastest
    )

    # Retrieve positioning data
    position = get_position(
        session_key,
        driver_number,
        fastest
    )

    track = merge_position_telemetry(
        position,
        telemetry
    )

    print("\nDRIVER TELEMETRY TEST")

    print("Driver:", driver["full_name"])
    print("Team:", driver["team_name"])
    print("Lap number:", fastest["lap_number"])
    print("Lap time:", fastest["lap_duration"], "s")
    print("Top speed:", telemetry["speed"].max(), "km/h")
    print("Telemetry samples:", len(telemetry))
    print("Position samples:", len(position))
    print("Aligned samples:", len(track))

    print(
        "Estimated distance:",
        round(telemetry["Distance"].max(), 2),
        "m"
    )

    print("\nAVAILABLE TELEMETRY CHANNELS")
    print(telemetry.columns.tolist())

    print("\nFIRST FIVE TELEMETRY SAMPLES")
    print(telemetry.head().to_string(index=False))

print("\nDATA LOADER TEST FINISHED")
