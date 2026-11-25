
import numpy as np
import pandas as pd

GRAVITY = 9.81


# Validate and prepare telemetry data
def prepare_telemetry(telemetry):

    data = telemetry.copy()

    required = [
        "Time",
        "Distance",
        "speed",
        "brake",
        "throttle"
    ]

    for column in required:

        if column not in data.columns:
            raise ValueError(
                f"Missing telemetry channel: {column}"
            )

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(subset=required)

    data = data.sort_values("Time")

    data = data.drop_duplicates("Time")

    data = data.reset_index(drop=True)

    if len(data) < 3:
        raise ValueError(
            "Insufficient telemetry samples."
        )

    # Convert speed to metres per second
    data["Speed_ms"] = data["speed"] / 3.6

    time = data["Time"].to_numpy()

    speed = data["Speed_ms"].to_numpy()

    # Numerical differentiation
    data["Acceleration"] = np.gradient(
        speed,
        time
    )

    # Smooth acceleration to reduce noise
    data["Smoothed_Acceleration"] = (
        data["Acceleration"]
        .rolling(
            window=3,
            center=True,
            min_periods=1
        )
        .mean()
    )

    # Positive values represent deceleration
    data["Deceleration"] = (
        -data["Smoothed_Acceleration"]
    )

    # Convert deceleration to G-force
    data["Braking_G"] = (
        data["Deceleration"] / GRAVITY
    )

    # Convert brake activation to binary values
    data["Brake_Status"] = (
        data["brake"] > 0
    ).astype(int)

    # Calculate normalised lap progress
    distance = (
        data["Distance"]
        - data["Distance"].iloc[0]
    )

    if distance.max() <= 0:
        raise ValueError(
            "Invalid lap distance."
        )

    data["Lap_Progress"] = (
        distance / distance.max()
    )

    return data


# Calculate time spent in a selected state
def calculate_state_duration(data, condition):

    time = data["Time"].to_numpy()

    dt = np.diff(time)

    condition = np.asarray(
        condition,
        dtype=bool
    )

    return float(
        np.sum(dt[condition[:-1]])
    )


# Calculate overall driver performance
def calculate_performance(
    telemetry,
    driver_name,
    lap_time
):

    data = prepare_telemetry(telemetry)

    sampled_duration = (
        data["Time"].iloc[-1]
        - data["Time"].iloc[0]
    )

    distance = (
        data["Distance"].iloc[-1]
        - data["Distance"].iloc[0]
    )

    braking_duration = calculate_state_duration(
        data,
        data["Brake_Status"] == 1
    )

    full_throttle_duration = (
        calculate_state_duration(
            data,
            data["throttle"] >= 95
        )
    )

    coasting_condition = np.logical_and(
        data["throttle"] <= 5,
        data["Brake_Status"] == 0
    )

    coasting_duration = (
        calculate_state_duration(
            data,
            coasting_condition
        )
    )

    braking_samples = data[
        data["Brake_Status"] == 1
    ]

    peak_braking_g = (
        max(0, braking_samples["Braking_G"].max())
        if not braking_samples.empty
        else 0
    )

    results = {
        "Driver": driver_name,

        "Lap Time (s)": lap_time,

        "Top Speed (km/h)":
            data["speed"].max(),

        "Minimum Speed (km/h)":
            data["speed"].min(),

        "Average Sampled Speed (km/h)":
            data["speed"].mean(),

        "Estimated Average Speed (km/h)":
            distance / sampled_duration * 3.6,

        "Estimated Distance (m)":
            distance,

        "Telemetry Samples":
            len(data),

        "Sampled Duration (s)":
            sampled_duration,

        "Telemetry Coverage (%)":
            sampled_duration / lap_time * 100,

        "Peak Braking G":
            peak_braking_g,

        "Peak Acceleration G":
            max(
                0,
                data["Smoothed_Acceleration"].max()
                / GRAVITY
            ),

        "Braking Duration (s)":
            braking_duration,

        "Braking Percentage (%)":
            braking_duration
            / sampled_duration * 100,

        "Full Throttle Duration (s)":
            full_throttle_duration,

        "Full Throttle Percentage (%)":
            full_throttle_duration
            / sampled_duration * 100,

        "Coasting Duration (s)":
            coasting_duration
    }

    return results


# Automatically detect individual braking events
def detect_braking_events(
    telemetry,
    driver_name,
    minimum_duration=0.3
):

    data = prepare_telemetry(telemetry)

    brake = data["Brake_Status"].to_numpy()

    # Detect rising and falling edges
    transitions = np.diff(
        np.concatenate(
            ([0], brake, [0])
        )
    )

    starts = np.where(
        transitions == 1
    )[0]

    ends = np.where(
        transitions == -1
    )[0]

    results = []

    event_number = 1

    for start, end in zip(starts, ends):

        last = min(
            end,
            len(data) - 1
        )

        duration = (
            data["Time"].iloc[last]
            - data["Time"].iloc[start]
        )

        # Remove very short braking events
        if duration < minimum_duration:
            continue

        event = data.iloc[start:last + 1]

        initial_speed = (
            event["speed"].iloc[0]
        )

        release_speed = (
            event["speed"].iloc[-1]
        )

        minimum_speed = (
            event["speed"].min()
        )

        start_distance = (
            event["Distance"].iloc[0]
        )

        end_distance = (
            event["Distance"].iloc[-1]
        )

        braking_distance = (
            end_distance - start_distance
        )

        speed_reduction = (
            initial_speed - release_speed
        )

        average_deceleration = (
            speed_reduction / 3.6
        ) / duration

        peak_deceleration = max(
            0,
            event["Deceleration"].max()
        )

        results.append({
            "Driver": driver_name,

            "Braking Event": event_number,

            "Start Distance (m)":
                start_distance,

            "End Distance (m)":
                end_distance,

            "Start Progress (%)":
                event["Lap_Progress"].iloc[0]
                * 100,

            "Braking Distance (m)":
                braking_distance,

            "Duration (s)":
                duration,

            "Initial Speed (km/h)":
                initial_speed,

            "Release Speed (km/h)":
                release_speed,

            "Minimum Speed (km/h)":
                minimum_speed,

            "Speed Reduction (km/h)":
                speed_reduction,

            "Average Deceleration (m/s2)":
                average_deceleration,

            "Peak Deceleration (m/s2)":
                peak_deceleration,

            "Peak Braking G":
                peak_deceleration / GRAVITY,

            "Boundary Event":
                start == 0 or end == len(data)
        })

        event_number += 1

    columns = [
        "Driver",
        "Braking Event",
        "Start Distance (m)",
        "End Distance (m)",
        "Start Progress (%)",
        "Braking Distance (m)",
        "Duration (s)",
        "Initial Speed (km/h)",
        "Release Speed (km/h)",
        "Minimum Speed (km/h)",
        "Speed Reduction (km/h)",
        "Average Deceleration (m/s2)",
        "Peak Deceleration (m/s2)",
        "Peak Braking G",
        "Boundary Event"
    ]

    return pd.DataFrame(
        results,
        columns=columns
    )


# Summarise braking performance
def calculate_braking_summary(
    events,
    driver_name
):

    if events.empty:

        return {
            "Driver": driver_name,
            "Braking Events": 0,
            "Total Braking Distance (m)": 0,
            "Total Braking Duration (s)": 0,
            "Peak Braking G": 0,
            "Average Deceleration (m/s2)": 0
        }

    total_duration = (
        events["Duration (s)"].sum()
    )

    weighted_deceleration = np.average(
        events["Average Deceleration (m/s2)"],
        weights=events["Duration (s)"]
    )

    return {
        "Driver": driver_name,

        "Braking Events":
            len(events),

        "Total Braking Distance (m)":
            events["Braking Distance (m)"].sum(),

        "Total Braking Duration (s)":
            total_duration,

        "Peak Braking G":
            events["Peak Braking G"].max(),

        "Average Deceleration (m/s2)":
            weighted_deceleration
    }


# Interpolate telemetry onto common lap progress
def interpolate_channel(
    data,
    progress,
    column
):

    unique = data.drop_duplicates(
        "Lap_Progress"
    )

    return np.interp(
        progress,
        unique["Lap_Progress"],
        unique[column]
    )


# Compare two drivers across a complete lap
def compare_laps(
    telemetry_a,
    telemetry_b,
    lap_time_a,
    lap_time_b,
    resolution=2000
):

    a = prepare_telemetry(telemetry_a)

    b = prepare_telemetry(telemetry_b)

    progress = np.linspace(
        0,
        1,
        resolution
    )

    # Estimate a common reference distance
    reference_distance = (
        a["Distance"].max()
        + b["Distance"].max()
    ) / 2

    time_a = interpolate_channel(
        a,
        progress,
        "Time"
    )

    time_b = interpolate_channel(
        b,
        progress,
        "Time"
    )

    # Reset elapsed time
    time_a = time_a - time_a[0]

    time_b = time_b - time_b[0]

    # Preserve the unadjusted sampled time delta
    raw_delta = time_b - time_a

    # Scale sampled time to recorded lap time
    time_a = (
        time_a / time_a[-1]
    ) * lap_time_a

    time_b = (
        time_b / time_b[-1]
    ) * lap_time_b

    # Positive values favour Driver A
    delta = time_b - time_a

    comparison = pd.DataFrame({
        "Lap Progress (%)":
            progress * 100,

        "Distance (m)":
            progress * reference_distance,

        "Driver A Time (s)":
            time_a,

        "Driver B Time (s)":
            time_b,

        "Driver A Speed (km/h)":
            interpolate_channel(
                a, progress, "speed"
            ),

        "Driver B Speed (km/h)":
            interpolate_channel(
                b, progress, "speed"
            ),

        "Driver A Throttle (%)":
            interpolate_channel(
                a, progress, "throttle"
            ),

        "Driver B Throttle (%)":
            interpolate_channel(
                b, progress, "throttle"
            ),

        "Raw Sampled Delta (s)":
            raw_delta,

        "Time Advantage (s)":
            delta
    })

    # Calculate local time gain per metre
    comparison["Local Time Gain"] = np.gradient(
        delta,
        comparison["Distance (m)"]
    )

    # Reduce noise in the local gain signal
    comparison["Smoothed Local Gain"] = (
        comparison["Local Time Gain"]
        .rolling(
            window=21,
            center=True,
            min_periods=1
        )
        .mean()
    )

    return comparison


# Automatically compare equal-length track sections
def analyse_track_sections(
    comparison,
    driver_a,
    driver_b,
    number_of_sections=20
):

    boundaries = np.linspace(
        0,
        100,
        number_of_sections + 1
    )

    progress = comparison[
        "Lap Progress (%)"
    ].to_numpy()

    results = []

    for i in range(number_of_sections):

        start = boundaries[i]

        end = boundaries[i + 1]

        section_progress = np.linspace(
            start,
            end,
            100
        )

        time_a = np.interp(
            [start, end],
            progress,
            comparison["Driver A Time (s)"]
        )

        time_b = np.interp(
            [start, end],
            progress,
            comparison["Driver B Time (s)"]
        )

        section_time_a = (
            time_a[1] - time_a[0]
        )

        section_time_b = (
            time_b[1] - time_b[0]
        )

        advantage = (
            section_time_b - section_time_a
        )

        speed_a = np.interp(
            section_progress,
            progress,
            comparison["Driver A Speed (km/h)"]
        )

        speed_b = np.interp(
            section_progress,
            progress,
            comparison["Driver B Speed (km/h)"]
        )

        if np.isclose(advantage, 0, atol=0.0001):
            faster_driver = "Equal"

        elif advantage > 0:
            faster_driver = driver_a

        else:
            faster_driver = driver_b

        results.append({
            "Section": i + 1,

            "Start Progress (%)":
                start,

            "End Progress (%)":
                end,

            "Start Distance (m)":
                np.interp(
                    start,
                    progress,
                    comparison["Distance (m)"]
                ),

            "End Distance (m)":
                np.interp(
                    end,
                    progress,
                    comparison["Distance (m)"]
                ),

            "Driver A Section Time (s)":
                section_time_a,

            "Driver B Section Time (s)":
                section_time_b,

            "Driver A Minimum Speed (km/h)":
                speed_a.min(),

            "Driver B Minimum Speed (km/h)":
                speed_b.min(),

            "Driver A Maximum Speed (km/h)":
                speed_a.max(),

            "Driver B Maximum Speed (km/h)":
                speed_b.max(),

            "Driver A Advantage (s)":
                advantage,

            "Faster Driver":
                faster_driver
        })

    return pd.DataFrame(results)


# Calculate time spent in each gear
def analyse_gear_usage(
    telemetry,
    driver_name
):

    data = prepare_telemetry(telemetry)

    if "n_gear" not in data.columns:
        return pd.DataFrame()

    time = data["Time"].to_numpy()

    dt = np.diff(time)

    gears = data["n_gear"].to_numpy()[:-1]

    gear_data = pd.DataFrame({
        "Gear": gears,
        "Duration (s)": dt
    })

    summary = (
        gear_data
        .dropna()
        .groupby("Gear", as_index=False)
        ["Duration (s)"]
        .sum()
    )

    summary["Driver"] = driver_name

    summary["Usage (%)"] = (
        summary["Duration (s)"]
        / summary["Duration (s)"].sum()
        * 100
    )

    return summary


# Run the complete engineering analysis
def run_analysis(
    telemetry_a,
    telemetry_b,
    driver_a,
    driver_b,
    lap_time_a,
    lap_time_b,
    number_of_sections=20
):

    prepared_a = prepare_telemetry(
        telemetry_a
    )

    prepared_b = prepare_telemetry(
        telemetry_b
    )

    performance = pd.DataFrame([
        calculate_performance(
            prepared_a,
            driver_a,
            lap_time_a
        ),

        calculate_performance(
            prepared_b,
            driver_b,
            lap_time_b
        )
    ])

    braking_a = detect_braking_events(
        prepared_a,
        driver_a
    )

    braking_b = detect_braking_events(
        prepared_b,
        driver_b
    )

    braking_events = pd.concat(
        [braking_a, braking_b],
        ignore_index=True
    )

    braking_summary = pd.DataFrame([
        calculate_braking_summary(
            braking_a,
            driver_a
        ),

        calculate_braking_summary(
            braking_b,
            driver_b
        )
    ])

    comparison = compare_laps(
        prepared_a,
        prepared_b,
        lap_time_a,
        lap_time_b
    )

    sections = analyse_track_sections(
        comparison,
        driver_a,
        driver_b,
        number_of_sections
    )

    gear_usage = pd.concat([
        analyse_gear_usage(
            prepared_a,
            driver_a
        ),

        analyse_gear_usage(
            prepared_b,
            driver_b
        )
    ], ignore_index=True)

    return {
        "telemetry_a": prepared_a,
        "telemetry_b": prepared_b,
        "performance": performance,
        "braking_events": braking_events,
        "braking_summary": braking_summary,
        "lap_comparison": comparison,
        "track_sections": sections,
        "gear_usage": gear_usage
    }
