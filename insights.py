
import numpy as np
import pandas as pd


# Generate automated engineering observations
def generate_insights(results, driver_a, driver_b):

    performance = results["performance"]

    braking = results["braking_summary"]

    sections = results["track_sections"]

    comparison = results["lap_comparison"]

    gear_usage = results["gear_usage"]

    a = performance.iloc[0]

    b = performance.iloc[1]

    insights = {
        "Lap Performance": [],
        "Speed Analysis": [],
        "Braking Analysis": [],
        "Throttle Analysis": [],
        "Track Section Analysis": [],
        "Gear Analysis": [],
        "Data Quality": []
    }

    # Compare recorded lap times
    delta = b["Lap Time (s)"] - a["Lap Time (s)"]

    if np.isclose(delta, 0):

        insights["Lap Performance"].append(
            "Both drivers recorded equal lap times."
        )

    else:

        faster = driver_a if delta > 0 else driver_b

        slower = driver_b if delta > 0 else driver_a

        percentage = (
            abs(delta)
            / max(a["Lap Time (s)"], b["Lap Time (s)"])
            * 100
        )

        insights["Lap Performance"].append(
            f"{faster} completed the selected lap "
            f"{abs(delta):.3f} seconds faster than "
            f"{slower}, representing a "
            f"{percentage:.3f}% reduction in lap time."
        )

    # Compare maximum speed
    speed_difference = (
        a["Top Speed (km/h)"]
        - b["Top Speed (km/h)"]
    )

    if np.isclose(speed_difference, 0):

        insights["Speed Analysis"].append(
            "Both drivers achieved the same "
            "maximum recorded speed."
        )

    else:

        faster = (
            driver_a
            if speed_difference > 0
            else driver_b
        )

        insights["Speed Analysis"].append(
            f"{faster} achieved a maximum speed "
            f"advantage of "
            f"{abs(speed_difference):.1f} km/h."
        )

    # Compare minimum speed
    minimum_difference = (
        a["Minimum Speed (km/h)"]
        - b["Minimum Speed (km/h)"]
    )

    insights["Speed Analysis"].append(
        f"Minimum recorded speeds were "
        f"{a['Minimum Speed (km/h)']:.1f} km/h "
        f"for {driver_a} and "
        f"{b['Minimum Speed (km/h)']:.1f} km/h "
        f"for {driver_b}. These minima may occur "
        f"at different track locations."
    )

    # Compare braking performance
    brake_a = braking.iloc[0]

    brake_b = braking.iloc[1]

    distance_difference = (
        brake_a["Total Braking Distance (m)"]
        - brake_b["Total Braking Distance (m)"]
    )

    if not np.isclose(distance_difference, 0):

        shorter = (
            driver_b
            if distance_difference > 0
            else driver_a
        )

        insights["Braking Analysis"].append(
            f"{shorter} recorded approximately "
            f"{abs(distance_difference):.1f} m less "
            f"total detected braking distance."
        )

    insights["Braking Analysis"].append(
        f"{driver_a} recorded "
        f"{int(brake_a['Braking Events'])} braking "
        f"events, compared with "
        f"{int(brake_b['Braking Events'])} "
        f"for {driver_b}."
    )

    g_difference = (
        brake_a["Peak Braking G"]
        - brake_b["Peak Braking G"]
    )

    if not np.isclose(g_difference, 0):

        higher_g = (
            driver_a
            if g_difference > 0
            else driver_b
        )

        insights["Braking Analysis"].append(
            f"{higher_g} achieved a higher estimated "
            f"peak longitudinal deceleration by "
            f"{abs(g_difference):.2f} G."
        )

    # Compare full throttle usage
    throttle_difference = (
        a["Full Throttle Percentage (%)"]
        - b["Full Throttle Percentage (%)"]
    )

    if not np.isclose(throttle_difference, 0):

        higher_throttle = (
            driver_a
            if throttle_difference > 0
            else driver_b
        )

        insights["Throttle Analysis"].append(
            f"{higher_throttle} spent "
            f"{abs(throttle_difference):.2f} percentage "
            f"points more of the sampled lap at "
            f"95% throttle or above."
        )

    insights["Throttle Analysis"].append(
        f"Full-throttle durations were "
        f"{a['Full Throttle Duration (s)']:.2f} s "
        f"and "
        f"{b['Full Throttle Duration (s)']:.2f} s "
        f"for {driver_a} and {driver_b}, respectively."
    )

    # Analyse track section performance
    advantages = sections[
        "Driver A Advantage (s)"
    ]

    tolerance = 0.0001

    sections_a = sections[
        advantages > tolerance
    ]

    sections_b = sections[
        advantages < -tolerance
    ]

    equal_sections = (
        len(sections)
        - len(sections_a)
        - len(sections_b)
    )

    insights["Track Section Analysis"].append(
        f"{driver_a} was faster in "
        f"{len(sections_a)} of {len(sections)} "
        f"estimated track sections, while "
        f"{driver_b} was faster in "
        f"{len(sections_b)}. "
        f"{equal_sections} sections were effectively equal."
    )

    # Identify strongest section for Driver A
    if not sections_a.empty:

        best_a = sections_a.loc[
            sections_a[
                "Driver A Advantage (s)"
            ].idxmax()
        ]

        insights["Track Section Analysis"].append(
            f"{driver_a}'s largest estimated section "
            f"advantage occurred in Section "
            f"{int(best_a['Section'])}, gaining "
            f"{best_a['Driver A Advantage (s)']:.3f} s "
            f"between approximately "
            f"{best_a['Start Distance (m)']:.0f} m "
            f"and {best_a['End Distance (m)']:.0f} m."
        )

    # Identify strongest section for Driver B
    if not sections_b.empty:

        best_b = sections_b.loc[
            sections_b[
                "Driver A Advantage (s)"
            ].idxmin()
        ]

        insights["Track Section Analysis"].append(
            f"{driver_b}'s largest estimated section "
            f"advantage occurred in Section "
            f"{int(best_b['Section'])}, gaining "
            f"{abs(best_b['Driver A Advantage (s)']):.3f} s "
            f"between approximately "
            f"{best_b['Start Distance (m)']:.0f} m "
            f"and {best_b['End Distance (m)']:.0f} m."
        )

    # Analyse dominant gear usage
    if not gear_usage.empty:

        for driver in [driver_a, driver_b]:

            selected = gear_usage[
                gear_usage["Driver"] == driver
            ]

            if not selected.empty:

                dominant = selected.loc[
                    selected["Usage (%)"].idxmax()
                ]

                insights["Gear Analysis"].append(
                    f"{driver} spent the largest "
                    f"proportion of sampled time in "
                    f"gear {int(dominant['Gear'])}, "
                    f"representing "
                    f"{dominant['Usage (%)']:.2f}% "
                    f"of recorded gear usage."
                )

    # Assess telemetry coverage
    coverage_a = a["Telemetry Coverage (%)"]

    coverage_b = b["Telemetry Coverage (%)"]

    insights["Data Quality"].append(
        f"Sampled time coverage was "
        f"{coverage_a:.2f}% for {driver_a} and "
        f"{coverage_b:.2f}% for {driver_b}."
    )

    if min(coverage_a, coverage_b) < 95:

        insights["Data Quality"].append(
            "Warning: At least one selected lap "
            "has less than 95% sampled time coverage. "
            "Interpret the comparison cautiously."
        )

    # Compare raw and normalised lap differences
    raw_delta = comparison[
        "Raw Sampled Delta (s)"
    ].iloc[-1]

    normalised_delta = comparison[
        "Time Advantage (s)"
    ].iloc[-1]

    discrepancy = abs(
        raw_delta - normalised_delta
    )

    insights["Data Quality"].append(
        f"The raw sampled and normalised lap "
        f"differences differ by "
        f"{discrepancy:.3f} seconds."
    )

    insights["Data Quality"].append(
        "Normalised timing is constrained to match "
        "the recorded lap duration. Agreement is "
        "not independent validation."
    )

    return insights


# Convert observations into an exportable report
def generate_report(
    results,
    driver_a,
    driver_b
):

    insights = generate_insights(
        results,
        driver_a,
        driver_b
    )

    lines = [
        "# F1 RaceLab Engineering Report",
        "",
        f"## {driver_a} vs {driver_b}",
        ""
    ]

    metadata = results.get("metadata")

    if isinstance(metadata, pd.DataFrame):

        if not metadata.empty:

            for key, value in metadata.iloc[0].items():

                lines.append(
                    f"- {key}: {value}"
                )

            lines.append("")

    for category, observations in insights.items():

        lines.append(f"## {category}")

        lines.append("")

        for observation in observations:

            lines.append(f"- {observation}")

        lines.append("")

    lines.extend([
        "## Engineering Limitations",
        "",
        "These observations describe telemetry differences.",
        "They do not establish the causes of lap-time differences.",
        "Tyre condition, traffic, fuel load, setup and weather "
        "may influence the results.",
        "",
        "Data source: OpenF1 API.",
        "",
        "Generated by F1 RaceLab."
    ])

    return "\n".join(lines)
