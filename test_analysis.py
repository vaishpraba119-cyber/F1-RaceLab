
from pathlib import Path

import numpy as np

from data_loader import (
    get_laps,
    get_fastest_lap,
    get_telemetry
)

from analysis import run_analysis


# Test configuration
SESSION_KEY = 9689

DRIVER_A = 4
DRIVER_B = 81

NAME_A = "Lando Norris"
NAME_B = "Oscar Piastri"


print("\nLOADING DRIVER DATA")

laps_a = get_laps(
    SESSION_KEY,
    DRIVER_A
)

laps_b = get_laps(
    SESSION_KEY,
    DRIVER_B
)

fastest_a = get_fastest_lap(laps_a)

fastest_b = get_fastest_lap(laps_b)

telemetry_a = get_telemetry(
    SESSION_KEY,
    DRIVER_A,
    fastest_a
)

telemetry_b = get_telemetry(
    SESSION_KEY,
    DRIVER_B,
    fastest_b
)


# Run the complete engineering analysis
print("\nRUNNING ENGINEERING ANALYSIS")

results = run_analysis(
    telemetry_a=telemetry_a,
    telemetry_b=telemetry_b,
    driver_a=NAME_A,
    driver_b=NAME_B,
    lap_time_a=float(
        fastest_a["lap_duration"]
    ),
    lap_time_b=float(
        fastest_b["lap_duration"]
    ),
    number_of_sections=20
)


# Display overall performance
print("\nDRIVER PERFORMANCE")

print(
    results["performance"]
    .round(3)
    .set_index("Driver")
    .T
    .to_string()
)


# Display individual braking events
print("\nBRAKING EVENTS")

print(
    results["braking_events"]
    .round(3)
    .to_string(index=False)
)


# Display braking summary
print("\nBRAKING SUMMARY")

print(
    results["braking_summary"]
    .round(3)
    .to_string(index=False)
)


# Display track section analysis
print("\nTRACK SECTION ANALYSIS")

print(
    results["track_sections"][[
        "Section",
        "Start Distance (m)",
        "End Distance (m)",
        "Driver A Advantage (s)",
        "Faster Driver"
    ]]
    .round(3)
    .to_string(index=False)
)


# Display gear usage
print("\nGEAR USAGE")

print(
    results["gear_usage"]
    .round(3)
    .to_string(index=False)
)


# Validate lap time calculations
recorded_delta = (
    fastest_b["lap_duration"]
    - fastest_a["lap_duration"]
)

calculated_delta = (
    results["lap_comparison"]
    ["Time Advantage (s)"]
    .iloc[-1]
)

section_delta = (
    results["track_sections"]
    ["Driver A Advantage (s)"]
    .sum()
)

raw_delta = (
    results["lap_comparison"]
    ["Raw Sampled Delta (s)"]
    .iloc[-1]
)

print("\nLAP TIME VALIDATION")

print(
    f"Recorded lap difference: "
    f"{recorded_delta:.6f} s"
)

print(
    f"Raw sampled difference: "
    f"{raw_delta:.6f} s"
)

print(
    f"Normalised lap difference: "
    f"{calculated_delta:.6f} s"
)

print(
    f"Sum of section differences: "
    f"{section_delta:.6f} s"
)


# Check internal mathematical consistency
assert np.isclose(
    recorded_delta,
    calculated_delta,
    atol=0.000001
)

assert np.isclose(
    recorded_delta,
    section_delta,
    atol=0.000001
)

assert len(
    results["track_sections"]
) == 20

assert (
    results["braking_events"]
    ["Duration (s)"] > 0
).all()

print("\nALL CONSISTENCY CHECKS PASSED")


# Create an output folder
output_dir = (
    Path(__file__).resolve().parent
    / "outputs"
)

output_dir.mkdir(exist_ok=True)


# Export all analysis results
print("\nEXPORTING RESULTS")

for name, dataframe in results.items():

    filename = output_dir / f"{name}.csv"

    dataframe.to_csv(
        filename,
        index=False
    )

    print(f"Saved: {filename.name}")


print("\nENGINEERING ANALYSIS TEST SUCCESSFUL")
