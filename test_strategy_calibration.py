from pathlib import Path

import numpy as np

from data_loader import (
    get_meetings,
    get_sessions,
    get_drivers,
    get_laps
)

from strategy_calibration import (
    calibrate_strategy,
    build_calibrated_parameters,
    run_calibrated_optimisation,
    plot_calibrated_degradation,
    plot_calibration_accuracy,
    plot_pit_loss,
    generate_calibration_insights
)

from strategy import (
    format_race_time,
    plot_strategy_timeline,
    plot_strategy_leaderboard,
    plot_strategy_lap_times
)


YEAR = 2025
GRAND_PRIX = "Bahrain"

FUEL_EFFECT = 0.03

print("=" * 65)
print("F1 RACELAB V4.0")
print("HISTORICAL STRATEGY CALIBRATION")
print("=" * 65)


# Find the historical race session.

meetings = get_meetings(YEAR)

selected = meetings[
    meetings["meeting_name"].str.contains(
        GRAND_PRIX,
        case=False,
        na=False
    )
]

if selected.empty:
    raise ValueError(
        "Grand Prix not found."
    )

meeting = selected.iloc[0]

sessions = get_sessions(
    int(meeting["meeting_key"])
)

races = sessions[
    sessions["session_name"] == "Race"
]

if races.empty:
    raise ValueError(
        "Race session not found."
    )

session = races.iloc[0]

SESSION_KEY = int(
    session["session_key"]
)

print("\nSelected race:")
print(meeting["meeting_name"])

print("\nSession key:")
print(SESSION_KEY)


# Load available drivers.

drivers = get_drivers(
    SESSION_KEY
)

DRIVERS = drivers[
    "driver_number"
].astype(int).tolist()

print("\nAvailable drivers:")
print(DRIVERS)

print(
    f"\nTotal drivers: {len(DRIVERS)}"
)


# Determine recorded race distance.

lap_counts = []

for number in DRIVERS:

    laps = get_laps(
        SESSION_KEY,
        number
    )

    if not laps.empty:

        lap_counts.append(
            int(
                laps["lap_number"].max()
            )
        )

if not lap_counts:
    raise ValueError(
        "No recorded race laps."
    )

TOTAL_LAPS = max(lap_counts)

print(
    f"\nMaximum recorded laps: {TOTAL_LAPS}"
)


# Fit historical tyre models.

print("\nCalibrating historical data...")

calibration = calibrate_strategy(
    session_key=SESSION_KEY,
    driver_numbers=DRIVERS,
    fuel_effect=FUEL_EFFECT
)

print("\nRaw eligible laps:")
print(
    len(calibration["raw_data"])
)

print("\nLaps after neutralisation filtering:")
print(
    len(calibration["green_data"])
)

print("\nFinal clean laps:")
print(
    len(calibration["clean_data"])
)

print("\nNeutralised periods:")

for period in calibration[
    "neutralised_periods"
]:

    print(period)

print("\nSkipped drivers:")
print(calibration["skipped"])

print("\nHistorical tyre models:")

print(
    calibration["summary"].to_string(
        index=False
    )
)


# Examine estimated pit loss.

pit_result = calibration[
    "pit_result"
]

print("\nPit-cycle observations:")
print(pit_result["samples"])

print("\nEstimated pit-cycle loss:")
print(pit_result["estimated_pit_loss"])


# Identify usable models.

parameters, rejected = (
    build_calibrated_parameters(
        calibration
    )
)

print("\nAccepted historical parameters:")

for compound, values in parameters.items():

    print(compound, values)

print("\nRejected models:")

for compound, reason in rejected.items():

    print(compound, reason)


# Run strategy optimisation.

print("\nOptimising race strategy...")

result = run_calibrated_optimisation(
    calibration=calibration,
    total_laps=TOTAL_LAPS,
    maximum_stops=2,
    minimum_stint=5
)

optimisation = result[
    "optimisation"
]

best = optimisation[
    "best_strategy"
]

print("\nParameter sources:")

print(
    result[
        "parameter_sources"
    ].to_string(index=False)
)

print("\nPit loss used:")
print(result["pit_loss"])

print("\nPit loss source:")
print(result["pit_source"])

print("\nBest strategy:")
print(best)

print("\nEstimated race time:")

print(
    format_race_time(
        best["Race Time (s)"]
    )
)

print("\nTop strategies:")

print(
    optimisation[
        "leaderboard"
    ].head(10).to_string(index=False)
)


# Engineering insights.

print("\nENGINEERING INSIGHTS")

for insight in (
    generate_calibration_insights(
        calibration
    )
):

    print("-", insight)


# Export results.

output_folder = Path(
    "outputs/strategy_calibration"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

calibration["raw_data"].to_csv(
    output_folder / "raw_calibration_data.csv",
    index=False
)

calibration["clean_data"].to_csv(
    output_folder / "clean_calibration_data.csv",
    index=False
)

calibration["summary"].to_csv(
    output_folder / "tyre_model_summary.csv",
    index=False
)

pit_result["data"].to_csv(
    output_folder / "pit_loss_estimates.csv",
    index=False
)

result["parameter_sources"].to_csv(
    output_folder / "calibrated_parameters.csv",
    index=False
)

optimisation["leaderboard"].to_csv(
    output_folder / "strategy_leaderboard.csv",
    index=False
)

optimisation["best_simulation"].to_csv(
    output_folder / "best_strategy_simulation.csv",
    index=False
)


# Generate interactive graphs.

figures = {
    "historical_tyre_degradation":
        plot_calibrated_degradation(
            calibration
        ),

    "model_accuracy":
        plot_calibration_accuracy(
            calibration
        ),

    "pit_loss":
        plot_pit_loss(
            calibration
        ),

    "strategy_timeline":
        plot_strategy_timeline(
            optimisation["best_simulation"]
        ),

    "strategy_leaderboard":
        plot_strategy_leaderboard(
            optimisation
        ),

    "strategy_lap_times":
        plot_strategy_lap_times(
            optimisation["best_simulation"]
        )
}

for name, figure in figures.items():

    figure.write_html(
        str(
            output_folder / f"{name}.html"
        ),
        include_plotlyjs=True
    )


# Basic consistency checks.

assert len(
    calibration["clean_data"]
) <= len(
    calibration["raw_data"]
)

assert len(
    optimisation["best_simulation"]
) == TOTAL_LAPS

assert np.isfinite(
    best["Race Time (s)"]
)

assert np.isclose(
    optimisation[
        "best_simulation"
    ]["Lap Time (s)"].sum(),
    best["Race Time (s)"],
    atol=0.001
)

for compound, values in (
    result["parameters"].items()
):

    assert values["base_time"] > 0

    assert values["degradation"] >= 0

    assert values["max_age"] > 0

print("\nAll basic consistency checks passed.")

print("\nFiles exported to:")

print(output_folder.resolve())

figures[
    "historical_tyre_degradation"
].show()

figures[
    "strategy_timeline"
].show()
