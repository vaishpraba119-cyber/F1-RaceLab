
from pathlib import Path

import pandas as pd

from data_loader import (
    get_laps,
    get_fastest_lap,
    get_position,
    merge_position_telemetry
)

from visualisations import (
    plot_all_telemetry,
    plot_lap_delta,
    plot_braking_events,
    plot_braking_summary,
    plot_track_sections,
    plot_gear_usage,
    plot_circuit_heatmap,
    plot_performance_map
)


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"

GRAPH_DIR = BASE_DIR / "graphs"

GRAPH_DIR.mkdir(exist_ok=True)


DRIVER_A = "Lando Norris"
DRIVER_B = "Oscar Piastri"

SESSION_KEY = 9689


# Load previously exported analysis results
print("\nLOADING ANALYSIS RESULTS")

a = pd.read_csv(
    OUTPUT_DIR / "telemetry_a.csv"
)

b = pd.read_csv(
    OUTPUT_DIR / "telemetry_b.csv"
)

events = pd.read_csv(
    OUTPUT_DIR / "braking_events.csv"
)

summary = pd.read_csv(
    OUTPUT_DIR / "braking_summary.csv"
)

comparison = pd.read_csv(
    OUTPUT_DIR / "lap_comparison.csv"
)

sections = pd.read_csv(
    OUTPUT_DIR / "track_sections.csv"
)

gear_usage = pd.read_csv(
    OUTPUT_DIR / "gear_usage.csv"
)


# Restore timestamp data types
a["date"] = pd.to_datetime(
    a["date"],
    utc=True
)

b["date"] = pd.to_datetime(
    b["date"],
    utc=True
)


# Retrieve position data
print("\nLOADING CIRCUIT POSITION DATA")

lap_a = get_fastest_lap(
    get_laps(SESSION_KEY, 4)
)

lap_b = get_fastest_lap(
    get_laps(SESSION_KEY, 81)
)

position_a = get_position(
    SESSION_KEY,
    4,
    lap_a
)

position_b = get_position(
    SESSION_KEY,
    81,
    lap_b
)

track_a = merge_position_telemetry(
    position_a,
    a
)

track_b = merge_position_telemetry(
    position_b,
    b
)


# Common speed scale for both drivers
speed_range = (
    min(a["speed"].min(), b["speed"].min()),
    max(a["speed"].max(), b["speed"].max())
)


# Generate all interactive figures
print("\nGENERATING INTERACTIVE GRAPHS")

figures = {

    "telemetry_comparison":
        plot_all_telemetry(
            a,
            b,
            DRIVER_A,
            DRIVER_B
        ),

    "lap_time_delta":
        plot_lap_delta(
            comparison,
            DRIVER_A,
            DRIVER_B,
            show_raw=True
        ),

    "braking_events":
        plot_braking_events(
            events,
            DRIVER_A,
            DRIVER_B
        ),

    "braking_summary":
        plot_braking_summary(
            summary,
            DRIVER_A,
            DRIVER_B
        ),

    "track_sections":
        plot_track_sections(
            sections,
            DRIVER_A,
            DRIVER_B
        ),

    "gear_usage":
        plot_gear_usage(
            gear_usage,
            DRIVER_A,
            DRIVER_B
        ),

    "driver_a_speed_map":
        plot_circuit_heatmap(
            track_a,
            DRIVER_A,
            channel="speed",
            value_range=speed_range
        ),

    "driver_b_speed_map":
        plot_circuit_heatmap(
            track_b,
            DRIVER_B,
            channel="speed",
            value_range=speed_range
        ),

    "driver_a_braking_map":
        plot_circuit_heatmap(
            track_a,
            DRIVER_A,
            channel="brake"
        ),

    "driver_b_braking_map":
        plot_circuit_heatmap(
            track_b,
            DRIVER_B,
            channel="brake"
        ),

    "performance_map":
        plot_performance_map(
            track_a,
            comparison,
            DRIVER_A,
            DRIVER_B
        )
}


# Export interactive HTML files
for name, figure in figures.items():

    filename = GRAPH_DIR / f"{name}.html"

    figure.write_html(
        str(filename),
        include_plotlyjs=True
    )

    print(f"Saved: {filename.name}")


# Export aligned position data
track_a.to_csv(
    OUTPUT_DIR / "track_a.csv",
    index=False
)

track_b.to_csv(
    OUTPUT_DIR / "track_b.csv",
    index=False
)


print("\nTOTAL GRAPHS GENERATED:", len(figures))

print("\nVISUALISATION TEST SUCCESSFUL")

print("\nGraphs saved in:")

print(GRAPH_DIR)
