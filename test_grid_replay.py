from pathlib import Path

from grid_replay import (
    get_all_driver_numbers,
    prepare_grid_replay,
    create_grid_replay,
    export_grid_data
)

from full_race_replay import (
    calculate_full_race_summary
)

SESSION_KEY = 9693

print("F1 RACELAB V4.0")
print("FULL-GRID REPLAY TEST")

drivers = get_all_driver_numbers(
    SESSION_KEY
)

print("\nAvailable drivers:")
print(drivers)

print(
    f"\nTotal available drivers: "
    f"{len(drivers)}"
)

print("\nLoading race replay...")

result = prepare_grid_replay(
    session_key=SESSION_KEY,
    driver_numbers=drivers,
    start_offset=1200,
    duration=120,
    interval=2.0,
    include_speed=True
)

print("\nDrivers loaded:")
print(len(result["drivers"]))

print("\nSkipped drivers:")
print(result["skipped"])

print("\nAdditional telemetry errors:")
print(result["information_errors"])

summary = calculate_full_race_summary(
    result
)

print("\nReplay summary:")
print(summary.to_string(index=False))

print("\nCreating full-grid animation...")

figure = create_grid_replay(
    result,
    playback_speed=10.0
)

output_folder = Path(
    "outputs/grid_replay"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

figure.write_html(
    str(
        output_folder / "grid_replay.html"
    ),
    include_plotlyjs=True,
    auto_play=False
)

summary.to_csv(
    output_folder / "grid_summary.csv",
    index=False
)

grid_data = export_grid_data(
    result
)

grid_data.to_csv(
    output_folder / "grid_telemetry.csv",
    index=False
)

print("\nTelemetry availability:")

for column in ["Lap", "Tyre Age", "Speed"]:

    available = (
        grid_data[column].notna().mean()
        * 100
    )

    print(
        f"{column}: "
        f"{available:.2f}% available"
    )

compound_coverage = (
    (
        grid_data["Compound"] != "UNKNOWN"
    ).mean() * 100
)

print(
    f"Tyre compound: "
    f"{compound_coverage:.2f}% available"
)

print("\nSample leaderboard data:")

print(
    grid_data[
        [
            "Position",
            "Driver",
            "Lap",
            "Compound",
            "Tyre Age",
            "Speed"
        ]
    ].head(20).to_string(index=False)
)

assert len(result["drivers"]) > 0

assert len(figure.frames) == len(
    result["time_axis"]
)

assert len(grid_data) == (
    len(result["drivers"])
    * len(result["time_axis"])
)

print(
    f"\nAnimation frames: "
    f"{len(figure.frames)}"
)

print(
    f"Exported telemetry rows: "
    f"{len(grid_data)}"
)

print("\nAll basic consistency checks passed.")

print("\nFiles exported to:")
print(output_folder.resolve())

figure.show()
