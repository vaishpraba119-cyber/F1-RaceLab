from pathlib import Path

from full_race_replay import (
    prepare_full_race,
    create_full_race_replay,
    calculate_full_race_summary,
    prepare_leaderboard
)

SESSION_KEY = 9693

DRIVERS = [
    4,   # Lando Norris
    81,  # Oscar Piastri
    16,  # Charles Leclerc
    44   # Lewis Hamilton
]

print("F1 RACELAB V4.0")
print("MULTI-DRIVER RACE REPLAY TEST")

print("\nLoading historical race data...")

result = prepare_full_race(
    session_key=SESSION_KEY,
    driver_numbers=DRIVERS,
    start_offset=120,
    duration=300,
    interval=1.0
)

print("\nSession:")
print(result["session"]["session_name"])

print("\nReplay window:")
print(result["start_time"])
print(result["end_time"])

print("\nDrivers loaded:")
print(len(result["drivers"]))

print("\nSkipped drivers:")
print(result["skipped"])

summary = calculate_full_race_summary(
    result
)

print("\nReplay summary:")
print(summary.to_string(index=False))

print("\nPreparing race leaderboard...")

leaderboard = prepare_leaderboard(
    result
)

print(leaderboard.head())

print("\nCreating animation...")

figure = create_full_race_replay(
    result,
    playback_speed=5.0
)

output_folder = Path(
    "outputs/full_race_replay"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

figure.write_html(
    str(
        output_folder / "full_race_replay.html"
    ),
    include_plotlyjs=True,
    auto_play=False
)

summary.to_csv(
    output_folder / "race_replay_summary.csv",
    index=False
)

leaderboard.to_csv(
    output_folder / "race_leaderboard.csv",
    index=False
)

for number, data in result["replay_data"].items():

    data.to_csv(
        output_folder / f"driver_{number}_replay.csv",
        index=False
    )

assert len(result["drivers"]) > 0

assert len(figure.frames) == len(
    result["time_axis"]
)

for number, data in result["replay_data"].items():

    assert len(data) == len(
        result["time_axis"]
    )

    assert data["Active"].any()

    assert data["Time"].is_monotonic_increasing

print(
    f"\nAnimation frames: "
    f"{len(figure.frames)}"
)

print("\nAll basic consistency checks passed.")

print("\nFiles exported to:")

print(output_folder.resolve())

figure.show()
