from pathlib import Path

from data_loader import (
    get_laps,
    get_fastest_lap
)

from race_replay import (
    load_replay_data,
    create_race_replay,
    calculate_replay_summary
)

SESSION_KEY = 9689

DRIVER_A = 4
DRIVER_B = 81

NAME_A = "Norris"
NAME_B = "Piastri"

print("F1 RACELAB V4.0")
print("RACE REPLAY TEST")

print("\nLoading driver laps...")

laps_a = get_laps(
    SESSION_KEY,
    DRIVER_A
)

laps_b = get_laps(
    SESSION_KEY,
    DRIVER_B
)

lap_a = get_fastest_lap(laps_a)
lap_b = get_fastest_lap(laps_b)

print(
    f"{NAME_A}: "
    f"{lap_a['lap_duration']:.3f} s"
)

print(
    f"{NAME_B}: "
    f"{lap_b['lap_duration']:.3f} s"
)

print("\nLoading position and telemetry data...")

data_a = load_replay_data(
    SESSION_KEY,
    DRIVER_A,
    lap_a
)

data_b = load_replay_data(
    SESSION_KEY,
    DRIVER_B,
    lap_b
)

print("\nPosition samples:")

print(f"{NAME_A}: {len(data_a)}")
print(f"{NAME_B}: {len(data_b)}")

summary = calculate_replay_summary(
    data_a,
    data_b,
    NAME_A,
    NAME_B
)

print("\nReplay summary:")

print(
    summary.to_string(index=False)
)

print("\nCreating animated replay...")

figure = create_race_replay(
    data_a,
    data_b,
    NAME_A,
    NAME_B,
    colour_a="#FF8700",
    colour_b="#FFFFFF",
    interval=0.2,
    playback_speed=2.0
)

output_folder = Path(
    "outputs/race_replay"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

figure.write_html(
    str(
        output_folder / "race_replay.html"
    ),
    include_plotlyjs=True,
    auto_play=False
)

summary.to_csv(
    output_folder / "replay_summary.csv",
    index=False
)

data_a.to_csv(
    output_folder / "norris_replay.csv",
    index=False
)

data_b.to_csv(
    output_folder / "piastri_replay.csv",
    index=False
)

assert len(data_a) > 0
assert len(data_b) > 0

assert data_a[
    "Replay Time"
].is_monotonic_increasing

assert data_b[
    "Replay Time"
].is_monotonic_increasing

assert len(figure.frames) > 0

print(
    f"\nAnimation frames: "
    f"{len(figure.frames)}"
)

print("\nAll basic replay checks passed.")

print("\nFiles exported to:")

print(output_folder.resolve())

figure.show()
