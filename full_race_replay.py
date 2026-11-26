import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_loader import (
    get_data,
    get_drivers,
    get_laps,
    get_fastest_lap,
    get_position
)


def load_session(session_key):

    sessions = get_data(
        "sessions",
        {"session_key": session_key}
    )

    if sessions.empty:
        raise ValueError("Session not found.")

    return sessions.iloc[0]


def load_race_positions(
    session_key,
    driver_number,
    start_time,
    end_time
):

    data = get_data(
        "location",
        {
            "session_key": session_key,
            "driver_number": driver_number,
            "date>": start_time.isoformat(),
            "date<": end_time.isoformat()
        }
    )

    if data.empty:
        return data

    data["date"] = pd.to_datetime(
        data["date"],
        utc=True
    )

    data = data.dropna(
        subset=["date", "x", "y"]
    )

    data = data.sort_values("date")

    data = data.drop_duplicates("date")

    data["Race Time"] = (
        data["date"] - start_time
    ).dt.total_seconds()

    return data.reset_index(drop=True)


def load_circuit(session_key, driver_number):

    laps = get_laps(
        session_key,
        driver_number
    )

    lap = get_fastest_lap(laps)

    position = get_position(
        session_key,
        driver_number,
        lap
    )

    if position.empty:
        raise ValueError(
            "Circuit position data unavailable."
        )

    return position.dropna(
        subset=["x", "y"]
    )


def load_race_order(session_key):

    positions = get_data(
        "position",
        {"session_key": session_key}
    )

    if positions.empty:
        return positions

    positions["date"] = pd.to_datetime(
        positions["date"],
        utc=True
    )

    return positions.sort_values("date")


def interpolate_driver(
    data,
    time_axis,
    maximum_gap=3.0
):

    original_time = data[
        "Race Time"
    ].to_numpy()

    x = np.interp(
        time_axis,
        original_time,
        data["x"].to_numpy(),
        left=np.nan,
        right=np.nan
    )

    y = np.interp(
        time_axis,
        original_time,
        data["y"].to_numpy(),
        left=np.nan,
        right=np.nan
    )

    right = np.searchsorted(
        original_time,
        time_axis,
        side="left"
    )

    left = np.clip(
        right - 1,
        0,
        len(original_time) - 1
    )

    right = np.clip(
        right,
        0,
        len(original_time) - 1
    )

    gaps = (
        original_time[right]
        - original_time[left]
    )

    valid = np.logical_and(
        time_axis >= original_time[0],
        time_axis <= original_time[-1]
    )

    valid = np.logical_and(
        valid,
        gaps <= maximum_gap
    )

    x[~valid] = np.nan
    y[~valid] = np.nan

    return pd.DataFrame({
        "Time": time_axis,
        "x": x,
        "y": y,
        "Active": valid
    })


def prepare_full_race(
    session_key,
    driver_numbers,
    start_offset=120,
    duration=300,
    interval=1.0
):

    if interval <= 0 or duration <= 0:
        raise ValueError(
            "Duration and interval must be positive."
        )

    if start_offset < 0:
        raise ValueError(
            "Start offset cannot be negative."
        )

    session = load_session(session_key)

    session_start = pd.to_datetime(
        session["date_start"],
        utc=True
    )

    session_end = pd.to_datetime(
        session["date_end"],
        utc=True
    )

    start_time = (
        session_start
        + pd.Timedelta(seconds=start_offset)
    )

    end_time = min(
        start_time
        + pd.Timedelta(seconds=duration),
        session_end
    )

    if end_time <= start_time:
        raise ValueError(
            "Selected window is outside the session."
        )

    actual_duration = (
        end_time - start_time
    ).total_seconds()

    time_axis = np.arange(
        0,
        actual_duration,
        interval
    )

    drivers = get_drivers(session_key)

    replay_data = {}
    driver_details = {}

    skipped = []

    for number in driver_numbers:

        print(f"Loading driver #{number}...")

        selected = drivers[
            drivers["driver_number"] == number
        ]

        if selected.empty:

            skipped.append(number)
            continue

        driver = selected.iloc[0]

        data = load_race_positions(
            session_key,
            number,
            start_time,
            end_time
        )

        if len(data) < 3:

            skipped.append(number)
            continue

        replay = interpolate_driver(
            data,
            time_axis
        )

        if not replay["Active"].any():

            skipped.append(number)
            continue

        colour = str(
            driver.get("team_colour", "FFFFFF")
        ).strip("#")

        if len(colour) != 6:
            colour = "FFFFFF"

        try:
            int(colour, 16)
        except ValueError:
            colour = "FFFFFF"

        replay_data[number] = replay

        driver_details[number] = {
            "Name": driver["full_name"],
            "Acronym": driver["name_acronym"],
            "Team": driver["team_name"],
            "Colour": "#" + colour,
            "Samples": len(data)
        }

    if not replay_data:

        raise ValueError(
            "No suitable driver position data."
        )

    circuit = load_circuit(
        session_key,
        next(iter(replay_data))
    )

    race_order = load_race_order(
        session_key
    )

    return {
        "session": session,
        "session_start": session_start,
        "start_time": start_time,
        "end_time": end_time,
        "time_axis": time_axis,
        "drivers": driver_details,
        "replay_data": replay_data,
        "circuit": circuit,
        "race_order": race_order,
        "skipped": skipped,
        "interval": interval
    }


def prepare_leaderboard(result):

    race_order = result["race_order"]

    timestamps = (
        result["start_time"]
        + pd.to_timedelta(
            result["time_axis"],
            unit="s"
        )
    )

    leaderboard = pd.DataFrame({
        "date": timestamps
    })

    for number in result["drivers"]:

        if race_order.empty:

            leaderboard[number] = np.nan
            continue

        positions = race_order[
            race_order["driver_number"] == number
        ][["date", "position"]]

        if positions.empty:

            leaderboard[number] = np.nan
            continue

        positions = positions.sort_values("date")

        matched = pd.merge_asof(
            leaderboard[["date"]],
            positions,
            on="date",
            direction="backward"
        )

        leaderboard[number] = matched["position"]

    return leaderboard


def format_elapsed(seconds):

    seconds = max(0, int(seconds))

    hours = seconds // 3600

    minutes = (seconds % 3600) // 60

    remaining = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining:02d}"
    )


def create_full_race_replay(
    result,
    playback_speed=5.0
):

    if playback_speed <= 0:
        raise ValueError(
            "Playback speed must be positive."
        )

    circuit = result["circuit"]

    drivers = result["drivers"]

    replay_data = result["replay_data"]

    time_axis = result["time_axis"]

    leaderboard = prepare_leaderboard(result)

    driver_numbers = list(drivers)

    session_offset = (
        result["start_time"]
        - result["session_start"]
    ).total_seconds()

    symbols = [
        "circle",
        "diamond",
        "square",
        "triangle-up",
        "cross",
        "triangle-down"
    ]

    def car_trace(number, index):

        row = replay_data[number].iloc[index]

        driver = drivers[number]

        active = bool(row["Active"])

        symbol = symbols[
            driver_numbers.index(number)
            % len(symbols)
        ]

        position = leaderboard.iloc[index][number]

        position_text = (
            f"P{int(position)}"
            if pd.notna(position)
            else "Position unavailable"
        )

        return go.Scatter(
            x=[row["x"]] if active else [],
            y=[row["y"]] if active else [],
            mode="markers+text",
            name=driver["Acronym"],
            text=[
                driver["Acronym"]
            ] if active else [],
            textposition="top center",
            marker=dict(
                size=14,
                color=driver["Colour"],
                symbol=symbol,
                line=dict(
                    color="white",
                    width=1
                )
            ),
            textfont=dict(
                color=driver["Colour"],
                size=12
            ),
            hovertext=[
                f"{driver['Name']}<br>"
                f"{driver['Team']}<br>"
                f"{position_text}"
            ] if active else [],
            hoverinfo="text",
            cliponaxis=False
        )

    def leaderboard_text(index):

        entries = []

        for number in driver_numbers:

            position = leaderboard.iloc[index][number]

            acronym = drivers[number]["Acronym"]

            colour = drivers[number]["Colour"]

            entries.append((
                position,
                acronym,
                colour
            ))

        entries.sort(
            key=lambda entry: (
                entry[0]
                if pd.notna(entry[0])
                else 999
            )
        )

        lines = ["<b>SELECTED DRIVERS</b><br>"]

        for position, acronym, colour in entries:

            label = (
                f"P{int(position)}"
                if pd.notna(position)
                else "—"
            )

            lines.append(
                f'<span style="color:{colour}">'
                f"{label} &nbsp; {acronym}"
                "</span>"
            )

        return "<br>".join(lines)

    def frame_title(index):

        elapsed = (
            session_offset
            + time_axis[index]
        )

        return (
            "F1 RaceLab — Race Replay"
            f" | Session elapsed: "
            f"{format_elapsed(elapsed)}"
        )

    def leaderboard_annotation(index):

        return dict(
            x=1.03,
            y=0.95,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            text=leaderboard_text(index),
            showarrow=False,
            align="left",
            font=dict(size=17),
            bgcolor="#171C25",
            bordercolor="#414955",
            borderwidth=1,
            borderpad=15
        )

    track_x = circuit["x"].to_numpy()
    track_y = circuit["y"].to_numpy()

    track_x = np.append(
        track_x,
        track_x[0]
    )

    track_y = np.append(
        track_y,
        track_y[0]
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=track_x,
            y=track_y,
            mode="lines",
            name="Circuit",
            line=dict(
                color="#414955",
                width=12
            ),
            hoverinfo="skip"
        )
    )

    figure.add_trace(
        go.Scatter(
            x=track_x,
            y=track_y,
            mode="lines",
            line=dict(
                color="#89919D",
                width=2
            ),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    for number in driver_numbers:

        figure.add_trace(
            car_trace(number, 0)
        )

    frames = []

    trace_indices = list(
        range(
            2,
            2 + len(driver_numbers)
        )
    )

    for index in range(len(time_axis)):

        frames.append(
            go.Frame(
                name=str(index),
                traces=trace_indices,
                data=[
                    car_trace(number, index)
                    for number in driver_numbers
                ],
                layout=go.Layout(
                    title_text=frame_title(index),
                    annotations=[
                        leaderboard_annotation(index)
                    ]
                )
            )
        )

    figure.frames = frames

    slider_indices = np.unique(
        np.linspace(
            0,
            len(time_axis) - 1,
            min(21, len(time_axis)),
            dtype=int
        )
    )

    slider_steps = []

    for index in slider_indices:

        slider_steps.append({
            "method": "animate",
            "label": format_elapsed(
                time_axis[index]
            ),
            "args": [
                [str(index)],
                {
                    "mode": "immediate",
                    "frame": {
                        "duration": 0,
                        "redraw": True
                    },
                    "transition": {
                        "duration": 0
                    }
                }
            ]
        })

    frame_duration = max(
        1,
        int(
            result["interval"]
            * 1000
            / playback_speed
        )
    )

    x_padding = max(
        np.ptp(track_x) * 0.12,
        1
    )

    y_padding = max(
        np.ptp(track_y) * 0.12,
        1
    )

    figure.update_layout(
        title=frame_title(0),
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=800,

        annotations=[
            leaderboard_annotation(0)
        ],

        xaxis=dict(
            visible=False,
            fixedrange=True,
            range=[
                track_x.min() - x_padding,
                track_x.max() + x_padding
            ]
        ),

        yaxis=dict(
            visible=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
            range=[
                track_y.min() - y_padding,
                track_y.max() + y_padding
            ]
        ),

        legend=dict(
            orientation="h",
            x=0,
            y=1.05
        ),

        margin=dict(
            l=30,
            r=240,
            t=110,
            b=150
        ),

        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.18,
                "showactive": False,

                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration":
                                        frame_duration,
                                    "redraw": True
                                },
                                "transition": {
                                    "duration": 0
                                },
                                "fromcurrent": True
                            }
                        ]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": False
                                },
                                "transition": {
                                    "duration": 0
                                }
                            }
                        ]
                    },
                    {
                        "label": "Restart",
                        "method": "animate",
                        "args": [
                            ["0"],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": True
                                },
                                "transition": {
                                    "duration": 0
                                }
                            }
                        ]
                    }
                ]
            }
        ],

        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Replay window: "
                },
                "steps": slider_steps,
                "pad": {"t": 30}
            }
        ]
    )

    return figure


def calculate_full_race_summary(result):

    rows = []

    for number, driver in result["drivers"].items():

        replay = result["replay_data"][number]

        rows.append({
            "Driver Number": number,
            "Driver": driver["Name"],
            "Team": driver["Team"],
            "Position Samples": driver["Samples"],
            "Animation Frames": len(replay),
            "Active Frames": int(
                replay["Active"].sum()
            ),
            "Coverage (%)": round(
                replay["Active"].mean() * 100,
                2
            )
        })

    return pd.DataFrame(rows)
