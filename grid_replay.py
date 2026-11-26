import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from data_loader import (
    get_data,
    get_drivers,
    get_laps,
    get_stints
)

from full_race_replay import (
    prepare_full_race,
    prepare_leaderboard,
    format_elapsed,
    calculate_full_race_summary
)


TYRE_COLOURS = {
    "SOFT": "#E10600",
    "MEDIUM": "#FFD700",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#00D26A",
    "WET": "#168BFF",
    "UNKNOWN": "#808080"
}


def get_all_driver_numbers(session_key):

    drivers = get_drivers(session_key)

    return drivers[
        "driver_number"
    ].astype(int).tolist()


def load_speed_data(
    session_key,
    driver_number,
    start_time,
    end_time
):

    data = get_data(
        "car_data",
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
        utc=True, format="ISO8601"
    )

    return (
        data.dropna(subset=["date", "speed"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )


def load_pit_stops(session_key):

    pits = get_data(
        "pit",
        {"session_key": session_key}
    )

    if pits.empty:
        return pits

    pits["date"] = pd.to_datetime(
        pits["date"],
        utc=True, format="ISO8601"
    )

    return pits.sort_values("date")


def prepare_driver_information(
    session_key,
    driver_number,
    timestamps,
    start_time,
    end_time,
    pits,
    include_speed=True
):

    information = pd.DataFrame({
        "date": timestamps
    })

    information["Lap"] = np.nan
    information["Compound"] = "UNKNOWN"
    information["Tyre Age"] = np.nan
    information["Speed"] = np.nan
    information["Recent Pit Stop"] = False

    errors = []
    def safe_load(fn, *args):
        try:
            return fn(*args)
        except Exception as exc:
            errors.append(f"{fn.__name__}: {exc}")
            return pd.DataFrame()

    laps = safe_load(get_laps, session_key, driver_number)
    stints = safe_load(get_stints, session_key, driver_number)

    if not laps.empty:

        laps = laps.copy()

        laps["date_start"] = pd.to_datetime(
            laps["date_start"],
            utc=True, format="ISO8601"
        )

        laps = laps.dropna(
            subset=["date_start"]
        )

        laps = laps.sort_values("date_start")

        matched = pd.merge_asof(
            information[["date"]],
            laps[
                [
                    "date_start",
                    "lap_number",
                    "lap_duration"
                ]
            ],
            left_on="date",
            right_on="date_start",
            direction="backward"
        )

        lap_end = (
            matched["date_start"]
            + pd.to_timedelta(
                matched["lap_duration"],
                unit="s"
            )
        )

        valid_lap = (
            matched["date"] <= lap_end
        )

        information["Lap"] = (
            matched["lap_number"].where(valid_lap)
        )

    if not stints.empty:

        for _, stint in stints.iterrows():

            mask = np.logical_and(
                information["Lap"]
                >= stint["lap_start"],
                information["Lap"]
                <= stint["lap_end"]
            )

            information.loc[
                mask,
                "Compound"
            ] = str(
                stint["compound"]
            ).upper()

            starting_age = stint.get(
                "tyre_age_at_start",
                np.nan
            )

            information.loc[
                mask,
                "Tyre Age"
            ] = (
                information.loc[mask, "Lap"]
                - stint["lap_start"]
                + starting_age
            )

    if include_speed:

        speed = safe_load(load_speed_data,
            session_key,
            driver_number,
            start_time,
            end_time
        )

        if not speed.empty:

            matched_speed = pd.merge_asof(
                information[["date"]],
                speed[["date", "speed"]],
                on="date",
                direction="nearest",
                tolerance=pd.Timedelta(
                    seconds=1
                )
            )

            information["Speed"] = (
                matched_speed["speed"]
            )

    if not pits.empty:

        driver_pits = pits[
            pits["driver_number"]
            == driver_number
        ]

        for _, pit in driver_pits.iterrows():

            pit_time = pit["date"]

            # Highlight a recorded pit event
            # for five seconds after its timestamp.

            mask = np.logical_and(
                information["date"] >= pit_time,
                information["date"]
                <= pit_time
                + pd.Timedelta(seconds=5)
            )

            information.loc[
                mask,
                "Recent Pit Stop"
            ] = True

    information.attrs["errors"] = errors
    return information


def prepare_grid_replay(
    session_key,
    driver_numbers=None,
    start_offset=120,
    duration=120,
    interval=2.0,
    include_speed=True
):

    if driver_numbers is None:

        driver_numbers = (
            get_all_driver_numbers(
                session_key
            )
        )

    if len(driver_numbers) > 22:
        raise ValueError(
            "Select no more than 22 drivers."
        )

    if duration > 300:
        raise ValueError(
            "Use replay windows of up to 300 seconds."
        )

    if interval < 1:
        raise ValueError(
            "Use a frame interval of at least one second."
        )

    result = prepare_full_race(
        session_key=session_key,
        driver_numbers=driver_numbers,
        start_offset=start_offset,
        duration=duration,
        interval=interval
    )

    timestamps = (
        result["start_time"]
        + pd.to_timedelta(
            result["time_axis"],
            unit="s"
        )
    )

    result["leaderboard"] = (
        prepare_leaderboard(result)
    )

    result["driver_information"] = {}

    result["information_errors"] = {}

    try:
        pits = load_pit_stops(session_key)
    except Exception as exc:
        pits = pd.DataFrame()
        result["information_errors"]["pit"] = str(exc)

    for number in result["drivers"]:

        print(
            f"Loading additional telemetry "
            f"for driver #{number}..."
        )

        try:

            information = (
                prepare_driver_information(
                    session_key,
                    number,
                    timestamps,
                    result["start_time"],
                    result["end_time"],
                    pits,
                    include_speed
                )
            )

        except Exception as error:

            result["information_errors"][
                number
            ] = str(error)

            information = pd.DataFrame({
                "date": timestamps,
                "Lap": np.nan,
                "Compound": "UNKNOWN",
                "Tyre Age": np.nan,
                "Speed": np.nan,
                "Recent Pit Stop": False
            })

        if information.attrs.get("errors"):
            result["information_errors"][number] = "; ".join(information.attrs["errors"])

        result["driver_information"][
            number
        ] = information

    return result


def create_leaderboard_rows(
    result,
    frame_index
):

    rows = []

    for number, driver in (
        result["drivers"].items()
    ):

        position = result[
            "leaderboard"
        ].iloc[frame_index][number]

        information = result[
            "driver_information"
        ][number].iloc[frame_index]

        replay = result[
            "replay_data"
        ][number].iloc[frame_index]

        rows.append({
            "Position": position,
            "Driver": driver["Acronym"],
            "Colour": driver["Colour"],
            "Lap": information["Lap"],
            "Compound": information["Compound"],
            "Tyre Age": information["Tyre Age"],
            "Speed": information["Speed"],
            "Recent Pit Stop":
                information["Recent Pit Stop"],
            "Active": replay["Active"]
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "Position",
            na_position="last"
        )
        .reset_index(drop=True)
    )


def format_number(value):

    if pd.isna(value):
        return "—"

    return str(int(round(value)))


def create_leaderboard_table(
    result,
    frame_index
):

    rows = create_leaderboard_rows(
        result,
        frame_index
    )

    compounds = []

    tyre_colours = []

    for compound in rows["Compound"]:

        compounds.append(
            compound[0]
            if compound in TYRE_COLOURS
            and compound != "UNKNOWN"
            else "—"
        )

        tyre_colours.append(
            TYRE_COLOURS.get(
                compound,
                "#808080"
            )
        )

    statuses = []

    for _, row in rows.iterrows():

        if row["Recent Pit Stop"]:
            statuses.append("PIT EVENT")

        elif not row["Active"]:
            statuses.append("NO GPS")

        else:
            statuses.append("TRACK")

    count = len(rows)

    row_colours = [
        "#171C25" if i % 2 == 0
        else "#202631"
        for i in range(count)
    ]

    return go.Table(
        columnwidth=[
            0.7,
            1.0,
            0.7,
            0.7,
            0.7,
            1.0,
            1.4
        ],

        header=dict(
            values=[
                "POS",
                "DRIVER",
                "LAP",
                "TYRE",
                "AGE",
                "KM/H",
                "STATUS"
            ],
            fill_color="#292F39",
            font=dict(
                color="white",
                size=11
            ),
            align="center",
            height=32,
            line_color="#0E1117"
        ),

        cells=dict(
            values=[
                [
                    format_number(x)
                    for x in rows["Position"]
                ],
                rows["Driver"].tolist(),
                [
                    format_number(x)
                    for x in rows["Lap"]
                ],
                compounds,
                [
                    format_number(x)
                    for x in rows["Tyre Age"]
                ],
                [
                    format_number(x)
                    for x in rows["Speed"]
                ],
                statuses
            ],

            fill_color=[
                row_colours
                for _ in range(7)
            ],

            font=dict(
                color=[
                    ["white"] * count,
                    rows["Colour"].tolist(),
                    ["white"] * count,
                    tyre_colours,
                    ["white"] * count,
                    ["white"] * count,
                    ["#C9CDD3"] * count
                ],
                size=11
            ),

            align="center",
            height=27,
            line_color="#0E1117"
        )
    )


def create_grid_markers(
    result,
    frame_index
):

    x_values = []
    y_values = []

    labels = []
    colours = []
    hover = []

    for number, driver in (
        result["drivers"].items()
    ):

        row = result[
            "replay_data"
        ][number].iloc[frame_index]

        if not row["Active"]:
            continue

        information = result[
            "driver_information"
        ][number].iloc[frame_index]

        x_values.append(row["x"])
        y_values.append(row["y"])

        labels.append(
            driver["Acronym"]
        )

        colours.append(
            driver["Colour"]
        )

        speed = format_number(
            information["Speed"]
        )

        hover.append(
            f"{driver['Name']}<br>"
            f"{driver['Team']}<br>"
            f"Lap: "
            f"{format_number(information['Lap'])}<br>"
            f"Speed: {speed} km/h<br>"
            f"Tyre: {information['Compound']}<br>"
            f"Tyre age: "
            f"{format_number(information['Tyre Age'])}"
        )

    return go.Scatter(
        x=x_values,
        y=y_values,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(
            size=10,
            color=colours
        ),
        marker=dict(
            size=11,
            color=colours,
            line=dict(
                color="white",
                width=1
            )
        ),
        hovertext=hover,
        hoverinfo="text",
        name="Drivers",
        showlegend=False,
        cliponaxis=False
    )


def create_grid_replay(
    result,
    playback_speed=10.0
):

    if playback_speed <= 0:
        raise ValueError(
            "Playback speed must be positive."
        )

    circuit = result["circuit"]

    time_axis = result["time_axis"]

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [
                {"type": "xy"},
                {"type": "table"}
            ]
        ],
        column_widths=[0.57, 0.43],
        horizontal_spacing=0.04
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

    figure.add_trace(
        go.Scatter(
            x=track_x,
            y=track_y,
            mode="lines",
            line=dict(
                color="#414955",
                width=12
            ),
            hoverinfo="skip",
            showlegend=False
        ),
        row=1,
        col=1
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
            hoverinfo="skip",
            showlegend=False
        ),
        row=1,
        col=1
    )

    figure.add_trace(
        create_grid_markers(result, 0),
        row=1,
        col=1
    )

    figure.add_trace(
        create_leaderboard_table(
            result,
            0
        ),
        row=1,
        col=2
    )

    session_offset = (
        result["start_time"]
        - result["session_start"]
    ).total_seconds()

    def frame_title(index):

        elapsed = (
            session_offset
            + time_axis[index]
        )

        return (
            "F1 RaceLab — Full-Grid Replay"
            f" | {format_elapsed(elapsed)}"
        )

    frames = []

    for index in range(len(time_axis)):

        frames.append(
            go.Frame(
                name=str(index),
                traces=[2, 3],
                data=[
                    create_grid_markers(
                        result,
                        index
                    ),
                    create_leaderboard_table(
                        result,
                        index
                    )
                ],
                layout=go.Layout(
                    title_text=frame_title(index)
                )
            )
        )

    figure.frames = frames

    slider_indices = np.unique(
        np.linspace(
            0,
            len(time_axis) - 1,
            min(12, len(time_axis)),
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

    buttons = []

    for speed in [1, 2, 5, 10, 20]:

        buttons.append({
            "label": f"{speed}x",
            "method": "animate",
            "args": [
                None,
                {
                    "mode": "immediate",
                    "frame": {
                        "duration": max(
                            1,
                            int(
                                result["interval"]
                                * 1000
                                / speed
                            )
                        ),
                        "redraw": True
                    },
                    "transition": {
                        "duration": 0
                    },
                    "fromcurrent": True
                }
            ]
        })

    x_padding = max(
        np.ptp(track_x) * 0.12,
        1
    )

    y_padding = max(
        np.ptp(track_y) * 0.12,
        1
    )

    figure.update_xaxes(
        visible=False,
        fixedrange=True,
        range=[
            track_x.min() - x_padding,
            track_x.max() + x_padding
        ],
        row=1,
        col=1
    )

    figure.update_yaxes(
        visible=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=1,
        range=[
            track_y.min() - y_padding,
            track_y.max() + y_padding
        ],
        row=1,
        col=1
    )

    figure.update_layout(
        title=frame_title(0),
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=850,

        margin=dict(
            l=25,
            r=25,
            t=100,
            b=170
        ),

        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.17,
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
            },
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.55,
                "y": -0.17,
                "showactive": True,
                "buttons": buttons
            }
        ],

        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Replay window: "
                },
                "steps": slider_steps,
                "pad": {"t": 25}
            }
        ]
    )

    return figure


def export_grid_data(result):

    tables = []

    for index in range(
        len(result["time_axis"])
    ):

        rows = create_leaderboard_rows(
            result,
            index
        )

        rows["Timestamp"] = (
            result["start_time"]
            + pd.Timedelta(
                seconds=float(
                    result["time_axis"][index]
                )
            )
        )

        tables.append(rows)

    return pd.concat(
        tables,
        ignore_index=True
    )
