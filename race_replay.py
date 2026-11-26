import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from data_loader import (
    get_position,
    get_telemetry,
    merge_position_telemetry,
    get_lap_times
)


def load_replay_data(session_key, driver_number, lap):

    position = get_position(
        session_key,
        driver_number,
        lap
    )

    telemetry = get_telemetry(
        session_key,
        driver_number,
        lap
    )

    if position.empty:
        raise ValueError("No position data available.")

    data = merge_position_telemetry(
        position,
        telemetry
    )

    start, end = get_lap_times(lap)

    data["Replay Time"] = (
        data["date"] - start
    ).dt.total_seconds()

    data = data.sort_values("Replay Time")

    data = data.drop_duplicates(
        subset=["Replay Time"]
    )

    data = data.dropna(
        subset=["x", "y", "speed", "throttle", "brake"]
    )

    if len(data) < 3:
        raise ValueError("Insufficient replay data.")

    return data.reset_index(drop=True)


def interpolate_replay(data, time_axis):

    original_time = data["Replay Time"].to_numpy()

    result = pd.DataFrame({
        "Time": time_axis
    })

    for channel in ["x", "y", "speed", "throttle"]:

        result[channel] = np.interp(
            time_axis,
            original_time,
            data[channel].to_numpy(),
            left=np.nan,
            right=np.nan
        )

    indices = np.searchsorted(
        original_time,
        time_axis,
        side="right"
    ) - 1

    indices = np.clip(
        indices,
        0,
        len(data) - 1
    )

    result["Brake"] = (
        data["brake"].to_numpy()[indices] > 0
    )

    result["Active"] = np.logical_and(
        time_axis >= original_time[0],
        time_axis <= original_time[-1]
    )

    return result


def prepare_replay(data_a, data_b, interval=0.2):

    if interval <= 0:
        raise ValueError("Interval must be positive.")

    maximum_time = max(
        data_a["Replay Time"].max(),
        data_b["Replay Time"].max()
    )

    time_axis = np.arange(
        0,
        maximum_time,
        interval
    )

    time_axis = np.append(
        time_axis,
        maximum_time
    )

    replay_a = interpolate_replay(
        data_a,
        time_axis
    )

    replay_b = interpolate_replay(
        data_b,
        time_axis
    )

    return replay_a, replay_b


def create_car_marker(
    row,
    driver_name,
    colour,
    symbol="circle",
    text_position="top left"
):

    active = bool(row["Active"])

    hover = ""

    if active:

        brake_status = (
            "ON" if row["Brake"] else "OFF"
        )

        hover = (
            f"{driver_name}<br>"
            f"Time: {row['Time']:.2f} s<br>"
            f"Speed: {row['speed']:.1f} km/h<br>"
            f"Throttle: {row['throttle']:.0f}%<br>"
            f"Brake: {brake_status}"
        )

    return go.Scatter(
        x=[row["x"]] if active else [],
        y=[row["y"]] if active else [],
        mode="markers+text",
        name=driver_name,
        text=[driver_name] if active else [],
        textposition=text_position,
        hovertext=[hover] if active else [],
        hoverinfo="text",
        marker=dict(
            size=17,
            symbol=symbol,
            color=colour,
            line=dict(
                color="#FFFFFF",
                width=1.5
            )
        ),
        textfont=dict(
            color=colour,
            size=13
        ),
        cliponaxis=False
    )


def create_speed_gauge(row, driver_name, colour):

    active = bool(row["Active"])

    speed = (
        float(row["speed"])
        if active else 0
    )

    return go.Indicator(
        mode="gauge+number",
        value=speed,
        number=dict(
            suffix=" km/h" if active else "",
            font=dict(
                size=27,
                color=colour
            )
        ),
        title=dict(
            text=(
                driver_name
                if active
                else f"{driver_name} — No sample"
            ),
            font=dict(size=15)
        ),
        gauge=dict(
            axis=dict(
                range=[0, 380],
                tickwidth=1,
                tickcolor="#A0A0A0"
            ),
            bar=dict(color=colour),
            bgcolor="#171C25",
            borderwidth=0,
            steps=[
                {
                    "range": [0, 100],
                    "color": "#222831"
                },
                {
                    "range": [100, 250],
                    "color": "#303846"
                },
                {
                    "range": [250, 380],
                    "color": "#414A59"
                }
            ]
        )
    )


def create_pedal_trace(row, driver_name, colour):

    active = bool(row["Active"])

    if active:

        throttle = float(
            np.clip(row["throttle"], 0, 100)
        )

        brake = (
            100 if row["Brake"] else 0
        )

        labels = [
            f"{throttle:.0f}%",
            "ON" if brake else "OFF"
        ]

    else:

        throttle = 0
        brake = 0

        labels = [
            "N/A",
            "N/A"
        ]

    return go.Bar(
        x=["Throttle", "Brake"],
        y=[throttle, brake],
        marker_color=[
            "#00D26A",
            "#E10600"
        ],
        text=labels,
        textposition="outside",
        name=driver_name,
        showlegend=False,
        hoverinfo="skip"
    )


def create_race_replay(
    data_a,
    data_b,
    driver_a,
    driver_b,
    colour_a="#FF8700",
    colour_b="#FFFFFF",
    interval=0.2,
    playback_speed=1.0
):

    if playback_speed <= 0:
        raise ValueError(
            "Playback speed must be positive."
        )

    replay_a, replay_b = prepare_replay(
        data_a,
        data_b,
        interval
    )

    frame_duration = max(
        1,
        int(
            interval * 1000 / playback_speed
        )
    )

    figure = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [
                {"type": "xy", "rowspan": 3},
                {"type": "indicator"}
            ],
            [
                None,
                {"type": "indicator"}
            ],
            [
                None,
                {"type": "xy"}
            ]
        ],
        column_widths=[0.66, 0.34],
        row_heights=[0.34, 0.34, 0.32],
        horizontal_spacing=0.08,
        vertical_spacing=0.08
    )

    track_x = data_a["x"].to_numpy()
    track_y = data_a["y"].to_numpy()

    # Close the visual circuit outline.
    track_x = np.append(track_x, track_x[0])
    track_y = np.append(track_y, track_y[0])

    figure.add_trace(
        go.Scatter(
            x=track_x,
            y=track_y,
            mode="lines",
            line=dict(
                color="#414955",
                width=12
            ),
            name="Circuit",
            hoverinfo="skip"
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
            showlegend=False,
            hoverinfo="skip"
        ),
        row=1,
        col=1
    )

    start_index = 0

    for i in range(len(replay_a)):

        if (
            replay_a.iloc[i]["Active"]
            or replay_b.iloc[i]["Active"]
        ):

            start_index = i
            break

    row_a = replay_a.iloc[start_index]
    row_b = replay_b.iloc[start_index]

    figure.add_trace(
        create_car_marker(
            row_a,
            driver_a,
            colour_a,
            "circle",
            "top left"
        ),
        row=1,
        col=1
    )

    figure.add_trace(
        create_car_marker(
            row_b,
            driver_b,
            colour_b,
            "diamond",
            "bottom right"
        ),
        row=1,
        col=1
    )

    figure.add_trace(
        create_speed_gauge(
            row_a,
            driver_a,
            colour_a
        ),
        row=1,
        col=2
    )

    figure.add_trace(
        create_speed_gauge(
            row_b,
            driver_b,
            colour_b
        ),
        row=2,
        col=2
    )

    figure.add_trace(
        create_pedal_trace(
            row_a,
            driver_a,
            colour_a
        ),
        row=3,
        col=2
    )

    figure.add_trace(
        create_pedal_trace(
            row_b,
            driver_b,
            colour_b
        ),
        row=3,
        col=2
    )

    # Distinguish the two drivers' pedal bars.
    figure.data[6].marker.pattern.shape = ""
    figure.data[7].marker.pattern.shape = "/"

    figure.data[6].showlegend = True
    figure.data[7].showlegend = True

    frames = []

    for i in range(len(replay_a)):

        row_a = replay_a.iloc[i]
        row_b = replay_b.iloc[i]

        pedal_a = create_pedal_trace(
            row_a,
            driver_a,
            colour_a
        )

        pedal_b = create_pedal_trace(
            row_b,
            driver_b,
            colour_b
        )

        pedal_a.marker.pattern.shape = ""
        pedal_b.marker.pattern.shape = "/"

        pedal_a.showlegend = True
        pedal_b.showlegend = True

        frames.append(
            go.Frame(
                name=str(i),
                traces=[2, 3, 4, 5, 6, 7],
                data=[
                    create_car_marker(
                        row_a,
                        driver_a,
                        colour_a,
                        "circle",
                        "top left"
                    ),
                    create_car_marker(
                        row_b,
                        driver_b,
                        colour_b,
                        "diamond",
                        "bottom right"
                    ),
                    create_speed_gauge(
                        row_a,
                        driver_a,
                        colour_a
                    ),
                    create_speed_gauge(
                        row_b,
                        driver_b,
                        colour_b
                    ),
                    pedal_a,
                    pedal_b
                ],
                layout=go.Layout(
                    title_text=(
                        "F1 RaceLab — Advanced Lap Replay"
                        f" | {row_a['Time']:.1f} s"
                    )
                )
            )
        )

    figure.frames = frames

    slider_indices = np.linspace(
        0,
        len(replay_a) - 1,
        min(16, len(replay_a)),
        dtype=int
    )

    slider_indices = np.unique(
        np.append(
            slider_indices,
            start_index
        )
    )

    slider_steps = []

    for i in slider_indices:

        slider_steps.append({
            "method": "animate",
            "label": (
                f"{replay_a.iloc[i]['Time']:.1f}"
            ),
            "args": [
                [str(i)],
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

    all_x = np.concatenate([
        data_a["x"].to_numpy(),
        data_b["x"].to_numpy()
    ])

    all_y = np.concatenate([
        data_a["y"].to_numpy(),
        data_b["y"].to_numpy()
    ])

    x_padding = max(
        np.ptp(all_x) * 0.08,
        1
    )

    y_padding = max(
        np.ptp(all_y) * 0.08,
        1
    )

    figure.update_xaxes(
        visible=False,
        fixedrange=True,
        range=[
            all_x.min() - x_padding,
            all_x.max() + x_padding
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
            all_y.min() - y_padding,
            all_y.max() + y_padding
        ],
        row=1,
        col=1
    )

    figure.update_yaxes(
        range=[0, 125],
        ticksuffix="%",
        gridcolor="#303640",
        fixedrange=True,
        row=3,
        col=2
    )

    figure.update_xaxes(
        fixedrange=True,
        row=3,
        col=2
    )

    figure.update_layout(
        title=(
            "F1 RaceLab — Advanced Lap Replay"
            f" | {replay_a.iloc[start_index]['Time']:.1f} s"
        ),
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=800,
        barmode="group",
        bargap=0.25,

        legend=dict(
            orientation="h",
            x=0,
            y=1.05,
            font=dict(size=12)
        ),

        margin=dict(
            l=25,
            r=25,
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
                "bgcolor": "#242A34",
                "bordercolor": "#414955",

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
                                "frame": {
                                    "duration": 0,
                                    "redraw": False
                                },
                                "mode": "immediate",
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
                            [str(start_index)],
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
                "active": int(
                    np.where(
                        slider_indices == start_index
                    )[0][0]
                ),
                "x": 0,
                "len": 1,
                "y": -0.04,
                "currentvalue": {
                    "prefix": "Replay position: ",
                    "suffix": " s"
                },
                "pad": {"t": 20},
                "steps": slider_steps
            }
        ]
    )

    return figure


def calculate_replay_summary(
    data_a,
    data_b,
    driver_a,
    driver_b
):

    results = []

    for data, driver in [
        (data_a, driver_a),
        (data_b, driver_b)
    ]:

        results.append({
            "Driver": driver,

            "Position Samples": len(data),

            "Replay Duration (s)": round(
                data["Replay Time"].max()
                - data["Replay Time"].min(),
                3
            ),

            "Maximum Speed (km/h)": round(
                data["speed"].max(),
                2
            ),

            "Average Sampled Speed (km/h)": round(
                data["speed"].mean(),
                2
            ),

            "Maximum Throttle (%)": round(
                data["throttle"].max(),
                2
            )
        })

    return pd.DataFrame(results)
