
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots


# Default visualisation colours
COLOR_A = "#FF8700"
COLOR_B = "#3693FF"
TEMPLATE = "plotly_dark"


# Telemetry channel labels
LABELS = {
    "speed": "Speed (km/h)",
    "throttle": "Throttle (%)",
    "Brake_Status": "Brake status",
    "brake": "Brake status",
    "rpm": "Engine RPM",
    "n_gear": "Gear",
    "drs": "DRS raw status",
    "Acceleration": "Acceleration (m/s²)",
    "Smoothed_Acceleration": "Acceleration (m/s²)",
    "Deceleration": "Deceleration (m/s²)",
    "Braking_G": "Longitudinal deceleration (G)"
}


# Apply consistent graph styling
def style_figure(
    fig,
    title,
    x_label=None,
    y_label=None,
    height=450
):

    fig.update_layout(
        template=TEMPLATE,

        title=dict(
            text=title,
            x=0.02,
            y=0.98
        ),

        height=height,

        margin=dict(
            l=55,
            r=35,
            t=100,
            b=65
        ),

        legend=dict(
            orientation="h",
            y=1.02,
            yanchor="bottom",
            x=0
        ),

        hovermode="x unified",

        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",

        xaxis_title=x_label,
        yaxis_title=y_label
    )

    return fig


# Display unavailable data message
def empty(title):

    fig = style_figure(
        go.Figure(),
        title
    )

    fig.add_annotation(
        text="Data unavailable",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False
    )

    return fig


# Plot individual telemetry channels
def plot_telemetry(
    telemetry_a,
    telemetry_b,
    driver_a,
    driver_b,
    channel="speed",
    color_a=COLOR_A,
    color_b=COLOR_B
):

    fig = go.Figure()

    for data, name, color in [
        (telemetry_a, driver_a, color_a),
        (telemetry_b, driver_b, color_b)
    ]:

        if channel in data:

            line_shape = (
                "hv"
                if channel in [
                    "Brake_Status",
                    "n_gear",
                    "drs"
                ]
                else "linear"
            )

            fig.add_scatter(
                x=data["Lap_Progress"] * 100,
                y=data[channel],
                name=name,
                line=dict(
                    color=color,
                    shape=line_shape,
                    width=2
                )
            )

    style_figure(
        fig,
        LABELS.get(channel, channel) + " comparison",
        "Lap progress (%)",
        LABELS.get(channel, channel)
    )

    fig.update_xaxes(
        range=[0, 100]
    )

    return fig


# Plot all available telemetry channels
def plot_all_telemetry(
    telemetry_a,
    telemetry_b,
    driver_a,
    driver_b,
    color_a=COLOR_A,
    color_b=COLOR_B
):

    channels = [
        c for c in [
            "speed",
            "throttle",
            "Brake_Status",
            "n_gear",
            "rpm",
            "drs",
            "Braking_G"
        ]
        if c in telemetry_a or c in telemetry_b
    ]

    if not channels:

        return empty("Telemetry")

    fig = make_subplots(
        rows=len(channels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025
    )

    for i, channel in enumerate(channels, 1):

        for data, name, color in [
            (telemetry_a, driver_a, color_a),
            (telemetry_b, driver_b, color_b)
        ]:

            if channel in data:

                line_shape = (
                    "hv"
                    if channel in [
                        "Brake_Status",
                        "n_gear",
                        "drs"
                    ]
                    else "linear"
                )

                fig.add_scatter(
                    x=data["Lap_Progress"] * 100,
                    y=data[channel],
                    name=name,
                    legendgroup=name,
                    showlegend=(i == 1),

                    line=dict(
                        color=color,
                        shape=line_shape
                    ),

                    row=i,
                    col=1
                )

        fig.update_yaxes(
            title_text=LABELS[channel],
            row=i,
            col=1
        )

    style_figure(
        fig,
        "All telemetry channels",
        height=230 * len(channels) + 100
    )

    fig.update_xaxes(
        range=[0, 100]
    )

    fig.update_xaxes(
        title_text="Lap progress (%)",
        row=len(channels),
        col=1
    )

    return fig


# Plot cumulative lap time advantage
def plot_lap_delta(
    comparison,
    driver_a,
    driver_b,
    color_a=COLOR_A,
    color_b=COLOR_B,
    show_raw=False
):

    x = comparison["Distance (m)"]

    y = comparison["Time Advantage (s)"]

    fig = go.Figure()

    # Positive and negative time advantages
    for values, name, color in [
        (
            y.clip(lower=0),
            driver_a,
            color_a
        ),
        (
            y.clip(upper=0),
            driver_b,
            color_b
        )
    ]:

        fig.add_scatter(
            x=x,
            y=values,
            name=name + " ahead",

            fill="tozeroy",

            line=dict(
                color=color,
                width=2
            ),

            customdata=y,

            hovertemplate=(
                "Distance: %{x:.0f} m"
                "<br>Delta: %{customdata:+.3f} s"
                "<extra></extra>"
            )
        )

    # Optional raw sampled delta
    if show_raw:

        fig.add_scatter(
            x=x,
            y=comparison["Raw Sampled Delta (s)"],
            name="Raw sampled delta",

            line=dict(
                color="#BBBBBB",
                dash="dash"
            )
        )

    # Zero reference line
    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray"
    )

    # Final lap time annotation
    fig.add_annotation(
        x=x.iloc[-1],
        y=y.iloc[-1],

        text=(
            f"Finish: {y.iloc[-1]:+.3f} s"
        ),

        showarrow=True,
        ax=-75,
        ay=-35
    )

    style_figure(
        fig,
        "Estimated cumulative lap time advantage",
        "Estimated reference distance (m)",
        "Time advantage (s)"
    )

    # Prevent negative distance values
    fig.update_xaxes(
        range=[0, float(x.max())]
    )

    fig.update_yaxes(
        rangemode="tozero"
    )

    return fig


# Plot detected braking events
def plot_braking_events(
    events,
    driver_a,
    driver_b,
    metric="Peak Braking G",
    color_a=COLOR_A,
    color_b=COLOR_B
):

    if events.empty:

        return empty("Braking events")

    fig = px.scatter(
        events,

        x="Start Progress (%)",
        y=metric,
        color="Driver",

        color_discrete_map={
            driver_a: color_a,
            driver_b: color_b
        },

        hover_data=[
            "Braking Event",
            "Duration (s)",
            "Braking Distance (m)",
            "Initial Speed (km/h)",
            "Release Speed (km/h)"
        ]
    )

    fig.update_traces(
        marker_size=11
    )

    style_figure(
        fig,
        "Detected braking events",
        "Lap progress (%)",
        metric
    )

    fig.update_xaxes(
        range=[0, 100]
    )

    return fig


# Plot overall braking performance
def plot_braking_summary(
    summary,
    driver_a,
    driver_b,
    metric="Total Braking Distance (m)",
    color_a=COLOR_A,
    color_b=COLOR_B
):

    fig = px.bar(
        summary,

        x="Driver",
        y=metric,
        color="Driver",

        color_discrete_map={
            driver_a: color_a,
            driver_b: color_b
        }
    )

    return style_figure(
        fig,
        "Braking summary",
        "Driver",
        metric
    )


# Plot estimated track section performance
def plot_track_sections(
    sections,
    driver_a,
    driver_b,
    color_a=COLOR_A,
    color_b=COLOR_B
):

    y = sections["Driver A Advantage (s)"]

    colors = np.where(
        y >= 0,
        color_a,
        color_b
    )

    fig = go.Figure(
        go.Bar(
            x=sections["Section"],
            y=y,

            marker_color=colors,

            text=sections["Faster Driver"],

            hovertemplate=(
                "Section %{x}"
                "<br>Advantage: %{y:+.3f} s"
                "<br>Faster driver: %{text}"
                "<extra></extra>"
            )
        )
    )

    fig.add_hline(
        y=0,
        line_color="gray",
        line_dash="dot"
    )

    return style_figure(
        fig,
        "Estimated section time advantage",
        "Section",
        f"{driver_a} advantage (s)"
    )


# Plot time-weighted gear usage
def plot_gear_usage(
    gear_usage,
    driver_a,
    driver_b,
    color_a=COLOR_A,
    color_b=COLOR_B
):

    if gear_usage.empty:

        return empty("Gear usage")

    fig = px.bar(
        gear_usage,

        x="Gear",
        y="Usage (%)",
        color="Driver",

        barmode="group",

        color_discrete_map={
            driver_a: color_a,
            driver_b: color_b
        }
    )

    return style_figure(
        fig,
        "Time-weighted gear usage",
        "Gear",
        "Usage (%)"
    )


# Apply circuit map formatting
def map_layout(fig, title):

    style_figure(
        fig,
        title,
        height=550
    )

    fig.update_layout(
        hovermode="closest",
        showlegend=False
    )

    fig.update_xaxes(
        visible=False
    )

    fig.update_yaxes(
        visible=False,
        scaleanchor="x",
        scaleratio=1
    )

    return fig


# Generate interactive circuit heatmaps
def plot_circuit_heatmap(
    track,
    driver_name,
    channel="speed",
    value_range=None
):

    if (
        track.empty
        or channel not in track
        or not {"x", "y"}.issubset(track.columns)
    ):

        return empty(
            driver_name + " — map unavailable"
        )

    fig = go.Figure()

    # Circuit outline
    fig.add_scatter(
        x=track.x,
        y=track.y,

        mode="lines",

        line=dict(
            color="#555555",
            width=3
        ),

        hoverinfo="skip"
    )

    # Heatmap markers
    marker = dict(
        size=7,
        color=track[channel],
        colorscale="Turbo",
        showscale=True,

        colorbar=dict(
            title=LABELS.get(channel, channel)
        )
    )

    # Common colour scale for both drivers
    if value_range is not None:

        marker.update(
            cmin=value_range[0],
            cmax=value_range[1]
        )

    # Binary braking heatmap
    if channel == "brake":

        marker.update(
            color=(
                track[channel] > 0
            ).astype(int),

            colorscale=[
                [0, "#777777"],
                [1, "#E10600"]
            ],

            cmin=0,
            cmax=1,

            colorbar=dict(
                tickvals=[0, 1],
                ticktext=["Off", "On"]
            )
        )

    fig.add_scatter(
        x=track.x,
        y=track.y,

        mode="markers",

        marker=marker,

        text=track[channel],

        hovertemplate=(
            LABELS.get(channel, channel)
            + ": %{text}<extra></extra>"
        )
    )

    # First available position sample
    fig.add_scatter(
        x=[track.x.iloc[0]],
        y=[track.y.iloc[0]],

        mode="markers",

        marker=dict(
            symbol="square",
            color="white",
            size=9
        ),

        name="First sample"
    )

    return map_layout(
        fig,
        driver_name
        + " — "
        + LABELS.get(channel, channel)
    )


# Generate estimated local performance map
def plot_performance_map(
    track,
    comparison,
    driver_a,
    driver_b,
    color_a=COLOR_A,
    color_b=COLOR_B
):

    if (
        track.empty
        or not {"x", "y", "Distance"}.issubset(track.columns)
        or track["Distance"].nunique() < 2
    ):

        return empty("Local performance map")

    # Normalise track distance
    p = (
        track.Distance - track.Distance.min()
    ) / (
        track.Distance.max() - track.Distance.min()
    ) * 100

    # Interpolate local time gain
    gain = np.interp(
        p,
        comparison["Lap Progress (%)"],
        comparison["Smoothed Local Gain"]
    ) * 100000

    fig = go.Figure()

    # Circuit outline
    fig.add_scatter(
        x=track.x,
        y=track.y,

        mode="lines",

        line=dict(
            color="#555555",
            width=3
        ),

        hoverinfo="skip"
    )

    # Identify faster driver at each estimated location
    colors = np.where(
        gain >= 0,
        color_a,
        color_b
    )

    fig.add_scatter(
        x=track.x,
        y=track.y,

        mode="markers",

        marker=dict(
            size=8,
            color=colors
        ),

        customdata=gain,

        hovertemplate=(
            "Local gain: %{customdata:+.1f} ms/100 m"
            "<extra></extra>"
        )
    )

    return map_layout(
        fig,
        f"Estimated local gain: "
        f"{driver_a} (+) / {driver_b} (-)"
    )


# Plot session weather
def plot_weather(weather):

    if (
        weather.empty
        or "date" not in weather
    ):

        return empty("Weather")

    weather = weather.copy()

    weather["date"] = pd.to_datetime(
        weather["date"],
        utc=True,
        errors="coerce"
    )

    weather = weather.dropna(
        subset=["date"]
    ).sort_values("date")

    if weather.empty:

        return empty("Weather")

    fig = go.Figure()

    for channel in [
        "air_temperature",
        "track_temperature"
    ]:

        if channel in weather:

            fig.add_scatter(
                x=weather["date"],
                y=weather[channel],

                name=channel.replace(
                    "_", " "
                ).title()
            )

    return style_figure(
        fig,
        "Session temperatures",
        "Time (UTC)",
        "Temperature (°C)"
    )
