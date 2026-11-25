
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots


COLOR_A = "#FF8700"
COLOR_B = "#3693FF"

TEMPLATE = "plotly_dark"


# Apply consistent styling to all graphs
def style_figure(
    fig,
    title,
    x_label=None,
    y_label=None,
    height=450
):

    fig.update_layout(
        title=title,
        template=TEMPLATE,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        hovermode="x unified",
        margin=dict(
            l=50,
            r=30,
            t=70,
            b=50
        ),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0
        )
    )

    return fig


# Generate individual telemetry comparison
def plot_telemetry(
    telemetry_a,
    telemetry_b,
    driver_a,
    driver_b,
    channel="speed",
    color_a=COLOR_A,
    color_b=COLOR_B
):

    labels = {
        "speed": "Speed (km/h)",
        "throttle": "Throttle (%)",
        "Brake_Status": "Brake Status",
        "rpm": "Engine RPM",
        "n_gear": "Gear",
        "drs": "DRS Status Code",
        "Acceleration": "Acceleration (m/s²)",
        "Smoothed_Acceleration": "Acceleration (m/s²)",
        "Deceleration": "Deceleration (m/s²)",
        "Braking_G": "Longitudinal Deceleration (G)"
    }

    label = labels.get(
        channel,
        channel
    )

    fig = go.Figure()

    for data, name, color in [
        (telemetry_a, driver_a, color_a),
        (telemetry_b, driver_b, color_b)
    ]:

        if channel not in data.columns:
            continue

        line_shape = (
            "hv"
            if channel in [
                "Brake_Status",
                "n_gear",
                "drs"
            ]
            else "linear"
        )

        fig.add_trace(
            go.Scatter(
                x=data["Lap_Progress"] * 100,
                y=data[channel],
                name=name,
                mode="lines",
                line=dict(
                    color=color,
                    width=2,
                    shape=line_shape
                )
            )
        )

    if channel == "Brake_Status":

        fig.update_yaxes(
            tickvals=[0, 1],
            ticktext=["OFF", "ON"]
        )

    return style_figure(
        fig,
        f"{label} Comparison",
        "Normalised Lap Progress (%)",
        label
    )


# Generate complete telemetry dashboard
def plot_all_telemetry(
    telemetry_a,
    telemetry_b,
    driver_a,
    driver_b,
    color_a=COLOR_A,
    color_b=COLOR_B
):

    channels = [
        ("speed", "Speed (km/h)"),
        ("throttle", "Throttle (%)"),
        ("Brake_Status", "Brake Status"),
        ("n_gear", "Gear"),
        ("rpm", "Engine RPM"),
        ("drs", "DRS Status Code"),
        ("Braking_G", "Deceleration (G)")
    ]

    channels = [
        (channel, label)
        for channel, label in channels
        if channel in telemetry_a.columns
        or channel in telemetry_b.columns
    ]

    fig = make_subplots(
        rows=len(channels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=[
            label
            for _, label in channels
        ]
    )

    for row, (channel, label) in enumerate(
        channels,
        start=1
    ):

        for data, name, color in [
            (telemetry_a, driver_a, color_a),
            (telemetry_b, driver_b, color_b)
        ]:

            if channel not in data.columns:
                continue

            fig.add_trace(
                go.Scatter(
                    x=data["Lap_Progress"] * 100,
                    y=data[channel],
                    name=name,
                    legendgroup=name,
                    showlegend=(row == 1),
                    mode="lines",
                    line=dict(
                        color=color,
                        width=2,
                        shape=(
                            "hv"
                            if channel in [
                                "Brake_Status",
                                "n_gear",
                                "drs"
                            ]
                            else "linear"
                        )
                    )
                ),
                row=row,
                col=1
            )

        fig.update_yaxes(
            title_text=label,
            row=row,
            col=1
        )

        if channel == "Brake_Status":

            fig.update_yaxes(
                tickvals=[0, 1],
                ticktext=["OFF", "ON"],
                row=row,
                col=1
            )

    fig.update_xaxes(
        title_text="Normalised Lap Progress (%)",
        row=len(channels),
        col=1
    )

    fig.update_layout(
        title="Complete Telemetry Comparison",
        template=TEMPLATE,
        height=230 * len(channels),
        hovermode="x unified"
    )

    return fig


# Plot cumulative lap time difference
def plot_lap_delta(
    comparison,
    driver_a,
    driver_b,
    color_a=COLOR_A,
    color_b=COLOR_B,
    show_raw=False
):

    x = comparison["Distance (m)"]

    delta = comparison["Time Advantage (s)"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=delta.clip(lower=0),
            fill="tozeroy",
            name=f"{driver_a} ahead",
            line=dict(color=color_a)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=delta.clip(upper=0),
            fill="tozeroy",
            name=f"{driver_b} ahead",
            line=dict(color=color_b)
        )
    )

    if show_raw:

        fig.add_trace(
            go.Scatter(
                x=x,
                y=comparison[
                    "Raw Sampled Delta (s)"
                ],
                name="Raw sampled delta",
                line=dict(
                    color="white",
                    dash="dash",
                    width=1.5
                )
            )
        )

    fig.add_hline(
        y=0,
        line_color="grey",
        line_dash="dash"
    )

    return style_figure(
        fig,
        "Estimated Cumulative Lap Time Advantage",
        "Estimated Reference Distance (m)",
        "Time Advantage (s)",
        height=500
    )


# Plot braking event metrics
def plot_braking_events(
    events,
    driver_a,
    driver_b,
    metric="Peak Braking G",
    color_a=COLOR_A,
    color_b=COLOR_B
):

    if events.empty:

        fig = go.Figure()

        fig.add_annotation(
            text="No qualifying braking events detected",
            showarrow=False
        )

        return style_figure(
            fig,
            "Braking Analysis"
        )

    fig = px.scatter(
        events,
        x="Start Progress (%)",
        y=metric,
        color="Driver",
        hover_data=[
            "Braking Event",
            "Duration (s)",
            "Braking Distance (m)",
            "Initial Speed (km/h)",
            "Release Speed (km/h)"
        ],
        color_discrete_map={
            driver_a: color_a,
            driver_b: color_b
        }
    )

    fig.update_traces(
        marker=dict(
            size=12,
            opacity=0.85
        )
    )

    return style_figure(
        fig,
        f"Braking Events — {metric}",
        "Normalised Lap Progress (%)",
        metric
    )


# Plot overall braking comparison
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
        text_auto=".2f",
        color_discrete_map={
            driver_a: color_a,
            driver_b: color_b
        }
    )

    fig.update_layout(
        showlegend=False
    )

    return style_figure(
        fig,
        "Overall Braking Performance",
        "Driver",
        metric
    )


# Plot automatically generated track sections
def plot_track_sections(
    sections,
    driver_a,
    driver_b,
    color_a=COLOR_A,
    color_b=COLOR_B
):

    advantage = sections[
        "Driver A Advantage (s)"
    ]

    colors = np.where(
        advantage > 0,
        color_a,
        np.where(
            advantage < 0,
            color_b,
            "grey"
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=sections["Section"],
            y=advantage,
            marker_color=colors,
            customdata=sections[
                "Faster Driver"
            ],
            hovertemplate=(
                "Section %{x}<br>"
                "Advantage: %{y:.4f} s<br>"
                "Faster: %{customdata}"
                "<extra></extra>"
            )
        )
    )

    fig.add_hline(
        y=0,
        line_color="grey"
    )

    fig.update_xaxes(
        dtick=1
    )

    return style_figure(
        fig,
        f"Track Section Advantage — {driver_a} vs {driver_b}",
        "Track Section",
        "Driver A Advantage (s)"
    )


# Plot time spent in each gear
def plot_gear_usage(
    gear_usage,
    driver_a,
    driver_b,
    color_a=COLOR_A,
    color_b=COLOR_B
):

    if gear_usage.empty:

        fig = go.Figure()

        fig.add_annotation(
            text="Gear telemetry unavailable",
            showarrow=False
        )

        return style_figure(
            fig,
            "Gear Usage"
        )

    fig = px.bar(
        gear_usage,
        x="Gear",
        y="Usage (%)",
        color="Driver",
        barmode="group",
        text_auto=".1f",
        color_discrete_map={
            driver_a: color_a,
            driver_b: color_b
        }
    )

    fig.update_xaxes(dtick=1)

    return style_figure(
        fig,
        "Gear Usage Comparison",
        "Gear",
        "Sampled Time in Gear (%)"
    )


# Generate an interactive circuit heatmap
def plot_circuit_heatmap(
    track,
    driver_name,
    channel="speed",
    value_range=None
):

    fig = go.Figure()

    if track.empty:

        fig.add_annotation(
            text="Position data unavailable",
            showarrow=False
        )

        return style_figure(
            fig,
            "Circuit Heatmap"
        )

    if channel not in track.columns:

        fig.add_annotation(
            text=f"{channel} telemetry unavailable",
            showarrow=False
        )

        return style_figure(
            fig,
            "Circuit Heatmap"
        )

    track = track.copy()

    labels = {
        "speed": "Speed (km/h)",
        "throttle": "Throttle (%)",
        "brake": "Brake Status",
        "rpm": "Engine RPM",
        "n_gear": "Gear",
        "drs": "DRS Status Code"
    }

    label = labels.get(
        channel,
        channel
    )

    colorscale = "Turbo"

    if channel == "brake":

        track["brake"] = (
            track["brake"] > 0
        ).astype(int)

        colorscale = [
            [0, "#555555"],
            [0.4999, "#555555"],
            [0.5, "#FF3333"],
            [1, "#FF3333"]
        ]

        value_range = (0, 1)

    marker = dict(
        size=9,
        color=track[channel],
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=label)
    )

    if value_range is not None:

        marker["cmin"] = value_range[0]
        marker["cmax"] = value_range[1]

    if channel == "brake":

        marker["colorbar"]["tickvals"] = [0, 1]
        marker["colorbar"]["ticktext"] = [
            "OFF",
            "ON"
        ]

    # Draw the sampled racing line
    fig.add_trace(
        go.Scatter(
            x=track["x"],
            y=track["y"],
            mode="lines",
            line=dict(
                color="#555555",
                width=3
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # Overlay coloured telemetry samples
    fig.add_trace(
        go.Scatter(
            x=track["x"],
            y=track["y"],
            mode="markers",
            marker=marker,
            customdata=track[channel],
            hovertemplate=(
                label + ": %{customdata:.2f}"
                "<extra></extra>"
            ),
            showlegend=False
        )
    )

    # Mark first available position sample
    fig.add_trace(
        go.Scatter(
            x=[track["x"].iloc[0]],
            y=[track["y"].iloc[0]],
            mode="markers",
            marker=dict(
                color="white",
                size=12,
                symbol="square"
            ),
            name="First Sample"
        )
    )

    fig.update_xaxes(
        visible=False
    )

    fig.update_yaxes(
        visible=False,
        scaleanchor="x",
        scaleratio=1
    )

    fig = style_figure(
        fig,
        f"{driver_name} — {label} Heatmap",
        height=650
    )

    fig.update_layout(
        hovermode="closest"
    )

    return fig


# Map estimated local performance advantage
def plot_performance_map(
    track,
    comparison,
    driver_a,
    driver_b,
    color_a=COLOR_A,
    color_b=COLOR_B
):

    fig = go.Figure()

    if track.empty:

        fig.add_annotation(
            text="Position data unavailable",
            showarrow=False
        )

        return style_figure(
            fig,
            "Performance Map"
        )

    track = track.copy()

    distance = (
        track["Distance"]
        - track["Distance"].min()
    )

    if distance.max() <= 0:

        raise ValueError(
            "Invalid position distance"
        )

    progress = (
        distance / distance.max()
    ) * 100

    # Convert local time gain into ms per 100 m
    local_gain = np.interp(
        progress,
        comparison["Lap Progress (%)"],
        comparison["Smoothed Local Gain"]
    ) * 100000

    track["Local Gain"] = local_gain

    fig.add_trace(
        go.Scatter(
            x=track["x"],
            y=track["y"],
            mode="lines",
            line=dict(
                color="#555555",
                width=3
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    for condition, name, color in [
        (
            track["Local Gain"] >= 0,
            driver_a,
            color_a
        ),
        (
            track["Local Gain"] < 0,
            driver_b,
            color_b
        )
    ]:

        selected = track[condition]

        fig.add_trace(
            go.Scatter(
                x=selected["x"],
                y=selected["y"],
                mode="markers",
                name=f"{name} gaining",
                marker=dict(
                    color=color,
                    size=9
                ),
                customdata=selected[
                    "Local Gain"
                ],
                hovertemplate=(
                    "Estimated local gain: "
                    "%{customdata:.2f} ms/100 m"
                    "<extra></extra>"
                )
            )
        )

    fig.update_xaxes(
        visible=False
    )

    fig.update_yaxes(
        visible=False,
        scaleanchor="x",
        scaleratio=1
    )

    fig = style_figure(
        fig,
        "Estimated Local Performance Advantage",
        height=650
    )

    fig.update_layout(
        hovermode="closest"
    )

    return fig


# Plot available weather conditions
def plot_weather(weather):

    fig = go.Figure()

    if weather.empty:

        fig.add_annotation(
            text="Weather data unavailable",
            showarrow=False
        )

        return style_figure(
            fig,
            "Weather Conditions"
        )

    weather = weather.copy()

    weather["date"] = pd.to_datetime(
        weather["date"],
        utc=True
    )

    weather = weather.sort_values("date")

    for channel, label, color in [
        (
            "air_temperature",
            "Air Temperature",
            "#3693FF"
        ),
        (
            "track_temperature",
            "Track Temperature",
            "#FF8700"
        )
    ]:

        if channel not in weather.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=weather["date"],
                y=weather[channel],
                name=label,
                mode="lines",
                line=dict(
                    color=color,
                    width=2
                )
            )
        )

    return style_figure(
        fig,
        "Session Temperature Conditions",
        "Time (UTC)",
        "Temperature (°C)"
    )
