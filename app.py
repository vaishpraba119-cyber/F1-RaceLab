
import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import html

from data_loader import (
    get_years,
    get_meetings,
    get_sessions,
    get_drivers,
    get_laps,
    get_fastest_lap,
    get_specific_lap,
    get_telemetry,
    get_position,
    merge_position_telemetry,
    get_weather,
    get_stints
)

from analysis import run_analysis

from insights import (
    generate_insights,
    generate_report
)

from visualisations import (
    plot_telemetry,
    plot_all_telemetry,
    plot_lap_delta,
    plot_braking_events,
    plot_braking_summary,
    plot_track_sections,
    plot_gear_usage,
    plot_circuit_heatmap,
    plot_performance_map,
    plot_weather
)


# Configure the web application
st.set_page_config(
    page_title="F1 RaceLab",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom dashboard styling
st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stMetric"] {
        border: 1px solid #333333;
        border-radius: 12px;
        padding: 16px;
    }

    .racelab-title {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: -2px;
        margin-bottom: 0px;
    }

    .racelab-subtitle {
        color: #999999;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .driver-card {
        border: 1px solid #444444;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Format lap times
def format_lap_time(seconds):

    milliseconds = round(float(seconds) * 1000)

    minutes, remaining = divmod(
        milliseconds,
        60000
    )

    seconds, milliseconds = divmod(
        remaining,
        1000
    )

    return (
        f"{minutes}:{seconds:02d}."
        f"{milliseconds:03d}"
    )


# Create downloadable CSV files
def download_csv(data, filename, label):

    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}"
    )


# Package analysis results into a ZIP file
def create_zip(results, report):

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as archive:

        for name, data in results.items():

            if isinstance(data, pd.DataFrame):

                archive.writestr(
                    f"{name}.csv",
                    data.to_csv(index=False)
                )

        archive.writestr(
            "engineering_report.md",
            report
        )

    buffer.seek(0)

    return buffer.getvalue()


# Retrieve optional API data safely
def optional_data(function, *args):

    try:

        return function(*args), None

    except Exception as error:

        return pd.DataFrame(), str(error)


# Display interactive Plotly charts
def display_chart(figure, key):

    st.plotly_chart(
        figure,
        use_container_width=True,
        key=key,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "scale": 3
            }
        }
    )


# Display application heading
st.markdown(
    """
    <div class="racelab-title">
        <span style="color:#E10600">F1</span> RaceLab
    </div>

    <div class="racelab-subtitle">
        Motorsport Telemetry & Performance Analytics Platform
    </div>
    """,
    unsafe_allow_html=True
)


# Initialise application state
if "analysis_results" not in st.session_state:

    st.session_state.analysis_results = None


# Sidebar configuration
st.sidebar.title("Race Configuration")

st.sidebar.caption(
    "Select a historical Formula 1 session."
)


try:

    # Select season
    year = st.sidebar.selectbox(
        "Season",
        get_years()
    )

    # Retrieve Grand Prix events
    meetings = get_meetings(year)

    if meetings.empty:

        st.warning(
            "No historical events available."
        )

        st.stop()

    # Exclude pre-season testing
    meetings = meetings[
        meetings["meeting_name"].str.contains(
            "Grand Prix",
            case=False,
            na=False
        )
    ]

    if meetings.empty:

        st.warning(
            "No Grand Prix events available."
        )

        st.stop()

    meeting_options = dict(
        zip(
            meetings["meeting_name"],
            meetings["meeting_key"]
        )
    )

    # Select Grand Prix
    meeting_name = st.sidebar.selectbox(
        "Grand Prix",
        list(meeting_options.keys()),
        key=f"meeting_{year}"
    )

    meeting_key = int(
        meeting_options[meeting_name]
    )

    # Retrieve available sessions
    sessions = get_sessions(meeting_key)

    if sessions.empty:

        st.warning(
            "No completed sessions available."
        )

        st.stop()

    session_options = dict(
        zip(
            sessions["session_name"],
            sessions["session_key"]
        )
    )

    session_names = list(
        session_options.keys()
    )

    default_session = (
        session_names.index("Qualifying")
        if "Qualifying" in session_names
        else 0
    )

    # Select session
    session_name = st.sidebar.selectbox(
        "Session",
        session_names,
        index=default_session,
        key=f"session_{meeting_key}"
    )

    session_key = int(
        session_options[session_name]
    )

    # Retrieve participating drivers
    drivers = get_drivers(session_key)

    if len(drivers) < 2:

        st.warning(
            "Insufficient driver data for comparison."
        )

        st.stop()

    driver_options = {}

    for _, driver in drivers.iterrows():

        label = (
            f'{driver["full_name"]} '
            f'(#{driver["driver_number"]})'
        )

        driver_options[label] = int(
            driver["driver_number"]
        )

    st.sidebar.divider()

    st.sidebar.subheader("Driver Selection")

    # Select Driver A
    driver_label_a = st.sidebar.selectbox(
        "Driver A",
        list(driver_options.keys()),
        key=f"driver_a_{session_key}"
    )

    driver_number_a = driver_options[
        driver_label_a
    ]

    # Prevent selecting the same driver twice
    remaining_drivers = [
        name
        for name in driver_options
        if name != driver_label_a
    ]

    # Select Driver B
    driver_label_b = st.sidebar.selectbox(
        "Driver B",
        remaining_drivers,
        key=f"driver_b_{session_key}_{driver_number_a}"
    )

    driver_number_b = driver_options[
        driver_label_b
    ]

    driver_a = drivers[
        drivers["driver_number"] == driver_number_a
    ].iloc[0]

    driver_b = drivers[
        drivers["driver_number"] == driver_number_b
    ].iloc[0]

    name_a = driver_a["full_name"]

    name_b = driver_b["full_name"]

    # Retrieve available laps
    laps_a = get_laps(
        session_key,
        driver_number_a
    )

    laps_b = get_laps(
        session_key,
        driver_number_b
    )

    if laps_a.empty or laps_b.empty:

        st.warning(
            "One or both drivers have no completed laps. "
            "Please select different drivers."
        )

        st.stop()

    # Select lap comparison mode
    lap_mode = st.sidebar.radio(
        "Lap Selection",
        [
            "Fastest recorded lap",
            "Specific lap"
        ]
    )

    if lap_mode == "Fastest recorded lap":

        lap_a = get_fastest_lap(laps_a)

        lap_b = get_fastest_lap(laps_b)

    else:

        lap_number_a = st.sidebar.selectbox(
            f"{name_a} lap",
            laps_a["lap_number"].tolist(),
            key=f"lap_a_{session_key}_{driver_number_a}"
        )

        lap_number_b = st.sidebar.selectbox(
            f"{name_b} lap",
            laps_b["lap_number"].tolist(),
            key=f"lap_b_{session_key}_{driver_number_b}"
        )

        lap_a = get_specific_lap(
            laps_a,
            lap_number_a
        )

        lap_b = get_specific_lap(
            laps_b,
            lap_number_b
        )

    # Analysis settings
    st.sidebar.divider()

    st.sidebar.subheader("Analysis Settings")

    number_of_sections = st.sidebar.slider(
        "Number of track sections",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )

    color_a = st.sidebar.color_picker(
        "Driver A colour",
        "#FF8700"
    )

    color_b = st.sidebar.color_picker(
        "Driver B colour",
        "#3693FF"
    )

    include_context = st.sidebar.checkbox(
        "Include tyre and weather data",
        value=True
    )

    # Generate configuration identifier
    configuration = (
        year,
        meeting_key,
        session_key,
        driver_number_a,
        driver_number_b,
        int(lap_a["lap_number"]),
        int(lap_b["lap_number"]),
        number_of_sections,
        include_context
    )

except Exception as error:

    st.error(
        f"Unable to retrieve session data: {error}"
    )

    st.stop()


# Generate analysis button
generate = st.sidebar.button(
    "Generate Analysis",
    type="primary",
    use_container_width=True
)

st.sidebar.divider()

st.sidebar.caption(
    "Data source: OpenF1 API"
)

st.sidebar.caption(
    "Historical telemetry · 2023 onwards"
)

st.sidebar.caption(
    "F1 RaceLab v2.1"
)


# Retrieve and process selected telemetry
if generate:

    st.session_state.analysis_results = None

    progress_bar = st.progress(
        0,
        text="Initialising analysis..."
    )

    try:

        progress_bar.progress(
            10,
            text=f"Retrieving {name_a} telemetry..."
        )

        telemetry_a = get_telemetry(
            session_key,
            driver_number_a,
            lap_a
        )

        progress_bar.progress(
            25,
            text=f"Retrieving {name_b} telemetry..."
        )

        telemetry_b = get_telemetry(
            session_key,
            driver_number_b,
            lap_b
        )

        progress_bar.progress(
            40,
            text="Running engineering calculations..."
        )

        results = run_analysis(
            telemetry_a=telemetry_a,
            telemetry_b=telemetry_b,
            driver_a=name_a,
            driver_b=name_b,
            lap_time_a=float(
                lap_a["lap_duration"]
            ),
            lap_time_b=float(
                lap_b["lap_duration"]
            ),
            number_of_sections=number_of_sections
        )

        progress_bar.progress(
            60,
            text="Retrieving circuit positioning..."
        )

        warnings = []

        position_a, error_a = optional_data(
            get_position,
            session_key,
            driver_number_a,
            lap_a
        )

        position_b, error_b = optional_data(
            get_position,
            session_key,
            driver_number_b,
            lap_b
        )

        if error_a:

            warnings.append(
                f"{name_a} positioning: {error_a}"
            )

        if error_b:

            warnings.append(
                f"{name_b} positioning: {error_b}"
            )

        # Align position and telemetry data
        track_a, merge_error_a = optional_data(
            merge_position_telemetry,
            position_a,
            results["telemetry_a"]
        )

        track_b, merge_error_b = optional_data(
            merge_position_telemetry,
            position_b,
            results["telemetry_b"]
        )

        if merge_error_a:

            warnings.append(
                f"{name_a} position alignment: "
                f"{merge_error_a}"
            )

        if merge_error_b:

            warnings.append(
                f"{name_b} position alignment: "
                f"{merge_error_b}"
            )

        results["track_a"] = track_a

        results["track_b"] = track_b

        progress_bar.progress(
            80,
            text="Retrieving session context..."
        )

        weather = pd.DataFrame()

        stints_a = pd.DataFrame()

        stints_b = pd.DataFrame()

        if include_context:

            weather, error = optional_data(
                get_weather,
                session_key
            )

            if error:

                warnings.append(
                    f"Weather: {error}"
                )

            stints_a, error = optional_data(
                get_stints,
                session_key,
                driver_number_a
            )

            if error:

                warnings.append(
                    f"{name_a} tyres: {error}"
                )

            stints_b, error = optional_data(
                get_stints,
                session_key,
                driver_number_b
            )

            if error:

                warnings.append(
                    f"{name_b} tyres: {error}"
                )

        results["weather"] = weather

        results["stints_a"] = stints_a

        results["stints_b"] = stints_b

        # Save analysis metadata
        results["metadata"] = pd.DataFrame([
            {
                "Season": year,
                "Grand Prix": meeting_name,
                "Session": session_name,
                "Session Key": session_key,
                "Driver A": name_a,
                "Driver B": name_b,
                "Driver A Lap": lap_a["lap_number"],
                "Driver B Lap": lap_b["lap_number"],
                "Driver A Lap Time": lap_a["lap_duration"],
                "Driver B Lap Time": lap_b["lap_duration"],
                "Track Sections": number_of_sections,
                "Data Source": "OpenF1 API"
            }
        ])

        progress_bar.progress(
            90,
            text="Generating engineering insights..."
        )

        # Generate automated observations
        insights = generate_insights(
            results,
            name_a,
            name_b
        )

        # Generate engineering report
        report = generate_report(
            results,
            name_a,
            name_b
        )

        # Store results between dashboard interactions
        st.session_state.analysis_results = {
            "configuration": configuration,
            "results": results,
            "warnings": warnings,
            "insights": insights,
            "report": report
        }

        progress_bar.progress(
            100,
            text="Analysis complete!"
        )

        progress_bar.empty()

        st.success(
            "Analysis generated successfully."
        )

    except Exception as error:

        progress_bar.empty()

        st.error(
            f"Analysis failed: {error}"
        )

        st.stop()


# Display selected event
st.subheader(
    f"{year} {meeting_name}"
)

st.caption(
    f"{session_name} | "
    f"{name_a} vs {name_b}"
)


# Display driver information cards
column_a, column_b = st.columns(2)

for column, driver, lap, color in [
    (column_a, driver_a, lap_a, color_a),
    (column_b, driver_b, lap_b, color_b)
]:

    with column:

        safe_name = html.escape(
            str(driver["full_name"])
        )

        safe_team = html.escape(
            str(driver.get("team_name", "Unknown"))
        )

        st.markdown(
            f"""
            <div class="driver-card"
                 style="border-top:4px solid {color}">

                <h2 style="color:{color};margin:0">
                    {safe_name}
                </h2>

                <p>
                    {safe_team} |
                    #{int(driver["driver_number"])}
                </p>

                <h1>
                    {format_lap_time(lap["lap_duration"])}
                </h1>

                <p>Lap {int(lap["lap_number"])}</p>

            </div>
            """,
            unsafe_allow_html=True
        )


# Check whether analysis is available
stored = st.session_state.analysis_results

if stored is None:

    st.info(
        "Select your preferred drivers and session, "
        "then click Generate Analysis."
    )

    st.stop()


# Prevent displaying outdated results
if stored["configuration"] != configuration:

    st.warning(
        "Configuration changed. "
        "Click Generate Analysis to update the results."
    )

    st.stop()


# Extract stored analysis results
results = stored["results"]

insights = stored["insights"]

report = stored["report"]

for warning in stored["warnings"]:

    st.warning(warning)


a = results["telemetry_a"]

b = results["telemetry_b"]

performance = results["performance"]

events = results["braking_events"]

summary = results["braking_summary"]

comparison = results["lap_comparison"]

sections = results["track_sections"]

gear_usage = results["gear_usage"]

track_a = results["track_a"]

track_b = results["track_b"]


# Create dashboard navigation
tabs = st.tabs([
    "Overview",
    "Telemetry",
    "Braking",
    "Lap Delta",
    "Track Sections",
    "Circuit Maps",
    "Gear Analysis",
    "Tyres & Weather",
    "Data Export",
    "Methodology",
    "Engineering Insights"
])


# Overview page
with tabs[0]:

    st.header("Performance Overview")

    lap_difference = float(
        lap_b["lap_duration"]
        - lap_a["lap_duration"]
    )

    if np.isclose(lap_difference, 0):

        faster_driver = "Equal"

    elif lap_difference > 0:

        faster_driver = name_a

    else:

        faster_driver = name_b

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Faster Driver",
        faster_driver
    )

    c2.metric(
        "Lap Difference",
        f"{abs(lap_difference):.3f} s"
    )

    c3.metric(
        "Maximum Speed",
        f'{performance["Top Speed (km/h)"].max():.0f} km/h'
    )

    c4.metric(
        "Peak Braking",
        f'{performance["Peak Braking G"].max():.2f} G'
    )

    st.subheader("Key Performance Insights")

    for category in [
        "Lap Performance",
        "Speed Analysis",
        "Throttle Analysis"
    ]:

        observations = insights.get(
            category,
            []
        )

        if observations:

            st.markdown(
                f"- {observations[0]}"
            )

    st.subheader("Driver Performance Comparison")

    st.dataframe(
        performance
        .set_index("Driver")
        .T
        .round(3),
        use_container_width=True
    )

    display_chart(
        plot_lap_delta(
            comparison,
            name_a,
            name_b,
            color_a,
            color_b
        ),
        "overview_delta"
    )

    st.caption(
        "Time differences are estimated using "
        "normalised lap-distance alignment."
    )


# Telemetry page
with tabs[1]:

    st.header("Telemetry Comparison")

    channels = {
        "Speed": "speed",
        "Throttle": "throttle",
        "Brake Status": "Brake_Status",
        "Engine RPM": "rpm",
        "Gear": "n_gear",
        "DRS Status": "drs",
        "Acceleration": "Smoothed_Acceleration",
        "Deceleration": "Deceleration",
        "Longitudinal Deceleration G": "Braking_G"
    }

    available_channels = {
        label: channel
        for label, channel in channels.items()
        if channel in a.columns
        or channel in b.columns
    }

    selected_channel = st.selectbox(
        "Telemetry Channel",
        list(available_channels.keys())
    )

    display_chart(
        plot_telemetry(
            a,
            b,
            name_a,
            name_b,
            available_channels[selected_channel],
            color_a,
            color_b
        ),
        "single_telemetry"
    )

    if st.checkbox(
        "Display all telemetry channels"
    ):

        display_chart(
            plot_all_telemetry(
                a,
                b,
                name_a,
                name_b,
                color_a,
                color_b
            ),
            "all_telemetry"
        )


# Braking analysis page
with tabs[2]:

    st.header("Braking Performance")

    st.dataframe(
        summary.round(3),
        use_container_width=True,
        hide_index=True
    )

    summary_metric = st.selectbox(
        "Overall Braking Metric",
        [
            "Braking Events",
            "Total Braking Distance (m)",
            "Total Braking Duration (s)",
            "Peak Braking G",
            "Average Deceleration (m/s2)"
        ]
    )

    display_chart(
        plot_braking_summary(
            summary,
            name_a,
            name_b,
            summary_metric,
            color_a,
            color_b
        ),
        "braking_summary_chart"
    )

    event_metric = st.selectbox(
        "Individual Braking Event Metric",
        [
            "Peak Braking G",
            "Braking Distance (m)",
            "Duration (s)",
            "Initial Speed (km/h)",
            "Release Speed (km/h)",
            "Speed Reduction (km/h)",
            "Average Deceleration (m/s2)"
        ]
    )

    display_chart(
        plot_braking_events(
            events,
            name_a,
            name_b,
            event_metric,
            color_a,
            color_b
        ),
        "braking_event_chart"
    )

    st.subheader("Detected Braking Events")

    st.dataframe(
        events.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Braking Insights")

    for observation in insights.get(
        "Braking Analysis",
        []
    ):

        st.markdown(
            f"- {observation}"
        )

    st.caption(
        "Braking events are detected from the binary "
        "brake signal. Event numbers are not necessarily "
        "equivalent corners between drivers."
    )


# Lap time delta page
with tabs[3]:

    st.header("Lap Time Delta")

    show_raw = st.checkbox(
        "Show raw sampled time difference",
        value=True
    )

    display_chart(
        plot_lap_delta(
            comparison,
            name_a,
            name_b,
            color_a,
            color_b,
            show_raw=show_raw
        ),
        "detailed_delta"
    )

    raw_delta = comparison[
        "Raw Sampled Delta (s)"
    ].iloc[-1]

    adjusted_delta = comparison[
        "Time Advantage (s)"
    ].iloc[-1]

    section_delta = sections[
        "Driver A Advantage (s)"
    ].sum()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Raw Sampled Delta",
        f"{raw_delta:+.3f} s"
    )

    c2.metric(
        "Normalised Delta",
        f"{adjusted_delta:+.3f} s"
    )

    c3.metric(
        "Sum of Section Deltas",
        f"{section_delta:+.3f} s"
    )

    st.info(
        f"Positive values favour {name_a}. "
        f"Negative values favour {name_b}."
    )

    st.caption(
        "The normalised delta is constrained to match "
        "the recorded lap difference."
    )


# Track section page
with tabs[4]:

    st.header("Track Section Analysis")

    display_chart(
        plot_track_sections(
            sections,
            name_a,
            name_b,
            color_a,
            color_b
        ),
        "section_chart"
    )

    best_a = sections.loc[
        sections["Driver A Advantage (s)"].idxmax()
    ]

    best_b = sections.loc[
        sections["Driver A Advantage (s)"].idxmin()
    ]

    c1, c2 = st.columns(2)

    c1.metric(
        f"Best relative section — {name_a}",
        f'Section {int(best_a["Section"])}',
        f'{best_a["Driver A Advantage (s)"]:+.3f} s'
    )

    c2.metric(
        f"Best relative section — {name_b}",
        f'Section {int(best_b["Section"])}',
        f'{-best_b["Driver A Advantage (s)"]:+.3f} s'
    )

    st.dataframe(
        sections.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Track Section Insights")

    for observation in insights.get(
        "Track Section Analysis",
        []
    ):

        st.markdown(
            f"- {observation}"
        )

    st.caption(
        "Sections are automatically generated using "
        "equal normalised lap-distance intervals. "
        "They are not official F1 sectors or corners."
    )


# Circuit heatmap page
with tabs[5]:

    st.header("Interactive Circuit Maps")

    map_channels = {
        "Speed": "speed",
        "Throttle": "throttle",
        "Braking": "brake",
        "Engine RPM": "rpm",
        "Gear": "n_gear",
        "DRS": "drs"
    }

    selected_map = st.selectbox(
        "Heatmap Channel",
        list(map_channels.keys())
    )

    channel = map_channels[selected_map]

    available_values = []

    for track in [track_a, track_b]:

        if not track.empty and channel in track.columns:

            available_values.extend(
                track[channel].dropna().tolist()
            )

    value_range = None

    if available_values:

        minimum = min(available_values)

        maximum = max(available_values)

        if maximum > minimum:

            value_range = (
                minimum,
                maximum
            )

    c1, c2 = st.columns(2)

    with c1:

        display_chart(
            plot_circuit_heatmap(
                track_a,
                name_a,
                channel,
                value_range
            ),
            "driver_a_map"
        )

    with c2:

        display_chart(
            plot_circuit_heatmap(
                track_b,
                name_b,
                channel,
                value_range
            ),
            "driver_b_map"
        )

    st.subheader("Estimated Local Performance")

    if (
        not track_a.empty
        and "Distance" in track_a.columns
        and track_a["Distance"].nunique() > 1
    ):

        display_chart(
            plot_performance_map(
                track_a,
                comparison,
                name_a,
                name_b,
                color_a,
                color_b
            ),
            "performance_map"
        )

    else:

        st.info(
            "Insufficient position data for "
            "the local performance map."
        )

    st.caption(
        "Position maps use sampled OpenF1 coordinates. "
        "Local time gain is an approximate "
        "distance-aligned estimate."
    )


# Gear analysis page
with tabs[6]:

    st.header("Gear Usage & Powertrain Telemetry")

    display_chart(
        plot_gear_usage(
            gear_usage,
            name_a,
            name_b,
            color_a,
            color_b
        ),
        "gear_usage_chart"
    )

    if "n_gear" in a.columns or "n_gear" in b.columns:

        display_chart(
            plot_telemetry(
                a,
                b,
                name_a,
                name_b,
                "n_gear",
                color_a,
                color_b
            ),
            "gear_telemetry"
        )

    else:

        st.info("Gear telemetry unavailable.")

    if "rpm" in a.columns or "rpm" in b.columns:

        display_chart(
            plot_telemetry(
                a,
                b,
                name_a,
                name_b,
                "rpm",
                color_a,
                color_b
            ),
            "rpm_telemetry"
        )

    else:

        st.info("RPM telemetry unavailable.")

    if not gear_usage.empty:

        st.dataframe(
            gear_usage.round(3),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Gear Usage Insights")

    for observation in insights.get(
        "Gear Analysis",
        []
    ):

        st.markdown(
            f"- {observation}"
        )


# Tyre and weather page
with tabs[7]:

    st.header("Tyre Strategy & Weather")

    if not include_context:

        st.info(
            "Enable tyre and weather data in the sidebar "
            "and regenerate the analysis."
        )

    else:

        c1, c2 = st.columns(2)

        for column, name, stints, lap in [
            (
                c1,
                name_a,
                results["stints_a"],
                lap_a
            ),
            (
                c2,
                name_b,
                results["stints_b"],
                lap_b
            )
        ]:

            with column:

                st.subheader(name)

                if stints.empty:

                    st.info(
                        "Tyre data unavailable."
                    )

                else:

                    required = [
                        "lap_start",
                        "lap_end"
                    ]

                    if all(
                        col in stints.columns
                        for col in required
                    ):

                        selected_stint = stints[
                            np.logical_and(
                                stints["lap_start"]
                                <= lap["lap_number"],
                                stints["lap_end"]
                                >= lap["lap_number"]
                            )
                        ]

                        if not selected_stint.empty:

                            stint = selected_stint.iloc[0]

                            compound = stint.get(
                                "compound",
                                "Unknown"
                            )

                            st.metric(
                                "Selected Lap Compound",
                                str(compound)
                            )

                            age_at_start = stint.get(
                                "tyre_age_at_start",
                                np.nan
                            )

                            if pd.notna(age_at_start):

                                estimated_age = (
                                    age_at_start
                                    + lap["lap_number"]
                                    - stint["lap_start"]
                                )

                                st.metric(
                                    "Estimated Tyre Age "
                                    "at Lap Start",
                                    f"{int(estimated_age)} laps"
                                )

                        else:

                            st.info(
                                "No matching tyre stint "
                                "for this lap."
                            )

                    columns = [
                        "stint_number",
                        "compound",
                        "lap_start",
                        "lap_end",
                        "tyre_age_at_start"
                    ]

                    available = [
                        col for col in columns
                        if col in stints.columns
                    ]

                    st.dataframe(
                        stints[available],
                        use_container_width=True,
                        hide_index=True
                    )

        st.subheader("Session Weather")

        weather = results["weather"]

        if weather.empty or "date" not in weather.columns:

            st.info(
                "Weather data unavailable."
            )

        else:

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

                st.info(
                    "No valid weather timestamps available."
                )

            else:

                display_chart(
                    plot_weather(weather),
                    "weather_chart"
                )

                # Display nearest weather sample
                for name, lap in [
                    (name_a, lap_a),
                    (name_b, lap_b)
                ]:

                    lap_start = pd.to_datetime(
                        lap["date_start"],
                        utc=True
                    )

                    nearest_index = (
                        weather["date"] - lap_start
                    ).abs().idxmin()

                    nearest = weather.loc[
                        nearest_index
                    ]

                    st.subheader(
                        f"Nearest weather sample — {name}"
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Air Temperature",
                        f'{nearest.get("air_temperature", "N/A")} °C'
                    )

                    c2.metric(
                        "Track Temperature",
                        f'{nearest.get("track_temperature", "N/A")} °C'
                    )

                    c3.metric(
                        "Humidity",
                        f'{nearest.get("humidity", "N/A")}%'
                    )

                    c4.metric(
                        "Wind Speed",
                        f'{nearest.get("wind_speed", "N/A")} m/s'
                    )

                    offset = abs(
                        (
                            nearest["date"] - lap_start
                        ).total_seconds()
                    )

                    st.caption(
                        f"Weather sample offset: "
                        f"{offset:.1f} seconds"
                    )

                    if offset > 120:

                        st.warning(
                            "The nearest weather sample "
                            "is more than two minutes "
                            "from the selected lap start."
                        )


# Export page
with tabs[8]:

    st.header("Export Analysis Results")

    st.caption(
        "Download the complete engineering dataset "
        "and automatically generated report."
    )

    st.download_button(
        "Download Complete Analysis ZIP",
        data=create_zip(results, report),
        file_name=(
            f"F1_RaceLab_{year}_{session_key}_"
            f"{driver_number_a}_vs_{driver_number_b}.zip"
        ),
        mime="application/zip",
        type="primary"
    )

    st.download_button(
        "Download Engineering Report",
        data=report,
        file_name=(
            f"F1_RaceLab_Report_{year}_"
            f"{session_key}_{driver_number_a}_"
            f"{driver_number_b}.md"
        ),
        mime="text/markdown",
        key="export_report"
    )

    st.subheader("Individual Data Files")

    export_options = {
        "Driver Performance": "performance",
        "Driver A Telemetry": "telemetry_a",
        "Driver B Telemetry": "telemetry_b",
        "Braking Events": "braking_events",
        "Braking Summary": "braking_summary",
        "Lap Comparison": "lap_comparison",
        "Track Sections": "track_sections",
        "Gear Usage": "gear_usage",
        "Driver A Position Data": "track_a",
        "Driver B Position Data": "track_b",
        "Weather": "weather",
        "Driver A Tyre Stints": "stints_a",
        "Driver B Tyre Stints": "stints_b",
        "Analysis Metadata": "metadata"
    }

    for label, key in export_options.items():

        if key in results:

            if not results[key].empty:

                download_csv(
                    results[key],
                    f"{key}.csv",
                    f"Download {label}"
                )


# Engineering methodology page
with tabs[9]:

    st.header("Engineering Methodology")

    st.markdown(
        """
        ### Project Objective

        F1 RaceLab is an independent motorsport
        telemetry analysis platform designed to compare
        Formula 1 driver performance using historical
        time-series data.

        ### 1. Data Acquisition

        Historical telemetry is retrieved from the
        OpenF1 REST API and processed using Python,
        NumPy and Pandas.

        ### 2. Distance Estimation

        Vehicle speed is converted from kilometres
        per hour to metres per second. Numerical
        trapezoidal integration estimates travelled
        distance.
        """
    )

    st.latex(
        r"v_{m/s}=\frac{v_{km/h}}{3.6}"
    )

    st.latex(
        r"s=\sum_{i=1}^{n-1}"
        r"\frac{v_i+v_{i+1}}{2}"
        r"(t_{i+1}-t_i)"
    )

    st.markdown(
        """
        ### 3. Acceleration and Deceleration

        Numerical differentiation estimates
        longitudinal acceleration. A three-sample
        moving average reduces measurement noise.
        """
    )

    st.latex(
        r"a=\frac{dv}{dt}"
    )

    st.latex(
        r"G_{\mathrm{deceleration}}="
        r"\frac{-a}{9.81}"
    )

    st.markdown(
        """
        ### 4. Braking Event Detection

        Rising and falling edges of the binary brake
        signal identify individual braking events.

        Events shorter than 0.3 seconds are excluded
        from the individual braking event analysis.

        ### 5. Lap Alignment

        Each driver's estimated travelled distance
        is normalised to a common lap-progress axis.

        Interpolation enables comparison across
        different sampling timestamps.
        """
    )

    st.latex(
        r"p=\frac{s}{s_{\mathrm{total}}}"
    )

    st.markdown(
        """
        ### 6. Lap Time Advantage

        Elapsed sampled time is scaled to match
        the recorded lap duration.
        """
    )

    st.latex(
        r"\Delta t(p)=t_B(p)-t_A(p)"
    )

    st.markdown(
        """
        Positive values favour Driver A.
        Negative values favour Driver B.

        ### 7. Circuit Mapping

        Position samples are matched with car telemetry
        using nearest-timestamp alignment with a
        maximum tolerance of 0.5 seconds.

        ### 8. Gear Usage

        Gear usage is calculated using the elapsed
        time between consecutive telemetry samples,
        rather than simply counting samples.
        """
    )

    st.latex(
        r"U_g=\frac{t_g}{t_{\mathrm{valid}}}\times100"
    )

    st.markdown(
        """
        ### 9. Automated Engineering Insights

        A deterministic rule-based analysis engine
        generates observations from calculated
        performance metrics.

        The engine compares lap times, maximum speed,
        braking performance, throttle usage, gear usage,
        track section performance and data quality.

        No generative AI or paid API is required.

        ### 10. Data Quality Assessment

        Sampled time coverage is calculated by
        comparing the observed telemetry time span
        with the recorded lap duration.
        """
    )

    st.latex(
        r"C=\frac{t_{\mathrm{last}}-t_{\mathrm{first}}}"
        r"{t_{\mathrm{lap}}}\times100"
    )

    st.markdown(
        """
        A coverage value below 95% triggers a warning.

        This is a project-defined heuristic and does
        not guarantee continuous or accurate sampling.

        ### Limitations

        - Historical telemetry coverage begins in 2023.
        - Some sessions and telemetry channels may
          be incomplete or unavailable.
        - Braking G is estimated from vehicle speed,
          not directly measured by an accelerometer.
        - Brake status is binary and does not measure
          brake pedal pressure.
        - Normalised lap alignment is approximate.
        - Track sections are not official F1 corners.
        - Fastest recorded laps may include deleted laps.
        - DRS values are displayed as raw status codes.
        - Normalised lap-time consistency is not
          independent timing validation.
        - Sampled time coverage does not detect every
          possible internal telemetry gap.
        - Tyres, weather, traffic, setup and fuel load
          may affect performance comparisons.
        - Automated observations identify differences
          but do not establish causation.

        ### Technologies

        Python, NumPy, Pandas, Requests,
        Plotly, Streamlit and REST APIs.

        ### Data Source

        [OpenF1 API](https://openf1.org/)

        This project is independent and is not
        affiliated with Formula 1 or its teams.
        """
    )


# Automated engineering insights page
with tabs[10]:

    st.header("Automated Engineering Insights")

    st.caption(
        "Data-driven observations generated from "
        "the selected telemetry comparison."
    )

    icons = {
        "Lap Performance": "🏁",
        "Speed Analysis": "⚡",
        "Braking Analysis": "🔴",
        "Throttle Analysis": "📈",
        "Track Section Analysis": "📍",
        "Gear Analysis": "⚙️",
        "Data Quality": "🔬"
    }

    # Display data quality metrics
    coverage_a = performance.iloc[0][
        "Telemetry Coverage (%)"
    ]

    coverage_b = performance.iloc[1][
        "Telemetry Coverage (%)"
    ]

    raw_difference = comparison[
        "Raw Sampled Delta (s)"
    ].iloc[-1]

    normalised_difference = comparison[
        "Time Advantage (s)"
    ].iloc[-1]

    discrepancy = abs(
        raw_difference - normalised_difference
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        f"{name_a} Coverage",
        f"{coverage_a:.2f}%"
    )

    c2.metric(
        f"{name_b} Coverage",
        f"{coverage_b:.2f}%"
    )

    c3.metric(
        "Raw vs Normalised Difference",
        f"{discrepancy:.3f} s"
    )

    if min(coverage_a, coverage_b) < 95:

        st.warning(
            "At least one lap has less than 95% "
            "sampled time coverage."
        )

    st.divider()

    # Display generated observations
    for category, observations in insights.items():

        if observations:

            with st.expander(
                f"{icons.get(category, '')} {category}",
                expanded=True
            ):

                for observation in observations:

                    st.markdown(
                        f"- {observation}"
                    )

    st.warning(
        "These observations identify differences "
        "in recorded performance. They do not prove "
        "why one driver was faster."
    )

    st.subheader("Generated Engineering Report")

    with st.expander(
        "Preview Full Engineering Report"
    ):

        st.markdown(report)

    st.download_button(
        "Download Engineering Report",
        data=report,
        file_name=(
            f"F1_RaceLab_Report_{year}_"
            f"{session_key}_{driver_number_a}_"
            f"{driver_number_b}.md"
        ),
        mime="text/markdown",
        type="primary",
        key="insights_report"
    )


# Application footer
st.divider()

st.caption(
    "F1 RaceLab v2.1 | "
    "Independent Motorsport Engineering Project | "
    "Powered by Python and OpenF1"
)
