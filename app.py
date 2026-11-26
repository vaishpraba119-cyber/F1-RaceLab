
import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import re
import html

from data_loader import (
    get_years, get_meetings, get_sessions,
    get_drivers, get_laps, get_fastest_lap,
    get_specific_lap, get_telemetry, get_position,
    merge_position_telemetry, get_weather, get_stints
)

from analysis import run_analysis
from insights import generate_insights, generate_report

from visualisations import (
    plot_telemetry, plot_all_telemetry,
    plot_lap_delta, plot_braking_events,
    plot_braking_summary, plot_track_sections,
    plot_gear_usage, plot_circuit_heatmap,
    plot_performance_map, plot_weather
)

st.set_page_config(
    page_title="F1 RaceLab",
    page_icon="🏎️",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 4rem;
    max-width: 1500px;
}
[data-testid="stMetric"] {
    border: 1px solid #353942;
    border-radius: 12px;
    padding: 12px;
}
[data-testid="stMetricValue"] {
    font-size: clamp(18px, 2.5vw, 32px);
}
.driver-card {
    border: 1px solid #353942;
    border-radius: 12px;
    padding: 18px;
    overflow-wrap: anywhere;
}
.driver-card img {
    width: 85px;
    height: 85px;
    object-fit: contain;
    float: right;
}
.driver-card h3 {
    margin: 0;
}
.driver-card strong {
    font-size: 30px;
}
</style>
""", unsafe_allow_html=True)


# Format lap times
def lap_time(value):

    m, s = divmod(round(float(value) * 1000), 60000)
    sec, ms = divmod(s, 1000)

    return f"{m}:{sec:02d}.{ms:03d}"


# Retrieve team colours
def team_color(driver, fallback):

    value = str(
        driver.get("team_colour", "")
    ).strip().lstrip("#")

    if re.fullmatch("[0-9a-fA-F]{6}", value):
        return "#" + value

    return fallback


# Handle optional API data
def optional(fn, *args):

    try:
        return fn(*args), None

    except Exception as e:
        return pd.DataFrame(), str(e)


# Display interactive charts
def chart(fig, key):

    st.plotly_chart(
        fig,
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


# Display data tables
def table(data):

    st.dataframe(
        data.round(3),
        use_container_width=True,
        hide_index=True
    )


# Display driver cards
def card(driver, lap, color):

    name = html.escape(str(driver["full_name"]))

    team = html.escape(
        str(driver.get("team_name", "Unknown"))
    )

    url = str(driver.get("headshot_url", ""))

    photo = (
        f'<img src="{html.escape(url, quote=True)}" '
        f'alt="Driver portrait">'
        if url.startswith("https://")
        else ""
    )

    # Avoid indentation-related Markdown rendering issues
    card_html = (
        f'<div class="driver-card" '
        f'style="border-top:4px solid {color}">'
        f'{photo}'
        f'<h3 style="color:{color}">{name}</h3>'
        f'<p>{team} · #{int(driver["driver_number"])}</p>'
        f'<strong>{lap_time(lap["lap_duration"])}</strong>'
        f'<p>Lap {int(lap["lap_number"])}</p>'
        f'</div>'
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True
    )


# Package results for downloading
def zip_results(results, report):

    buf = io.BytesIO()

    with zipfile.ZipFile(
        buf, "w", zipfile.ZIP_DEFLATED
    ) as z:

        for k, v in results.items():

            if isinstance(v, pd.DataFrame):

                z.writestr(
                    k + ".csv",
                    v.to_csv(index=False)
                )

        z.writestr(
            "engineering_report.md",
            report
        )

    return buf.getvalue()


# Select preferred default values
def choose(label, options, preferred, key):

    return st.sidebar.selectbox(
        label,
        options,
        index=(
            options.index(preferred)
            if preferred in options
            else 0
        ),
        key=key
    )


# Read shareable URL parameters
def qp_int(key, default):

    try:
        return int(st.query_params.get(key, default))

    except (ValueError, TypeError):
        return default


# Application heading
st.markdown(
    '<h1 style="font-size:46px;margin-bottom:0">'
    '<span style="color:#E10600">F1</span> RaceLab'
    '</h1>',
    unsafe_allow_html=True
)

st.caption(
    "Motorsport telemetry & performance analytics · v4.0"
)


# Dashboard navigation
pages = [
    "Overview",
    "Full-Grid Replay",
    "Multi-Driver Replay",
    "Advanced Lap Replay",
    "Historical Strategy Calibration",
    "Monte Carlo Strategy",
    "Telemetry",
    "Braking",
    "Lap Delta",
    "Track Sections",
    "Circuit Maps",
    "Gear Analysis",
    "Tyres & Weather",
    "Tyre Degradation",
    "ML Lap Predictor",
    "Undercut & Overcut",
    "Strategy Optimiser",
    "Engineering Insights",
    "Data Export",
    "Methodology",
    "About"
]

page = st.sidebar.radio(
    "Dashboard navigation",
    pages
)

# Full-grid historical replay, independent of two-driver analysis.
if page == "Full-Grid Replay":
    from grid_replay import prepare_grid_replay, create_grid_replay, export_grid_data
    st.header("Full-Grid Race Replay")
    st.caption("Historical position samples and recorded timing; missing telemetry is shown as unavailable.")
    try:
        years = get_years()
        year = st.sidebar.selectbox("Season", years, index=years.index(2025) if 2025 in years else 0, key="grid_year")
        meetings = get_meetings(year)
        if meetings.empty:
            st.info("No meetings available."); st.stop()
        meetings = meetings[meetings.meeting_name.str.contains("Grand Prix", case=False, na=False)]
        if meetings.empty:
            st.info("No Grand Prix available."); st.stop()
        names = dict(zip(meetings.meeting_key.astype(int), meetings.meeting_name))
        meeting = st.sidebar.selectbox("Grand Prix", list(names), format_func=names.get)
        sessions = get_sessions(meeting)
        if sessions.empty:
            st.info("No completed sessions."); st.stop()
        races = sessions[sessions.session_name == "Race"]
        if races.empty:
            st.info("No completed race."); st.stop()
        session = int(races.iloc[0].session_key)
        drivers = get_drivers(session)
        if drivers.empty:
            st.info("No drivers available."); st.stop()
        labels = dict(zip(drivers.driver_number.astype(int), drivers.full_name))
        selected = st.multiselect("Drivers", list(labels), default=list(labels)[:22], format_func=labels.get, max_selections=22)
        c = st.columns(3)
        offset = c[0].number_input("Start offset (seconds)", min_value=0, value=120, step=30)
        duration = c[1].slider("Window (seconds)", 30, 300, 120, 30)
        interval = c[2].slider("Frame interval (seconds)", 1, 5, 2)
        speed = st.checkbox("Include speed telemetry", value=True)
        cfg = (session, tuple(selected), offset, duration, interval, speed)
        if st.button("Load replay", type="primary", disabled=not selected):
            st.session_state.pop("grid4", None)
            with st.spinner("Loading historical grid data..."):
                result = prepare_grid_replay(session, selected, offset, duration, interval, speed)
                st.session_state.grid4 = (cfg, result)
        saved = st.session_state.get("grid4")
        if saved and saved[0] == cfg:
            result = saved[1]
            st.metric("Drivers loaded", len(result["drivers"]))
            if result.get("information_errors"):
                with st.expander("Partial telemetry warnings"):
                    st.write(result["information_errors"])
            chart(create_grid_replay(result), "grid4chart")
            data = export_grid_data(result)
            cols = st.columns(3)
            for col, field in zip(cols, ["Lap", "Tyre Age", "Speed"]):
                col.metric(field + " coverage", f"{data[field].notna().mean():.1%}")
            st.download_button("Download replay CSV", data.to_csv(index=False), "grid_replay.csv", "text/csv")
    except Exception as exc:
        st.error(f"Replay unavailable: {exc}")
    st.stop()

# Historical replay and calibrated strategy pages.
if page in ["Multi-Driver Replay", "Historical Strategy Calibration", "Monte Carlo Strategy"]:
    try:
        st.header(page)
        if page != "Monte Carlo Strategy":
            years = get_years()
            y = st.sidebar.selectbox("Season", years, index=years.index(2025) if 2025 in years else 0)
            meetings = get_meetings(y)
            if meetings.empty:
                st.info("No meetings available."); st.stop()
            meetings = meetings[meetings.meeting_name.str.contains("Grand Prix", case=False, na=False)]
            names = dict(zip(meetings.meeting_key.astype(int), meetings.meeting_name))
            if not names:
                st.info("No Grand Prix available."); st.stop()
            mk = st.sidebar.selectbox("Grand Prix", list(names), format_func=names.get)
            sessions = get_sessions(mk)
            if sessions.empty:
                st.info("No sessions available."); st.stop()
            races = sessions[sessions.session_name == "Race"]
            if races.empty:
                st.info("No completed race available."); st.stop()
            sk = int(races.iloc[0].session_key)
            drivers = get_drivers(sk)
            labels = dict(zip(drivers.driver_number.astype(int), drivers.full_name))
            selected = st.multiselect("Drivers", list(labels), default=list(labels)[:4], format_func=labels.get)
            st.caption(f"{y} {names[mk]}")
        if page == "Multi-Driver Replay":
            from full_race_replay import prepare_full_race, create_full_race_replay, calculate_full_race_summary
            offset = st.number_input("Session start offset (s)", min_value=0, value=120)
            duration = st.slider("Replay window (s)", 30, 300, 120, 30)
            interval = st.slider("Frame interval (s)", 1, 5, 2)
            playback = st.slider("Playback speed", 1.0, 10.0, 5.0)
            cfg = (sk, tuple(selected), offset, duration, interval)
            if st.button("Load multi-driver replay", disabled=not selected, type="primary"):
                st.session_state.pop("multi4", None)
                with st.spinner("Loading historical positions..."):
                    st.session_state.multi4 = (cfg, prepare_full_race(sk, selected, offset, duration, interval))
            saved = st.session_state.get("multi4")
            if saved and saved[0] == cfg:
                result = saved[1]
                chart(create_full_race_replay(result, playback), "multi4chart")
                table(calculate_full_race_summary(result))
                if result.get("skipped"):
                    st.write(result["skipped"])
        elif page == "Historical Strategy Calibration":
            from strategy_calibration import calibrate_strategy, run_calibrated_optimisation, plot_calibrated_degradation, plot_calibration_accuracy, plot_pit_loss, generate_calibration_insights
            from strategy import plot_strategy_timeline, plot_strategy_lap_times
            fuel = st.number_input("Assumed fuel effect (s/lap)", 0.0, 0.2, 0.03, step=0.01)
            laps = st.number_input("Simulation race laps", 20, 100, 57)
            stops = st.radio("Maximum stops", [1, 2], index=1, horizontal=True)
            cfg = (sk, tuple(selected), fuel)
            if st.button("Calibrate historical models", disabled=not selected, type="primary"):
                st.session_state.pop("cal4", None)
                st.session_state.pop("calopt4", None)
                with st.spinner("Loading historical laps and fitting tyre models..."):
                    st.session_state.cal4 = (cfg, calibrate_strategy(sk, selected, fuel))
            saved = st.session_state.get("cal4")
            if saved and saved[0] == cfg:
                cal = saved[1]
                table(cal["summary"])
                if not cal["race_control_available"]:
                    st.warning("Race control data unavailable; neutralised laps may remain.")
                chart(plot_calibrated_degradation(cal), "cal4deg")
                if not cal["summary"].empty:
                    chart(plot_calibration_accuracy(cal), "cal4acc")
                chart(plot_pit_loss(cal), "cal4pit")
                for item in generate_calibration_insights(cal): st.write(item)
                st.download_button("Download calibration data", cal["clean_data"].to_csv(index=False), "calibration.csv")
                optcfg = (cfg, laps, stops)
                if st.button("Optimise calibrated strategy"):
                    st.session_state.pop("calopt4", None)
                    st.session_state.calopt4 = (optcfg, run_calibrated_optimisation(cal, laps, stops))
                opt = st.session_state.get("calopt4")
                if opt and opt[0] == optcfg:
                    result = opt[1]
                    table(result["parameter_sources"])
                    st.write("Pit loss source:", result["pit_source"])
                    st.write("Rejected models:", result["rejected_models"])
                    sim = result["optimisation"]
                    chart(plot_strategy_timeline(sim["best_simulation"]), "cal4timeline")
                    chart(plot_strategy_lap_times(sim["best_simulation"]), "cal4laps")
                    table(sim["leaderboard"].drop(columns=["Compounds", "Pit Lap Values"], errors="ignore"))
                    st.session_state["calibrated_mc_parameters"] = dict(tyre_parameters=result["parameters"], pit_loss=result["pit_loss"], warmup_penalty=result["warmup_penalty"], fuel_effect=fuel, source=f"{y} {names[mk]}")
                st.caption("Historical fits are observational; rejected models use illustrative fallback parameters.")
        else:
            from monte_carlo import run_monte_carlo, plot_win_probability, plot_race_time_distribution, plot_strategy_regret, plot_risk_return, plot_condition_analysis, plot_scenario_frequency, generate_monte_carlo_insights
            from race_conditions import DEFAULT_CONDITIONS
            calibrated = st.session_state.get("calibrated_mc_parameters")
            use_cal = st.checkbox("Use last calibrated strategy parameters", disabled=not calibrated)
            params = dict(calibrated) if use_cal and calibrated else {}
            source = params.pop("source", "Illustrative defaults")
            st.caption("Parameter source: " + source)
            laps = st.number_input("Race laps", 20, 100, 57)
            stops = st.radio("Maximum stops", [1, 2], index=1, horizontal=True)
            candidates = st.slider("Candidate strategies", 5, 50, 20)
            simulations = st.slider("Simulated scenarios", 100, 5000, 2000, 100)
            seed = st.number_input("Random seed", 0, 2147483647, 42)
            risk = st.slider("Risk weight", 0.0, 1.0, 0.5)
            conditions = DEFAULT_CONDITIONS.copy()
            with st.expander("Scenario assumptions", expanded=True):
                for key in ["safety_car_probability", "vsc_probability", "rain_probability"]:
                    conditions[key] = st.slider(key.replace("_", " ").title(), 0.0, 1.0, float(conditions[key]), 0.05)
            cfg = (laps, stops, candidates, simulations, seed, risk, repr(conditions), repr(params), source)
            if st.button("Run Monte Carlo simulation", type="primary"):
                st.session_state.pop("mc4", None)
                with st.spinner("Simulating shared race scenarios..."):
                    st.session_state.mc4 = (cfg, run_monte_carlo(laps, maximum_stops=stops, number_of_candidates=candidates, number_of_simulations=simulations, conditions=conditions, seed=seed, risk_weight=risk, **params))
            saved = st.session_state.get("mc4")
            if saved and saved[0] == cfg:
                result = saved[1]
                st.metric("Strategy evaluations", result["number_of_candidates"] * result["number_of_simulations"])
                st.write("Risk-ranked best:", result["best_risk_strategy"]["Strategy"])
                table(result["leaderboard"])
                for i, fn in enumerate([plot_win_probability, plot_race_time_distribution, plot_strategy_regret, plot_risk_return, plot_condition_analysis, plot_scenario_frequency]):
                    chart(fn(result), f"mc4chart{i}")
                for item in generate_monte_carlo_insights(result): st.write(item)
                st.download_button("Download simulation leaderboard", result["leaderboard"].to_csv(index=False), "monte_carlo.csv")
            st.warning("Probabilities are conditional on simulated assumptions and candidate strategies, not actual race-win forecasts. Rain is a pace adjustment, not a wet-tyre strategy model.")
    except Exception as exc:
        st.error(f"Module unavailable: {exc}")
    st.stop()

# Independent v4.0 modelling pages
if page in ['Tyre Degradation', 'ML Lap Predictor', 'Undercut & Overcut', 'Strategy Optimiser']:
    try:
        if page in ['Tyre Degradation', 'ML Lap Predictor']:
            st.sidebar.header('Historical race selection')
            y = st.sidebar.selectbox('Season', get_years(), index=get_years().index(2025) if 2025 in get_years() else 0, key='model_year')
            meetings = get_meetings(y)
            meetings = meetings[meetings.meeting_name.str.contains('Grand Prix', case=False, na=False)]
            if meetings.empty:
                st.info('No races available.'); st.stop()
            names = dict(zip(meetings.meeting_key.astype(int), meetings.meeting_name))
            mk3 = st.sidebar.selectbox('Grand Prix', list(names), format_func=names.get, key=f'model_gp{y}')
            sessions3 = get_sessions(mk3)
            races = sessions3[sessions3.session_name == 'Race']
            if races.empty:
                st.info('No completed race available.'); st.stop()
            sk3 = int(races.iloc[0].session_key)
            drivers3 = get_drivers(sk3)
            labels = dict(zip(drivers3.driver_number.astype(int), drivers3.full_name))
            if not labels:
                st.info('No drivers available.'); st.stop()
            st.header(f'{y} {names[mk3]} — {page}')
            if page == 'Tyre Degradation':
                from tyre_model import analyse_tyre_stints, plot_tyre_degradation, plot_tyre_prediction, generate_tyre_insights
                n = st.selectbox('Driver', list(labels), format_func=labels.get)
                degree = st.radio('Regression', [1, 2], format_func=lambda x: 'Linear' if x == 1 else 'Quadratic', horizontal=True)
                cfg = (sk3, n, degree)
                if st.button('Fit tyre models', type='primary'):
                    st.session_state.pop('tyre3', None)
                    with st.spinner('Retrieving race laps and fitting stints...'):
                        st.session_state.tyre3 = (cfg, analyse_tyre_stints(sk3, n, degree))
                saved = st.session_state.get('tyre3')
                if saved and saved[0] == cfg:
                    result = saved[1]
                    table(result['summary'])
                    if not result['summary'].empty:
                        chart(plot_tyre_degradation(result), 'tyre3chart')
                        eligible = [k for k,v in result['models'].items() if v['mae'] < v['baseline_mae']]
                        if eligible:
                            stint = st.selectbox('Stint passing baseline comparison', eligible)
                            horizon = st.slider('Extrapolation laps', 1, 10, 5)
                            chart(plot_tyre_prediction(result['models'][stint], future_laps=horizon), 'tyre3prediction')
                        else:
                            st.info('No fitted stint beat its baseline; extrapolation disabled.')
                    for item in generate_tyre_insights(result):
                        st.write('• ' + item)
                    for key in ['summary', 'clean_data', 'raw_data']:
                        st.download_button('Download '+key, result[key].to_csv(index=False), key+'.csv', 'text/csv', key='tyre3'+key)
                st.caption('Observed pace trends combine tyre wear, fuel, traffic and track evolution. Baseline success does not guarantee future accuracy.')
            else:
                from lap_predictor import (FEATURES, build_race_dataset, train_lap_predictor, predict_single_lap, plot_model_comparison, plot_prediction_accuracy, plot_prediction_errors, plot_driver_predictions, generate_prediction_insights)
                selected = st.multiselect('Training drivers', list(labels), default=list(labels)[:4], format_func=labels.get)
                cfg = (sk3, tuple(selected))
                if st.button('Train and evaluate models', type='primary', disabled=not selected):
                    st.session_state.pop('ml3', None)
                    with st.spinner('Building dataset and training models...'):
                        data, skipped = build_race_dataset(sk3, selected)
                        st.session_state.ml3 = (cfg, train_lap_predictor(data), skipped)
                saved = st.session_state.get('ml3')
                if saved and saved[0] == cfg:
                    result, skipped = saved[1:]
                    st.subheader(result['best_model_name'])
                    cols = st.columns(3)
                    for col, key in zip(cols, ['training_samples','validation_samples','testing_samples']):
                        col.metric(key.replace('_',' ').title(), result[key])
                    if not skipped.empty:
                        st.warning('Some drivers were skipped.'); table(skipped)
                    table(result['validation_leaderboard']); table(result['test_summary'])
                    chart(plot_model_comparison(result), 'ml3leader')
                    chart(plot_prediction_accuracy(result), 'ml3accuracy')
                    chart(plot_prediction_errors(result), 'ml3errors')
                    nums = result['test_predictions']['Driver Number'].unique().tolist()
                    n = st.selectbox('Test driver', nums, format_func=lambda x: labels.get(int(x), str(x)))
                    chart(plot_driver_predictions(result, n), 'ml3driver')
                    st.subheader('Single-lap what-if prediction')
                    sample = result['testing_data'].iloc[0]
                    with st.form('ml3form'):
                        values = {}
                        for feature in FEATURES:
                            if feature == 'Driver Number':
                                values[feature] = st.selectbox(feature, [str(x) for x in selected])
                            elif feature == 'Compound':
                                values[feature] = st.selectbox(feature, sorted(result['training_data'][feature].unique()))
                            else:
                                values[feature] = st.number_input(feature, value=float(sample[feature]) if pd.notna(sample[feature]) else 0.0)
                        submitted = st.form_submit_button('Predict lap time')
                    if submitted:
                        st.metric('Estimated lap time', f'{predict_single_lap(result, values):.3f} s')
                    for item in generate_prediction_insights(result): st.write('• '+item)
                    for key in ['validation_leaderboard','test_summary','test_predictions']:
                        st.download_button('Download '+key, result[key].to_csv(index=False), key+'.csv','text/csv',key='ml3'+key)
                st.caption('Within-race chronological evaluation only. Lap-start splits may allow boundary overlap; unseen-race accuracy is not established.')
        else:
            from strategy import (DEFAULT_TYRES, simulate_strategy, optimise_strategy, simulate_undercut_overcut, compare_strategies, format_race_time, plot_strategy_lap_times, plot_strategy_timeline, plot_strategy_leaderboard, plot_undercut_overcut, plot_strategy_comparison, generate_strategy_insights)
            st.header(page)
            st.caption('Illustrative dry-race simulation, not calibrated to a specific circuit.')
            laps = st.sidebar.slider('Race laps', 20, 80, 57)
            loss = st.sidebar.number_input('Pit loss (s)', 0.0, 60.0, 22.0)
            fuel = st.sidebar.number_input('Fuel effect (s/lap)', 0.0, 0.2, 0.03, step=0.01)
            warm = st.sidebar.number_input('Warm-up penalty (s)', 0.0, 10.0, 0.8)
            minimum = st.sidebar.slider('Minimum stint length', 1, 10, 5)
            params = {}
            with st.expander('Tyre model parameters', expanded=True):
                for compound, defaults in DEFAULT_TYRES.items():
                    st.markdown('**'+compound+'**')
                    c = st.columns(4)
                    params[compound] = dict(base_time=c[0].number_input('Base time (s)', 30.0, 300.0, float(defaults['base_time']), key=compound+'base'), degradation=c[1].number_input('Trend (s/lap)', 0.0, 2.0, float(defaults['degradation']), step=0.005, format='%.3f', key=compound+'deg'), quadratic=c[2].number_input('Quadratic term', 0.0, 0.1, 0.0, step=0.001, format='%.3f', key=compound+'quad'), max_age=c[3].number_input('Maximum stint laps', 1, 100, int(defaults['max_age']), key=compound+'age'))
            if page == 'Undercut & Overcut':
                original = st.selectbox('Starting compound', list(params))
                new = st.selectbox('New compound', [x for x in params if x != original])
                lo = max(minimum, laps-params[new]['max_age'])
                hi = min(laps-minimum, params[original]['max_age'])
                if lo > hi:
                    st.warning('No feasible reference pit lap for these settings.'); st.stop()
                pit = st.number_input('Reference pit lap', lo, hi, min(max(20,lo),hi))
                window = st.slider('Pit window', 1, 15, 5)
                result = simulate_undercut_overcut(laps, original, new, pit, params, loss, fuel, warm, window, minimum)
                chart(plot_undercut_overcut(result), 'under3')
                table(result)
                st.download_button('Download pit timing results', result.to_csv(index=False), 'undercut_overcut.csv','text/csv')
                st.caption('Positive advantage means earlier elapsed time than the reference. Both strategies represent the same hypothetical car.')
            else:
                stops = st.radio('Maximum stops', [1,2], index=1, horizontal=True)
                cfg = (laps,loss,fuel,warm,minimum,stops,repr(params))
                if st.button('Optimise race strategy', type='primary'):
                    st.session_state.pop('strategy3', None)
                    with st.spinner('Searching feasible strategies...'):
                        st.session_state.strategy3 = (cfg, optimise_strategy(laps, params, loss, fuel, warm, minimum, stops))
                saved = st.session_state.get('strategy3')
                if saved and saved[0] == cfg:
                    result = saved[1]; best = result['best_strategy']; sim = result['best_simulation']
                    st.subheader(best['Strategy'])
                    c = st.columns(3)
                    c[0].metric('Race time', format_race_time(best['Race Time (s)']))
                    c[1].metric('Pit laps', best['Pit Laps'])
                    c[2].metric('Feasible strategies', result['feasible_strategies'])
                    chart(plot_strategy_timeline(sim), 'str3timeline')
                    chart(plot_strategy_lap_times(sim), 'str3laps')
                    chart(plot_strategy_leaderboard(result), 'str3rank')
                    table(result['leaderboard'].drop(columns=['Compounds','Pit Lap Values']))
                    candidates = result['leaderboard']
                    rank = st.selectbox('Compare against ranked strategy', candidates['Rank'].tolist(), index=min(1,len(candidates)-1))
                    other = candidates[candidates['Rank']==rank].iloc[0]
                    reference = simulate_strategy(laps, other['Compounds'], other['Pit Lap Values'], params, loss, fuel, warm, minimum)
                    chart(plot_strategy_comparison(compare_strategies(sim, reference, 'Optimal', 'Alternative'), 'Optimal', 'Alternative'), 'str3compare')
                    for item in generate_strategy_insights(result): st.write('• '+item)
                    st.download_button('Download strategies', result['all_strategies'].to_csv(index=False), 'strategies.csv','text/csv')
                    st.download_button('Download optimal lap times', sim.to_csv(index=False), 'optimal_laps.csv','text/csv')
            st.warning('Simplified dry-compound rule; traffic, safety cars, weather changes and tyre availability are not modelled.')
    except Exception as e:
        st.error(f'Modelling unavailable: {e}')
    st.caption('F1 RaceLab v4.0 · Independent educational project')
    st.stop()

st.sidebar.divider()
st.sidebar.header("Race configuration")


# Initialise application state
if "racelab_v22" not in st.session_state:

    st.session_state.racelab_v22 = None


try:

    # Select season
    years = get_years()

    year = choose(
        "Season",
        years,
        qp_int("year", 2025),
        "year22"
    )

    # Retrieve Grand Prix events
    meetings = get_meetings(year)

    if meetings.empty:

        st.warning("No events available.")
        st.stop()

    meetings = meetings[
        meetings.meeting_name.str.contains(
            "Grand Prix",
            case=False,
            na=False
        )
    ]

    if meetings.empty:

        st.warning("No Grand Prix events available.")
        st.stop()

    mo = dict(zip(
        meetings.meeting_key.astype(int),
        meetings.meeting_name
    ))

    preferred_meeting = qp_int("meeting", -1)

    mk = st.sidebar.selectbox(
        "Grand Prix",
        list(mo),
        index=(
            list(mo).index(preferred_meeting)
            if preferred_meeting in mo
            else 0
        ),
        format_func=mo.get,
        key=f"m22{year}"
    )

    # Retrieve completed sessions
    sessions = get_sessions(mk)

    if sessions.empty:

        st.warning("No completed sessions.")
        st.stop()

    so = dict(zip(
        sessions.session_key.astype(int),
        sessions.session_name
    ))

    default = next(
        (k for k, v in so.items() if v == "Qualifying"),
        next(iter(so))
    )

    preferred_session = qp_int("session", default)

    sk = st.sidebar.selectbox(
        "Session",
        list(so),
        index=(
            list(so).index(preferred_session)
            if preferred_session in so
            else list(so).index(default)
        ),
        format_func=so.get,
        key=f"s22{mk}"
    )

    # Retrieve drivers
    drivers = get_drivers(sk)

    if len(drivers) < 2:

        st.warning("Insufficient driver data.")
        st.stop()

    records = {
        int(r.driver_number): r
        for _, r in drivers.iterrows()
    }

    nums = list(records)

    def driver_label(n):

        return f'{records[n]["full_name"]} (#{n})'

    st.sidebar.subheader("Drivers")

    # Default Driver A: Lando Norris
    da = qp_int("a", 4)

    na = st.sidebar.selectbox(
        "Driver A",
        nums,
        index=nums.index(da) if da in nums else 0,
        format_func=driver_label,
        key=f"a22{sk}"
    )

    remaining = [
        n for n in nums if n != na
    ]

    # Default Driver B: Oscar Piastri
    db = qp_int("b", 81)

    nb = st.sidebar.selectbox(
        "Driver B",
        remaining,
        index=(
            remaining.index(db)
            if db in remaining
            else 0
        ),
        format_func=driver_label,
        key=f"b22{sk}_{na}"
    )

    driver_a = records[na]
    driver_b = records[nb]

    name_a = driver_a.full_name
    name_b = driver_b.full_name

    # Retrieve laps
    laps_a = get_laps(sk, na)
    laps_b = get_laps(sk, nb)

    if laps_a.empty or laps_b.empty:

        st.warning(
            "Select drivers with completed laps."
        )

        st.stop()

    mode = st.sidebar.radio(
        "Lap selection",
        [
            "Fastest recorded lap",
            "Specific lap"
        ],
        index=int(
            st.query_params.get("mode", "fast")
            == "specific"
        )
    )

    if mode == "Specific lap":

        la = choose(
            name_a + " lap",
            laps_a.lap_number.tolist(),
            qp_int("la", -1),
            f"la22{sk}_{na}"
        )

        lb = choose(
            name_b + " lap",
            laps_b.lap_number.tolist(),
            qp_int("lb", -1),
            f"lb22{sk}_{nb}"
        )

        lap_a = get_specific_lap(laps_a, la)
        lap_b = get_specific_lap(laps_b, lb)

    else:

        lap_a = get_fastest_lap(laps_a)
        lap_b = get_fastest_lap(laps_b)

    # Analysis settings
    st.sidebar.subheader("Analysis settings")

    ns = qp_int("sections", 20)

    ns = ns if ns in range(5, 51, 5) else 20

    count = st.sidebar.slider(
        "Track sections",
        5, 50, ns, 5
    )

    # Automatic team colours
    ca = team_color(driver_a, "#FF8700")
    cb = team_color(driver_b, "#3693FF")

    # Distinguish drivers from the same team
    if ca.lower() == cb.lower():

        cb = (
            "#FFFFFF"
            if ca.lower() != "#ffffff"
            else "#3693FF"
        )

    custom = st.sidebar.checkbox(
        "Custom driver colours"
    )

    if custom:

        ca = st.sidebar.color_picker(
            "Driver A colour",
            ca,
            key=f"ca22{sk}_{na}"
        )

        cb = st.sidebar.color_picker(
            "Driver B colour",
            cb,
            key=f"cb22{sk}_{nb}"
        )

    context = st.sidebar.checkbox(
        "Include tyres and weather",
        value=(
            st.query_params.get("context", "1") != "0"
        )
    )

    config = (
        year, mk, sk, na, nb,
        int(lap_a.lap_number),
        int(lap_b.lap_number),
        count, context
    )

except Exception as e:

    st.error(f"Unable to load session: {e}")
    st.stop()


# Generate engineering analysis
if st.sidebar.button(
    "Generate Analysis",
    type="primary",
    use_container_width=True
):

    st.session_state.racelab_v22 = None

    bar = st.progress(
        0,
        text="Loading telemetry"
    )

    try:

        ta = get_telemetry(sk, na, lap_a)

        bar.progress(
            25,
            text="Loading second driver"
        )

        tb = get_telemetry(sk, nb, lap_b)

        bar.progress(
            40,
            text="Running engineering analysis"
        )

        r = run_analysis(
            ta, tb,
            name_a, name_b,
            float(lap_a.lap_duration),
            float(lap_b.lap_duration),
            count
        )

        warnings = []

        bar.progress(
            60,
            text="Loading circuit positions"
        )

        for suffix, n, lap in [
            ("a", na, lap_a),
            ("b", nb, lap_b)
        ]:

            pos, err = optional(
                get_position, sk, n, lap
            )

            if err:
                warnings.append(err)

            track, err = optional(
                merge_position_telemetry,
                pos,
                r["telemetry_" + suffix]
            )

            r["track_" + suffix] = track

            if err:
                warnings.append(err)

        bar.progress(
            80,
            text="Loading session context"
        )

        for key, fn, args in [
            ("weather", get_weather, (sk,)),
            ("stints_a", get_stints, (sk, na)),
            ("stints_b", get_stints, (sk, nb))
        ]:

            r[key], err = (
                optional(fn, *args)
                if context
                else (pd.DataFrame(), None)
            )

            if err:
                warnings.append(err)

        # Analysis metadata
        r["metadata"] = pd.DataFrame([{
            "Season": year,
            "Grand Prix": mo[mk],
            "Session": so[sk],
            "Session Key": sk,
            "Driver A": name_a,
            "Driver B": name_b,
            "Driver A Lap": int(lap_a.lap_number),
            "Driver B Lap": int(lap_b.lap_number),
            "Driver A Lap Time": float(lap_a.lap_duration),
            "Driver B Lap Time": float(lap_b.lap_duration),
            "Track Sections": count
        }])

        observations = generate_insights(
            r, name_a, name_b
        )

        report = generate_report(
            r, name_a, name_b
        )

        st.session_state.racelab_v22 = dict(
            config=config,
            results=r,
            insights=observations,
            report=report,
            warnings=warnings
        )

        bar.progress(100, text="Complete")
        bar.empty()

    except Exception as e:

        bar.empty()
        st.error(f"Analysis failed: {e}")


# Create shareable comparison URL
if st.sidebar.button(
    "Create shareable comparison URL"
):

    st.query_params.update(
        year=str(year),
        meeting=str(mk),
        session=str(sk),
        a=str(na),
        b=str(nb),
        la=str(int(lap_a.lap_number)),
        lb=str(int(lap_b.lap_number)),
        mode=(
            "specific"
            if mode == "Specific lap"
            else "fast"
        ),
        sections=str(count),
        context=str(int(context))
    )

    st.sidebar.success(
        "Copy the address from your browser. "
        "Recipients must click Generate Analysis."
    )

st.sidebar.caption(
    "Historical OpenF1 data · 2023 onwards"
)


# About page
if page == "About":

    st.header("About F1 RaceLab")

    st.write(
        "An independent engineering software project "
        "for exploring Formula 1 telemetry and "
        "driver performance."
    )

    st.subheader("Developer")

    st.write(
        "Vaishnavi Prabagaran — Electrical and "
        "Computer Systems Engineering student, "
        "Monash University."
    )

    st.markdown(
        "[Source code on GitHub]"
        "(https://github.com/vaishpraba119-cyber/F1-RaceLab)"
        " · [OpenF1 data source](https://openf1.org/)"
    )

    st.caption(
        "Independent educational project. "
        "Not affiliated with Formula 1 or its teams."
    )

    st.stop()


# Methodology page
if page == "Methodology":

    st.header("Engineering methodology")

    st.write(
        "Speed is converted to m/s and integrated "
        "using the trapezoidal rule. Numerical "
        "differentiation and a three-sample moving "
        "average estimate longitudinal acceleration."
    )

    st.latex(
        r"s=\sum_i\frac{v_i+v_{i+1}}{2}(t_{i+1}-t_i)"
    )

    st.latex(
        r"G=-\frac{dv/dt}{9.81}"
    )

    st.write(
        "Binary brake transitions identify events; "
        "events shorter than 0.3 seconds are excluded. "
        "Gear usage is time weighted. Position samples "
        "are matched to telemetry within 0.5 seconds."
    )

    st.latex(
        r"p=s/s_{total},\quad \Delta t(p)=t_B(p)-t_A(p)"
    )

    st.write(
        "Lap time is interpolated over normalised "
        "distance and scaled to recorded duration. "
        "Positive delta favours A. Sections are "
        "equal-distance intervals, not official corners. "
        "Coverage measures sampled time span, "
        "not continuity."
    )

    st.warning(
        "Speed-derived G is not directly measured. "
        "Brake status is not pressure. Distance "
        "alignment is approximate; normalised final "
        "delta is constrained, not independent validation. "
        "Deleted laps, missing channels, tyre condition, "
        "traffic, setup, fuel and weather may affect "
        "comparisons. DRS codes are raw. Insights do "
        "not establish causation."
    )

    st.stop()


# Display race and driver information
st.subheader(f"{year} {mo[mk]}")
st.caption(so[sk])

c1, c2 = st.columns(2)

with c1:
    card(driver_a, lap_a, ca)

with c2:
    card(driver_b, lap_b, cb)


if page == "Advanced Lap Replay":
    from race_replay import load_replay_data, create_race_replay, calculate_replay_summary
    st.header("Advanced Lap Replay")
    interval = st.slider("Frame interval (s)", 0.2, 1.0, 0.4, 0.2)
    playback = st.slider("Playback speed", 0.5, 5.0, 1.0, 0.5)
    replay_cfg = (sk, na, nb, int(lap_a.lap_number), int(lap_b.lap_number))
    try:
        if st.button("Load lap replay", type="primary"):
            st.session_state.pop("lap4", None)
            with st.spinner("Loading lap positions and telemetry..."):
                st.session_state.lap4 = (replay_cfg, load_replay_data(sk, na, lap_a), load_replay_data(sk, nb, lap_b))
        saved = st.session_state.get("lap4")
        if saved and saved[0] == replay_cfg:
            da, db = saved[1:]
            chart(create_race_replay(da, db, name_a, name_b, ca, cb, interval, playback), "lap4chart")
            table(calculate_replay_summary(da, db, name_a, name_b))
    except Exception as exc:
        st.error(f"Lap replay unavailable: {exc}")
    st.stop()

# Retrieve stored analysis
stored = st.session_state.racelab_v22

if stored is None or stored["config"] != config:

    st.info(
        "Choose a session and two drivers, "
        "then click Generate Analysis in the sidebar."
    )

    st.stop()

r = stored["results"]
insights = stored["insights"]
report = stored["report"]

for warning in stored["warnings"]:

    st.warning(warning)

a = r["telemetry_a"]
b = r["telemetry_b"]

perf = r["performance"]
events = r["braking_events"]
summary = r["braking_summary"]
comp = r["lap_comparison"]
sections = r["track_sections"]
gear = r["gear_usage"]

tracks = [
    r["track_a"],
    r["track_b"]
]


# Display automated observations
def observations(category):

    for item in insights.get(category, []):

        st.markdown("- " + item)


# Display telemetry channel
def telemetry(channel, key):

    chart(
        plot_telemetry(
            a, b, name_a, name_b,
            channel, ca, cb
        ),
        key
    )


# Display lap delta
def delta(raw=False):

    chart(
        plot_lap_delta(
            comp, name_a, name_b,
            ca, cb,
            show_raw=raw
        ),
        "delta22"
    )


# Overview page
if page == "Overview":

    st.header("Performance overview")

    d = float(
        lap_b.lap_duration - lap_a.lap_duration
    )

    winner = driver_a if d > 0 else driver_b

    st.subheader(
        "Equal lap times"
        if np.isclose(d, 0)
        else f"{winner.full_name} is faster by {abs(d):.3f} s"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Lap difference",
        f"{abs(d):.3f} s"
    )

    c2.metric(
        "Maximum speed",
        f'{perf["Top Speed (km/h)"].max():.0f} km/h'
    )

    c3.metric(
        "Peak braking",
        f'{perf["Peak Braking G"].max():.2f} G'
    )

    st.subheader("Key findings")

    for category in [
        "Lap Performance",
        "Speed Analysis",
        "Throttle Analysis"
    ]:

        observations(category)

    st.dataframe(
        perf.set_index("Driver").T.round(3),
        use_container_width=True
    )

    delta()


# Telemetry page
elif page == "Telemetry":

    st.header("Telemetry comparison")

    channels = {
        "Speed": "speed",
        "Throttle": "throttle",
        "Brake status": "Brake_Status",
        "Engine RPM": "rpm",
        "Gear": "n_gear",
        "DRS": "drs",
        "Acceleration": "Smoothed_Acceleration",
        "Deceleration": "Deceleration",
        "Braking G": "Braking_G"
    }

    available = {
        k: v for k, v in channels.items()
        if v in a or v in b
    }

    selected = st.selectbox(
        "Channel",
        list(available)
    )

    telemetry(
        available[selected],
        "telemetry22"
    )

    if st.checkbox("All channels"):

        chart(
            plot_all_telemetry(
                a, b, name_a, name_b, ca, cb
            ),
            "all22"
        )


# Braking analysis page
elif page == "Braking":

    st.header("Braking performance")

    table(summary)

    metric = st.selectbox(
        "Summary metric",
        [c for c in summary if c != "Driver"]
    )

    chart(
        plot_braking_summary(
            summary, name_a, name_b,
            metric, ca, cb
        ),
        "bs22"
    )

    metric = st.selectbox(
        "Event metric",
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

    chart(
        plot_braking_events(
            events, name_a, name_b,
            metric, ca, cb
        ),
        "be22"
    )

    table(events)

    observations("Braking Analysis")

    st.caption(
        "Event numbers do not necessarily "
        "identify matching corners."
    )


# Lap delta page
elif page == "Lap Delta":

    st.header("Lap time delta")

    delta(
        st.checkbox(
            "Show raw sampled delta",
            True
        )
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Raw",
        f'{comp["Raw Sampled Delta (s)"].iloc[-1]:+.3f} s'
    )

    c2.metric(
        "Normalised",
        f'{comp["Time Advantage (s)"].iloc[-1]:+.3f} s'
    )

    c3.metric(
        "Section sum",
        f'{sections["Driver A Advantage (s)"].sum():+.3f} s'
    )

    st.caption(
        f"Positive favours {name_a}; "
        f"negative favours {name_b}. "
        "Normalised final delta is constrained "
        "to recorded timing."
    )


# Track section page
elif page == "Track Sections":

    st.header("Track section analysis")

    chart(
        plot_track_sections(
            sections, name_a, name_b, ca, cb
        ),
        "sections22"
    )

    table(sections)

    observations("Track Section Analysis")

    st.caption(
        "Equal normalised distance intervals, "
        "not official corners."
    )


# Circuit heatmap page
elif page == "Circuit Maps":

    st.header("Circuit heatmaps")

    channel = st.selectbox(
        "Map channel",
        [
            "speed",
            "throttle",
            "brake",
            "rpm",
            "n_gear",
            "drs"
        ]
    )

    values = [
        t[channel].dropna()
        for t in tracks
        if not t.empty and channel in t
    ]

    scale = None

    if values:

        v = pd.concat(values)

        if not v.empty and v.max() > v.min():

            scale = (v.min(), v.max())

    for i, (col, t, name) in enumerate(
        zip(
            st.columns(2),
            tracks,
            [name_a, name_b]
        )
    ):

        with col:

            chart(
                plot_circuit_heatmap(
                    t, name, channel, scale
                ),
                f"map22{i}"
            )

    chart(
        plot_performance_map(
            tracks[0], comp,
            name_a, name_b, ca, cb
        ),
        "gain22"
    )

    st.caption(
        "Local gain is an approximate "
        "distance-aligned estimate, "
        "not exact GPS alignment."
    )


# Gear analysis page
elif page == "Gear Analysis":

    st.header("Gear and powertrain")

    chart(
        plot_gear_usage(
            gear, name_a, name_b, ca, cb
        ),
        "gear22"
    )

    for channel in ["n_gear", "rpm"]:

        if channel in a or channel in b:

            telemetry(
                channel,
                channel + "22"
            )

    table(gear)

    observations("Gear Analysis")


# Tyres and weather page
elif page == "Tyres & Weather":

    st.header("Tyres and weather")

    if not context:

        st.info(
            "Enable session context and "
            "regenerate analysis."
        )

    else:

        for col, name, lap, key in zip(
            st.columns(2),
            [name_a, name_b],
            [lap_a, lap_b],
            ["stints_a", "stints_b"]
        ):

            with col:

                st.subheader(name)

                t = r[key]

                if t.empty:

                    st.info("Tyre data unavailable.")
                    continue

                if {
                    "lap_start",
                    "lap_end"
                }.issubset(t.columns):

                    selected = t[
                        np.logical_and(
                            t.lap_start <= lap.lap_number,
                            t.lap_end >= lap.lap_number
                        )
                    ]

                    if not selected.empty:

                        s = selected.iloc[0]

                        st.metric(
                            "Compound",
                            str(s.get("compound", "Unknown"))
                        )

                        age = s.get(
                            "tyre_age_at_start",
                            np.nan
                        )

                        if pd.notna(age):

                            estimated_age = (
                                age
                                + lap.lap_number
                                - s.lap_start
                            )

                            st.metric(
                                "Estimated tyre age at lap start",
                                f"{int(estimated_age)} laps"
                            )

                table(t)

        w = r["weather"].copy()

        chart(
            plot_weather(w),
            "weather22"
        )

        if not w.empty and "date" in w:

            w["date"] = pd.to_datetime(
                w.date,
                utc=True,
                errors="coerce"
            )

            w = w.dropna(
                subset=["date"]
            ).reset_index(drop=True)

            for name, lap in [
                (name_a, lap_a),
                (name_b, lap_b)
            ]:

                if w.empty:
                    break

                offset = (
                    w.date - pd.to_datetime(
                        lap.date_start,
                        utc=True
                    )
                ).abs()

                idx = offset.idxmin()

                row = w.loc[idx]

                st.subheader(
                    name + " — nearest weather sample"
                )

                cols = st.columns(4)

                for col, label, key, unit in zip(
                    cols,
                    ["Air", "Track", "Humidity", "Wind"],
                    [
                        "air_temperature",
                        "track_temperature",
                        "humidity",
                        "wind_speed"
                    ],
                    ["°C", "°C", "%", "m/s"]
                ):

                    col.metric(
                        label,
                        f'{row.get(key, "N/A")} {unit}'
                    )

                seconds = offset.loc[idx].total_seconds()

                st.caption(
                    f"Sample offset: {seconds:.1f} s"
                )

                if seconds > 120:

                    st.warning(
                        "Weather sample is more than "
                        "two minutes from lap start."
                    )


# Engineering insights page
elif page == "Engineering Insights":

    st.header("Automated engineering insights")

    for category, items in insights.items():

        if items:

            with st.expander(
                category,
                expanded=True
            ):

                observations(category)

    st.warning(
        "Observations do not establish causation."
    )

    st.download_button(
        "Download engineering report",
        report,
        "engineering_report.md",
        "text/markdown"
    )

    with st.expander("Report preview"):

        st.markdown(report)


# Data export page
elif page == "Data Export":

    st.header("Export analysis")

    st.download_button(
        "Download complete analysis ZIP",
        zip_results(r, report),
        f"F1_RaceLab_{sk}_{na}_{nb}.zip",
        "application/zip",
        type="primary"
    )

    st.download_button(
        "Download report",
        report,
        "engineering_report.md",
        "text/markdown"
    )

    for key, data in r.items():

        if (
            isinstance(data, pd.DataFrame)
            and not data.empty
        ):

            st.download_button(
                "Download " + key.replace("_", " "),
                data.to_csv(index=False),
                key + ".csv",
                "text/csv",
                key="dl22" + key
            )


# Application footer
st.divider()

st.caption(
    "F1 RaceLab v4.0 · "
    "Independent educational project · OpenF1"
)
