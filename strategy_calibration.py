import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error

from data_loader import (
    get_data,
    get_drivers,
    get_laps,
    get_stints
)

from strategy import (
    optimise_strategy,
    DEFAULT_TYRES
)


DRY_COMPOUNDS = [
    "SOFT",
    "MEDIUM",
    "HARD"
]

TYRE_COLOURS = {
    "SOFT": "#E10600",
    "MEDIUM": "#FFD700",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#00D26A",
    "WET": "#168BFF"
}


def load_race_control(session_key):

    data = get_data(
        "race_control",
        {"session_key": session_key}
    )

    if data.empty:
        return data

    data["date"] = pd.to_datetime(
        data["date"],
        utc=True,
        errors="coerce"
    )

    return data.dropna(
        subset=["date"]
    ).sort_values("date")


def get_neutralised_periods(
    race_control,
    session_end
):

    periods = []

    if race_control.empty:
        return periods

    active_start = None

    for _, row in race_control.iterrows():

        message = str(
            row.get("message", "")
        ).upper()

        category = str(
            row.get("category", "")
        ).upper()

        flag = str(
            row.get("flag", "")
        ).upper()

        timestamp = row["date"]

        start_event = (
            "SAFETY CAR DEPLOYED" in message
            or "VSC DEPLOYED" in message
            or "VIRTUAL SAFETY CAR DEPLOYED"
            in message
            or flag == "RED"
        )

        end_event = (
            "SAFETY CAR IN THIS LAP" in message
            or "VSC ENDING" in message
            or "VSC ENDED" in message
            or "SAFETY CAR ENDING" in message
            or (
                flag == "GREEN"
                and category == "FLAG"
                and str(
                    row.get("scope", "")
                ).upper() == "TRACK"
            )
        )

        if start_event:

            if active_start is None:
                active_start = timestamp

        elif end_event:

            if active_start is not None:

                periods.append((
                    active_start,
                    timestamp
                    + pd.Timedelta(seconds=120)
                ))

                active_start = None

    if active_start is not None:

        periods.append((
            active_start,
            session_end
        ))

    return periods


def load_driver_calibration_data(
    session_key,
    driver_number,
    fuel_effect=0.03
):

    laps = get_laps(
        session_key,
        driver_number
    )

    stints = get_stints(
        session_key,
        driver_number
    )

    if laps.empty or stints.empty:
        return pd.DataFrame()

    laps = laps.copy()

    laps["date_start"] = pd.to_datetime(
        laps["date_start"],
        utc=True,
        errors="coerce"
    )

    laps["date_end"] = (
        laps["date_start"]
        + pd.to_timedelta(
            laps["lap_duration"],
            unit="s"
        )
    )

    rows = []

    for _, stint in stints.iterrows():

        compound = str(
            stint["compound"]
        ).upper()

        if compound not in DRY_COMPOUNDS:
            continue

        first_lap = stint["lap_start"]
        last_lap = stint["lap_end"]

        starting_age = stint.get(
            "tyre_age_at_start",
            0
        )

        if pd.isna(starting_age):
            continue

        selected = laps[
            np.logical_and(
                laps["lap_number"] >= first_lap,
                laps["lap_number"] <= last_lap
            )
        ].copy()

        for _, lap in selected.iterrows():

            lap_number = int(
                lap["lap_number"]
            )

            lap_time = float(
                lap["lap_duration"]
            )

            if lap_number == 1:
                continue

            if lap_number == first_lap:
                continue

            if lap_number == last_lap:
                continue

            if lap.get(
                "is_pit_out_lap",
                False
            ) == True:
                continue

            tyre_age = (
                lap_number
                - first_lap
                + starting_age
            )

            corrected_time = (
                lap_time
                + fuel_effect
                * (lap_number - 1)
            )

            rows.append({
                "Driver Number": driver_number,
                "Lap": lap_number,
                "Stint": int(
                    stint["stint_number"]
                ),
                "Compound": compound,
                "Tyre Age": tyre_age,
                "Lap Time (s)": lap_time,
                "Fuel Corrected Time (s)":
                    corrected_time,
                "Start Time": lap["date_start"],
                "End Time": lap["date_end"]
            })

    return pd.DataFrame(rows)


def remove_neutralised_laps(
    data,
    periods
):

    if data.empty:
        return data

    valid = np.ones(
        len(data),
        dtype=bool
    )

    for start, end in periods:

        overlap = np.logical_and(
            data["Start Time"] <= end,
            data["End Time"] >= start
        )

        valid = np.logical_and(
            valid,
            np.logical_not(
                overlap.to_numpy()
            )
        )

    return data.loc[
        valid
    ].reset_index(drop=True)


def remove_abnormal_laps(
    data,
    threshold=3.5
):

    if data.empty:
        return data

    groups = []

    for _, group in data.groupby(
        ["Driver Number", "Stint"]
    ):

        group = group.copy()

        if len(group) < 5:
            continue

        times = group[
            "Fuel Corrected Time (s)"
        ].to_numpy()

        median = np.median(times)

        mad = np.median(
            np.abs(times - median)
        )

        if mad > 0:

            scores = (
                0.6745
                * np.abs(times - median)
                / mad
            )

            group = group.loc[
                scores <= threshold
            ]

        groups.append(group)

    if not groups:
        return pd.DataFrame(
            columns=data.columns
        )

    return pd.concat(
        groups,
        ignore_index=True
    )


def fit_compound_model(
    data,
    compound,
    minimum_samples=12
):

    selected = data[
        data["Compound"] == compound
    ].copy()

    selected = selected.sort_values(
        "Start Time"
    )

    if len(selected) < minimum_samples:
        return None

    split = int(
        len(selected) * 0.75
    )

    training = selected.iloc[:split]
    testing = selected.iloc[split:]

    if training["Tyre Age"].nunique() < 3:
        return None

    if len(testing) < 3:
        return None

    x_train = training[
        ["Tyre Age"]
    ].to_numpy()

    y_train = training[
        "Fuel Corrected Time (s)"
    ].to_numpy()

    x_test = testing[
        ["Tyre Age"]
    ].to_numpy()

    y_test = testing[
        "Fuel Corrected Time (s)"
    ].to_numpy()

    model = HuberRegressor(
        max_iter=2000
    )

    model.fit(
        x_train,
        y_train
    )

    predictions = model.predict(
        x_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    baseline = np.full(
        len(y_test),
        np.median(y_train)
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline
    )

    # Refit using all retained historical
    # observations for strategy calibration.

    final_model = HuberRegressor(
        max_iter=2000
    )

    final_model.fit(
        selected[
            ["Tyre Age"]
        ].to_numpy(),
        selected[
            "Fuel Corrected Time (s)"
        ].to_numpy()
    )

    base_time = float(
        final_model.intercept_
    )

    degradation = float(
        final_model.coef_[0]
    )

    residuals = (
        selected[
            "Fuel Corrected Time (s)"
        ].to_numpy()
        - final_model.predict(
            selected[
                ["Tyre Age"]
            ].to_numpy()
        )
    )

    return {
        "compound": compound,
        "model": final_model,
        "base_time": base_time,
        "degradation": degradation,
        "mae": float(mae),
        "baseline_mae": float(baseline_mae),
        "beats_baseline": bool(
            mae < baseline_mae
        ),
        "residual_std": float(
            np.std(residuals, ddof=1)
        ),
        "samples": len(selected),
        "training_samples": len(training),
        "testing_samples": len(testing),
        "drivers": selected[
            "Driver Number"
        ].nunique(),
        "minimum_age": int(
            selected["Tyre Age"].min()
        ),
        "maximum_age": int(
            selected["Tyre Age"].max()
        ),
        "data": selected
    }


def estimate_pit_loss(
    session_key,
    driver_numbers,
    neutralised_periods
):

    pits = get_data(
        "pit",
        {"session_key": session_key}
    )

    if pits.empty:
        return {
            "estimated_pit_loss": None,
            "samples": 0,
            "data": pd.DataFrame()
        }

    pits = pits[
        pits["driver_number"].isin(
            driver_numbers
        )
    ].copy()

    rows = []

    for number in driver_numbers:

        laps = get_laps(
            session_key,
            number
        )

        if laps.empty:
            continue

        laps = laps.copy()

        laps["date_start"] = pd.to_datetime(
            laps["date_start"],
            utc=True,
            errors="coerce"
        )

        laps["date_end"] = (
            laps["date_start"]
            + pd.to_timedelta(
                laps["lap_duration"],
                unit="s"
            )
        )

        driver_pits = pits[
            pits["driver_number"] == number
        ]

        pit_laps = set(
            driver_pits[
                "lap_number"
            ].dropna().astype(int)
        )

        for _, pit in driver_pits.iterrows():

            if pd.isna(pit["lap_number"]):
                continue

            lap_number = int(
                pit["lap_number"]
            )

            affected = laps[
                laps["lap_number"].isin([
                    lap_number,
                    lap_number + 1
                ])
            ]

            reference = laps[
                laps["lap_number"].isin([
                    lap_number - 3,
                    lap_number - 2,
                    lap_number - 1,
                    lap_number + 2,
                    lap_number + 3,
                    lap_number + 4
                ])
            ]

            excluded = (
                pit_laps
                | {x + 1 for x in pit_laps}
                | {1}
            )

            reference = reference[
                ~reference[
                    "lap_number"
                ].isin(excluded)
            ]

            if (
                len(affected) != 2
                or len(reference) < 3
            ):
                continue

            combined = pd.concat([
                affected,
                reference
            ])

            neutralised = False

            for start, end in neutralised_periods:

                overlap = np.logical_and(
                    combined["date_start"] <= end,
                    combined["date_end"] >= start
                )

                if overlap.any():
                    neutralised = True
                    break

            if neutralised:
                continue

            reference_time = float(
                reference[
                    "lap_duration"
                ].median()
            )

            loss = (
                affected[
                    "lap_duration"
                ].sum()
                - 2 * reference_time
            )

            # Reject implausible or heavily
            # disrupted pit-cycle estimates.

            if not 5 <= loss <= 60:
                continue

            rows.append({
                "Driver Number": number,
                "Pit Lap": lap_number,
                "Reference Lap Time (s)":
                    reference_time,
                "Estimated Pit Loss (s)":
                    float(loss)
            })

    data = pd.DataFrame(rows)

    if data.empty:
        return {
            "estimated_pit_loss": None,
            "samples": 0,
            "data": data
        }

    return {
        "estimated_pit_loss": float(
            data[
                "Estimated Pit Loss (s)"
            ].median()
        ),
        "samples": len(data),
        "data": data
    }


def calibrate_strategy(
    session_key,
    driver_numbers=None,
    fuel_effect=0.03
):

    if fuel_effect < 0:
        raise ValueError(
            "Fuel effect cannot be negative."
        )

    drivers = get_drivers(
        session_key
    )

    if driver_numbers is None:

        driver_numbers = drivers[
            "driver_number"
        ].astype(int).tolist()

    sessions = get_data(
        "sessions",
        {"session_key": session_key}
    )

    if sessions.empty:
        raise ValueError(
            "Session not found."
        )

    session = sessions.iloc[0]

    session_end = pd.to_datetime(
        session["date_end"],
        utc=True
    )

    race_control = load_race_control(
        session_key
    )

    periods = get_neutralised_periods(
        race_control,
        session_end
    )

    tables = []
    skipped = {}

    for number in driver_numbers:

        print(
            f"Calibrating driver #{number}..."
        )

        try:

            data = load_driver_calibration_data(
                session_key,
                number,
                fuel_effect
            )

            if data.empty:

                skipped[number] = (
                    "No suitable dry-tyre laps"
                )

            else:
                tables.append(data)

        except Exception as error:

            skipped[number] = str(error)

    if not tables:

        raise ValueError(
            "No suitable calibration data."
        )

    raw_data = pd.concat(
        tables,
        ignore_index=True
    )

    green_data = remove_neutralised_laps(
        raw_data,
        periods
    )

    clean_data = remove_abnormal_laps(
        green_data
    )

    if clean_data.empty:

        raise ValueError(
            "No laps remain after filtering."
        )

    models = {}
    summary_rows = []

    for compound in DRY_COMPOUNDS:

        model = fit_compound_model(
            clean_data,
            compound
        )

        if model is None:
            continue

        models[compound] = model

        summary_rows.append({
            "Compound": compound,
            "Samples": model["samples"],
            "Drivers": model["drivers"],
            "Base Time (s)": model["base_time"],
            "Degradation (s/lap)":
                model["degradation"],
            "MAE (s)": model["mae"],
            "Baseline MAE (s)":
                model["baseline_mae"],
            "Residual Std (s)":
                model["residual_std"],
            "Maximum Observed Age":
                model["maximum_age"],
            "Beats Baseline":
                model["beats_baseline"]
        })

    pit_result = estimate_pit_loss(
        session_key,
        driver_numbers,
        periods
    )

    return {
        "session": session,
        "raw_data": raw_data,
        "green_data": green_data,
        "clean_data": clean_data,
        "models": models,
        "summary": pd.DataFrame(summary_rows),
        "pit_result": pit_result,
        "neutralised_periods": periods,
        "race_control_available":
            not race_control.empty,
        "fuel_effect": fuel_effect,
        "skipped": skipped
    }


def build_calibrated_parameters(
    calibration
):

    parameters = {}

    rejected = {}

    for compound in DRY_COMPOUNDS:

        model = calibration[
            "models"
        ].get(compound)

        if model is None:

            rejected[compound] = (
                "Insufficient historical data"
            )

            continue

        if not model["beats_baseline"]:

            rejected[compound] = (
                "Model did not beat baseline"
            )

            continue

        if model["degradation"] < 0:

            rejected[compound] = (
                "Negative degradation trend"
            )

            continue

        if model["base_time"] <= 0:

            rejected[compound] = (
                "Invalid estimated base time"
            )

            continue

        parameters[compound] = {
            "base_time":
                model["base_time"],
            "degradation":
                model["degradation"],
            "quadratic": 0.0,
            "max_age": max(
                1,
                model["maximum_age"]
            )
        }

    return parameters, rejected


def run_calibrated_optimisation(
    calibration,
    total_laps,
    maximum_stops=2,
    minimum_stint=5
):

    calibrated, rejected = (
        build_calibrated_parameters(
            calibration
        )
    )

    # Missing or unreliable compounds use
    # the existing illustrative assumptions.
    # Match their base-time scale to the
    # accepted historical models.

    parameters = {}

    offsets = [
        calibrated[c]["base_time"]
        - DEFAULT_TYRES[c]["base_time"]
        for c in calibrated
    ]

    offset = (
        float(np.median(offsets))
        if offsets else 0.0
    )

    sources = []

    for compound in DRY_COMPOUNDS:

        if compound in calibrated:

            parameters[compound] = (
                calibrated[compound].copy()
            )

            source = "Historical model"

        else:

            parameters[compound] = (
                DEFAULT_TYRES[compound].copy()
            )

            parameters[compound][
                "base_time"
            ] += offset

            source = "Assumed fallback"

        sources.append({
            "Compound": compound,
            "Source": source,
            **parameters[compound]
        })

    pit_result = calibration[
        "pit_result"
    ]

    historical_pit_loss = pit_result[
        "estimated_pit_loss"
    ]

    if (
        historical_pit_loss is not None
        and pit_result["samples"] >= 3
    ):

        pit_loss = historical_pit_loss

        pit_source = (
            "Historical pit-cycle estimate"
        )

        # Historical pit-cycle loss already
        # includes some warm-up effects.

        warmup_penalty = 0.0

    else:

        pit_loss = 22.0
        pit_source = "Assumed fallback"
        warmup_penalty = 0.8

    result = optimise_strategy(
        total_laps=total_laps,
        tyre_parameters=parameters,
        pit_loss=pit_loss,
        fuel_effect=calibration[
            "fuel_effect"
        ],
        warmup_penalty=warmup_penalty,
        minimum_stint=minimum_stint,
        maximum_stops=maximum_stops
    )

    return {
        "optimisation": result,
        "parameters": parameters,
        "parameter_sources":
            pd.DataFrame(sources),
        "rejected_models": rejected,
        "pit_loss": pit_loss,
        "pit_source": pit_source,
        "warmup_penalty": warmup_penalty
    }


def plot_calibrated_degradation(
    calibration
):

    figure = go.Figure()

    for compound, model in (
        calibration["models"].items()
    ):

        data = model["data"]

        colour = TYRE_COLOURS[
            compound
        ]

        figure.add_trace(
            go.Scatter(
                x=data["Tyre Age"],
                y=data[
                    "Fuel Corrected Time (s)"
                ],
                mode="markers",
                name=f"{compound} observations",
                marker=dict(
                    color=colour,
                    size=5,
                    opacity=0.45
                )
            )
        )

        ages = np.linspace(
            model["minimum_age"],
            model["maximum_age"],
            100
        )

        predicted = model[
            "model"
        ].predict(
            ages.reshape(-1, 1)
        )

        figure.add_trace(
            go.Scatter(
                x=ages,
                y=predicted,
                mode="lines",
                name=f"{compound} model",
                line=dict(
                    color=colour,
                    width=3
                )
            )
        )

    figure.update_layout(
        title="Historical Tyre Performance",
        xaxis_title="Tyre age (laps)",
        yaxis_title="Fuel-corrected lap time (s)",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=600
    )

    return figure


def plot_calibration_accuracy(
    calibration
):

    summary = calibration["summary"]

    figure = go.Figure()

    if not summary.empty:

        figure.add_trace(
            go.Bar(
                x=summary["Compound"],
                y=summary["MAE (s)"],
                name="Historical model",
                marker_color="#E10600"
            )
        )

        figure.add_trace(
            go.Bar(
                x=summary["Compound"],
                y=summary["Baseline MAE (s)"],
                name="Median baseline",
                marker_color="#89919D"
            )
        )

    figure.update_layout(
        title="Held-out Model Error",
        yaxis_title="Mean absolute error (s)",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    return figure


def plot_pit_loss(calibration):

    data = calibration[
        "pit_result"
    ]["data"]

    figure = go.Figure()

    if not data.empty:

        figure.add_trace(
            go.Box(
                y=data[
                    "Estimated Pit Loss (s)"
                ],
                name="Pit-cycle loss",
                boxpoints="all",
                marker_color="#E10600"
            )
        )

    figure.update_layout(
        title="Historical Pit-Cycle Loss",
        yaxis_title="Estimated time loss (s)",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117"
    )

    return figure


def generate_calibration_insights(
    calibration
):

    insights = []

    raw = len(
        calibration["raw_data"]
    )

    clean = len(
        calibration["clean_data"]
    )

    insights.append(
        f"Retained {clean} of {raw} "
        "eligible historical dry-tyre laps."
    )

    insights.append(
        f"Identified "
        f"{len(calibration['neutralised_periods'])} "
        "approximate neutralised periods."
    )

    if not calibration[
        "race_control_available"
    ]:

        insights.append(
            "Warning: race-control data unavailable."
        )

    for compound, model in (
        calibration["models"].items()
    ):

        trend = model["degradation"]

        insights.append(
            f"{compound}: estimated trend "
            f"{trend:+.3f} seconds per lap, "
            f"held-out MAE "
            f"{model['mae']:.3f} seconds."
        )

        if not model["beats_baseline"]:

            insights.append(
                f"{compound} did not outperform "
                "the median baseline."
            )

        if trend < 0:

            insights.append(
                f"{compound} has a negative trend. "
                "This may reflect fuel correction, "
                "track evolution or driver effects."
            )

    pit_result = calibration[
        "pit_result"
    ]

    if pit_result[
        "estimated_pit_loss"
    ] is not None:

        insights.append(
            "Median estimated pit-cycle loss: "
            f"{pit_result['estimated_pit_loss']:.2f} "
            f"seconds from "
            f"{pit_result['samples']} observations."
        )

    insights.append(
        "Historical associations are not "
        "isolated measurements of tyre wear."
    )

    return insights
