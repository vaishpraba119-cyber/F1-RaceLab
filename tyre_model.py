
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_loader import get_laps, get_stints


COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFD700",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#3693FF",
    "UNKNOWN": "#888888"
}


# Retrieve race laps and tyre information
def load_tyre_data(session_key, driver_number):

    laps = get_laps(session_key, driver_number)
    stints = get_stints(session_key, driver_number)

    if laps.empty:
        raise ValueError("No completed laps available.")

    if stints.empty:
        raise ValueError("No tyre stint data available.")

    results = []

    for _, stint in stints.iterrows():

        start = int(stint["lap_start"])
        end = int(stint["lap_end"])

        selected = laps[
            np.logical_and(
                laps["lap_number"] >= start,
                laps["lap_number"] <= end
            )
        ].copy()

        if selected.empty:
            continue

        initial_age = stint.get("tyre_age_at_start", np.nan)

        if pd.isna(initial_age):
            continue

        selected["Compound"] = str(
            stint.get("compound", "UNKNOWN")
        ).upper()

        selected["Stint"] = int(
            stint["stint_number"]
        )

        selected["Tyre Age"] = (
            selected["lap_number"]
            - start
            + initial_age
        )

        selected["Stint Lap"] = (
            selected["lap_number"]
            - start
            + 1
        )

        selected["Driver Number"] = driver_number

        selected["Lap Time (s)"] = pd.to_numeric(
            selected["lap_duration"],
            errors="coerce"
        )

        selected["Pit In Candidate"] = (
            selected["lap_number"] == end
        )

        results.append(selected)

    if not results:
        raise ValueError("No laps matched tyre stints.")

    return pd.concat(
        results,
        ignore_index=True
    ).sort_values(
        "lap_number"
    ).reset_index(drop=True)


# Remove unsuitable laps before regression
def clean_tyre_data(data, outlier_threshold=3.5):

    data = data.copy()

    data = data.dropna(
        subset=[
            "Lap Time (s)",
            "Tyre Age",
            "Compound"
        ]
    )

    data = data[
        data["Lap Time (s)"] > 0
    ]

    # Remove opening race lap
    data = data[
        data["lap_number"] > 1
    ]

    # Remove recorded pit-out laps
    if "is_pit_out_lap" in data:

        data = data[
            data["is_pit_out_lap"] != True
        ]

    # Remove final lap of each stint conservatively
    if "Pit In Candidate" in data:

        data = data[
            data["Pit In Candidate"] == False
        ]

    cleaned = []

    # Detect unusual lap times separately per stint
    for _, group in data.groupby("Stint"):

        group = group.copy()

        median = group["Lap Time (s)"].median()

        mad = np.median(
            np.abs(
                group["Lap Time (s)"] - median
            )
        )

        if mad > 0:

            modified_z = (
                0.6745
                * (
                    group["Lap Time (s)"]
                    - median
                )
                / mad
            )

            group = group[
                np.abs(modified_z)
                <= outlier_threshold
            ]

        cleaned.append(group)

    if not cleaned:
        return pd.DataFrame(
            columns=data.columns
        )

    return pd.concat(
        cleaned,
        ignore_index=True
    ).sort_values(
        "lap_number"
    ).reset_index(drop=True)


# Create regression model
def create_model(degree=1):

    if degree not in [1, 2]:
        raise ValueError(
            "Only linear and quadratic models are supported."
        )

    return make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression()
    )


# Fit regression model to one tyre stint
def fit_stint_model(data, degree=1):

    data = data.sort_values(
        "Tyre Age"
    ).reset_index(drop=True)

    if len(data) < 8:

        raise ValueError(
            "At least 8 clean laps are required."
        )

    X = data[
        ["Tyre Age"]
    ].to_numpy()

    y = data[
        "Lap Time (s)"
    ].to_numpy()

    # Chronological training and testing split
    split = int(len(data) * 0.75)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    model = create_model(degree)

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = (
        r2_score(y_test, predictions)
        if len(y_test) >= 2
        and np.var(y_test) > 0
        else np.nan
    )

    # Constant lap-time baseline
    baseline_predictions = np.full(
        len(y_test),
        y_train[-1]
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    # Refit final descriptive model using all clean laps
    final_model = create_model(degree)

    final_model.fit(X, y)

    coefficients = final_model.named_steps[
        "linearregression"
    ].coef_

    min_age = float(X.min())
    max_age = float(X.max())

    age_range = np.linspace(
        min_age,
        max_age,
        100
    ).reshape(-1, 1)

    fitted_times = final_model.predict(
        age_range
    )

    average_trend = (
        fitted_times[-1]
        - fitted_times[0]
    ) / (max_age - min_age)

    prediction_data = pd.DataFrame({
        "Tyre Age": age_range.flatten(),
        "Predicted Lap Time (s)": fitted_times
    })

    validation_data = pd.DataFrame({
        "Tyre Age": X_test.flatten(),
        "Actual Lap Time (s)": y_test,
        "Predicted Lap Time (s)": predictions,
        "Baseline Lap Time (s)": baseline_predictions
    })

    return {
        "model": final_model,
        "degree": degree,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "baseline_mae": float(baseline_mae),
        "average_trend": float(average_trend),
        "coefficients": coefficients,
        "min_age": min_age,
        "max_age": max_age,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "prediction_data": prediction_data,
        "validation_data": validation_data,
        "actual_data": data
    }


# Analyse all available tyre stints
def analyse_tyre_stints(
    session_key,
    driver_number,
    degree=1
):

    raw = load_tyre_data(
        session_key,
        driver_number
    )

    clean = clean_tyre_data(raw)

    models = {}
    summary = []
    skipped = []

    for stint_number, group in clean.groupby("Stint"):

        if len(group) < 8:

            skipped.append({
                "Stint": stint_number,
                "Reason": "Fewer than 8 clean laps"
            })

            continue

        result = fit_stint_model(
            group,
            degree
        )

        compound = group["Compound"].iloc[0]

        result["compound"] = compound
        result["stint"] = int(stint_number)

        models[int(stint_number)] = result

        summary.append({
            "Stint": int(stint_number),
            "Compound": compound,
            "Clean Laps": len(group),
            "Minimum Tyre Age": result["min_age"],
            "Maximum Tyre Age": result["max_age"],
            "Lap Time Trend (s/lap)": result["average_trend"],
            "MAE (s)": result["mae"],
            "RMSE (s)": result["rmse"],
            "R2": result["r2"],
            "Baseline MAE (s)": result["baseline_mae"],
            "Beats Baseline": (
                result["mae"]
                < result["baseline_mae"]
            )
        })

    return {
        "raw_data": raw,
        "clean_data": clean,
        "models": models,
        "summary": pd.DataFrame(summary),
        "skipped": pd.DataFrame(skipped)
    }


# Predict lap times at selected tyre ages
def predict_tyre_performance(
    model_result,
    future_laps=5
):

    if not 0 <= future_laps <= 10:

        raise ValueError(
            "Prediction horizon must be 0 to 10 laps."
        )

    ages = np.arange(
        model_result["min_age"],
        model_result["max_age"] + future_laps + 1
    )

    predictions = model_result["model"].predict(
        ages.reshape(-1, 1)
    )

    return pd.DataFrame({
        "Tyre Age": ages,
        "Predicted Lap Time (s)": predictions,
        "Extrapolated": (
            ages > model_result["max_age"]
        )
    })


# Plot actual and fitted stint performance
def plot_tyre_degradation(results):

    fig = go.Figure()

    for stint, result in results["models"].items():

        compound = result["compound"]

        color = COMPOUND_COLORS.get(
            compound,
            "#888888"
        )

        actual = result["actual_data"]
        fitted = result["prediction_data"]

        label = f"Stint {stint} - {compound}"

        fig.add_scatter(
            x=actual["Tyre Age"],
            y=actual["Lap Time (s)"],
            mode="markers",
            name=label + " actual",
            marker=dict(
                color=color,
                size=7
            )
        )

        fig.add_scatter(
            x=fitted["Tyre Age"],
            y=fitted["Predicted Lap Time (s)"],
            mode="lines",
            name=label + " fitted",
            line=dict(
                color=color,
                width=3
            )
        )

    fig.update_layout(
        template="plotly_dark",
        title="Estimated tyre stint performance",
        xaxis_title="Tyre age at lap start",
        yaxis_title="Lap time (s)",
        height=550,
        hovermode="closest",
        legend=dict(
            orientation="h",
            y=-0.2
        )
    )

    return fig


# Plot future tyre performance
def plot_tyre_prediction(
    model_result,
    future_laps=5
):

    data = predict_tyre_performance(
        model_result,
        future_laps
    )

    observed = data[
        data["Extrapolated"] == False
    ]

    future = data[
        data["Extrapolated"] == True
    ]

    actual = model_result["actual_data"]

    color = COMPOUND_COLORS.get(
        model_result["compound"],
        "#888888"
    )

    fig = go.Figure()

    fig.add_scatter(
        x=actual["Tyre Age"],
        y=actual["Lap Time (s)"],
        mode="markers",
        name="Actual laps",
        marker=dict(
            color=color,
            size=7
        )
    )

    fig.add_scatter(
        x=observed["Tyre Age"],
        y=observed["Predicted Lap Time (s)"],
        name="Fitted performance",
        line=dict(
            color=color,
            width=3
        )
    )

    if not future.empty:

        future = pd.concat([
            observed.tail(1),
            future
        ])

        fig.add_scatter(
            x=future["Tyre Age"],
            y=future["Predicted Lap Time (s)"],
            name="Extrapolated prediction",
            line=dict(
                color=color,
                width=3,
                dash="dash"
            )
        )

    fig.add_vline(
        x=model_result["max_age"],
        line_dash="dot",
        line_color="gray"
    )

    fig.update_layout(
        template="plotly_dark",
        title="Future stint performance estimate",
        xaxis_title="Tyre age at lap start",
        yaxis_title="Lap time (s)",
        height=500
    )

    return fig


# Generate statistical model observations
def generate_tyre_insights(results):

    observations = []

    for stint, model in results["models"].items():

        compound = model["compound"]
        trend = model["average_trend"]

        if trend > 0.01:

            interpretation = (
                f"Lap times increased by approximately "
                f"{trend:.3f} seconds per lap."
            )

        elif trend < -0.01:

            interpretation = (
                f"Lap times improved by approximately "
                f"{abs(trend):.3f} seconds per lap."
            )

        else:

            interpretation = (
                "Lap times remained relatively stable."
            )

        observations.append(
            f"Stint {stint} ({compound}): "
            + interpretation
        )

        observations.append(
            f"Stint {stint}: chronological test MAE "
            f"was {model['mae']:.3f} seconds, "
            f"compared with a baseline MAE of "
            f"{model['baseline_mae']:.3f} seconds."
        )

        if model["mae"] >= model["baseline_mae"]:

            observations.append(
                f"Stint {stint}: regression did not "
                "outperform the constant lap-time baseline."
            )

    observations.append(
        "Observed trends include the combined effects "
        "of tyre wear, fuel consumption, traffic, "
        "track evolution and driver behaviour."
    )

    observations.append(
        "Future predictions are extrapolations and "
        "should not be treated as guaranteed lap times."
    )

    return observations
