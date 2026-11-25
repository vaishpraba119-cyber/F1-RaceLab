
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_loader import (
    get_drivers,
    get_weather
)

from tyre_model import load_tyre_data


# Numerical machine learning features
NUMERIC_FEATURES = [
    "Tyre Age",
    "Stint Lap",
    "lap_number",
    "Previous Lap Time",
    "Previous Three Lap Average",
    "air_temperature",
    "track_temperature",
    "humidity",
    "rainfall"
]


# Categorical machine learning features
CATEGORICAL_FEATURES = [
    "Compound",
    "Driver Number"
]


FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


TARGET = "Lap Time (s)"


# Retrieve and prepare weather information
def prepare_weather(session_key):

    weather = get_weather(session_key)

    if weather.empty or "date" not in weather:

        return pd.DataFrame()

    weather = weather.copy()

    weather["date"] = pd.to_datetime(
        weather["date"],
        utc=True,
        errors="coerce"
    )

    columns = [
        "date",
        "air_temperature",
        "track_temperature",
        "humidity",
        "rainfall"
    ]

    available = [
        column for column in columns
        if column in weather.columns
    ]

    return weather[available].dropna(
        subset=["date"]
    ).sort_values("date")


# Prepare one driver's race data
def prepare_driver_data(
    session_key,
    driver_number,
    weather
):

    data = load_tyre_data(
        session_key,
        driver_number
    )

    data = data.copy()

    data["date_start"] = pd.to_datetime(
        data["date_start"],
        utc=True,
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "date_start",
            TARGET,
            "Tyre Age"
        ]
    )

    data = data.sort_values(
        "lap_number"
    ).drop_duplicates(
        "lap_number"
    ).reset_index(drop=True)

    # Identify suitable completed laps
    valid = np.logical_and(
        data[TARGET] > 0,
        data["lap_number"] > 1
    )

    if "is_pit_out_lap" in data:

        valid = np.logical_and(
            valid,
            data["is_pit_out_lap"] == False
        )

    if "Pit In Candidate" in data:

        valid = np.logical_and(
            valid,
            data["Pit In Candidate"] == False
        )

    # Previous lap features must not contain
    # information from the target lap
    data["Eligible Lap Time"] = data[
        TARGET
    ].where(valid)

    grouped = data.groupby(
        "Stint"
    )["Eligible Lap Time"]

    data["Previous Lap Time"] = (
        grouped.shift(1)
    )

    data["Previous Three Lap Average"] = (
        grouped.transform(
            lambda values:
            values.shift(1).rolling(
                window=3,
                min_periods=1
            ).mean()
        )
    )

    # Ensure the previous lap was consecutive
    previous_number = data.groupby(
        "Stint"
    )["lap_number"].shift(1)

    consecutive = (
        data["lap_number"]
        - previous_number
        == 1
    )

    data = data[
        np.logical_and(
            valid,
            consecutive
        )
    ].copy()

    data = data.dropna(
        subset=[
            "Previous Lap Time",
            "Previous Three Lap Average"
        ]
    )

    # Attach only weather observations that
    # existed before the target lap started
    if not weather.empty:

        data = pd.merge_asof(
            data.sort_values("date_start"),
            weather.sort_values("date"),
            left_on="date_start",
            right_on="date",
            direction="backward",
            tolerance=pd.Timedelta(minutes=10)
        )

    # Handle missing weather channels
    for column in [
        "air_temperature",
        "track_temperature",
        "humidity",
        "rainfall"
    ]:

        if column not in data:

            data[column] = np.nan

    data["Driver Number"] = str(
        driver_number
    )

    data["Compound"] = (
        data["Compound"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    for column in NUMERIC_FEATURES:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    return data.reset_index(drop=True)


# Construct a multi-driver race dataset
def build_race_dataset(
    session_key,
    driver_numbers=None
):

    drivers = get_drivers(session_key)

    if drivers.empty:

        raise ValueError(
            "No drivers available."
        )

    if driver_numbers is None:

        driver_numbers = (
            drivers["driver_number"]
            .astype(int)
            .tolist()
        )

    try:

        weather = prepare_weather(
            session_key
        )

    except Exception:

        weather = pd.DataFrame()

    datasets = []
    skipped = []

    for number in driver_numbers:

        try:

            data = prepare_driver_data(
                session_key,
                number,
                weather
            )

            if data.empty:

                raise ValueError(
                    "No eligible laps."
                )

            datasets.append(data)

        except Exception as error:

            skipped.append({
                "Driver Number": number,
                "Reason": str(error)
            })

    if not datasets:

        raise ValueError(
            "No usable race data."
        )

    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    combined = combined.sort_values(
        "date_start"
    ).reset_index(drop=True)

    return combined, pd.DataFrame(skipped)


# Create feature preprocessing pipeline
def create_preprocessor():

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])


# Define candidate machine learning algorithms
def create_models():

    estimators = {
        "Linear Regression":
            LinearRegression(),

        "Ridge Regression":
            Ridge(alpha=10.0),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=4,
                random_state=42,
                loss="huber"
            )
    }

    models = {}

    for name, estimator in estimators.items():

        models[name] = Pipeline([
            (
                "preprocessor",
                create_preprocessor()
            ),
            (
                "model",
                estimator
            )
        ])

    return models


# Split data using chronological race timestamps
def chronological_split(data):

    data = data.sort_values(
        "date_start"
    ).reset_index(drop=True)

    if len(data) < 60:

        raise ValueError(
            "At least 60 eligible laps are required."
        )

    timestamps = (
        data["date_start"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(timestamps) < 10:

        raise ValueError(
            "Insufficient unique timestamps."
        )

    train_end = timestamps.iloc[
        int(len(timestamps) * 0.60)
    ]

    validation_end = timestamps.iloc[
        int(len(timestamps) * 0.80)
    ]

    train = data[
        data["date_start"] < train_end
    ].copy()

    validation = data[
        np.logical_and(
            data["date_start"] >= train_end,
            data["date_start"] < validation_end
        )
    ].copy()

    test = data[
        data["date_start"] >= validation_end
    ].copy()

    if min(
        len(train),
        len(validation),
        len(test)
    ) < 10:

        raise ValueError(
            "Insufficient samples in chronological split."
        )

    return train, validation, test


# Calculate model evaluation metrics
def evaluate_predictions(
    actual,
    predicted
):

    return {
        "MAE (s)": float(
            mean_absolute_error(
                actual,
                predicted
            )
        ),

        "RMSE (s)": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted
                )
            )
        ),

        "R2": float(
            r2_score(
                actual,
                predicted
            )
        )
    }


# Train and evaluate candidate models
def train_lap_predictor(data):

    train, validation, test = (
        chronological_split(data)
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    models = create_models()

    validation_results = []

    # Previous-lap prediction baseline
    baseline_validation = (
        validation["Previous Lap Time"]
        .to_numpy()
    )

    baseline_metrics = evaluate_predictions(
        y_validation,
        baseline_validation
    )

    validation_results.append({
        "Model": "Previous Lap Baseline",
        **baseline_metrics
    })

    # Compare models using validation data
    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predicted = model.predict(
            X_validation
        )

        metrics = evaluate_predictions(
            y_validation,
            predicted
        )

        validation_results.append({
            "Model": name,
            **metrics
        })

    leaderboard = pd.DataFrame(
        validation_results
    ).sort_values(
        "MAE (s)"
    ).reset_index(drop=True)

    # Select model before examining test results
    best_name = leaderboard.iloc[0]["Model"]

    combined = pd.concat(
        [train, validation],
        ignore_index=True
    )

    best_model = None

    if best_name == "Previous Lap Baseline":

        test_predictions = (
            test["Previous Lap Time"]
            .to_numpy()
        )

    else:

        best_model = create_models()[
            best_name
        ]

        best_model.fit(
            combined[FEATURES],
            combined[TARGET]
        )

        test_predictions = best_model.predict(
            X_test
        )

    # Independent chronological test evaluation
    test_metrics = evaluate_predictions(
        y_test,
        test_predictions
    )

    baseline_test = evaluate_predictions(
        y_test,
        test["Previous Lap Time"]
    )

    predictions = test[[
        "date_start",
        "Driver Number",
        "lap_number",
        "Compound",
        "Tyre Age"
    ]].copy()

    predictions["Actual Lap Time (s)"] = (
        y_test.to_numpy()
    )

    predictions["Predicted Lap Time (s)"] = (
        test_predictions
    )

    predictions["Baseline Lap Time (s)"] = (
        test["Previous Lap Time"].to_numpy()
    )

    predictions["Prediction Error (s)"] = (
        predictions["Predicted Lap Time (s)"]
        - predictions["Actual Lap Time (s)"]
    )

    predictions["Absolute Error (s)"] = (
        predictions["Prediction Error (s)"]
        .abs()
    )

    test_summary = pd.DataFrame([
        {
            "Model": best_name,
            **test_metrics
        },
        {
            "Model": "Previous Lap Baseline",
            **baseline_test
        }
    ])

    if best_name == "Previous Lap Baseline":

        test_summary = (
            test_summary.drop_duplicates(
                "Model"
            )
        )

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "validation_leaderboard": leaderboard,
        "test_summary": test_summary,
        "test_predictions": predictions,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test,
        "training_samples": len(train),
        "validation_samples": len(validation),
        "testing_samples": len(test),
        "training_data": train,
        "validation_data": validation,
        "testing_data": test
    }


# Predict a single lap from supplied features
def predict_single_lap(
    results,
    feature_values
):

    row = pd.DataFrame([
        feature_values
    ])

    missing = [
        feature for feature in FEATURES
        if feature not in row.columns
    ]

    if missing:

        raise ValueError(
            f"Missing features: {missing}"
        )

    if results["best_model"] is None:

        return float(
            row["Previous Lap Time"].iloc[0]
        )

    prediction = results[
        "best_model"
    ].predict(
        row[FEATURES]
    )

    return float(prediction[0])


# Plot candidate model comparison
def plot_model_comparison(results):

    data = results[
        "validation_leaderboard"
    ]

    fig = px.bar(
        data,
        x="Model",
        y="MAE (s)",
        color="Model",
        text_auto=".3f"
    )

    fig.update_layout(
        template="plotly_dark",
        title="Validation model comparison",
        xaxis_title="Machine learning model",
        yaxis_title="Mean absolute error (s)",
        showlegend=False,
        height=500
    )

    return fig


# Plot predicted versus actual lap times
def plot_prediction_accuracy(results):

    data = results[
        "test_predictions"
    ]

    fig = px.scatter(
        data,
        x="Actual Lap Time (s)",
        y="Predicted Lap Time (s)",
        color="Driver Number",
        hover_data=[
            "lap_number",
            "Compound",
            "Tyre Age",
            "Absolute Error (s)"
        ]
    )

    minimum = min(
        data["Actual Lap Time (s)"].min(),
        data["Predicted Lap Time (s)"].min()
    )

    maximum = max(
        data["Actual Lap Time (s)"].max(),
        data["Predicted Lap Time (s)"].max()
    )

    fig.add_scatter(
        x=[minimum, maximum],
        y=[minimum, maximum],
        mode="lines",
        name="Perfect prediction",
        line=dict(
            color="white",
            dash="dash"
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Held-out test prediction accuracy",
        height=550
    )

    return fig


# Plot prediction error distribution
def plot_prediction_errors(results):

    data = results[
        "test_predictions"
    ]

    fig = px.histogram(
        data,
        x="Prediction Error (s)",
        nbins=30
    )

    fig.add_vline(
        x=0,
        line_color="white",
        line_dash="dash"
    )

    fig.update_layout(
        template="plotly_dark",
        title="Test prediction error distribution",
        xaxis_title="Predicted minus actual time (s)",
        yaxis_title="Number of laps",
        height=450
    )

    return fig


# Plot lap predictions for a selected driver
def plot_driver_predictions(
    results,
    driver_number
):

    data = results[
        "test_predictions"
    ]

    selected = data[
        data["Driver Number"].astype(str)
        == str(driver_number)
    ].sort_values("lap_number")

    fig = go.Figure()

    for column, color in [
        (
            "Actual Lap Time (s)",
            "#FFFFFF"
        ),
        (
            "Predicted Lap Time (s)",
            "#E10600"
        ),
        (
            "Baseline Lap Time (s)",
            "#3693FF"
        )
    ]:

        fig.add_scatter(
            x=selected["lap_number"],
            y=selected[column],
            name=column,
            mode="lines+markers",
            line=dict(color=color)
        )

    fig.update_layout(
        template="plotly_dark",
        title=f"Driver #{driver_number} - test predictions",
        xaxis_title="Race lap",
        yaxis_title="Lap time (s)",
        height=500
    )

    return fig


# Generate model evaluation insights
def generate_prediction_insights(results):

    model = results["best_model_name"]

    mae = results[
        "test_metrics"
    ]["MAE (s)"]

    baseline = results[
        "baseline_test_metrics"
    ]["MAE (s)"]

    insights = [
        f"Selected model: {model}.",

        f"Held-out test MAE: "
        f"{mae:.3f} seconds.",

        f"Previous-lap baseline MAE: "
        f"{baseline:.3f} seconds."
    ]

    if baseline > 0:

        improvement = (
            (baseline - mae)
            / baseline
            * 100
        )

        if improvement > 0:

            insights.append(
                f"Prediction error decreased by "
                f"{improvement:.2f}% "
                "relative to the baseline."
            )

        elif improvement < 0:

            insights.append(
                f"Prediction error increased by "
                f"{abs(improvement):.2f}% "
                "relative to the baseline."
            )

        else:

            insights.append(
                "The selected model matched "
                "baseline performance."
            )

    insights.append(
        "Evaluation represents one-lap-ahead "
        "predictions within the selected race."
    )

    insights.append(
        "Performance on unseen circuits or "
        "future races has not been established."
    )

    return insights
