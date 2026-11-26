
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    get_meetings,
    get_sessions
)

from lap_predictor import (
    FEATURES,
    build_race_dataset,
    train_lap_predictor,
    predict_single_lap,
    plot_model_comparison,
    plot_prediction_accuracy,
    plot_prediction_errors,
    plot_driver_predictions,
    generate_prediction_insights
)


YEAR = 2025
GRAND_PRIX = "Bahrain Grand Prix"

DRIVER_NUMBERS = [
    4,
    81,
    1,
    16
]


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "machine_learning"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 65)
print("F1 RACELAB - MACHINE LEARNING TEST")
print("=" * 65)


# Locate the selected race
meetings = get_meetings(YEAR)

meeting = meetings[
    meetings["meeting_name"] == GRAND_PRIX
]

if meeting.empty:

    raise ValueError(
        "Grand Prix not found."
    )

meeting_key = int(
    meeting.iloc[0]["meeting_key"]
)

sessions = get_sessions(meeting_key)

race = sessions[
    sessions["session_name"] == "Race"
]

if race.empty:

    raise ValueError(
        "Race session not found."
    )

session_key = int(
    race.iloc[0]["session_key"]
)


print("\nSeason:", YEAR)
print("Grand Prix:", GRAND_PRIX)
print("Session key:", session_key)


# Build the historical dataset
print("\nBUILDING MACHINE LEARNING DATASET")

data, skipped = build_race_dataset(
    session_key,
    DRIVER_NUMBERS
)

print("Eligible laps:", len(data))

print(
    "Drivers:",
    data["Driver Number"].nunique()
)

print(
    "Compounds:",
    data["Compound"].unique().tolist()
)

print("\nINPUT FEATURES")

for feature in FEATURES:

    print("-", feature)

if not skipped.empty:

    print("\nSKIPPED DRIVERS")

    print(
        skipped.to_string(
            index=False
        )
    )


# Train candidate models
print("\nTRAINING MACHINE LEARNING MODELS")

results = train_lap_predictor(data)


print("\nDATASET SPLIT")

print(
    "Training samples:",
    results["training_samples"]
)

print(
    "Validation samples:",
    results["validation_samples"]
)

print(
    "Testing samples:",
    results["testing_samples"]
)


print("\nVALIDATION LEADERBOARD")

print(
    results["validation_leaderboard"]
    .round(4)
    .to_string(index=False)
)


print("\nSELECTED MODEL")

print(
    results["best_model_name"]
)


print("\nINDEPENDENT TEST RESULTS")

print(
    results["test_summary"]
    .round(4)
    .to_string(index=False)
)


print("\nAUTOMATED ENGINEERING INSIGHTS")

for insight in generate_prediction_insights(
    results
):

    print("-", insight)


# Verify chronological separation
train = results["training_data"]
validation = results["validation_data"]
test = results["testing_data"]

assert (
    train["date_start"].max()
    < validation["date_start"].min()
)

assert (
    validation["date_start"].max()
    < test["date_start"].min()
)

assert (
    len(train)
    + len(validation)
    + len(test)
    == len(data)
)

assert np.isfinite(
    results["test_predictions"][
        "Predicted Lap Time (s)"
    ]
).all()

assert results[
    "test_metrics"
]["MAE (s)"] >= 0


# Verify single-lap prediction
sample = test.iloc[0]

feature_values = {
    feature: sample[feature]
    for feature in FEATURES
}

prediction = predict_single_lap(
    results,
    feature_values
)

expected = results[
    "test_predictions"
]["Predicted Lap Time (s)"].iloc[0]

assert np.isclose(
    prediction,
    expected
)


print("\nSINGLE LAP PREDICTION TEST")

print(
    "Driver:",
    sample["Driver Number"]
)

print(
    "Race lap:",
    sample["lap_number"]
)

print(
    "Actual lap time:",
    round(sample["Lap Time (s)"], 3)
)

print(
    "Predicted lap time:",
    round(prediction, 3)
)

print(
    "Absolute error:",
    round(
        abs(
            prediction
            - sample["Lap Time (s)"]
        ),
        3
    )
)


# Export machine learning datasets
data.to_csv(
    OUTPUT_DIR / "race_dataset.csv",
    index=False
)

results["validation_leaderboard"].to_csv(
    OUTPUT_DIR / "model_leaderboard.csv",
    index=False
)

results["test_summary"].to_csv(
    OUTPUT_DIR / "test_summary.csv",
    index=False
)

results["test_predictions"].to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)


# Generate interactive graphs
figures = {
    "model_comparison":
        plot_model_comparison(results),

    "prediction_accuracy":
        plot_prediction_accuracy(results),

    "prediction_errors":
        plot_prediction_errors(results)
}

for number in DRIVER_NUMBERS:

    figures[f"driver_{number}_predictions"] = (
        plot_driver_predictions(
            results,
            number
        )
    )

for name, figure in figures.items():

    filename = (
        OUTPUT_DIR
        / f"{name}.html"
    )

    figure.write_html(
        str(filename),
        include_plotlyjs=True
    )

    print("Saved:", filename.name)


print("\n" + "=" * 65)
print("ALL CONSISTENCY CHECKS PASSED")
print("MACHINE LEARNING TEST SUCCESSFUL")
print("=" * 65)

print("\nOutput directory:")
print(OUTPUT_DIR)
