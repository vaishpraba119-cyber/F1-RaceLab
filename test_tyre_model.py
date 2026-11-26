
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    get_meetings,
    get_sessions,
    get_drivers
)

from tyre_model import (
    analyse_tyre_stints,
    predict_tyre_performance,
    plot_tyre_degradation,
    plot_tyre_prediction,
    generate_tyre_insights,
    fit_stint_model
)


YEAR = 2025
GRAND_PRIX = "Australian Grand Prix"
DRIVER_NUMBER = 4
DRIVER_NAME = "Lando Norris"

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "tyre_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 65)
print("F1 RACELAB - TYRE MODEL TEST")
print("=" * 65)


# Test regression using known synthetic data
print("\nTEST 1: NUMERICAL VALIDATION")

ages = np.arange(20)

synthetic = pd.DataFrame({
    "Tyre Age": ages,
    "Lap Time (s)": 90 + 0.15 * ages
})

synthetic_result = fit_stint_model(
    synthetic,
    degree=1
)

assert np.isclose(
    synthetic_result["average_trend"],
    0.15,
    atol=1e-8
)

assert synthetic_result["mae"] < 1e-8

assert synthetic_result["rmse"] < 1e-8

print("Known trend: 0.150 s/lap")

print(
    "Calculated trend:",
    round(
        synthetic_result["average_trend"],
        6
    )
)

print("Synthetic regression test PASSED")


# Retrieve race information
print("\nTEST 2: HISTORICAL DATA ACQUISITION")

meetings = get_meetings(YEAR)

selected = meetings[
    meetings["meeting_name"] == GRAND_PRIX
]

if selected.empty:

    raise ValueError(
        "Grand Prix not found."
    )

meeting_key = int(
    selected.iloc[0]["meeting_key"]
)

sessions = get_sessions(meeting_key)

race = sessions[
    sessions["session_name"] == "Race"
]

if race.empty:

    raise ValueError(
        "Race session unavailable."
    )

session_key = int(
    race.iloc[0]["session_key"]
)

drivers = get_drivers(session_key)

assert DRIVER_NUMBER in (
    drivers["driver_number"].astype(int).tolist()
)

print("Season:", YEAR)
print("Grand Prix:", GRAND_PRIX)
print("Session key:", session_key)
print("Driver:", DRIVER_NAME)


# Run both regression models
for degree in [1, 2]:

    model_name = (
        "linear"
        if degree == 1
        else "quadratic"
    )

    print("\n" + "=" * 65)
    print(model_name.upper(), "REGRESSION")
    print("=" * 65)

    results = analyse_tyre_stints(
        session_key,
        DRIVER_NUMBER,
        degree
    )

    print(
        "\nRaw laps:",
        len(results["raw_data"])
    )

    print(
        "Clean laps:",
        len(results["clean_data"])
    )

    print(
        "Fitted stints:",
        len(results["models"])
    )

    if not results["skipped"].empty:

        print("\nSKIPPED STINTS")

        print(
            results["skipped"].to_string(
                index=False
            )
        )

    if results["summary"].empty:

        print(
            "Insufficient clean laps "
            "for regression."
        )

        continue

    print("\nMODEL SUMMARY")

    print(
        results["summary"]
        .round(3)
        .to_string(index=False)
    )

    print("\nAUTOMATED INSIGHTS")

    for observation in generate_tyre_insights(
        results
    ):

        print("-", observation)

    # Export source data
    results["raw_data"].to_csv(
        OUTPUT_DIR / f"{model_name}_raw_laps.csv",
        index=False
    )

    results["clean_data"].to_csv(
        OUTPUT_DIR / f"{model_name}_clean_laps.csv",
        index=False
    )

    results["summary"].to_csv(
        OUTPUT_DIR / f"{model_name}_summary.csv",
        index=False
    )

    # Export interactive tyre chart
    figure = plot_tyre_degradation(
        results
    )

    figure.write_html(
        str(
            OUTPUT_DIR
            / f"{model_name}_degradation.html"
        ),
        include_plotlyjs=True
    )

    # Validate and export individual stint models
    for stint, model in results["models"].items():

        assert model["mae"] >= 0
        assert model["rmse"] >= 0

        assert (
            model["training_samples"]
            + model["testing_samples"]
            == len(model["actual_data"])
        )

        assert (
            model["max_age"]
            > model["min_age"]
        )

        prediction = predict_tyre_performance(
            model,
            future_laps=5
        )

        assert np.isfinite(
            prediction["Predicted Lap Time (s)"]
        ).all()

        prediction.to_csv(
            OUTPUT_DIR
            / f"{model_name}_stint_{stint}_prediction.csv",
            index=False
        )

        model["validation_data"].to_csv(
            OUTPUT_DIR
            / f"{model_name}_stint_{stint}_validation.csv",
            index=False
        )

        figure = plot_tyre_prediction(
            model,
            future_laps=5
        )

        figure.write_html(
            str(
                OUTPUT_DIR
                / f"{model_name}_stint_{stint}.html"
            ),
            include_plotlyjs=True
        )

    print("\nMODEL CONSISTENCY CHECKS PASSED")


print("\n" + "=" * 65)
print("TYRE MODEL TEST FINISHED")
print("=" * 65)

print("Output directory:")
print(OUTPUT_DIR)
