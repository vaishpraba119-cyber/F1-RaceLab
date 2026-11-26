
from pathlib import Path

from data_loader import (
    get_meetings,
    get_sessions
)

from tyre_model import (
    analyse_tyre_stints,
    plot_tyre_degradation,
    plot_tyre_prediction,
    generate_tyre_insights
)


YEAR = 2025
GRAND_PRIX = "Bahrain Grand Prix"

DRIVERS = {
    4: "Lando Norris",
    81: "Oscar Piastri",
    1: "Max Verstappen",
    16: "Charles Leclerc"
}


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "dry_race_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Find the race session
meetings = get_meetings(YEAR)

meeting = meetings[
    meetings["meeting_name"] == GRAND_PRIX
]

if meeting.empty:
    raise ValueError("Grand Prix not found.")

meeting_key = int(
    meeting.iloc[0]["meeting_key"]
)

sessions = get_sessions(meeting_key)

race = sessions[
    sessions["session_name"] == "Race"
]

if race.empty:
    raise ValueError("Race session not found.")

session_key = int(
    race.iloc[0]["session_key"]
)


print("=" * 65)
print("F1 RACELAB - DRY RACE VALIDATION")
print("=" * 65)

print("Season:", YEAR)
print("Grand Prix:", GRAND_PRIX)
print("Session key:", session_key)


successful_models = 0
total_models = 0


# Analyse each driver
for driver_number, driver_name in DRIVERS.items():

    print("\n" + "=" * 65)
    print(driver_name.upper())
    print("=" * 65)

    for degree in [1, 2]:

        model_name = (
            "linear"
            if degree == 1
            else "quadratic"
        )

        print("\nMODEL:", model_name.upper())

        try:

            results = analyse_tyre_stints(
                session_key,
                driver_number,
                degree
            )

            summary = results["summary"]

            print(
                "Raw laps:",
                len(results["raw_data"])
            )

            print(
                "Clean laps:",
                len(results["clean_data"])
            )

            if summary.empty:

                print("No suitable stints.")

                continue

            print(
                summary.round(3).to_string(
                    index=False
                )
            )

            total_models += len(summary)

            successful_models += int(
                summary["Beats Baseline"].sum()
            )

            prefix = (
                f"{driver_number}_{model_name}"
            )

            summary.to_csv(
                OUTPUT_DIR
                / f"{prefix}_summary.csv",
                index=False
            )

            results["clean_data"].to_csv(
                OUTPUT_DIR
                / f"{prefix}_clean_laps.csv",
                index=False
            )

            figure = plot_tyre_degradation(
                results
            )

            figure.write_html(
                str(
                    OUTPUT_DIR
                    / f"{prefix}_degradation.html"
                ),
                include_plotlyjs=True
            )

            # Only export future predictions
            # for models that beat the baseline
            for stint, model in results["models"].items():

                if model["mae"] >= model["baseline_mae"]:

                    print(
                        f"Stint {stint}: "
                        "prediction rejected - "
                        "failed baseline comparison."
                    )

                    continue

                figure = plot_tyre_prediction(
                    model,
                    future_laps=5
                )

                figure.write_html(
                    str(
                        OUTPUT_DIR
                        / f"{prefix}_stint_{stint}.html"
                    ),
                    include_plotlyjs=True
                )

            print("\nENGINEERING INSIGHTS")

            for observation in generate_tyre_insights(
                results
            ):

                print("-", observation)

        except Exception as error:

            print(
                f"Analysis unavailable: {error}"
            )


print("\n" + "=" * 65)
print("FINAL MODEL VALIDATION")
print("=" * 65)

print("Total fitted models:", total_models)

print(
    "Models beating baseline:",
    successful_models
)

print(
    "Models failing baseline:",
    total_models - successful_models
)

if total_models > 0:

    success_rate = (
        successful_models
        / total_models
        * 100
    )

    print(
        f"Baseline success rate: "
        f"{success_rate:.2f}%"
    )

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nDRY RACE TEST FINISHED")
