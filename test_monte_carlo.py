from pathlib import Path

import numpy as np
import pandas as pd

from strategy import (
    DEFAULT_TYRES,
    format_race_time,
    simulate_strategy
)

from race_conditions import (
    DEFAULT_CONDITIONS,
    generate_race_conditions
)

from monte_carlo import (
    simulate_candidate,
    run_monte_carlo,
    plot_win_probability,
    plot_race_time_distribution,
    plot_strategy_regret,
    plot_risk_return,
    plot_condition_analysis,
    plot_scenario_frequency,
    generate_monte_carlo_insights
)


TOTAL_LAPS = 57

NUMBER_OF_SIMULATIONS = 5000

NUMBER_OF_CANDIDATES = 30

SEED = 42

RISK_WEIGHT = 0.5

FUEL_EFFECT = 0.03


print("=" * 70)
print("F1 RACELAB V4.0")
print("MONTE CARLO RACE STRATEGY SIMULATOR")
print("=" * 70)


# Load previous historical calibration.

calibration_folder = Path(
    "outputs/strategy_calibration"
)

parameter_file = (
    calibration_folder
    / "calibrated_parameters.csv"
)

pit_file = (
    calibration_folder
    / "pit_loss_estimates.csv"
)

tyre_parameters = {
    compound: values.copy()
    for compound, values
    in DEFAULT_TYRES.items()
}

pit_loss = 22.0

warmup_penalty = 0.8


if parameter_file.exists():

    parameters = pd.read_csv(
        parameter_file
    )

    for _, row in parameters.iterrows():

        compound = row["Compound"]

        tyre_parameters[compound] = {
            "base_time":
                float(row["base_time"]),

            "degradation":
                float(row["degradation"]),

            "quadratic":
                float(row["quadratic"]),

            "max_age":
                int(row["max_age"])
        }

    print(
        "\nLoaded historical calibration parameters."
    )

    print(
        parameters.to_string(index=False)
    )

else:

    print(
        "\nCalibration file not found. "
        "Using illustrative default parameters."
    )


if pit_file.exists():

    try:

        pit_data = pd.read_csv(pit_file)

        if len(pit_data) >= 3:

            pit_loss = float(
                pit_data[
                    "Estimated Pit Loss (s)"
                ].median()
            )

            warmup_penalty = 0.0

            print(
                "\nLoaded historical pit-cycle loss."
            )

    except (
        pd.errors.EmptyDataError,
        KeyError
    ):

        print(
            "\nPit calibration unavailable. "
            "Using default pit loss."
        )


print(
    f"\nPit-cycle loss: {pit_loss:.3f} s"
)

print(
    f"Warm-up penalty: {warmup_penalty:.3f} s"
)


# Test 1: deterministic consistency.

print(
    "\nTEST 1: Deterministic consistency"
)

zero_uncertainty = (
    DEFAULT_CONDITIONS.copy()
)

zero_uncertainty.update({
    "safety_car_probability": 0.0,
    "vsc_probability": 0.0,
    "rain_probability": 0.0,
    "pit_loss_std": 0.0,
    "lap_time_std": 0.0,
    "degradation_uncertainty": 0.0,
    "base_time_std": 0.0
})

deterministic_scenarios = (
    generate_race_conditions(
        total_laps=TOTAL_LAPS,
        number_of_simulations=10,
        conditions=zero_uncertainty,
        seed=SEED
    )
)

reference_compounds = (
    "SOFT",
    "MEDIUM"
)

reference_pits = (20,)

reference = simulate_strategy(
    total_laps=TOTAL_LAPS,
    compounds=reference_compounds,
    pit_laps=reference_pits,
    tyre_parameters=tyre_parameters,
    pit_loss=pit_loss,
    fuel_effect=FUEL_EFFECT,
    warmup_penalty=warmup_penalty
)

reference_time = float(
    reference["Lap Time (s)"].sum()
)

deterministic = simulate_candidate(
    total_laps=TOTAL_LAPS,
    compounds=reference_compounds,
    pit_laps=reference_pits,
    scenarios=deterministic_scenarios,
    tyre_parameters=tyre_parameters,
    pit_loss=pit_loss,
    fuel_effect=FUEL_EFFECT,
    warmup_penalty=warmup_penalty
)

assert np.allclose(
    deterministic["race_times"],
    reference_time,
    atol=0.001
)

print(
    f"Original strategy: "
    f"{reference_time:.6f} s"
)

print(
    f"Monte Carlo without uncertainty: "
    f"{deterministic['race_times'][0]:.6f} s"
)

print("PASS")


# Test 2: reproducibility.

print(
    "\nTEST 2: Random seed reproducibility"
)

scenario_a = generate_race_conditions(
    total_laps=TOTAL_LAPS,
    number_of_simulations=100,
    seed=SEED
)

scenario_b = generate_race_conditions(
    total_laps=TOTAL_LAPS,
    number_of_simulations=100,
    seed=SEED
)

assert np.array_equal(
    scenario_a["track_status"],
    scenario_b["track_status"]
)

assert np.array_equal(
    scenario_a["lap_noise"],
    scenario_b["lap_noise"]
)

assert np.array_equal(
    scenario_a["rain"],
    scenario_b["rain"]
)

print("PASS")


# Test 3: safety-car pit-loss reduction.

print(
    "\nTEST 3: Safety-car pit-loss reduction"
)

controlled = generate_race_conditions(
    total_laps=TOTAL_LAPS,
    number_of_simulations=1,
    conditions=zero_uncertainty,
    seed=SEED
)

controlled["track_status"][0, 19] = 2

controlled["pit_factor"][0, 19] = 0.55

controlled_result = simulate_candidate(
    total_laps=TOTAL_LAPS,
    compounds=reference_compounds,
    pit_laps=reference_pits,
    scenarios=controlled,
    tyre_parameters=tyre_parameters,
    pit_loss=pit_loss,
    fuel_effect=FUEL_EFFECT,
    warmup_penalty=warmup_penalty
)

expected_saving = (
    pit_loss * (1 - 0.55)
)

actual_saving = (
    reference_time
    - controlled_result["race_times"][0]
)

assert np.isclose(
    expected_saving,
    actual_saving,
    atol=0.001
)

print(
    f"Expected pit saving: "
    f"{expected_saving:.3f} s"
)

print(
    f"Calculated pit saving: "
    f"{actual_saving:.3f} s"
)

print("PASS")


# Run full Monte Carlo analysis.

print("\nRunning full simulation...")

print(
    f"Race laps: {TOTAL_LAPS}"
)

print(
    f"Candidate strategies: "
    f"{NUMBER_OF_CANDIDATES}"
)

print(
    f"Random race scenarios: "
    f"{NUMBER_OF_SIMULATIONS:,}"
)

result = run_monte_carlo(
    total_laps=TOTAL_LAPS,
    tyre_parameters=tyre_parameters,
    pit_loss=pit_loss,
    fuel_effect=FUEL_EFFECT,
    warmup_penalty=warmup_penalty,
    minimum_stint=5,
    maximum_stops=2,
    number_of_candidates=
        NUMBER_OF_CANDIDATES,
    number_of_simulations=
        NUMBER_OF_SIMULATIONS,
    seed=SEED,
    risk_weight=RISK_WEIGHT
)

leaderboard = result[
    "leaderboard"
]

print("\nSIMULATION STATISTICS")

for key, value in result[
    "scenario_statistics"
].items():

    print(f"{key}: {value}")


print("\nTOP 10 RISK-ADJUSTED STRATEGIES")

columns = [
    "Risk Rank",
    "Strategy",
    "Mean Race Time (s)",
    "Win Probability (%)",
    "Mean Regret (s)",
    "CVaR95 Regret (s)",
    "Risk Score"
]

print(
    leaderboard[
        columns
    ].head(10).round(3).to_string(
        index=False
    )
)


print("\nLOWEST-RISK STRATEGY")

best = result[
    "best_risk_strategy"
]

print(best["Strategy"])

print(
    "Expected race time:",
    format_race_time(
        best["Mean Race Time (s)"]
    )
)

print(
    f"Win probability: "
    f"{best['Win Probability (%)']:.2f}%"
)

print(
    f"Expected regret: "
    f"{best['Mean Regret (s)']:.3f} s"
)

print(
    f"Worst 5% average regret: "
    f"{best['CVaR95 Regret (s)']:.3f} s"
)


print("\nENGINEERING INSIGHTS")

for insight in (
    generate_monte_carlo_insights(result)
):

    print("-", insight)


# Final consistency checks.

print("\nTEST 4: Probability consistency")

assert np.isclose(
    leaderboard[
        "Win Probability (%)"
    ].sum(),
    100,
    atol=0.000001
)

print("PASS")


print("\nTEST 5: Simulation dimensions")

assert result["race_times"].shape == (
    NUMBER_OF_SIMULATIONS,
    result["number_of_candidates"]
)

print("PASS")


print("\nTEST 6: Valid race times")

assert np.isfinite(
    result["race_times"].to_numpy()
).all()

assert (
    result["race_times"].to_numpy() > 0
).all()

print("PASS")


print("\nTEST 7: Valid strategy regret")

assert (
    result["regret"].to_numpy()
    >= -0.000001
).all()

assert np.allclose(
    result["regret"].min(axis=1),
    0,
    atol=0.000001
)

print("PASS")


print("\nTEST 8: Valid risk metrics")

assert (
    leaderboard["CVaR95 Regret (s)"]
    >= leaderboard["Mean Regret (s)"]
    - 0.000001
).all()

assert leaderboard[
    "Risk Score"
].is_monotonic_increasing

print("PASS")


# Export all results.

output_folder = Path(
    "outputs/monte_carlo"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

leaderboard.to_csv(
    output_folder / "monte_carlo_leaderboard.csv",
    index=False
)

result["race_times"].to_csv(
    output_folder / "simulated_race_times.csv",
    index=False
)

result["regret"].to_csv(
    output_folder / "strategy_regret.csv",
    index=False
)

result["scenario_summary"].to_csv(
    output_folder / "race_scenarios.csv",
    index=False
)

result["condition_analysis"].to_csv(
    output_folder / "condition_analysis.csv",
    index=False
)

pd.DataFrame([
    result["scenario_statistics"]
]).to_csv(
    output_folder / "scenario_statistics.csv",
    index=False
)


# Generate interactive visualisations.

figures = {
    "win_probability":
        plot_win_probability(result),

    "race_time_distribution":
        plot_race_time_distribution(result),

    "strategy_risk":
        plot_strategy_regret(result),

    "risk_return":
        plot_risk_return(result),

    "condition_analysis":
        plot_condition_analysis(result),

    "scenario_frequency":
        plot_scenario_frequency(result)
}

for name, figure in figures.items():

    figure.write_html(
        str(
            output_folder / f"{name}.html"
        ),
        include_plotlyjs=True
    )


print("\n" + "=" * 70)

print("ALL 8 CONSISTENCY TESTS PASSED")

print(
    f"Total strategy evaluations: "
    f"{NUMBER_OF_SIMULATIONS * result['number_of_candidates']:,}"
)

print("\nFiles exported to:")

print(output_folder.resolve())

print("=" * 70)


figures["win_probability"].show()

figures["strategy_risk"].show()

figures["risk_return"].show()
