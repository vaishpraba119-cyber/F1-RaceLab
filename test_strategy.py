
from pathlib import Path

import numpy as np

from strategy import (
    DEFAULT_TYRES,
    calculate_lap_time,
    simulate_strategy,
    estimate_strategy_time,
    optimise_strategy,
    simulate_undercut_overcut,
    compare_strategies,
    format_race_time,
    plot_strategy_lap_times,
    plot_strategy_timeline,
    plot_strategy_leaderboard,
    plot_undercut_overcut,
    plot_strategy_comparison,
    generate_strategy_insights
)


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
    / "strategy_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TOTAL_LAPS = 57
PIT_LOSS = 22.0
FUEL_EFFECT = 0.03
WARMUP_PENALTY = 0.8
MINIMUM_STINT = 5


print("=" * 65)
print("F1 RACELAB - STRATEGY SIMULATOR")
print("=" * 65)


# Test known mathematical result
print("\nTEST 1: NUMERICAL VALIDATION")

lap_time = calculate_lap_time(
    compound="SOFT",
    tyre_age=10,
    race_lap=1,
    tyre_parameters=DEFAULT_TYRES,
    fuel_effect=0
)

assert np.isclose(
    lap_time,
    96.4
)

print("Expected lap time: 96.400 s")
print("Calculated lap time:", lap_time)
print("Numerical validation PASSED")


# Simulate reference one-stop strategy
print("\nTEST 2: REFERENCE STRATEGY")

reference = simulate_strategy(
    total_laps=TOTAL_LAPS,
    compounds=["SOFT", "MEDIUM"],
    pit_laps=[20],
    tyre_parameters=DEFAULT_TYRES,
    pit_loss=PIT_LOSS,
    fuel_effect=FUEL_EFFECT,
    warmup_penalty=WARMUP_PENALTY,
    minimum_stint=MINIMUM_STINT
)

reference_time = reference[
    "Cumulative Time (s)"
].iloc[-1]

assert len(reference) == TOTAL_LAPS

assert reference["Pit Stop"].sum() == 1

assert np.isclose(
    reference["Pit Loss (s)"].sum(),
    PIT_LOSS
)

assert np.isclose(
    reference["Warmup Penalty (s)"].sum(),
    WARMUP_PENALTY
)

# Validate analytical time calculation
analytical_time = estimate_strategy_time(
    TOTAL_LAPS,
    ["SOFT", "MEDIUM"],
    [20],
    DEFAULT_TYRES,
    PIT_LOSS,
    FUEL_EFFECT,
    WARMUP_PENALTY,
    MINIMUM_STINT
)

assert np.isclose(
    reference_time,
    analytical_time
)

print("Strategy: SOFT - MEDIUM")
print("Pit lap: 20")

print(
    "Estimated race time:",
    format_race_time(reference_time)
)

print("Analytical consistency test PASSED")


# Test invalid strategy rejection
print("\nTEST 3: INVALID STRATEGY VALIDATION")

try:

    simulate_strategy(
        total_laps=57,
        compounds=["SOFT", "MEDIUM"],
        pit_laps=[2]
    )

except ValueError:

    print("Invalid short stint rejected")

else:

    raise AssertionError(
        "Invalid strategy was accepted."
    )


# Optimise one-stop and two-stop strategies
print("\nTEST 4: STRATEGY OPTIMISATION")

results = optimise_strategy(
    total_laps=TOTAL_LAPS,
    tyre_parameters=DEFAULT_TYRES,
    pit_loss=PIT_LOSS,
    fuel_effect=FUEL_EFFECT,
    warmup_penalty=WARMUP_PENALTY,
    minimum_stint=MINIMUM_STINT,
    maximum_stops=2,
    top_n=20
)

best = results["best_strategy"]

best_simulation = results[
    "best_simulation"
]

print(
    "Candidates tested:",
    results["tested_strategies"]
)

print(
    "Feasible strategies:",
    results["feasible_strategies"]
)

print("\nTOP 10 STRATEGIES")

columns = [
    "Rank",
    "Stops",
    "Strategy",
    "Pit Laps",
    "Race Time (s)",
    "Gap to Best (s)"
]

print(
    results["leaderboard"][columns]
    .head(10)
    .round(3)
    .to_string(index=False)
)

assert best[
    "Race Time (s)"
] <= reference_time + 1e-8

assert np.isclose(
    best["Race Time (s)"],
    best_simulation[
        "Cumulative Time (s)"
    ].iloc[-1]
)

print("\nBEST STRATEGY")

print(
    "Compounds:",
    best["Strategy"]
)

print(
    "Pit laps:",
    best["Pit Laps"]
)

print(
    "Race time:",
    format_race_time(
        best["Race Time (s)"]
    )
)

print(
    "Time saved:",
    round(
        reference_time
        - best["Race Time (s)"],
        3
    ),
    "seconds"
)


# Test undercut and overcut simulation
print("\nTEST 5: UNDERCUT AND OVERCUT")

undercut = simulate_undercut_overcut(
    total_laps=TOTAL_LAPS,
    original_compound="SOFT",
    new_compound="MEDIUM",
    reference_pit_lap=20,
    tyre_parameters=DEFAULT_TYRES,
    pit_loss=PIT_LOSS,
    fuel_effect=FUEL_EFFECT,
    warmup_penalty=WARMUP_PENALTY,
    window=5,
    minimum_stint=MINIMUM_STINT
)

print(
    undercut.round(3).to_string(
        index=False
    )
)

reference_row = undercut[
    undercut["Strategy Type"] == "Reference"
].iloc[0]

assert np.isclose(
    reference_row["Post-stop Advantage (s)"],
    0
)

assert np.isclose(
    reference_row["Final Race Advantage (s)"],
    0
)


# Compare reference and optimised strategies
comparison = compare_strategies(
    best_simulation,
    reference,
    "Optimised Strategy",
    "Reference Strategy"
)

assert np.isclose(
    comparison["Advantage (s)"].iloc[-1],
    reference_time - best["Race Time (s)"]
)


print("\nAUTOMATED ENGINEERING INSIGHTS")

for insight in generate_strategy_insights(
    results
):

    print("-", insight)


# Export CSV files
reference.to_csv(
    OUTPUT_DIR / "reference_strategy.csv",
    index=False
)

best_simulation.to_csv(
    OUTPUT_DIR / "optimised_strategy.csv",
    index=False
)

results["leaderboard"].to_csv(
    OUTPUT_DIR / "strategy_leaderboard.csv",
    index=False
)

results["all_strategies"].to_csv(
    OUTPUT_DIR / "all_strategies.csv",
    index=False
)

undercut.to_csv(
    OUTPUT_DIR / "undercut_overcut.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_DIR / "strategy_comparison.csv",
    index=False
)


# Export interactive visualisations
figures = {
    "reference_lap_times":
        plot_strategy_lap_times(reference),

    "optimised_lap_times":
        plot_strategy_lap_times(best_simulation),

    "strategy_timeline":
        plot_strategy_timeline(best_simulation),

    "strategy_leaderboard":
        plot_strategy_leaderboard(results),

    "undercut_overcut":
        plot_undercut_overcut(undercut),

    "strategy_comparison":
        plot_strategy_comparison(
            comparison,
            "Optimised Strategy",
            "Reference Strategy"
        )
}

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
print("STRATEGY SIMULATOR TEST SUCCESSFUL")
print("=" * 65)

print("\nOutput directory:")
print(OUTPUT_DIR)
