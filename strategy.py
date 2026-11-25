
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from itertools import product, combinations


COMPOUND_COLORS = {
    "SOFT": "#E10600",
    "MEDIUM": "#FFD700",
    "HARD": "#FFFFFF",
    "INTERMEDIATE": "#39B54A",
    "WET": "#3693FF"
}


# Default illustrative tyre parameters
DEFAULT_TYRES = {
    "SOFT": {
        "base_time": 95.0,
        "degradation": 0.14,
        "quadratic": 0.0,
        "max_age": 25
    },

    "MEDIUM": {
        "base_time": 95.7,
        "degradation": 0.05,
        "quadratic": 0.0,
        "max_age": 40
    },

    "HARD": {
        "base_time": 96.2,
        "degradation": 0.035,
        "quadratic": 0.0,
        "max_age": 50
    }
}


# Calculate estimated lap time
def calculate_lap_time(
    compound,
    tyre_age,
    race_lap,
    tyre_parameters,
    fuel_effect=0.03
):

    if compound not in tyre_parameters:

        raise ValueError(
            f"Unknown compound: {compound}"
        )

    if tyre_age < 0 or race_lap < 1:

        raise ValueError(
            "Invalid tyre age or race lap."
        )

    parameters = tyre_parameters[compound]

    base = parameters["base_time"]

    linear = parameters["degradation"]

    quadratic = parameters.get(
        "quadratic",
        0.0
    )

    lap_time = (
        base
        + linear * tyre_age
        + quadratic * tyre_age ** 2
        - fuel_effect * (race_lap - 1)
    )

    if not np.isfinite(lap_time) or lap_time <= 0:

        raise ValueError(
            "Invalid simulated lap time."
        )

    return float(lap_time)


# Simulate a complete race strategy
def simulate_strategy(
    total_laps,
    compounds,
    pit_laps,
    tyre_parameters=None,
    pit_loss=22.0,
    fuel_effect=0.03,
    warmup_penalty=0.8,
    minimum_stint=5,
    enforce_two_compounds=True
):

    if tyre_parameters is None:

        tyre_parameters = DEFAULT_TYRES

    compounds = list(compounds)
    pit_laps = list(pit_laps)

    if total_laps < 1:

        raise ValueError(
            "Race distance must be positive."
        )

    if len(compounds) != len(pit_laps) + 1:

        raise ValueError(
            "Each stint requires one compound."
        )

    if (
        pit_laps != sorted(set(pit_laps))
        or any(
            lap <= 0 or lap >= total_laps
            for lap in pit_laps
        )
    ):

        raise ValueError(
            "Invalid pit-stop laps."
        )

    if min(
        pit_loss,
        fuel_effect,
        warmup_penalty
    ) < 0:

        raise ValueError(
            "Simulation penalties cannot be negative."
        )

    if (
        enforce_two_compounds
        and len(set(compounds)) < 2
    ):

        raise ValueError(
            "At least two compounds are required."
        )

    boundaries = (
        [0]
        + pit_laps
        + [total_laps]
    )

    rows = []

    cumulative_time = 0.0

    for stint, compound in enumerate(
        compounds,
        start=1
    ):

        start = boundaries[stint - 1]
        end = boundaries[stint]

        stint_length = end - start

        if stint_length < minimum_stint:

            raise ValueError(
                "Stint is shorter than minimum length."
            )

        maximum_age = tyre_parameters[
            compound
        ].get(
            "max_age",
            total_laps
        )

        if stint_length > maximum_age:

            raise ValueError(
                f"{compound} stint exceeds age limit."
            )

        for lap in range(
            start + 1,
            end + 1
        ):

            tyre_age = lap - start - 1

            lap_time = calculate_lap_time(
                compound,
                tyre_age,
                lap,
                tyre_parameters,
                fuel_effect
            )

            # Pit loss assigned to the in-lap
            pit_stop = lap in pit_laps

            pit_penalty = (
                pit_loss
                if pit_stop
                else 0.0
            )

            # Warm-up penalty on first lap
            # after a tyre change
            warmup = (
                warmup_penalty
                if stint > 1 and tyre_age == 0
                else 0.0
            )

            adjusted_time = (
                lap_time
                + pit_penalty
                + warmup
            )

            cumulative_time += adjusted_time

            rows.append({
                "Lap": lap,
                "Stint": stint,
                "Compound": compound,
                "Tyre Age": tyre_age,
                "Base Lap Time (s)": lap_time,
                "Pit Loss (s)": pit_penalty,
                "Warmup Penalty (s)": warmup,
                "Lap Time (s)": adjusted_time,
                "Cumulative Time (s)": cumulative_time,
                "Pit Stop": pit_stop
            })

    return pd.DataFrame(rows)


# Calculate strategy time efficiently
# using the sum of arithmetic and quadratic series
def estimate_strategy_time(
    total_laps,
    compounds,
    pit_laps,
    tyre_parameters,
    pit_loss,
    fuel_effect,
    warmup_penalty,
    minimum_stint
):

    boundaries = (
        [0]
        + list(pit_laps)
        + [total_laps]
    )

    total_time = 0.0

    for i, compound in enumerate(compounds):

        length = (
            boundaries[i + 1]
            - boundaries[i]
        )

        parameters = tyre_parameters[
            compound
        ]

        if (
            length < minimum_stint
            or length > parameters.get(
                "max_age",
                total_laps
            )
        ):

            return np.inf

        sum_age = (
            length * (length - 1) / 2
        )

        sum_age_squared = (
            length
            * (length - 1)
            * (2 * length - 1)
            / 6
        )

        total_time += (
            length * parameters["base_time"]
            + parameters["degradation"] * sum_age
            + parameters.get(
                "quadratic",
                0.0
            ) * sum_age_squared
        )

    # Race-wide fuel effect
    total_time -= (
        fuel_effect
        * total_laps
        * (total_laps - 1)
        / 2
    )

    # Pit-stop and tyre warm-up penalties
    total_time += len(pit_laps) * (
        pit_loss + warmup_penalty
    )

    return float(total_time)


# Search all feasible one-stop and two-stop strategies
def optimise_strategy(
    total_laps,
    tyre_parameters=None,
    pit_loss=22.0,
    fuel_effect=0.03,
    warmup_penalty=0.8,
    minimum_stint=5,
    maximum_stops=2,
    top_n=20
):

    if tyre_parameters is None:

        tyre_parameters = DEFAULT_TYRES

    if maximum_stops not in [1, 2]:

        raise ValueError(
            "Maximum stops must be 1 or 2."
        )

    if (
        total_laps < 2 * minimum_stint
        or minimum_stint < 1
        or top_n < 1
    ):

        raise ValueError(
            "Invalid optimisation settings."
        )

    if min(
        pit_loss,
        fuel_effect,
        warmup_penalty
    ) < 0:

        raise ValueError(
            "Penalties cannot be negative."
        )

    available = [
        compound
        for compound in [
            "SOFT",
            "MEDIUM",
            "HARD"
        ]
        if compound in tyre_parameters
    ]

    if len(available) < 2:

        raise ValueError(
            "At least two dry compounds are required."
        )

    results = []

    tested = 0

    for stops in range(
        1,
        maximum_stops + 1
    ):

        pit_candidates = range(
            minimum_stint,
            total_laps - minimum_stint + 1
        )

        for pit_laps in combinations(
            pit_candidates,
            stops
        ):

            for compounds in product(
                available,
                repeat=stops + 1
            ):

                # Simplified dry-race compound rule
                if len(set(compounds)) < 2:

                    continue

                tested += 1

                race_time = estimate_strategy_time(
                    total_laps,
                    compounds,
                    pit_laps,
                    tyre_parameters,
                    pit_loss,
                    fuel_effect,
                    warmup_penalty,
                    minimum_stint
                )

                if not np.isfinite(race_time):

                    continue

                results.append({
                    "Stops": stops,

                    "Strategy": " - ".join(
                        compounds
                    ),

                    "Pit Laps": ", ".join(
                        map(str, pit_laps)
                    ),

                    "Race Time (s)": race_time,

                    "Compounds": compounds,

                    "Pit Lap Values": pit_laps
                })

    if not results:

        raise ValueError(
            "No feasible strategies found."
        )

    leaderboard = pd.DataFrame(
        results
    ).sort_values(
        "Race Time (s)"
    ).reset_index(drop=True)

    leaderboard["Gap to Best (s)"] = (
        leaderboard["Race Time (s)"]
        - leaderboard["Race Time (s)"].iloc[0]
    )

    leaderboard.insert(
        0,
        "Rank",
        np.arange(
            1,
            len(leaderboard) + 1
        )
    )

    best = leaderboard.iloc[0]

    best_simulation = simulate_strategy(
        total_laps=total_laps,
        compounds=best["Compounds"],
        pit_laps=best["Pit Lap Values"],
        tyre_parameters=tyre_parameters,
        pit_loss=pit_loss,
        fuel_effect=fuel_effect,
        warmup_penalty=warmup_penalty,
        minimum_stint=minimum_stint
    )

    return {
        "leaderboard": leaderboard.head(top_n),
        "all_strategies": leaderboard,
        "best_strategy": best.to_dict(),
        "best_simulation": best_simulation,
        "tested_strategies": tested,
        "feasible_strategies": len(leaderboard)
    }


# Compare earlier and later pit-stop timing
def simulate_undercut_overcut(
    total_laps,
    original_compound,
    new_compound,
    reference_pit_lap,
    tyre_parameters=None,
    pit_loss=22.0,
    fuel_effect=0.03,
    warmup_penalty=0.8,
    window=5,
    minimum_stint=5
):

    if tyre_parameters is None:

        tyre_parameters = DEFAULT_TYRES

    if window < 0:

        raise ValueError(
            "Pit window cannot be negative."
        )

    reference = simulate_strategy(
        total_laps,
        [original_compound, new_compound],
        [reference_pit_lap],
        tyre_parameters,
        pit_loss,
        fuel_effect,
        warmup_penalty,
        minimum_stint
    )

    reference_time = reference[
        "Cumulative Time (s)"
    ].iloc[-1]

    results = []

    for pit_lap in range(
        reference_pit_lap - window,
        reference_pit_lap + window + 1
    ):

        try:

            simulation = simulate_strategy(
                total_laps,
                [original_compound, new_compound],
                [pit_lap],
                tyre_parameters,
                pit_loss,
                fuel_effect,
                warmup_penalty,
                minimum_stint
            )

        except ValueError:

            continue

        race_time = simulation[
            "Cumulative Time (s)"
        ].iloc[-1]

        # Compare after both strategies
        # have completed their pit stops
        comparison_lap = max(
            pit_lap,
            reference_pit_lap
        ) + 1

        candidate_elapsed = simulation.loc[
            simulation["Lap"] == comparison_lap,
            "Cumulative Time (s)"
        ].iloc[0]

        reference_elapsed = reference.loc[
            reference["Lap"] == comparison_lap,
            "Cumulative Time (s)"
        ].iloc[0]

        if pit_lap < reference_pit_lap:

            strategy_type = "Undercut"

        elif pit_lap > reference_pit_lap:

            strategy_type = "Overcut"

        else:

            strategy_type = "Reference"

        results.append({
            "Pit Lap": pit_lap,

            "Strategy Type": strategy_type,

            "Comparison Lap": comparison_lap,

            "Post-stop Advantage (s)": (
                reference_elapsed
                - candidate_elapsed
            ),

            "Race Time (s)": race_time,

            "Final Race Advantage (s)": (
                reference_time
                - race_time
            )
        })

    return pd.DataFrame(results)


# Compare two complete race strategies
def compare_strategies(
    strategy_a,
    strategy_b,
    name_a="Strategy A",
    name_b="Strategy B"
):

    if len(strategy_a) != len(strategy_b):

        raise ValueError(
            "Race distances must match."
        )

    comparison = pd.DataFrame({
        "Lap": strategy_a["Lap"].to_numpy(),

        name_a: strategy_a[
            "Cumulative Time (s)"
        ].to_numpy(),

        name_b: strategy_b[
            "Cumulative Time (s)"
        ].to_numpy()
    })

    comparison["Advantage (s)"] = (
        comparison[name_b]
        - comparison[name_a]
    )

    return comparison


# Format race time for display
def format_race_time(seconds):

    hours = int(seconds // 3600)

    minutes = int(
        seconds % 3600 // 60
    )

    remaining = seconds % 60

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{remaining:06.3f}"
    )


# Plot simulated lap times
def plot_strategy_lap_times(simulation):

    fig = px.scatter(
        simulation,

        x="Lap",
        y="Lap Time (s)",
        color="Compound",

        color_discrete_map=COMPOUND_COLORS,

        hover_data=[
            "Stint",
            "Tyre Age",
            "Pit Loss (s)",
            "Warmup Penalty (s)"
        ]
    )

    for lap in simulation.loc[
        simulation["Pit Stop"],
        "Lap"
    ]:

        fig.add_vline(
            x=lap,
            line_dash="dash",
            line_color="gray"
        )

    fig.update_layout(
        template="plotly_dark",
        title="Simulated race lap times",
        height=500
    )

    return fig


# Plot tyre strategy timeline
def plot_strategy_timeline(simulation):

    fig = go.Figure()

    for stint, data in simulation.groupby(
        "Stint"
    ):

        compound = data["Compound"].iloc[0]

        fig.add_bar(
            x=[len(data)],
            y=["Race strategy"],

            orientation="h",

            name=compound,

            marker_color=COMPOUND_COLORS.get(
                compound,
                "#888888"
            ),

            text=[
                f"{compound} - {len(data)} laps"
            ],

            textposition="inside",

            textfont=dict(
                color="#111111"
                if compound in ["MEDIUM", "HARD"]
                else "#FFFFFF"
            )
        )

    fig.update_layout(
        template="plotly_dark",
        barmode="stack",
        title="Tyre strategy timeline",
        xaxis_title="Race laps",
        height=250
    )

    return fig


# Plot ranked strategy candidates
def plot_strategy_leaderboard(results):

    data = results[
        "leaderboard"
    ].copy()

    data["Label"] = (
        "#"
        + data["Rank"].astype(str)
        + " "
        + data["Strategy"]
        + " | Pit "
        + data["Pit Laps"]
    )

    fig = px.bar(
        data,

        x="Gap to Best (s)",
        y="Label",

        color="Stops",

        orientation="h",

        hover_data=[
            "Race Time (s)",
            "Pit Laps"
        ]
    )

    fig.update_yaxes(
        autorange="reversed"
    )

    fig.update_layout(
        template="plotly_dark",
        title="Optimised race strategies",
        yaxis_title="Strategy",
        height=650
    )

    return fig


# Plot undercut and overcut results
def plot_undercut_overcut(data):

    fig = px.bar(
        data,

        x="Pit Lap",
        y="Post-stop Advantage (s)",

        color="Strategy Type",

        color_discrete_map={
            "Undercut": "#3693FF",
            "Overcut": "#E10600",
            "Reference": "#888888"
        },

        hover_data=[
            "Comparison Lap",
            "Final Race Advantage (s)"
        ]
    )

    fig.add_hline(
        y=0,
        line_color="white",
        line_dash="dot"
    )

    fig.update_layout(
        template="plotly_dark",
        title="Undercut and overcut advantage",
        height=450
    )

    return fig


# Plot cumulative strategy advantage
def plot_strategy_comparison(
    comparison,
    name_a="Strategy A",
    name_b="Strategy B"
):

    fig = go.Figure()

    fig.add_scatter(
        x=comparison["Lap"],

        y=comparison[
            "Advantage (s)"
        ].clip(lower=0),

        name=name_a + " ahead",

        fill="tozeroy",

        line=dict(
            color="#3693FF"
        )
    )

    fig.add_scatter(
        x=comparison["Lap"],

        y=comparison[
            "Advantage (s)"
        ].clip(upper=0),

        name=name_b + " ahead",

        fill="tozeroy",

        line=dict(
            color="#E10600"
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dot",
        line_color="gray"
    )

    fig.update_layout(
        template="plotly_dark",
        title="Cumulative race strategy advantage",
        xaxis_title="Race lap",
        yaxis_title="Advantage (s)",
        height=500
    )

    return fig


# Generate automated strategy observations
def generate_strategy_insights(results):

    best = results["best_strategy"]

    observations = [
        f"Fastest simulated strategy: "
        f"{best['Strategy']}.",

        f"Recommended simulated pit laps: "
        f"{best['Pit Laps']}.",

        f"Estimated race time: "
        f"{format_race_time(best['Race Time (s)'])}.",

        f"Evaluated "
        f"{results['tested_strategies']:,} "
        "candidate strategies.",

        f"Found "
        f"{results['feasible_strategies']:,} "
        "feasible strategies."
    ]

    all_strategies = results[
        "all_strategies"
    ]

    one_stop = all_strategies[
        all_strategies["Stops"] == 1
    ]

    two_stop = all_strategies[
        all_strategies["Stops"] == 2
    ]

    if (
        not one_stop.empty
        and not two_stop.empty
    ):

        difference = (
            one_stop["Race Time (s)"].min()
            - two_stop["Race Time (s)"].min()
        )

        faster = (
            "two-stop"
            if difference > 0
            else "one-stop"
        )

        observations.append(
            f"The {faster} strategy is "
            f"{abs(difference):.3f} seconds faster "
            "under the selected assumptions."
        )

    observations.append(
        "Results are deterministic simulations, "
        "not guaranteed race outcomes."
    )

    observations.append(
        "Traffic, safety cars, tyre availability, "
        "weather changes and driver interactions "
        "are not explicitly simulated."
    )

    return observations
