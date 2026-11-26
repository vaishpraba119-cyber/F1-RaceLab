import numpy as np
import pandas as pd
import plotly.graph_objects as go

from race_conditions import (
    generate_race_conditions,
    summarise_race_conditions,
    get_scenario_statistics
)

from strategy import (
    DEFAULT_TYRES,
    optimise_strategy,
    format_race_time
)


BACKGROUND = "#0E1117"

COLOURS = [
    "#E10600",
    "#FFD700",
    "#FFFFFF",
    "#00D26A",
    "#168BFF",
    "#FF8700",
    "#A855F7",
    "#00CED1"
]


def simulate_candidate(
    total_laps,
    compounds,
    pit_laps,
    scenarios,
    tyre_parameters=None,
    pit_loss=22.0,
    fuel_effect=0.03,
    warmup_penalty=0.8
):

    if tyre_parameters is None:
        tyre_parameters = DEFAULT_TYRES

    compounds = tuple(compounds)

    pit_laps = tuple(
        int(lap) for lap in pit_laps
    )

    if total_laps != scenarios["total_laps"]:
        raise ValueError(
            "Scenario race distance does not match."
        )

    if len(compounds) != len(pit_laps) + 1:
        raise ValueError(
            "Invalid number of compounds."
        )

    if tuple(sorted(set(pit_laps))) != pit_laps:
        raise ValueError(
            "Pit laps must be unique and ordered."
        )

    if any(
        lap <= 0 or lap >= total_laps
        for lap in pit_laps
    ):
        raise ValueError(
            "Invalid pit lap."
        )

    number_of_simulations = scenarios[
        "number_of_simulations"
    ]

    lap_times = np.zeros(
        (
            number_of_simulations,
            total_laps
        )
    )

    race_laps = np.arange(
        1,
        total_laps + 1
    )

    boundaries = (
        (0,)
        + pit_laps
        + (total_laps,)
    )

    for stint, compound in enumerate(
        compounds
    ):

        if compound not in tyre_parameters:
            raise ValueError(
                f"Unknown compound: {compound}"
            )

        start = boundaries[stint]

        end = boundaries[stint + 1]

        length = end - start

        parameters = tyre_parameters[
            compound
        ]

        if length > parameters["max_age"]:
            raise ValueError(
                f"{compound} stint exceeds "
                "maximum permitted age."
            )

        ages = np.arange(length)

        base = parameters["base_time"]

        degradation = parameters[
            "degradation"
        ]

        quadratic = parameters.get(
            "quadratic",
            0.0
        )

        base_offsets = scenarios[
            "base_time_offsets"
        ][compound][:, None]

        degradation_factors = scenarios[
            "degradation_factors"
        ][compound][:, None]

        stint_times = (
            base
            + base_offsets
            + degradation
            * degradation_factors
            * ages[None, :]
            + quadratic * ages[None, :] ** 2
            - fuel_effect
            * (
                race_laps[start:end][None, :]
                - 1
            )
        )

        lap_times[:, start:end] = (
            stint_times
        )

    # Apply race-condition pace changes.

    lap_times *= scenarios[
        "pace_factor"
    ]

    lap_times += scenarios[
        "lap_noise"
    ]

    # Apply pit-cycle losses.
    # A pit on lap N is recorded at index N-1.

    for pit_lap in pit_laps:

        index = pit_lap - 1

        variable_loss = np.maximum(
            0,
            pit_loss
            + scenarios["pit_noise"][:, index]
        )

        actual_loss = (
            variable_loss
            * scenarios["pit_factor"][:, index]
        )

        lap_times[:, index] += actual_loss

        if pit_lap < total_laps:

            lap_times[:, pit_lap] += (
                warmup_penalty
            )

    if np.any(lap_times <= 0):
        raise ValueError(
            "Simulation generated invalid lap times."
        )

    return {
        "lap_times": lap_times,
        "race_times": lap_times.sum(axis=1),
        "compounds": compounds,
        "pit_laps": pit_laps
    }


def get_candidate_strategies(
    total_laps,
    tyre_parameters=None,
    pit_loss=22.0,
    fuel_effect=0.03,
    warmup_penalty=0.8,
    minimum_stint=5,
    maximum_stops=2,
    number_of_candidates=20
):

    result = optimise_strategy(
        total_laps=total_laps,
        tyre_parameters=tyre_parameters,
        pit_loss=pit_loss,
        fuel_effect=fuel_effect,
        warmup_penalty=warmup_penalty,
        minimum_stint=minimum_stint,
        maximum_stops=maximum_stops,
        top_n=number_of_candidates
    )

    return result["leaderboard"].copy()


def calculate_cvar(
    values,
    confidence=0.95
):

    values = np.sort(
        np.asarray(values, dtype=float)
    )

    tail_size = max(
        1,
        int(
            np.ceil(
                len(values)
                * (1 - confidence)
            )
        )
    )

    return float(
        values[-tail_size:].mean()
    )


def run_monte_carlo(
    total_laps,
    tyre_parameters=None,
    pit_loss=22.0,
    fuel_effect=0.03,
    warmup_penalty=0.8,
    minimum_stint=5,
    maximum_stops=2,
    number_of_candidates=20,
    number_of_simulations=2000,
    conditions=None,
    seed=42,
    risk_weight=0.5
):

    if tyre_parameters is None:
        tyre_parameters = DEFAULT_TYRES

    if number_of_candidates < 1:
        raise ValueError(
            "At least one candidate is required."
        )

    if risk_weight < 0:
        raise ValueError(
            "Risk weight cannot be negative."
        )

    candidates = get_candidate_strategies(
        total_laps=total_laps,
        tyre_parameters=tyre_parameters,
        pit_loss=pit_loss,
        fuel_effect=fuel_effect,
        warmup_penalty=warmup_penalty,
        minimum_stint=minimum_stint,
        maximum_stops=maximum_stops,
        number_of_candidates=number_of_candidates
    )

    if candidates.empty:
        raise ValueError(
            "No feasible candidate strategies."
        )

    scenarios = generate_race_conditions(
        total_laps=total_laps,
        number_of_simulations=
            number_of_simulations,
        conditions=conditions,
        seed=seed
    )

    scenario_summary = (
        summarise_race_conditions(
            scenarios
        )
    )

    labels = []

    results = []

    metadata = []

    for _, candidate in candidates.iterrows():

        compounds = tuple(
            candidate["Compounds"]
        )

        pit_laps = tuple(
            candidate["Pit Lap Values"]
        )

        label = (
            candidate["Strategy"]
            + " | Pit "
            + ", ".join(
                str(lap)
                for lap in pit_laps
            )
        )

        simulation = simulate_candidate(
            total_laps=total_laps,
            compounds=compounds,
            pit_laps=pit_laps,
            scenarios=scenarios,
            tyre_parameters=tyre_parameters,
            pit_loss=pit_loss,
            fuel_effect=fuel_effect,
            warmup_penalty=warmup_penalty
        )

        labels.append(label)

        results.append(
            simulation["race_times"]
        )

        metadata.append({
            "Strategy": label,
            "Compounds": compounds,
            "Pit Laps": pit_laps,
            "Stops": len(pit_laps),
            "Deterministic Time (s)":
                candidate["Race Time (s)"]
        })

    # Rows = simulations
    # Columns = candidate strategies

    race_times = np.column_stack(
        results
    )

    scenario_best = race_times.min(
        axis=1
    )

    regret = (
        race_times
        - scenario_best[:, None]
    )

    # Split probability equally among tied winners.

    winners = np.isclose(
        race_times,
        scenario_best[:, None],
        rtol=0,
        atol=0.000001
    )

    win_shares = (
        winners
        / winners.sum(
            axis=1,
            keepdims=True
        )
    )

    win_probability = (
        win_shares.mean(axis=0)
        * 100
    )

    rows = []

    for index, info in enumerate(
        metadata
    ):

        times = race_times[:, index]

        gaps = regret[:, index]

        mean_time = float(times.mean())

        std_time = float(
            times.std(ddof=1)
            if len(times) > 1 else 0
        )

        mean_regret = float(
            gaps.mean()
        )

        tail_regret = calculate_cvar(
            gaps
        )

        risk_score = (
            mean_regret
            + risk_weight * tail_regret
        )

        rows.append({
            **info,

            "Mean Race Time (s)":
                mean_time,

            "Median Race Time (s)":
                float(np.median(times)),

            "Std Dev (s)":
                std_time,

            "P05 Race Time (s)":
                float(
                    np.percentile(times, 5)
                ),

            "P95 Race Time (s)":
                float(
                    np.percentile(times, 95)
                ),

            "Win Probability (%)":
                float(win_probability[index]),

            "Mean Regret (s)":
                mean_regret,

            "P95 Regret (s)":
                float(
                    np.percentile(gaps, 95)
                ),

            "CVaR95 Regret (s)":
                tail_regret,

            "Within 2s of Best (%)":
                float(
                    (gaps <= 2).mean() * 100
                ),

            "Risk Score":
                risk_score
        })

    leaderboard = pd.DataFrame(rows)

    leaderboard = leaderboard.sort_values(
        [
            "Risk Score",
            "Mean Race Time (s)"
        ]
    ).reset_index(drop=True)

    leaderboard.insert(
        0,
        "Risk Rank",
        np.arange(
            1,
            len(leaderboard) + 1
        )
    )

    race_time_table = pd.DataFrame(
        race_times,
        columns=labels
    )

    regret_table = pd.DataFrame(
        regret,
        columns=labels
    )

    # Evaluate strategies under different conditions.

    condition_rows = []

    masks = {
        "All scenarios":
            np.ones(
                number_of_simulations,
                dtype=bool
            ),

        "Green and dry":
            (
                ~scenario_summary["Safety Car"]
                & ~scenario_summary["VSC"]
                & ~scenario_summary["Rain"]
            ).to_numpy(),

        "Safety car":
            scenario_summary[
                "Safety Car"
            ].to_numpy(),

        "VSC":
            scenario_summary[
                "VSC"
            ].to_numpy(),

        "Rain pace scenario":
            scenario_summary[
                "Rain"
            ].to_numpy()
    }

    for condition, mask in masks.items():

        count = int(mask.sum())

        if count == 0:
            continue

        for index, label in enumerate(
            labels
        ):

            condition_rows.append({
                "Condition": condition,
                "Strategy": label,
                "Scenarios": count,

                "Mean Race Time (s)":
                    float(
                        race_times[
                            mask, index
                        ].mean()
                    ),

                "Mean Regret (s)":
                    float(
                        regret[
                            mask, index
                        ].mean()
                    ),

                "Win Probability (%)":
                    float(
                        win_shares[
                            mask, index
                        ].mean() * 100
                    )
            })

    return {
        "leaderboard": leaderboard,

        "race_times": race_time_table,

        "regret": regret_table,

        "scenario_summary":
            scenario_summary,

        "scenario_statistics":
            get_scenario_statistics(
                scenarios
            ),

        "condition_analysis":
            pd.DataFrame(condition_rows),

        "scenarios": scenarios,

        "best_risk_strategy":
            leaderboard.iloc[0].to_dict(),

        "best_mean_strategy":
            leaderboard.loc[
                leaderboard[
                    "Mean Race Time (s)"
                ].idxmin()
            ].to_dict(),

        "best_probability_strategy":
            leaderboard.loc[
                leaderboard[
                    "Win Probability (%)"
                ].idxmax()
            ].to_dict(),

        "number_of_simulations":
            number_of_simulations,

        "number_of_candidates":
            len(candidates),

        "risk_weight": risk_weight,

        "seed": seed
    }


def style_figure(figure, title):

    figure.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        height=550,
        margin=dict(
            l=50,
            r=40,
            t=70,
            b=50
        )
    )

    return figure


def plot_win_probability(
    result,
    top_n=10
):

    data = result[
        "leaderboard"
    ].nlargest(
        top_n,
        "Win Probability (%)"
    ).sort_values(
        "Win Probability (%)"
    )

    figure = go.Figure(
        go.Bar(
            x=data["Win Probability (%)"],
            y=data["Strategy"],
            orientation="h",
            marker_color="#E10600",
            text=[
                f"{value:.1f}%"
                for value in data[
                    "Win Probability (%)"
                ]
            ],
            textposition="auto"
        )
    )

    figure.update_xaxes(
        title="Probability of fastest strategy (%)"
    )

    return style_figure(
        figure,
        "Strategy Win Probability"
    )


def plot_race_time_distribution(
    result,
    top_n=5
):

    selected = result[
        "leaderboard"
    ].head(top_n)

    figure = go.Figure()

    for index, label in enumerate(
        selected["Strategy"]
    ):

        figure.add_trace(
            go.Histogram(
                x=result["race_times"][label],
                name=label,
                opacity=0.55,
                nbinsx=60,
                marker_color=COLOURS[
                    index % len(COLOURS)
                ]
            )
        )

    figure.update_layout(
        barmode="overlay",
        xaxis_title="Simulated race time (s)",
        yaxis_title="Number of simulations"
    )

    return style_figure(
        figure,
        "Race Time Distributions"
    )


def plot_strategy_regret(
    result,
    top_n=8
):

    selected = result[
        "leaderboard"
    ].head(top_n)

    figure = go.Figure()

    for index, label in enumerate(
        selected["Strategy"]
    ):

        figure.add_trace(
            go.Box(
                y=result["regret"][label],
                name=label,
                boxpoints=False,
                marker_color=COLOURS[
                    index % len(COLOURS)
                ]
            )
        )

    figure.update_yaxes(
        title="Time behind scenario-best strategy (s)"
    )

    return style_figure(
        figure,
        "Strategy Risk and Regret"
    )


def plot_risk_return(result):

    data = result["leaderboard"]

    figure = go.Figure(
        go.Scatter(
            x=data["CVaR95 Regret (s)"],
            y=data["Mean Regret (s)"],
            mode="markers",
            text=data["Strategy"],
            marker=dict(
                size=(
                    10
                    + data["Win Probability (%)"]
                    * 0.5
                ),
                color=data[
                    "Win Probability (%)"
                ],
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(
                    title="Win %"
                ),
                line=dict(
                    color="white",
                    width=0.5
                )
            ),
            hovertemplate=(
                "%{text}<br>"
                "Tail regret: %{x:.3f}s<br>"
                "Mean regret: %{y:.3f}s"
                "<extra></extra>"
            )
        )
    )

    figure.update_layout(
        xaxis_title="Worst 5% average regret (s)",
        yaxis_title="Expected regret (s)"
    )

    return style_figure(
        figure,
        "Strategy Risk-Return Analysis"
    )


def plot_condition_analysis(
    result,
    top_n=5
):

    selected = result[
        "leaderboard"
    ].head(top_n)["Strategy"]

    data = result[
        "condition_analysis"
    ]

    data = data[
        data["Strategy"].isin(selected)
    ]

    figure = go.Figure()

    for index, strategy in enumerate(
        selected
    ):

        subset = data[
            data["Strategy"] == strategy
        ]

        figure.add_trace(
            go.Bar(
                x=subset["Condition"],
                y=subset["Mean Regret (s)"],
                name=strategy,
                marker_color=COLOURS[
                    index % len(COLOURS)
                ]
            )
        )

    figure.update_layout(
        barmode="group",
        yaxis_title="Mean regret (s)"
    )

    return style_figure(
        figure,
        "Performance Under Different Race Conditions"
    )


def plot_scenario_frequency(result):

    statistics = result[
        "scenario_statistics"
    ]

    labels = [
        "Safety Car",
        "Virtual Safety Car",
        "Rain Pace Scenario"
    ]

    values = [
        statistics[
            "safety_car_percentage"
        ],
        statistics[
            "vsc_percentage"
        ],
        statistics[
            "rain_percentage"
        ]
    ]

    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=[
                "#FFD700",
                "#FF8700",
                "#168BFF"
            ],
            text=[
                f"{value:.1f}%"
                for value in values
            ],
            textposition="auto"
        )
    )

    figure.update_yaxes(
        title="Scenarios containing event (%)",
        range=[0, 100]
    )

    return style_figure(
        figure,
        "Simulated Race Conditions"
    )


def generate_monte_carlo_insights(result):

    best_risk = result[
        "best_risk_strategy"
    ]

    best_mean = result[
        "best_mean_strategy"
    ]

    best_probability = result[
        "best_probability_strategy"
    ]

    return [

        f"Evaluated "
        f"{result['number_of_candidates']} "
        f"candidate strategies across "
        f"{result['number_of_simulations']:,} "
        "shared race scenarios.",

        "Lowest-risk strategy: "
        f"{best_risk['Strategy']}.",

        "Lowest expected race time: "
        f"{best_mean['Strategy']} — "
        f"{format_race_time(best_mean['Mean Race Time (s)'])}.",

        "Highest probability of being fastest: "
        f"{best_probability['Strategy']} — "
        f"{best_probability['Win Probability (%)']:.2f}%.",

        "Lowest-risk strategy expected regret: "
        f"{best_risk['Mean Regret (s)']:.3f} seconds.",

        "Lowest-risk strategy average regret "
        "in the worst 5% of scenarios: "
        f"{best_risk['CVaR95 Regret (s)']:.3f} seconds.",

        "Probability of finishing within two "
        "seconds of the scenario-best strategy: "
        f"{best_risk['Within 2s of Best (%)']:.2f}%.",

        "Probabilities are conditional on the "
        "assumed scenario distributions and "
        "selected candidate strategies.",

        "Weather is represented by a simplified "
        "pace penalty, not a wet-tyre strategy model."
    ]
