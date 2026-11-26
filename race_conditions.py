import numpy as np
import pandas as pd


DEFAULT_CONDITIONS = {
    "safety_car_probability": 0.45,
    "vsc_probability": 0.30,
    "rain_probability": 0.15,
    "safety_car_duration_min": 3,
    "safety_car_duration_max": 6,
    "vsc_duration_min": 1,
    "vsc_duration_max": 3,
    "safety_car_pace_factor": 1.40,
    "vsc_pace_factor": 1.25,
    "safety_car_pit_factor": 0.55,
    "vsc_pit_factor": 0.70,
    "rain_pace_factor": 1.12,
    "rain_duration_min": 5,
    "rain_duration_max": 15,
    "pit_loss_std": 1.5,
    "lap_time_std": 0.35,
    "degradation_uncertainty": 0.25,
    "base_time_std": 0.25
}


def validate_conditions(conditions):

    for key in [
        "safety_car_probability",
        "vsc_probability",
        "rain_probability"
    ]:

        if not 0 <= conditions[key] <= 1:
            raise ValueError(
                f"{key} must be between 0 and 1."
            )

    for prefix in [
        "safety_car",
        "vsc",
        "rain"
    ]:

        minimum = conditions[
            f"{prefix}_duration_min"
        ]

        maximum = conditions[
            f"{prefix}_duration_max"
        ]

        if minimum < 1 or maximum < minimum:
            raise ValueError(
                f"Invalid {prefix} duration."
            )

    for key in [
        "safety_car_pit_factor",
        "vsc_pit_factor"
    ]:

        if not 0 <= conditions[key] <= 1:
            raise ValueError(
                f"Invalid {key}."
            )

    for key in [
        "safety_car_pace_factor",
        "vsc_pace_factor",
        "rain_pace_factor"
    ]:

        if conditions[key] < 1:
            raise ValueError(
                f"{key} must be at least 1."
            )

    for key in [
        "pit_loss_std",
        "lap_time_std",
        "degradation_uncertainty",
        "base_time_std"
    ]:

        if conditions[key] < 0:
            raise ValueError(
                f"{key} cannot be negative."
            )


def generate_race_conditions(
    total_laps,
    number_of_simulations=2000,
    conditions=None,
    seed=42
):

    if total_laps < 2:
        raise ValueError(
            "Race must contain at least two laps."
        )

    if number_of_simulations < 1:
        raise ValueError(
            "At least one simulation is required."
        )

    settings = DEFAULT_CONDITIONS.copy()

    if conditions is not None:
        settings.update(conditions)

    validate_conditions(settings)

    rng = np.random.default_rng(seed)

    shape = (
        number_of_simulations,
        total_laps
    )

    # 0 = green flag
    # 1 = virtual safety car
    # 2 = safety car

    track_status = np.zeros(
        shape,
        dtype=np.int8
    )

    rain = np.zeros(
        shape,
        dtype=bool
    )

    safety_car_events = (
        rng.random(number_of_simulations)
        < settings["safety_car_probability"]
    )

    vsc_events = (
        rng.random(number_of_simulations)
        < settings["vsc_probability"]
    )

    rain_events = (
        rng.random(number_of_simulations)
        < settings["rain_probability"]
    )

    for simulation in range(
        number_of_simulations
    ):

        if vsc_events[simulation]:

            start = int(
                rng.integers(0, total_laps)
            )

            duration = int(
                rng.integers(
                    settings["vsc_duration_min"],
                    settings["vsc_duration_max"] + 1
                )
            )

            end = min(
                start + duration,
                total_laps
            )

            track_status[
                simulation,
                start:end
            ] = 1

        if safety_car_events[simulation]:

            start = int(
                rng.integers(0, total_laps)
            )

            duration = int(
                rng.integers(
                    settings[
                        "safety_car_duration_min"
                    ],
                    settings[
                        "safety_car_duration_max"
                    ] + 1
                )
            )

            end = min(
                start + duration,
                total_laps
            )

            # Safety car takes priority over VSC.

            track_status[
                simulation,
                start:end
            ] = 2

        if rain_events[simulation]:

            start = int(
                rng.integers(0, total_laps)
            )

            duration = int(
                rng.integers(
                    settings["rain_duration_min"],
                    settings["rain_duration_max"] + 1
                )
            )

            end = min(
                start + duration,
                total_laps
            )

            rain[
                simulation,
                start:end
            ] = True

    pace_factor = np.ones(shape)

    pace_factor[track_status == 1] = (
        settings["vsc_pace_factor"]
    )

    pace_factor[track_status == 2] = (
        settings["safety_car_pace_factor"]
    )

    pace_factor[rain] *= (
        settings["rain_pace_factor"]
    )

    pit_factor = np.ones(shape)

    pit_factor[track_status == 1] = (
        settings["vsc_pit_factor"]
    )

    pit_factor[track_status == 2] = (
        settings["safety_car_pit_factor"]
    )

    lap_noise = rng.normal(
        0,
        settings["lap_time_std"],
        size=shape
    )

    pit_noise = rng.normal(
        0,
        settings["pit_loss_std"],
        size=shape
    )

    # Shared compound uncertainty across strategies.

    compounds = [
        "SOFT",
        "MEDIUM",
        "HARD"
    ]

    degradation_factors = {}

    base_time_offsets = {}

    for compound in compounds:

        degradation_factors[compound] = (
            np.maximum(
                0,
                rng.normal(
                    1,
                    settings[
                        "degradation_uncertainty"
                    ],
                    number_of_simulations
                )
            )
        )

        base_time_offsets[compound] = (
            rng.normal(
                0,
                settings["base_time_std"],
                number_of_simulations
            )
        )

    return {
        "total_laps": total_laps,
        "number_of_simulations":
            number_of_simulations,
        "track_status": track_status,
        "rain": rain,
        "pace_factor": pace_factor,
        "pit_factor": pit_factor,
        "lap_noise": lap_noise,
        "pit_noise": pit_noise,
        "degradation_factors":
            degradation_factors,
        "base_time_offsets":
            base_time_offsets,
        "settings": settings,
        "seed": seed
    }


def summarise_race_conditions(scenarios):

    status = scenarios["track_status"]

    rain = scenarios["rain"]

    safety_car_laps = (
        status == 2
    ).sum(axis=1)

    vsc_laps = (
        status == 1
    ).sum(axis=1)

    rain_laps = rain.sum(axis=1)

    return pd.DataFrame({
        "Simulation": np.arange(
            1,
            len(status) + 1
        ),
        "Safety Car": safety_car_laps > 0,
        "VSC": vsc_laps > 0,
        "Rain": rain_laps > 0,
        "Safety Car Laps": safety_car_laps,
        "VSC Laps": vsc_laps,
        "Rain Laps": rain_laps
    })


def get_scenario_statistics(scenarios):

    summary = summarise_race_conditions(
        scenarios
    )

    return {
        "simulations": len(summary),

        "safety_car_percentage": float(
            summary["Safety Car"].mean() * 100
        ),

        "vsc_percentage": float(
            summary["VSC"].mean() * 100
        ),

        "rain_percentage": float(
            summary["Rain"].mean() * 100
        ),

        "average_safety_car_laps": float(
            summary["Safety Car Laps"].mean()
        ),

        "average_vsc_laps": float(
            summary["VSC Laps"].mean()
        ),

        "average_rain_laps": float(
            summary["Rain Laps"].mean()
        )
    }
