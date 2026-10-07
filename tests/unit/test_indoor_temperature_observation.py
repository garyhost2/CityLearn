"""The indoor dry-bulb temperature observation must reflect the simulated dynamics.

Regression tests for the stale-observation defect: because ``apply_actions`` runs before
``next_time_step`` (since v2.4.0), ``LSTMDynamicsBuilding.update_indoor_dry_bulb_temperature``
wrote its prediction to the step the action was applied to, while the observation returned by
``step`` is built one step later. The observed value was therefore always the dataset's
uncontrolled series and never responded to the agent's HVAC action.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from citylearn.citylearn import CityLearnEnv


SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "data/datasets/citylearn_challenge_2023_phase_2_local_evaluation/schema.json"
)

OBSERVATION = "indoor_dry_bulb_temperature"

#: ``simulate_dynamics`` only turns on once the LSTM lookback buffer has filled, so the
#: first steps of an episode are not dynamics-controlled and must not be compared.
STEPS = 36


def _is_controlled(building) -> bool:
    return (
        getattr(building, "simulate_dynamics", False)
        and OBSERVATION in building.active_observations
    )


def _actions(env: CityLearnEnv, level: float):
    actions = []

    for building in env.buildings:
        building_actions = []

        for name in building.active_actions:
            if name == "cooling_device":
                building_actions.append(level)
            elif name == "cooling_or_heating_device":
                building_actions.append(-level)
            else:
                building_actions.append(0.0)

        actions.append(building_actions)

    return actions


def _observed_temperatures(level: float) -> np.ndarray:
    env = CityLearnEnv(str(SCHEMA), central_agent=False, random_seed=0)

    try:
        env.reset()
        observed = []

        for _ in range(STEPS):
            observations = env.step(_actions(env, level))[0]

            for building, observation in zip(env.buildings, observations):
                if not _is_controlled(building):
                    continue

                index = list(building.active_observations).index(OBSERVATION)
                observed.append(observation[index])

        return np.asarray(observed, dtype="float64")

    finally:
        env.close()


def test_indoor_temperature_observation_responds_to_cooling_action():
    """A controller must be able to observe the effect of its own HVAC action."""

    without_cooling = _observed_temperatures(0.0)
    with_cooling = _observed_temperatures(1.0)

    assert without_cooling.size > 0, 'no dynamics building exposed the observation'
    assert not np.allclose(without_cooling, with_cooling), (
        'the observed indoor dry-bulb temperature is identical with the cooling device idle '
        'and at full power, so the simulated dynamics are invisible to the agent'
    )
    assert with_cooling.mean() < without_cooling.mean(), (
        'cooling at full power should lower the observed indoor dry-bulb temperature'
    )


def test_indoor_temperature_observation_is_not_the_uncontrolled_series():
    """The observation must not be the dataset's uncontrolled trace."""

    env = CityLearnEnv(str(SCHEMA), central_agent=False, random_seed=0)

    try:
        env.reset()
        matches_uncontrolled = 0
        compared = 0

        for _ in range(STEPS):
            observations = env.step(_actions(env, 1.0))[0]

            for building, observation in zip(env.buildings, observations):
                if not _is_controlled(building):
                    continue

                index = list(building.active_observations).index(OBSERVATION)
                uncontrolled = building.energy_simulation.indoor_dry_bulb_temperature_without_control[
                    building.time_step
                ]
                compared += 1
                matches_uncontrolled += int(np.isclose(observation[index], uncontrolled))

        assert compared > 0
        assert matches_uncontrolled < compared, (
            f'the observation equalled the uncontrolled dataset value on all {compared} '
            'controlled building-steps'
        )

    finally:
        env.close()
