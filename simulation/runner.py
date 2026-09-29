# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Generic simulation runner.

All simulations are driven by YAML configuration files.
Every parameter must be explicitly specified in the YAML (Tenet 3).

Usage:
    python3 -m simulation.runner sample_configs/sample_run.yaml

YAML schema (all fields required unless marked otherwise):
    data:
      matrices_path: <path to pkl>

    agent:
      measurement_policy:
        state_measurement_decision_algorithm: entropy_threshold | dynamic_programming
        entropy_threshold: <float>       # if algorithm=entropy_threshold
        discount_factor: <float>         # if algorithm=dynamic_programming
        measurement_cost: <float>        # if algorithm=dynamic_programming
        uncertainty_weight: <float>      # if algorithm=dynamic_programming
        control_weight: <float>          # if algorithm=dynamic_programming
      tuning_policy:
        tuning_space: synthetic | data
        is_category_known: bool
        tuning_selection: given_constant | random_constant | convex | argmax
        constant_tuning_selection: <action name>  # if tuning_selection=given_constant
        tuning_cadence: <int>            # apply tuning every N steps; None between. 1 = every step

    environment:
      true_category: <name of the environment's true category>
      alpha:
        type: constant | decaying
        mean: <float>
        decay_rate: <float>              # 0.0 for constant
      number_of_steps: <int>
      number_of_runs: <int>
      initial_state: <state name> | random
      desired_distribution: [<float>, ...]

    costs:
      measurement_cost: <float>
      switching_cost: <float>

    metrics: {name: 0|1, ...}          # absent = off
      time_to_reach: {pattern: <regex with one capture group>}

    visualization:
      state_y_axis_positions: {<state name>: <int>, ...}

    output:
      plot: 0 | 1
      verbose: 0 | 1
"""

from typing import Dict, List, Any

import numpy as np
import pickle
import yaml

from core.configurator import Configurator
from core.metrics import (
    Metrics,
    compute_overall_cost,
    compute_final_cost,
    time_to_reach_regex_pattern,
)

from core.recorder import Recorder
from core.utils import metropolis_hastings_matrix
from environment.environment import Environment
from simulation.simulator import Simulator
from agents.agent import Agent


# ============================================================
# STRICT-LOOKUP HELPER (Tenet 3)
# ============================================================


def _require(config: Dict[str, Any], key: str, section: str = "") -> Any:
    """Return config[key] or raise KeyError with a descriptive message.

    Args:
        config: The config dict to look up in.
        key: The required key.
        section: Optional section name for error messages (e.g., "environment.alpha").

    Raises:
        KeyError: With a descriptive path indicating where the key is missing.
    """
    if key not in config:
        path = f"{section}.{key}" if section else key
        raise KeyError(
            f"Required config key missing: {path!r}. "
            "Per Tenet 3, all parameters must be explicit in the YAML."
        )
    return config[key]


# ============================================================
# INITIAL-STATE RESOLVER (state name OR "random")
# ============================================================


def _resolve_initial_state(
    initial_state_config_value: Any,
    state_names: List[str],
    rng: np.random.RandomState,
) -> int:
    """Convert `environment.initial_state` value to a state index.

    Accepted forms:
      - A state name string (case-sensitive) present in `state_names`
      - The special string `"random"` — samples uniformly across NON-absorbing
        states (all states except the last, which is assumed to be the
        absorbing/terminal state per the pkl convention).

    Per Tenet 3, integer indices are NOT accepted. Any other value raises.

    Args:
        initial_state_config_value: Value read from `environment.initial_state`
        state_names: List of state name strings (from pkl or YAML override)
        rng: numpy RandomState (used for the "random" branch)

    Returns:
        Integer state index.
    """
    if not isinstance(initial_state_config_value, str):
        raise TypeError(
            "environment.initial_state must be a state-name string or 'random', "
            f"got {type(initial_state_config_value).__name__}. Per Tenet 3, "
            "integer indices are not accepted."
        )
    if initial_state_config_value == "random":
        # Sample uniformly across non-absorbing (transient) states.
        # Convention: last state is absorbing (e.g., Death).
        number_of_states = len(state_names)
        if number_of_states < 2:
            raise ValueError(
                "Cannot sample a random initial state: need at least 2 states "
                "(one transient + one absorbing)."
            )
        return int(rng.randint(number_of_states - 1))
    if initial_state_config_value not in state_names:
        raise ValueError(
            f"environment.initial_state={initial_state_config_value!r} is not "
            f"a valid state name. Must be one of {state_names} or 'random'."
        )
    return state_names.index(initial_state_config_value)


# ============================================================
# STATE Y-AXIS POSITIONS RESOLVER
# ============================================================


def _resolve_state_y_axis_positions(
    state_y_axis_positions_map: Dict[str, int],
    state_names: List[str],
) -> List[int]:
    """Validate the visualization.state_y_axis_positions dict and return a list.

    The YAML dict maps state names to their desired y-axis position (0 = bottom).
    Per Tenet 3, the dict must be a bijection: cover ALL state names and use each
    of 0..number_of_states-1 exactly once. Any deviation raises loudly.

    Args:
        state_y_axis_positions_map: Dict from YAML, keyed by state name.
        state_names: List of state names (from pkl).

    Returns:
        List of length `len(state_names)` where entry `i` is the y-axis position
        for state index `i` (i.e., state_names[i]).
    """
    if not isinstance(state_y_axis_positions_map, dict):
        raise TypeError(
            "visualization.state_y_axis_positions must be a dict, got "
            f"{type(state_y_axis_positions_map).__name__}"
        )
    number_of_states = len(state_names)
    missing_state_names = [
        name for name in state_names if name not in state_y_axis_positions_map
    ]
    if missing_state_names:
        raise ValueError(
            "visualization.state_y_axis_positions is missing entries for "
            f"{missing_state_names!r}. Must cover all states: {state_names!r}"
        )
    extra_names_in_map = [
        name for name in state_y_axis_positions_map if name not in state_names
    ]
    if extra_names_in_map:
        raise ValueError(
            "visualization.state_y_axis_positions has entries that are not "
            f"state names: {extra_names_in_map!r}. Valid names: {state_names!r}"
        )
    provided_positions = sorted(state_y_axis_positions_map.values())
    expected_positions = list(range(number_of_states))
    if provided_positions != expected_positions:
        raise ValueError(
            "visualization.state_y_axis_positions values must be a permutation "
            f"of {expected_positions!r}, got {provided_positions!r}"
        )
    return [state_y_axis_positions_map[name] for name in state_names]


# ============================================================
# DATA LOADING (from YAML data section)
# ============================================================


def _build_effective_tuning_matrix(
    baseline_transition_matrix: np.ndarray, additive_tuning_effect: np.ndarray
) -> np.ndarray:
    """Row-stochastic matrix of an additive effect: rownorm(clip(P_0 + T)).

    The last state is absorbing (pkl convention), so its row is forced to e_n.
    """
    effective_tuning_matrix = np.clip(
        baseline_transition_matrix + additive_tuning_effect, 0, 1
    )
    number_of_states = effective_tuning_matrix.shape[0]
    for row_index in range(number_of_states - 1):
        if effective_tuning_matrix[row_index].sum() > 0:
            effective_tuning_matrix[row_index] /= effective_tuning_matrix[
                row_index
            ].sum()
    effective_tuning_matrix[-1] = np.zeros(number_of_states)
    effective_tuning_matrix[-1, -1] = 1.0
    return effective_tuning_matrix


def _load_matrices_from_yaml(matrices_path: str) -> Dict[str, Any]:
    """Load all transition and treatment matrices from the pkl.

    The data tuning basis is per category: B^(c,None) = P_0^(c) and
    B^(c,a) = rownorm(clip(P_0^(c) + T^(c,a))). A (category, action) pair
    without a fit in the pkl is stored as None and raises wherever it is used.

    Returns:
        Dict with 'transition_matrices' (one P_0 per category),
        'data_tuning_basis_by_category' (list over categories of lists over
        actions), 'data_tuning_matrix_names' (action names, 'None' first),
        'category_names', and 'state_names' (from pkl if present, else None).
    """
    with open(matrices_path, "rb") as f:
        results = pickle.load(f)

    # Organize by category
    categories = {}
    for result_entry in results:
        category_name = result_entry["category"]
        if category_name not in categories:
            categories[category_name] = {
                "P_0": result_entry["P_0"],
                "treatments": {},
            }
        if result_entry["action"] != "Unknown":
            categories[category_name]["treatments"][result_entry["action"]] = (
                result_entry["T"]
            )

    category_names = list(categories.keys())
    transition_matrices = [categories[c]["P_0"] for c in category_names]

    # Read state_names from the pkl if present.
    state_names_from_pkl = None
    if results and results[0].get("states"):
        state_names_from_pkl = list(results[0]["states"])

    # Action names: union over categories; 'None' (no treatment) first.
    tuning_action_names = sorted(
        {result_entry["action"] for result_entry in results} - {"Unknown", "None"}
    )
    data_tuning_matrix_names = ["None"] + tuning_action_names

    # Tuning basis per category, built from that category's own P_0 and T.
    data_tuning_basis_by_category = []
    for category_name in category_names:
        baseline_transition_matrix = categories[category_name]["P_0"]
        fitted_tuning_effects = categories[category_name]["treatments"]
        tuning_basis_row = [baseline_transition_matrix.copy()]
        for action_name in tuning_action_names:
            if action_name in fitted_tuning_effects:
                tuning_basis_row.append(
                    _build_effective_tuning_matrix(
                        baseline_transition_matrix, fitted_tuning_effects[action_name]
                    )
                )
            else:
                # No fit for this (category, action): stored as missing, never
                # substituted. Using it raises (core.utils.mix_tuning_basis).
                tuning_basis_row.append(None)
        data_tuning_basis_by_category.append(tuning_basis_row)

    return {
        "transition_matrices": transition_matrices,
        "data_tuning_basis_by_category": data_tuning_basis_by_category,
        "data_tuning_matrix_names": data_tuning_matrix_names,
        "category_names": category_names,
        "state_names": state_names_from_pkl,
    }


def _build_synthetic_tuning_basis(
    desired_distribution: np.ndarray,
    transition_matrices: List[np.ndarray],
    category_names: List[str],
    entropy_threshold: float,
) -> tuple:
    """One Metropolis-Hastings matrix per category (proposal = that category's P_0).

    A synthetic action is a designed matrix, so its effect does not depend on
    the category: every category's basis row holds the same matrices.

    Returns:
        (tuning_basis_by_category, tuning_matrix_names), with one action per
        category, named after it.
    """
    designed_tuning_matrices = [
        metropolis_hastings_matrix(
            desired_distribution, entropy_threshold, initial_matrix=P
        )
        for P in transition_matrices
    ]
    tuning_basis_by_category = [
        [M.copy() for M in designed_tuning_matrices] for _ in category_names
    ]
    return tuning_basis_by_category, list(category_names)


def _resolve_measurement_parameters(
    measurement_policy_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Read the measurement policy. Every field is required (Tenet 3).

    Parameters of the algorithm that is not selected are unused by it and are
    set to 0.0.
    """
    algorithm = _require(
        measurement_policy_config,
        "state_measurement_decision_algorithm",
        "agent.measurement_policy",
    )
    if algorithm == "dynamic_programming":
        return {
            "algorithm": algorithm,
            "entropy_threshold": 0.0,  # unused, but Agent takes it
            "discount_factor": _require(
                measurement_policy_config,
                "discount_factor",
                "agent.measurement_policy",
            ),
            "measurement_cost": _require(
                measurement_policy_config,
                "measurement_cost",
                "agent.measurement_policy",
            ),
            "uncertainty_weight": _require(
                measurement_policy_config,
                "uncertainty_weight",
                "agent.measurement_policy",
            ),
            "control_weight": _require(
                measurement_policy_config,
                "control_weight",
                "agent.measurement_policy",
            ),
        }
    if algorithm == "entropy_threshold":
        return {
            "algorithm": algorithm,
            "entropy_threshold": _require(
                measurement_policy_config,
                "entropy_threshold",
                "agent.measurement_policy",
            ),
            # DP params unused; supply sentinels
            "discount_factor": 0.0,
            "measurement_cost": 0.0,
            "uncertainty_weight": 0.0,
            "control_weight": 0.0,
        }
    raise ValueError(f"Unknown state_measurement_decision_algorithm: {algorithm!r}")


# ============================================================
# CONFIGURATION BUILDER
# ============================================================


def _create_config(
    transition_matrices: List[np.ndarray],
    number_of_steps: int,
    alpha_type: str,
    alpha_mean: float,
    alpha_decay_rate: float,
    desired_distribution: np.ndarray,
    state_names: List[str],
) -> Dict[str, Any]:
    """Create a simulation configuration from transition matrices.

    All parameters required (Tenet 3).
    """
    number_of_states = transition_matrices[0].shape[0]
    number_of_categories = len(transition_matrices)

    # Uniform category prior — agent's Bayesian prior. Note this is the
    # AGENT's prior, not the environment's true category (that's set separately).
    category_prior = np.ones(number_of_categories) / number_of_categories

    configurator = Configurator(
        number_of_states=number_of_states,
        number_of_categories=number_of_categories,
        category_prior=category_prior,
        transition_matrices=transition_matrices,
        alpha_type=alpha_type,
        alpha_mean=alpha_mean,
        alpha_decay_rate=alpha_decay_rate,
        desired_distribution=desired_distribution,
        number_of_steps=number_of_steps,
        state_names=state_names,
    )

    return configurator.get_config()


# ============================================================
# AGENT FACTORY
# ============================================================


def _create_agent(
    agent_config: Dict[str, Any],
    config: Dict[str, Any],
    true_category: int,
    rng: np.random.RandomState,
) -> Agent:
    """Build the parameterized Agent. Every field is required (Tenet 3).

    The tuning basis is read from config['tuning_basis_by_category'], the same
    basis the Environment applies.
    """
    measurement_policy_config = _require(agent_config, "measurement_policy", "agent")
    treatment_policy_config = _require(agent_config, "tuning_policy", "agent")
    measurement_parameters = _resolve_measurement_parameters(measurement_policy_config)

    return Agent(
        config=config,
        state_measurement_decision_algorithm=measurement_parameters["algorithm"],
        tuning_space=_require(
            treatment_policy_config, "tuning_space", "agent.tuning_policy"
        ),
        is_category_known=bool(
            _require(
                treatment_policy_config,
                "is_category_known",
                "agent.tuning_policy",
            )
        ),
        tuning_selection=_require(
            treatment_policy_config,
            "tuning_selection",
            "agent.tuning_policy",
        ),
        constant_tuning_selection=(
            _require(
                treatment_policy_config,
                "constant_tuning_selection",
                "agent.tuning_policy",
            )
            if treatment_policy_config.get("tuning_selection") == "given_constant"
            else None
        ),
        tuning_cadence=_require(
            treatment_policy_config, "tuning_cadence", "agent.tuning_policy"
        ),
        entropy_threshold=measurement_parameters["entropy_threshold"],
        discount_factor=measurement_parameters["discount_factor"],
        measurement_cost=measurement_parameters["measurement_cost"],
        uncertainty_weight=measurement_parameters["uncertainty_weight"],
        control_weight=measurement_parameters["control_weight"],
        true_category=true_category,
        rng=rng,
    )


# ============================================================
# SINGLE EPISODE
# ============================================================


def _run_episode(
    config: Dict[str, Any],
    agent_config: Dict[str, Any],
    number_of_steps: int,
    initial_state: int,
    true_category_index: int,
    measurement_cost: float,
    switching_cost: float,
    recorder: Recorder,
    plot: bool,
    verbose: bool,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    """Run a single simulation episode."""
    environment = Environment(config)
    environment.category.true_category = true_category_index
    environment.state = initial_state

    agent = _create_agent(
        agent_config=agent_config,
        config=config,
        true_category=environment.category.true_category,
        rng=rng,
    )

    simulator = Simulator(
        environment=environment,
        agent=agent,
        recorder=recorder,
        measurement_cost=measurement_cost,
        switching_cost=switching_cost,
    )

    if verbose:
        print(f"Running agent {agent} for {number_of_steps} steps...")

    simulator.run(number_of_steps)

    episode = recorder.episodes[-1]
    metrics = Metrics(episode, config)

    if verbose:
        number_of_states = config["number_of_states"]
        state_names = config["state_names"] or [str(i) for i in range(number_of_states)]
        print(f"  Measurements: {metrics.total_measurements}/{number_of_steps}")
        print(f"  Final state: {environment.state} ({state_names[environment.state]})")
        state_rates = metrics.state_rates
        if state_rates:
            for state_index in range(number_of_states):
                percentage = state_rates.get(state_index, 0) * 100
                print(
                    f"  State {state_index} ({state_names[state_index]}): "
                    f"{percentage:.1f}%"
                )

    if plot:
        simulator.plot_results(output_dir=recorder.output_dir)

    return {
        "simulator": simulator,
        "agent": agent,
        "environment": environment,
        "metrics": metrics,
    }


# ============================================================
# PUBLIC API: YAML-DRIVEN RUNNER
# ============================================================


def run_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Run simulation(s) from a YAML configuration file."""
    top_level_config = _load_yaml(yaml_path)

    data_config = _require(top_level_config, "data")
    agent_config = _require(top_level_config, "agent")
    environment_config = _require(top_level_config, "environment")
    costs_config = _require(top_level_config, "costs")
    metrics_config = _require(top_level_config, "metrics")
    visualization_config = _require(top_level_config, "visualization")
    output_config = _require(top_level_config, "output")

    alpha_config = _require(environment_config, "alpha", "environment")

    # Load matrices from pkl
    matrices_path = _require(data_config, "matrices_path", "data")
    matrix_data = _load_matrices_from_yaml(matrices_path)
    transition_matrices = matrix_data["transition_matrices"]
    category_names = matrix_data["category_names"]

    # True category is an environment property — REQUIRED, no fallback.
    true_category_name = _require(environment_config, "true_category", "environment")
    if true_category_name not in category_names:
        raise ValueError(
            f"environment.true_category={true_category_name!r} not in pkl "
            f"categories: {category_names}"
        )
    true_category_index = category_names.index(true_category_name)

    # Environment scalars
    number_of_steps = _require(environment_config, "number_of_steps", "environment")
    number_of_runs = _require(environment_config, "number_of_runs", "environment")
    initial_state_config_value = _require(
        environment_config, "initial_state", "environment"
    )
    desired_distribution = np.array(
        _require(environment_config, "desired_distribution", "environment")
    )

    # Dimension check: desired_distribution must match the pkl's state count
    n_states_from_pkl = transition_matrices[0].shape[0]
    if len(desired_distribution) != n_states_from_pkl:
        raise ValueError(
            f"environment.desired_distribution has {len(desired_distribution)} "
            f"entries but the pkl matrices are {n_states_from_pkl}x{n_states_from_pkl}. "
            "These must match."
        )

    # state_names: pkl is source of truth; YAML may explicitly override.
    state_names = environment_config.get("state_names") or matrix_data["state_names"]
    if state_names is None:
        raise KeyError(
            "state_names not in pkl and not provided in environment.state_names. "
            "Per Tenet 3, state names must be resolvable."
        )

    # Alpha
    alpha_type = _require(alpha_config, "type", "environment.alpha")
    alpha_mean = _require(alpha_config, "mean", "environment.alpha")
    alpha_decay_rate = _require(alpha_config, "decay_rate", "environment.alpha")

    # Costs
    measurement_cost = _require(costs_config, "measurement_cost", "costs")
    switching_cost = _require(costs_config, "switching_cost", "costs")

    # Output
    verbose = bool(_require(output_config, "verbose", "output"))
    plot = bool(_require(output_config, "plot", "output"))

    # Tuning basis per category — shared by the Environment (true effects)
    # and the Agent (its model of them).
    tuning_policy_config = _require(agent_config, "tuning_policy", "agent")
    tuning_space = _require(tuning_policy_config, "tuning_space", "agent.tuning_policy")
    if tuning_space == "data":
        tuning_basis_by_category = matrix_data["data_tuning_basis_by_category"]
        tuning_matrix_names = matrix_data["data_tuning_matrix_names"]
    elif tuning_space == "synthetic":
        measurement_parameters = _resolve_measurement_parameters(
            _require(agent_config, "measurement_policy", "agent")
        )
        tuning_basis_by_category, tuning_matrix_names = _build_synthetic_tuning_basis(
            desired_distribution=desired_distribution,
            transition_matrices=transition_matrices,
            category_names=category_names,
            entropy_threshold=measurement_parameters["entropy_threshold"],
        )
    else:
        raise ValueError(f"Unknown agent.tuning_policy.tuning_space: {tuning_space!r}")

    # Build simulation config
    config = _create_config(
        transition_matrices=transition_matrices,
        number_of_steps=number_of_steps,
        alpha_type=alpha_type,
        alpha_mean=alpha_mean,
        alpha_decay_rate=alpha_decay_rate,
        desired_distribution=desired_distribution,
        state_names=state_names,
    )
    # Visualization: validate + resolve state_y_axis_positions.
    state_y_axis_positions_map = _require(
        visualization_config, "state_y_axis_positions", "visualization"
    )
    state_y_axis_positions_by_index = _resolve_state_y_axis_positions(
        state_y_axis_positions_map, state_names
    )

    config["category_names"] = category_names
    config["tuning_matrix_names"] = tuning_matrix_names
    config["tuning_basis_by_category"] = tuning_basis_by_category
    config["state_y_axis_positions_by_index"] = state_y_axis_positions_by_index

    if verbose:
        print(f"YAML Config: {yaml_path}")
        print(f"  Measurement policy: {agent_config['measurement_policy']}")
        print(f"  Tuning policy: {agent_config['tuning_policy']}")

        print(f"  Alpha: {alpha_type}, mean={alpha_mean}, decay={alpha_decay_rate}")
        print(f"  Steps: {number_of_steps}, Runs: {number_of_runs}")
        print(f"  Categories: {len(category_names)} ({category_names})")
        print(
            f"  True category (env): {true_category_name} "
            f"(index {true_category_index})"
        )
        print(f"  Data: {matrices_path}")
        print(f"  Treatments: {tuning_matrix_names}")
        print("  Tuning actions fitted per category:")
        for category_index, category_name in enumerate(category_names):
            fitted_tuning_matrix_names = [
                name
                for name, basis_matrix in zip(
                    tuning_matrix_names, tuning_basis_by_category[category_index]
                )
                if basis_matrix is not None
            ]
            print(f"    {category_name}: {fitted_tuning_matrix_names}")
        print(f"  Costs: measurement={measurement_cost}, switching={switching_cost}")

    shared_recorder = Recorder()
    rng = np.random.RandomState()

    all_results = []
    for run_index in range(number_of_runs):
        if verbose and number_of_runs > 1:
            print(f"\nRun {run_index + 1}/{number_of_runs}")

        initial_state_index = _resolve_initial_state(
            initial_state_config_value, state_names, rng
        )

        result = _run_episode(
            config=config,
            agent_config=agent_config,
            number_of_steps=number_of_steps,
            initial_state=initial_state_index,
            true_category_index=true_category_index,
            measurement_cost=measurement_cost,
            switching_cost=switching_cost,
            recorder=shared_recorder,
            plot=(plot and run_index == 0),
            verbose=verbose,
            rng=rng,
        )

        metrics_object = result["metrics"]
        requested_metrics = _extract_requested_metrics(
            metrics_config=metrics_config,
            metrics_object=metrics_object,
            episode=shared_recorder.episodes[-1],
        )
        result["requested_metrics"] = requested_metrics
        all_results.append(result)

        if verbose:
            print(f"  Requested metrics: {requested_metrics}")

    shared_recorder.save_to_csv(shared_recorder.output_dir)
    output_dir = shared_recorder.output_dir

    if verbose:
        print(f"\nAll {number_of_runs} episodes saved to: {output_dir}")
        print(
            f"  Episodes: {shared_recorder.episode_counter} "
            f"({shared_recorder.aggregate_stats['total_steps']} total steps)"
        )

    return {
        "config": top_level_config,
        "runs": all_results,
        "number_of_runs": number_of_runs,
        "output_dir": output_dir,
        "logger": shared_recorder,
    }


# ============================================================
# HELPERS (private)
# ============================================================


def _load_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _extract_requested_metrics(
    metrics_config: Dict,
    metrics_object: Metrics,
    episode: Dict,
) -> Dict[str, Any]:
    """Extract only the metrics requested in YAML config from a Metrics object.

    metrics_config uses 1/0 flags to turn metrics on/off; missing metric keys
    are treated as OFF (0). This is a semantic on/off flag pattern, NOT a
    fake default value (see Tenet 3).
    """
    requested = {}

    if metrics_config.get("measurement_frequency", 0):
        requested["measurement_frequency"] = metrics_object.measurement_frequency
    if metrics_config.get("state_rates", 0):
        requested["state_rates"] = metrics_object.state_rates

    time_to_reach_config = metrics_config.get("time_to_reach", 0)
    if time_to_reach_config:
        if not isinstance(time_to_reach_config, dict):
            raise TypeError(
                "metrics.time_to_reach must be a dict with a 'pattern' entry. "
                "Per Tenet 3, no shorthand form is accepted."
            )
        user_pattern = _require(
            time_to_reach_config, "pattern", "metrics.time_to_reach"
        )
        # Read state history from the episode log (Tenet 1: everything from logs).
        state_history_indices = [step["true_state_before"] for step in episode["steps"]]
        # Get state_names from the shared config threaded through the runner.
        # Note: metrics_object._config holds the same config dict.
        state_names = None
        if metrics_object._config:
            inner_config = metrics_object._config.get("config", metrics_object._config)
            state_names = inner_config.get("state_names")
        if state_names is None:
            raise KeyError(
                "state_names not available for time_to_reach.pattern; the "
                "config must contain it."
            )
        requested["time_to_reach"] = time_to_reach_regex_pattern(
            state_history_indices=state_history_indices,
            state_names=state_names,
            user_pattern=user_pattern,
        )

    if metrics_config.get("tracking_error", 0):
        requested["tracking_error"] = metrics_object.tracking_error
    if metrics_config.get("control_error", 0):
        requested["control_error"] = metrics_object.control_error
    if metrics_config.get("average_entropy", 0):
        requested["average_entropy"] = metrics_object.average_entropy
    if metrics_config.get("desired_distribution_distance", 0):
        requested["desired_distribution_distance"] = (
            metrics_object.desired_distribution_distance
        )
    if metrics_config.get("tuning_matrix_switches", 0):
        requested["tuning_matrix_switches"] = metrics_object.tuning_matrix_switches
    if metrics_config.get("overall_cost", 0):
        step_costs = [step.get("cost", 0.0) for step in episode["steps"]]
        requested["overall_cost"] = compute_overall_cost(step_costs, episode)

    return requested


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        yaml_path = "sample_configs/sample_run.yaml"
    else:
        yaml_path = sys.argv[1]
    run_from_yaml(yaml_path)
