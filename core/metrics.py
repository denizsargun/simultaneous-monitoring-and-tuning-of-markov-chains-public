# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Unified metrics computation for simulations.

Single source of truth for all metric calculations. All metrics are computed
from logged episode data per Tenet #1.

Structure:
    - Metrics class: stateful, initialized with episode(s), computes all metrics
    - Module-level functions: L2 distances, MLE errors, cost computation
"""

import numpy as np
import pickle
import re
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from typing import Dict, Any, List, Optional, Union

from core.utils import compute_steady_state


# Separator token used to join state names into a single trace string for
# regex matching. State names must not contain this character.
STATE_TRACE_SEPARATOR = ","


def time_to_reach_regex_pattern(
    state_history_indices: List[int],
    state_names: List[str],
    user_pattern: str,
) -> Optional[int]:
    """Return the state-index where a user-provided regex first matches, else None.

    The state trace is concatenated into a string using STATE_TRACE_SEPARATOR
    (with a trailing separator). The user's regex is run against that string.
    The regex must contain EXACTLY ONE capture group; the reported time is the
    STATE INDEX at which that capture group begins.

    Per Tenet 3, no state name may contain the separator character. Any
    regex with != 1 capture groups raises.

    Args:
        state_history_indices: Per-step state indices (list of int).
        state_names: List of state names indexed by state.
        user_pattern: Raw Python regex over the concatenated trace.

    Returns:
        Integer step index of the capture group's start, or None if no match.

    Example:
        state_names = ["Progress", "Disease Control", "Death"]
        trace       = [0, 0, 1, 1, 0, 2]
        trace_str   = "Progress,Progress,Disease Control,Disease Control,Progress,Death,"
        user_pattern= "(?:Progress,)+(?:Disease Control,)+((?:Progress,)|(?:Death,))"
        → capture group begins at char position of the 5th token → state index 4
    """
    for state_name in state_names:
        if STATE_TRACE_SEPARATOR in state_name:
            raise ValueError(
                f"state_name={state_name!r} contains the reserved separator "
                f"{STATE_TRACE_SEPARATOR!r}. Rename or use a different separator."
            )
    compiled_regex = re.compile(user_pattern)
    if compiled_regex.groups != 1:
        raise ValueError(
            "time_to_reach.pattern regex must contain exactly ONE capture group; "
            f"got {compiled_regex.groups} in pattern={user_pattern!r}."
        )
    trace_string = "".join(
        state_names[state_index] + STATE_TRACE_SEPARATOR
        for state_index in state_history_indices
    )
    match = compiled_regex.search(trace_string)
    if match is None:
        return None
    # Convert char position of capture group start → state index by counting
    # separator occurrences in the pre-match portion.
    prefix = trace_string[: match.start(1)]
    return prefix.count(STATE_TRACE_SEPARATOR)


# ============================================================
# METRICS CLASS — computed from logged episodes
# ============================================================


class Metrics:
    """
    Computes all simulation metrics from logged episode data.

    Accepts a single episode or multiple episodes. For single episodes,
    properties return scalar values. For multiple episodes, properties
    return dicts with 'mean' and 'std' keys.

    All data is read from logged .pkl files or in-memory episode dicts
    per Tenet #1 (everything computed from logs).
    """

    def __init__(
        self,
        episodes: Union[str, Path, Dict, List],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Metrics from logged episodes.

        Args:
            episodes: One of:
                - str/Path to a single .pkl file
                - str/Path to a directory containing episode_*.pkl files
                - dict (single in-memory episode)
                - list of any of the above
            config: Config dict (optional). If None, attempts to load from
                    config.pkl in the same directory as the first .pkl file.
        """
        self._episodes = self._load_episodes(episodes)
        self._config = config or self._try_load_config(episodes)
        self._is_multi = len(self._episodes) > 1
        self._per_episode_metrics = [self._compute_single(ep) for ep in self._episodes]

    # ----------------------------------------------------------
    # Episode loading
    # ----------------------------------------------------------

    @staticmethod
    def _load_episodes(source) -> List[Dict]:
        """Normalize any input to a list of episode dicts."""
        if isinstance(source, dict):
            return [source]
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_file() and path.suffix == ".pkl":
                with open(path, "rb") as f:
                    data = pickle.load(f)
                return [data] if isinstance(data, dict) else data
            if path.is_dir():
                episode_files = sorted(path.glob("episode_*.pkl"))
                episodes = []
                for ep_file in episode_files:
                    with open(ep_file, "rb") as f:
                        episodes.append(pickle.load(f))
                return episodes
            raise FileNotFoundError(f"Not a valid file or directory: {path}")
        if isinstance(source, list):
            all_episodes = []
            for item in source:
                all_episodes.extend(Metrics._load_episodes(item))
            return all_episodes
        raise TypeError(f"Unsupported episode source type: {type(source)}")

    @staticmethod
    def _try_load_config(source) -> Optional[Dict]:
        """Try to load config.pkl from the same directory as the source."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_file():
                config_path = path.parent / "config.pkl"
            elif path.is_dir():
                config_path = path / "config.pkl"
            else:
                return None
            if config_path.exists():
                with open(config_path, "rb") as f:
                    return pickle.load(f)
        return None

    # ----------------------------------------------------------
    # Single-episode computation (one pass)
    # ----------------------------------------------------------

    def _compute_single(self, episode: Dict) -> Dict[str, Any]:
        """Compute all metrics from a single episode in one pass through steps."""
        # Per Tenet 3, these are required — no silent fake fallbacks.
        steps = episode["steps"]
        number_of_states = episode["number_of_states"]
        _n_categories = episode["number_of_categories"]  # noqa: reserved for future use

        if not steps:
            return {}

        # Accumulators
        state_counts = np.zeros(number_of_states)
        first_arrival = [None] * number_of_states
        transition_counts = {}
        total_measurements = 0
        entropy_sum = 0.0
        cost_sum = 0.0
        reward_sum = 0.0
        costs_list = []
        tuning_matrix_switches = 0
        switching_decisions = []
        prev_tuning_weights = None

        for step in steps:
            s_before = step["true_state_before"]
            s_after = step["true_state_after"]

            # State visits
            if s_before is not None and 0 <= s_before < number_of_states:
                state_counts[s_before] += 1

            # First arrival
            if s_before is not None and first_arrival[s_before] is None:
                first_arrival[s_before] = step["step"]

            # Transition counts
            if s_before is not None and s_after is not None:
                key = f"{s_before}->{s_after}"
                transition_counts[key] = transition_counts.get(key, 0) + 1

            # Measurements
            if step["measurement_decision"]:
                total_measurements += 1

            # Entropy
            entropy_sum += step.get("entropy", 0.0)

            # Costs
            step_cost = step.get("cost", 0.0)
            cost_sum += step_cost
            costs_list.append(step_cost)

            # Rewards
            reward_sum += step.get("reward", 0.0)

            # Treatment switches: the chosen action (weights) changed.
            # Same rule as the Simulator's switching cost.
            curr_tuning_weights = step["tuning_weights"]
            switched = False
            if prev_tuning_weights is not None:
                if np.linalg.norm(curr_tuning_weights - prev_tuning_weights) > 1e-10:
                    tuning_matrix_switches += 1
                    switched = True
            switching_decisions.append(1 if switched else 0)
            prev_tuning_weights = curr_tuning_weights

        number_of_steps = len(steps)
        total = state_counts.sum()
        state_distribution = state_counts / total if total > 0 else state_counts

        # State rates
        state_rates = {}
        for s in range(number_of_states):
            state_rates[s] = (
                state_counts[s] / number_of_steps if number_of_steps > 0 else 0.0
            )

        # Steps until state
        time_to_reach = {s: first_arrival[s] for s in range(number_of_states)}

        # Desired distribution distance: L2(state_rates, desired_distribution)
        desired_distribution_distance = None
        desired_distribution = None
        if self._config:
            cfg = self._config.get("config", self._config)
            desired_distribution = cfg.get("desired_distribution")
            if desired_distribution is not None:
                desired_distribution = np.array(desired_distribution)

        if desired_distribution is not None:
            actual_distribution = np.array(
                [state_rates.get(s, 0.0) for s in range(number_of_states)]
            )
            desired_distribution_distance = float(
                np.linalg.norm(actual_distribution - desired_distribution)
            )

        # Tracking error (from logs: belief vs true state one-hot)
        tracking_error = None
        beliefs_logged = [
            s
            for s in steps
            if s.get("belief_matrix") is not None
            and s.get("true_category") is not None
            and s.get("true_state_before") is not None
        ]
        if beliefs_logged:
            tracking_errors = []
            for s in beliefs_logged:
                belief = s["belief_matrix"]
                true_cat = s["true_category"]
                true_state = s["true_state_before"]
                if true_cat < belief.shape[0] and true_state < number_of_states:
                    one_hot = np.eye(number_of_states)[true_state]
                    tracking_errors.append(np.linalg.norm(belief[true_cat] - one_hot))
            if tracking_errors:
                tracking_error = float(np.mean(tracking_errors))

        # Control error (from logs: belief vs model steady state)
        control_error = None
        control_steps = [
            s
            for s in steps
            if s.get("belief_matrix") is not None
            and s.get("true_category") is not None
            and s.get("agent_P_model") is not None
        ]
        if control_steps:
            control_errors = []
            for s in control_steps:
                belief = s["belief_matrix"]
                true_cat = s["true_category"]
                P_model = s["agent_P_model"]
                if true_cat < len(P_model) and true_cat < belief.shape[0]:
                    model_ss = compute_steady_state(P_model[true_cat])
                    control_errors.append(
                        float(jensenshannon(belief[true_cat], model_ss))
                    )
            if control_errors:
                control_error = float(np.mean(control_errors))

        # Category correct & confidence
        category_correct = None
        category_confidence = None
        final_belief = episode.get("final_agent_belief")
        true_category = episode.get("true_category")
        if final_belief is not None and true_category is not None:
            # Marginalize to category beliefs
            cat_beliefs = final_belief.sum(axis=1)
            cat_sum = cat_beliefs.sum()
            if cat_sum > 0:
                cat_beliefs = cat_beliefs / cat_sum
            predicted_cat = np.argmax(cat_beliefs)
            category_correct = bool(predicted_cat == true_category)
            category_confidence = float(cat_beliefs[predicted_cat])

        # Product of (1-alpha)
        product_1_minus_alpha = None
        alpha_values = [
            s.get("alpha_value") for s in steps if s.get("alpha_value") is not None
        ]
        if alpha_values:
            product_1_minus_alpha = float(np.prod([1 - a for a in alpha_values]))

        # Cumulative costs
        cumulative_costs = np.cumsum(costs_list).tolist() if costs_list else []

        return {
            "total_steps": number_of_steps,
            "total_observations": total_measurements,
            "total_measurements": total_measurements,
            "measurement_frequency": (
                total_measurements / number_of_steps if number_of_steps > 0 else 0.0
            ),
            "average_entropy": (
                entropy_sum / number_of_steps if number_of_steps > 0 else 0.0
            ),
            "state_rates": state_rates,
            "time_to_reach": time_to_reach,
            "state_distribution": state_distribution.tolist(),
            "state_visits": {
                int(s): int(state_counts[s]) for s in range(number_of_states)
            },
            "transition_counts": transition_counts,
            "tuning_matrix_switches": tuning_matrix_switches,
            "switching_decisions": switching_decisions,
            "desired_distribution_distance": desired_distribution_distance,
            "tracking_error": tracking_error,
            "control_error": control_error,
            "category_correct": category_correct,
            "category_confidence": category_confidence,
            "product_1_minus_alpha": product_1_minus_alpha,
            "total_cost": cost_sum,
            "mean_step_cost": (
                cost_sum / number_of_steps if number_of_steps > 0 else 0.0
            ),
            "cumulative_costs": cumulative_costs,
            "total_reward": reward_sum,
        }

    # ----------------------------------------------------------
    # Properties — dispatch to single or multi
    # ----------------------------------------------------------

    def _get(self, key):
        """Return scalar for single episode, {mean, std} for multi."""
        if self._is_multi:
            values = [
                m.get(key) for m in self._per_episode_metrics if m.get(key) is not None
            ]
            if not values:
                return None
            if isinstance(values[0], (int, float, np.floating, np.integer)):
                return {"mean": float(np.mean(values)), "std": float(np.std(values))}
            if isinstance(values[0], bool):
                return {"mean": float(np.mean(values)), "std": float(np.std(values))}
            return values  # non-numeric: return raw list
        return self._per_episode_metrics[0].get(key)

    @property
    def total_steps(self):
        return self._get("total_steps")

    @property
    def total_observations(self):
        return self._get("total_observations")

    @property
    def total_measurements(self):
        return self._get("total_measurements")

    @property
    def measurement_frequency(self):
        return self._get("measurement_frequency")

    @property
    def average_entropy(self):
        return self._get("average_entropy")

    @property
    def state_rates(self):
        return self._get("state_rates")

    @property
    def time_to_reach(self):
        return self._get("time_to_reach")

    @property
    def state_distribution(self):
        return self._get("state_distribution")

    @property
    def state_visits(self):
        return self._get("state_visits")

    @property
    def transition_counts(self):
        return self._get("transition_counts")

    @property
    def tuning_matrix_switches(self):
        return self._get("tuning_matrix_switches")

    @property
    def switching_decisions(self):
        return self._get("switching_decisions")

    @property
    def desired_distribution_distance(self):
        return self._get("desired_distribution_distance")

    @property
    def tracking_error(self):
        return self._get("tracking_error")

    @property
    def control_error(self):
        return self._get("control_error")

    @property
    def category_correct(self):
        return self._get("category_correct")

    @property
    def category_confidence(self):
        return self._get("category_confidence")

    @property
    def product_1_minus_alpha(self):
        return self._get("product_1_minus_alpha")

    @property
    def total_cost(self):
        return self._get("total_cost")

    @property
    def mean_step_cost(self):
        return self._get("mean_step_cost")

    @property
    def cumulative_costs(self):
        return self._get("cumulative_costs")

    @property
    def total_reward(self):
        return self._get("total_reward")

    @property
    def episodes(self):
        """Access the raw episode data."""
        return self._episodes

    @property
    def config(self):
        """Access the config data."""
        return self._config

    # ----------------------------------------------------------
    # Convenience methods
    # ----------------------------------------------------------

    def compute_all(self) -> Dict[str, Any]:
        """Return all metrics as a flat dictionary."""
        if self._is_multi:
            return {
                prop: self._get(prop)
                for prop in [
                    "total_steps",
                    "total_observations",
                    "total_measurements",
                    "measurement_frequency",
                    "average_entropy",
                    "state_rates",
                    "time_to_reach",
                    "state_distribution",
                    "state_visits",
                    "transition_counts",
                    "tuning_matrix_switches",
                    "desired_distribution_distance",
                    "tracking_error",
                    "control_error",
                    "category_correct",
                    "category_confidence",
                    "product_1_minus_alpha",
                    "total_cost",
                    "mean_step_cost",
                    "total_reward",
                ]
            }
        return self._per_episode_metrics[0].copy()

    def print_metrics(self):
        """Print all metrics in a formatted way."""
        metrics = self.compute_all()
        n_episodes = len(self._episodes)
        print(
            f"\n=== Metrics ({n_episodes} episode{'s' if n_episodes > 1 else ''}) ==="
        )
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, dict) and "mean" in value:
                print(f"  {key}: {value['mean']:.6f} ± {value['std']:.6f}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            elif isinstance(value, (list, np.ndarray)):
                if len(str(value)) < 80:
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: [{len(value)} items]")
            else:
                print(f"  {key}: {value}")

    def __repr__(self):
        n = len(self._episodes)
        return f"Metrics(episodes={n}, multi={self._is_multi})"


# ============================================================
# L2 DISTANCE FUNCTIONS (module-level)
# ============================================================


def l2_distance_matrix(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute Frobenius norm (L2 distance) between two matrices.

    Args:
        A: First matrix
        B: Second matrix

    Returns:
        Frobenius norm ||A - B||_F
    """
    return float(np.linalg.norm(A - B, "fro"))


def l2_distance_steady_states(P: np.ndarray, desired_distribution: np.ndarray) -> float:
    """
    Compute L2 distance between the steady state of P and a desired steady state.

    Args:
        P: Transition matrix
        desired_distribution: Desired steady state distribution

    Returns:
        L2 norm ||steady_state(P) - desired_distribution||
    """
    actual_ss = compute_steady_state(P)
    return float(np.linalg.norm(actual_ss - desired_distribution))


def l2_distances_to_target(
    candidate_matrices: List[np.ndarray], desired_distribution: np.ndarray
) -> List[float]:
    """
    Compute L2 distance from each candidate matrix's steady state to desired steady state.

    Args:
        candidate_matrices: List of transition matrices
        desired_distribution: Desired steady state distribution

    Returns:
        List of L2 distances
    """
    return [
        l2_distance_steady_states(P, desired_distribution) for P in candidate_matrices
    ]


# ============================================================
# MLE ERROR FUNCTIONS (module-level)
# ============================================================


def calculate_mle_error_metrics(
    T_true: np.ndarray, T_estimated: np.ndarray
) -> Dict[str, float]:
    """
    Calculate error metrics for T matrix estimation.

    Args:
        T_true: True additive treatment matrix
        T_estimated: Estimated additive treatment matrix

    Returns:
        dict with MAE, MSE, RMSE, Max_Error, Frobenius_Norm
    """
    diff = T_estimated - T_true
    absolute_error = np.abs(diff)

    return {
        "MAE": float(np.mean(absolute_error)),
        "MSE": float(np.mean(diff**2)),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "Max_Error": float(np.max(absolute_error)),
        "Frobenius_Norm": float(np.linalg.norm(diff, "fro")),
    }


def calculate_element_errors(
    T_true: np.ndarray, T_estimated: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate element-wise error metrics for T matrix estimation.

    Args:
        T_true: True additive treatment matrix
        T_estimated: Estimated additive treatment matrix

    Returns:
        dict with 'absolute_error' and 'relative_error' matrices
    """
    absolute_error = np.abs(T_estimated - T_true)
    n = T_true.shape[0]

    relative_error = np.zeros_like(T_true)
    for i in range(n):
        for j in range(n):
            if abs(T_true[i, j]) > 1e-6:
                relative_error[i, j] = 100 * absolute_error[i, j] / abs(T_true[i, j])
            else:
                relative_error[i, j] = 0 if absolute_error[i, j] < 1e-6 else np.inf

    return {
        "absolute_error": absolute_error,
        "relative_error": relative_error,
    }


# ============================================================
# COST FUNCTIONS (module-level, ALL params required)
# ============================================================


def compute_step_cost(
    measurement_decision: int,
    tuning_switched: bool,
    measurement_cost: float,
    switching_cost: float,
) -> float:
    """
    Compute cost for a single simulation step.

    Cost = measurement_cost (if measured) + switching_cost (if treatment changed).

    Args:
        measurement_decision: 1 if agent measured, 0 otherwise
        tuning_switched: True if agent changed treatment from previous step
        measurement_cost: Cost per observation
        switching_cost: Cost per treatment switch

    Returns:
        Step cost (float)
    """
    cost = 0.0
    if measurement_decision:
        cost += measurement_cost
    if tuning_switched:
        cost += switching_cost
    return cost


def compute_final_cost(episode: Dict[str, Any]) -> float:
    """Terminal (episode-end) cost.

    Placeholder — returns 0.0 for now.
    TODO: implement based on terminal state, residual lifetime, or a
    JS-divergence penalty of belief vs desired steady state.

    Args:
        episode: Episode dict from Recorder.

    Returns:
        Final cost (float).
    """
    return 0.0


def compute_overall_cost(step_costs, episode: Dict[str, Any]) -> float:
    """Overall episode cost = sum of per-step costs + terminal cost.

    Args:
        step_costs: Iterable of per-step costs (from
            core.metrics.compute_step_cost).
        episode: Episode dict from Recorder.

    Returns:
        Overall cost (float).
    """
    return float(sum(step_costs)) + compute_final_cost(episode)


def time_to_reach_auc(episodes, target_state, number_of_steps, config=None):
    """
    Area under the empirical CDF of time-to-reach for a target state.

    Runs across multiple episodes. Censored episodes (never reached)
    are counted as reaching at number_of_steps.

    AUC ∈ [0, number_of_steps]:
      - High AUC → fast arrival (good if target is desired, bad if Death)
      - Low AUC → slow arrival

    Args:
        episodes: list of episode dicts (from Recorder)
        target_state: int, state index
        number_of_steps: simulation horizon
        config: optional config dict for Metrics

    Returns:
        float AUC value
    """
    times = []
    for ep in episodes:
        m = Metrics(ep, config)
        ttr = m.time_to_reach
        t = ttr.get(target_state)
        times.append(t if t is not None and t >= 0 else number_of_steps)

    times = np.sort(times)
    n = len(times)
    # Empirical CDF: F(t) = (number of arrivals <= t) / n
    # AUC = integral of F(t) dt from 0 to number_of_steps
    # = sum over arrivals: (number_of_steps - t_i) / n
    auc = sum(number_of_steps - t for t in times) / n
    return auc
