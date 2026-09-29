# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Parameterized Agent — composes measurement and treatment policies.

The agent's behavior is fully specified by two orthogonal top-level policies:

1. measurement_policy (when to observe the state):
   - state_measurement_decision_algorithm:
       "entropy_threshold" | "dynamic_programming"

2. tuning_policy (which tuning action to apply):
   - tuning_space: "synthetic" | "data"
   - is_category_known: bool  (True = one-hot on true category)
   - tuning_selection: "random_constant" | "given_constant" | "convex" | "argmax"
   - constant_tuning_selection: str  (required when tuning_selection="given_constant";
     must be one of the entries in tuning_matrix_names)

The agent chooses weights w over tuning actions. The tuning basis is per
category, B^(c,a) (config['tuning_basis_by_category']), so the same action has
a category-specific matrix T^(c) = sum_a w_a B^(c,a). The Environment applies
T^(C) for the true category; the agent's model of category c blends toward T^(c).
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

from agents.belief import Belief
from core.utils import compute_steady_state, mix_tuning_basis


# ============================================================
# TREATMENT SELECTION STRATEGIES
# ============================================================


def _select_random_constant(number_of_basis_tuning_matrices, rng):
    """Pick one basis matrix uniformly at random (at init). Returns one-hot weights + idx."""
    idx = rng.randint(number_of_basis_tuning_matrices)
    weights = np.zeros(number_of_basis_tuning_matrices)
    weights[idx] = 1.0
    return weights, idx


def _select_convex_over_categories(category_beliefs, number_of_basis_tuning_matrices):
    """Weights = normalized category beliefs (only valid when number_of_basis_tuning_matrices == number_of_categories)."""
    weights = np.array(category_beliefs).copy()
    total = weights.sum()
    if total > 0:
        weights = weights / total
    else:
        weights = (
            np.ones(number_of_basis_tuning_matrices) / number_of_basis_tuning_matrices
        )
    return weights


def _select_argmax_by_js_to_desired(
    tuning_basis_by_category, category_beliefs, modeled_categories, desired_distribution
):
    """Pick the action whose steady state is closest to π* (JS divergence).

    Each action's score is averaged over the modeled categories, weighted by
    the category belief: score_a = sum_c π_C(c) JS(ss(B^(c,a)), π*).
    """
    number_of_basis_tuning_matrices = len(tuning_basis_by_category[0])
    belief_mass = sum(category_beliefs[c] for c in modeled_categories)
    scores = np.zeros(number_of_basis_tuning_matrices)
    for i in range(number_of_basis_tuning_matrices):
        for c in modeled_categories:
            category_weight = category_beliefs[c] / belief_mass
            if category_weight == 0:
                continue
            try:
                ss = compute_steady_state(tuning_basis_by_category[c][i])
                category_score = jensenshannon(ss, desired_distribution)
            except Exception:
                category_score = np.inf
            scores[i] += category_weight * category_score
    idx = int(np.argmin(scores))
    weights = np.zeros(number_of_basis_tuning_matrices)
    weights[idx] = 1.0
    return weights, idx


# ============================================================
# MEASUREMENT DECISION ALGORITHMS
# ============================================================


class _EntropyThresholdMeasurement:
    """Measure when state-belief entropy >= threshold."""

    def __init__(self, entropy_threshold):
        self.entropy_threshold = entropy_threshold

    def decide(self, belief, current_state_estimate=None):
        ent = belief.state_entropy()
        decision = 1 if ent >= self.entropy_threshold else 0
        return decision, ent


class _DynamicProgrammingMeasurement:
    """
    Value-iteration measurement policy. Precomputes an optimal
    measure/don't-measure decision for each state.

    Requires knowledge of the true category (is_category_known=True).
    """

    def __init__(
        self,
        P_true,
        desired_distribution,
        discount_factor=0.95,
        measurement_cost=0.1,
        uncertainty_weight=0.5,
        control_weight=1.0,
    ):
        self.entropy_threshold = None  # for logging/plot compatibility
        number_of_states = P_true.shape[0]

        # Terminal control cost: JS from current steady state to π*
        current_ss = compute_steady_state(P_true)
        js_div = jensenshannon(current_ss, desired_distribution)
        terminal_cost = -control_weight * js_div

        # Value iteration
        V = np.zeros(number_of_states)
        V_new = np.zeros(number_of_states)
        max_iter = 1000
        tol = 1e-4

        for it in range(max_iter):
            for s in range(number_of_states):
                next_dist = P_true[s]
                unc = -uncertainty_weight * stats.entropy(next_dist, base=2)
                v_no = np.sum(P_true[s] * V) + unc
                if it == max_iter - 1:
                    v_no += terminal_cost
                v_meas = -measurement_cost + discount_factor * V[s]
                V_new[s] = max(v_no, v_meas)
            if np.max(np.abs(V_new - V)) < tol:
                break
            V = V_new.copy()

        # Extract policy
        self.policy = np.zeros(number_of_states, dtype=bool)
        for s in range(number_of_states):
            next_dist = P_true[s]
            unc = -uncertainty_weight * stats.entropy(next_dist, base=2)
            v_no = np.sum(P_true[s] * V) + unc + terminal_cost
            v_meas = -measurement_cost + discount_factor * V[s]
            self.policy[s] = v_meas > v_no

    def decide(self, belief, current_state_estimate=None):
        if current_state_estimate is None:
            current_state_estimate = int(np.argmax(belief.belief_matrix.sum(axis=0)))
        decision = int(self.policy[current_state_estimate])
        ent = belief.state_entropy()
        return decision, ent


# ============================================================
# THE PARAMETERIZED AGENT
# ============================================================


class Agent:
    """
    Composable agent with independent measurement and treatment policies.

    Args:
        config: dict from Configurator. Must include number_of_states, number_of_categories,
            category_prior, initial_distributions, transition_matrices,
            desired_distribution, alpha_types, alpha_means, alpha_decay_rates,
            number_of_steps, and (added by the runner) category_names,
            tuning_matrix_names, tuning_basis_by_category.
        state_measurement_decision_algorithm:
            "entropy_threshold" | "dynamic_programming"
        tuning_space: "synthetic" | "data"
        is_category_known: bool. If True, prior is one-hot on true_category.
        tuning_selection: "random_constant" | "given_constant" | "convex" | "argmax"
        entropy_threshold: float. Only used if algorithm=entropy_threshold.
        discount_factor: float. Only used if algorithm=dynamic_programming.
        measurement_cost: float. Only used if algorithm=dynamic_programming.
        true_category: int. Required if is_category_known=True or if
            algorithm=dynamic_programming.
        rng: numpy random generator (defaults to np.random module).

    Modeled categories are those with positive prior (all of them when the
    category is unknown, only the true one when it is known). The agent
    advances P_model only for them; a zero-prior category keeps zero belief
    forever, so its model never affects the belief. Every action the tuning
    selection can apply must be fitted for every modeled category, else
    __init__ raises (Tenet 3).
    """

    def __init__(
        self,
        config,
        state_measurement_decision_algorithm,
        tuning_space,
        is_category_known,
        tuning_selection,
        entropy_threshold,
        discount_factor,
        measurement_cost,
        uncertainty_weight,
        control_weight,
        true_category,
        constant_tuning_selection,
        tuning_cadence,
        rng,
    ):

        self.number_of_states = config["number_of_states"]
        self.number_of_categories = config["number_of_categories"]
        self.state_measurement_decision_algorithm = state_measurement_decision_algorithm
        self.tuning_space = tuning_space
        self.is_category_known = is_category_known
        self.tuning_selection = tuning_selection
        self.entropy_threshold_value = entropy_threshold
        self.discount_factor = discount_factor
        self.measurement_cost = measurement_cost
        self.true_category = true_category
        self.category_names = list(config["category_names"])
        self.rng = rng if rng is not None else np.random

        # ---- Category prior override ----
        category_prior = np.asarray(config["category_prior"]).copy()
        if is_category_known:
            if true_category is None:
                raise ValueError(
                    "is_category_known=True requires true_category to be set"
                )
            category_prior = np.zeros(self.number_of_categories)
            category_prior[true_category] = 1.0
        self.modeled_categories = [
            c for c in range(self.number_of_categories) if category_prior[c] > 0
        ]

        # ---- Belief state ----
        self.belief = Belief(self.number_of_states, self.number_of_categories)
        for c in range(self.number_of_categories):
            for s in range(self.number_of_states):
                self.belief.belief_matrix[c, s] = (
                    category_prior[c] * config["initial_distributions"][c][s]
                )

        # ---- Agent's internal transition model ----
        self.P_model = [P.copy() for P in config["transition_matrices"]]
        self.desired_distribution = np.asarray(config["desired_distribution"]).copy()
        self.alpha_sequences = self._estimate_alpha_sequences(config)
        self.number_of_steps = config["number_of_steps"]
        self.current_step = 0

        # ---- Tuning basis (per category, shared with the Environment) ----
        if tuning_space not in ("synthetic", "data"):
            raise ValueError(f"Unknown tuning_space: {tuning_space!r}")
        self.tuning_basis_by_category = config["tuning_basis_by_category"]
        self.basis_labels = list(config["tuning_matrix_names"])
        self.number_of_basis_tuning_matrices = len(self.basis_labels)
        if len(self.tuning_basis_by_category) != self.number_of_categories:
            raise ValueError(
                f"tuning_basis_by_category has {len(self.tuning_basis_by_category)} "
                f"rows but there are {self.number_of_categories} categories"
            )
        for c, tuning_basis_row in enumerate(self.tuning_basis_by_category):
            if len(tuning_basis_row) != self.number_of_basis_tuning_matrices:
                raise ValueError(
                    f"tuning basis row of category {self.category_names[c]!r} has "
                    f"{len(tuning_basis_row)} entries but tuning_matrix_names has "
                    f"{self.number_of_basis_tuning_matrices}"
                )

        # ---- Pre-compute selection state (for random_constant / given_constant) ----
        self._constant_weights = None
        self._constant_index = None
        if tuning_selection == "random_constant":
            self._constant_weights, self._constant_index = _select_random_constant(
                self.number_of_basis_tuning_matrices, self.rng
            )
        elif tuning_selection == "given_constant":
            # constant_tuning_selection must name one of the basis labels.
            # Per Tenet 3, no fallback — raise loudly on mismatch.
            if constant_tuning_selection is None:
                raise ValueError(
                    "tuning_selection='given_constant' requires "
                    "constant_tuning_selection to be set to one of "
                    f"{self.basis_labels!r}"
                )
            if constant_tuning_selection not in self.basis_labels:
                raise ValueError(
                    f"constant_tuning_selection={constant_tuning_selection!r} is "
                    f"not in basis_labels {self.basis_labels!r}"
                )
            self._constant_index = self.basis_labels.index(constant_tuning_selection)
            self._constant_weights = np.zeros(self.number_of_basis_tuning_matrices)
            self._constant_weights[self._constant_index] = 1.0

        # ---- Tuning cadence ----
        # Apply the selected tuning only every `tuning_cadence` steps; on the
        # steps in between, apply the "None" (no-treatment) matrix. cadence=1
        # reproduces the every-step behavior exactly.
        # Per Tenet 3, validate loudly: cadence must be a strict positive integer
        # (reject bool, float, str, and non-positive values — no coercion).
        if isinstance(tuning_cadence, bool) or not isinstance(
            tuning_cadence, (int, np.integer)
        ):
            raise TypeError(
                "tuning_cadence must be a positive integer, got "
                f"{tuning_cadence!r} of type {type(tuning_cadence).__name__}."
            )
        if tuning_cadence < 1:
            raise ValueError(
                f"tuning_cadence must be a positive integer (>= 1), got {tuning_cadence!r}."
            )
        self.tuning_cadence = int(tuning_cadence)
        self._none_index = (
            self.basis_labels.index("None") if "None" in self.basis_labels else None
        )
        if self.tuning_cadence > 1 and self._none_index is None:
            raise ValueError(
                "tuning_cadence > 1 requires a 'None' entry in the tuning basis "
                f"(off-steps apply no treatment); basis_labels={self.basis_labels!r}. "
                "Use tuning_space='data', whose basis includes 'None'."
            )

        # ---- Every selectable action must be fitted for every modeled category ----
        self._validate_selectable_tuning_is_fitted()

        # ---- Measurement decision algorithm setup ----
        if state_measurement_decision_algorithm == "entropy_threshold":
            self._measurement = _EntropyThresholdMeasurement(entropy_threshold)
        elif state_measurement_decision_algorithm == "dynamic_programming":
            if not is_category_known or true_category is None:
                raise ValueError(
                    "state_measurement_decision_algorithm='dynamic_programming' "
                    "requires is_category_known=True and true_category"
                )
            self._measurement = _DynamicProgrammingMeasurement(
                P_true=config["transition_matrices"][true_category],
                desired_distribution=self.desired_distribution,
                discount_factor=discount_factor,
                measurement_cost=measurement_cost,
                uncertainty_weight=uncertainty_weight,
                control_weight=control_weight,
            )

        else:
            raise ValueError(
                "Unknown state_measurement_decision_algorithm: "
                f"{state_measurement_decision_algorithm!r}"
            )

        # ---- History ----
        self.history = {
            "belief": [],
            "entropy": [],
            "decisions": [],
            "category_belief": [],
            "tuning_weights": [],  # weight vector over basis (recorded every step)
            # per step: list over categories of T^(c), None if not modeled
            "tuning_matrices_by_category": [],
            "belief_matrix": [],
        }

    # ==============================================================
    # Public API — required by Simulator
    # ==============================================================

    def decide_measurement(self):
        """Decide whether to request a state observation at this step."""
        current_state_estimate = self._current_state_estimate()
        decision, ent = self._measurement.decide(
            self.belief, current_state_estimate=current_state_estimate
        )
        pi = self.belief.state_belief()
        cat_belief = self.belief.category_belief()
        self.history["decisions"].append(decision)
        self.history["entropy"].append(ent)
        self.history["belief"].append(pi.copy())
        self.history["category_belief"].append(cat_belief.copy())
        self.history["belief_matrix"].append(self.belief.belief_matrix.copy())
        return decision

    def decide_tuning(self):
        """Choose the tuning action: weights w over the basis actions.

        Returns the weights; the Environment applies T^(C) = sum_a w_a B^(C,a)
        for the true category C. The agent records its own T^(c) for every
        modeled category (None for the others).

        Cadence gate: the selected tuning is applied only on steps that are
        multiples of `tuning_cadence`; on the steps in between the agent applies
        the 'None' (no-treatment) action. cadence=1 tunes every step.
        """
        if self.current_step % self.tuning_cadence == 0:
            weights = self._compute_weights()
        else:
            weights = np.zeros(self.number_of_basis_tuning_matrices)
            weights[self._none_index] = 1.0
        tuning_matrices_by_category = [None] * self.number_of_categories
        for c in self.modeled_categories:
            tuning_matrices_by_category[c] = mix_tuning_basis(
                self.tuning_basis_by_category[c],
                weights,
                self.basis_labels,
                self.category_names[c],
            )
        self.history["tuning_matrices_by_category"].append(tuning_matrices_by_category)
        self.history["tuning_weights"].append(weights.copy())
        return weights

    def process_measurement(self, measurement):
        """Update belief given (optional) observation and advance P_model."""
        self._update_P_model()
        if measurement is not None:
            self.belief.update_with_observation(measurement, self.P_model)
        else:
            self.belief.update_without_observation(self.P_model)
        self.current_step += 1

    # ==============================================================
    # Internals
    # ==============================================================

    def _compute_weights(self):
        """Return the basis-weight vector for the current step."""
        if self.tuning_selection in ("random_constant", "given_constant"):
            return self._constant_weights.copy()

        if self.tuning_selection == "convex":
            if self.tuning_space == "synthetic":
                # One basis matrix per category → weights = category beliefs
                cat_beliefs = self.belief.category_belief()
                return _select_convex_over_categories(
                    cat_beliefs, self.number_of_basis_tuning_matrices
                )
            # tuning_space=data: no natural category→drug mapping; uniform fallback.
            return (
                np.ones(self.number_of_basis_tuning_matrices)
                / self.number_of_basis_tuning_matrices
            )

        if self.tuning_selection == "argmax":
            weights, _ = _select_argmax_by_js_to_desired(
                self.tuning_basis_by_category,
                self.belief.category_belief(),
                self.modeled_categories,
                self.desired_distribution,
            )
            return weights

        raise ValueError(f"Unknown tuning_selection: {self.tuning_selection!r}")

    def _current_state_estimate(self):
        """Argmax of belief for the true category (or marginal if unknown)."""
        if self.true_category is not None:
            return int(np.argmax(self.belief.belief_matrix[self.true_category]))
        return int(np.argmax(self.belief.belief_matrix.sum(axis=0)))

    def _validate_selectable_tuning_is_fitted(self):
        """Raise if an action this agent can apply lacks a fit in a modeled category.

        given_constant applies its action (and 'None' on cadence off-steps);
        random_constant, convex and argmax can apply any action.
        """
        if self.tuning_selection == "given_constant":
            selectable_action_indices = {self._constant_index}
            if self.tuning_cadence > 1:
                selectable_action_indices.add(self._none_index)
        else:
            selectable_action_indices = set(range(self.number_of_basis_tuning_matrices))
        missing_fits = [
            (self.category_names[c], self.basis_labels[a])
            for c in self.modeled_categories
            for a in sorted(selectable_action_indices)
            if self.tuning_basis_by_category[c][a] is None
        ]
        if missing_fits:
            raise ValueError(
                f"tuning_selection={self.tuning_selection!r} can apply actions "
                "that have no fitted tuning matrix for a category the agent "
                f"models (is_category_known={self.is_category_known}): "
                f"(category, action) = {missing_fits!r}. Per Tenet 3 there is "
                "no fallback: choose an action fitted for these categories, or "
                "set is_category_known: true with a true_category that has it."
            )

    def _update_P_model(self):
        """Blend each modeled category's P_model toward its own tuning matrix T^(c)."""
        if not self.history["tuning_matrices_by_category"]:
            raise RuntimeError("process_measurement called before decide_tuning")
        if self.current_step < self.number_of_steps:
            tuning_matrices_by_category = self.history["tuning_matrices_by_category"][
                -1
            ]
            for c in self.modeled_categories:
                current_alpha = self.alpha_sequences[c, self.current_step]
                self.P_model[c] = (1 - current_alpha) * self.P_model[
                    c
                ] + current_alpha * tuning_matrices_by_category[c]

    @staticmethod
    def _estimate_alpha_sequences(config):
        """Expected-value alpha estimate per category (deterministic)."""
        n_cat = config["number_of_categories"]
        number_of_steps = config["number_of_steps"]
        t = np.arange(number_of_steps)
        seqs = np.zeros((n_cat, number_of_steps))
        for c in range(n_cat):
            if config["alpha_types"][c] == "decaying":
                seqs[c] = config["alpha_means"][c] * np.exp(
                    -config["alpha_decay_rates"][c] * t
                )
            else:
                seqs[c] = config["alpha_means"][c]
        return seqs

    # ==============================================================
    # Metadata
    # ==============================================================

    @property
    def entropy_threshold(self):
        """Backward-compat property used by Simulator logging."""
        if self.state_measurement_decision_algorithm == "entropy_threshold":
            return self.entropy_threshold_value
        return None

    @property
    def agent_type(self) -> str:
        return (
            f"Agent(meas={self.state_measurement_decision_algorithm},"
            f"space={self.tuning_space},"
            f"cat_known={self.is_category_known},"
            f"select={self.tuning_selection})"
        )

    def __repr__(self):
        return self.agent_type
