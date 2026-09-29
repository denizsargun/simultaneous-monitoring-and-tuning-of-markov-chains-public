# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Environment class for managing system state and providing measurements.
"""

import numpy as np
from core.utils import mix_tuning_basis
from environment.category import Category
from environment.state import State
from environment.alpha_generators import AlphaGenerator


class Environment:
    """
    Environment that manages the true system state and provides measurements.

    The environment only outputs measurements when requested and keeps
    all internal dynamics hidden. Data logging is handled by the Recorder
    in the Simulator — the environment has no history dict.
    """

    def __init__(self, config):
        """
        Initialize environment from configurator config.

        Args:
            config: Configuration dictionary from Configurator. Must contain:
                - number_of_states, number_of_categories
                - category_prior, initial_distributions
                - transition_matrices, number_of_steps
                - alpha_types, alpha_means, alpha_decay_rates (per-category lists)
                - category_names, tuning_matrix_names, tuning_basis_by_category
                  (added by the runner)
        """
        self.number_of_states = config["number_of_states"]
        self.number_of_categories = config["number_of_categories"]

        # Initialize category using prior from configurator
        self.category = Category(self.number_of_categories, config["category_prior"])

        # Initialize Markov chain state
        self.initial_states = [
            State(self.number_of_states, dist)
            for dist in config["initial_distributions"]
        ]
        self.state = self.initial_states[self.category.true_category].current_state

        # Current transition matrices — one per category, evolve via alpha
        # blending each step. Per Tenet 4, use the explicit name (not `self.P`).
        self.current_transition_matrices = [
            P.copy() for P in config["transition_matrices"]
        ]
        # Tuning basis B^(c,a): the true effect of each action in each category
        # (None where the data has no fit). The agent's weights w select the
        # true category's matrix T^(C) = sum_a w_a B^(C,a).
        self.tuning_basis_by_category = config["tuning_basis_by_category"]
        # Tuning matrices (applied T) — one per category. step() stores T^(C)
        # at the true-category slot. Then `current_transition_matrix` blends
        # toward `tuning_matrix` via: P = (1-α)P + α*tuning_matrix.
        self.tuning_matrices = [P.copy() for P in config["transition_matrices"]]

        # Environment creates AlphaGenerator for its true category only
        c = self.category.true_category
        self.alpha_generator = AlphaGenerator.create(
            alpha_type=config["alpha_types"][c],
            alpha_mean=config["alpha_means"][c],
            alpha_decay_rate=config["alpha_decay_rates"][c],
        )
        self.alpha_sequence = self.alpha_generator.generate_sequence(
            config["number_of_steps"]
        )

        self.number_of_steps = config["number_of_steps"]
        self.current_step = 0

        # Names for display
        self.state_names = config.get("state_names", None)
        self.category_names = config["category_names"]
        self.tuning_matrix_names = config["tuning_matrix_names"]
        # Visualization-only: maps each state index to its display y-axis position.
        # List of ints, length == number_of_states. Injected by the runner.
        self.state_y_axis_positions_by_index = config.get(
            "state_y_axis_positions_by_index", None
        )

    def step(self, measurement_decision, tuning_weights):
        """Environment step function. Execution order:

        1. Set the tuning matrix T^(C) = sum_a w_a B^(C,a) of the true category
        2. Blend current_transition_matrix toward tuning_matrix via alpha
        3. Transition to next state using updated current_transition_matrix
        4. Provide measurement (the new state) if agent requested

        The decision made at step k is M_{k+1} in the action A_k = (M_{k+1}, T_k),
        so a measurement reveals the post-transition state: Y_{k+1} = X_{k+1}.

        Args:
            measurement_decision: 1 if agent wants to measure, 0 otherwise
            tuning_weights: agent's weights over the tuning actions

        Returns:
            measurement: next state X_{k+1} if measurement_decision=1, None otherwise

        Raises:
            ValueError: if an action with positive weight has no fitted matrix
                for the true category (Tenet 3: no fallback).
        """
        c = self.category.true_category

        # Step 1: Apply the true category's own matrix for the chosen action
        self.tuning_matrices[c] = mix_tuning_basis(
            self.tuning_basis_by_category[c],
            tuning_weights,
            self.tuning_matrix_names,
            self.category_names[c],
        )

        # Step 2: Blend current transition matrix toward the tuning matrix
        if self.current_step < self.number_of_steps:
            current_alpha = self.alpha_sequence[self.current_step]
            self.current_transition_matrices[c] = (
                1 - current_alpha
            ) * self.current_transition_matrices[
                c
            ] + current_alpha * self.tuning_matrices[
                c
            ]
            self.current_step += 1

        # Step 3: Transition to next state
        current_transition_matrix = self.current_transition_matrices[c]
        transition_probs = current_transition_matrix[self.state]
        transition_probs = np.clip(transition_probs, 0, 1)
        transition_probs = transition_probs / transition_probs.sum()
        self.last_transition_probabilities = transition_probs.copy()
        self.state = np.random.choice(self.number_of_states, p=transition_probs)

        # Step 4: Provide measurement of the new state based on agent's decision
        measurement = self.state if measurement_decision == 1 else None

        return measurement

    def __repr__(self):
        return (
            f"Environment(number_of_states={self.number_of_states}, "
            f"number_of_categories={self.number_of_categories}, "
            f"true_category={self.category.true_category}, "
            f"alpha_generator={self.alpha_generator}, "
            f"step={self.current_step}/{self.number_of_steps})"
        )
