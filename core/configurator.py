# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Configurator class for setting up simulation parameters.
"""

import numpy as np


class Configurator:
    """
    Configurator class that determines the simulation parameters including:
    - Prior distribution on environment categories
    - Transition matrices for each category
    - Number of states
    - Desired steady state for agent's tuning policy
    - Per-category alpha parameters for environment's AlphaGenerators

    All information is passed to both environment and agent via get_config().
    """

    def __init__(
        self,
        number_of_states=4,
        number_of_categories=3,
        category_prior=None,
        transition_matrices=None,
        alpha_type="decaying",
        alpha_mean=0.01,
        alpha_decay_rate=0.001,
        desired_distribution=None,
        number_of_steps=1000,
        state_names=None,
    ):
        """
        Initialize configurator with simulation parameters.

        Args:
            number_of_states: Number of states in the Markov chain
            number_of_categories: Number of environment categories
            category_prior: Prior distribution over categories (optional)
            transition_matrices: List of transition matrices per category (optional)
            alpha_type: Type of alpha generator ('constant' or 'decaying').
                        Scalar or list of length number_of_categories.
            alpha_mean: Mean for the alpha generator. Scalar (broadcast to all
                        categories) or list of length number_of_categories.
            alpha_decay_rate: Decay rate for decaying alpha. Scalar or list.
            desired_distribution: Target steady state distribution (optional)
            number_of_steps: Number of simulation steps
            state_names: List of human-readable state names (optional)
        """
        self.number_of_states = number_of_states
        self.number_of_categories = number_of_categories
        self.number_of_steps = number_of_steps

        # Per-category alpha parameters — broadcast scalar to list
        self.alpha_types = self._broadcast(
            alpha_type, number_of_categories, "alpha_type"
        )
        self.alpha_means = self._broadcast_float(
            alpha_mean, number_of_categories, "alpha_mean"
        )
        self.alpha_decay_rates = self._broadcast_float(
            alpha_decay_rate, number_of_categories, "alpha_decay_rate"
        )

        # Set category prior distribution
        if category_prior is None:
            self.category_prior = np.random.dirichlet(np.ones(number_of_categories))
        else:
            self.category_prior = np.asarray(category_prior, dtype=float)

        # Set desired steady state for agent's tuning policy
        if desired_distribution is None:
            self.desired_distribution = np.random.dirichlet(np.ones(number_of_states))
        else:
            self.desired_distribution = np.array(desired_distribution) / np.sum(
                desired_distribution
            )

        # Generate transition matrices for each category
        if transition_matrices is None:
            self.transition_matrices = []
            for _ in range(number_of_categories):
                transition_matrix = np.random.dirichlet(
                    np.ones(number_of_states), size=number_of_states
                )
                self.transition_matrices.append(transition_matrix)
        else:
            self.transition_matrices = [P.copy() for P in transition_matrices]

        # Generate initial state distributions for each category
        self.initial_distributions = [
            np.random.dirichlet(np.ones(number_of_states))
            for _ in range(number_of_categories)
        ]

        # State names for display
        if state_names is not None:
            if len(state_names) != number_of_states:
                raise ValueError(
                    f"state_names length ({len(state_names)}) must match number_of_states ({number_of_states})"
                )
            self.state_names = list(state_names)
        else:
            self.state_names = [str(i) for i in range(number_of_states)]

    @staticmethod
    def _broadcast(val, n, name):
        """Broadcast a scalar string to a list of length n."""
        if isinstance(val, str):
            return [val] * n
        if len(val) != n:
            raise ValueError(f"{name} must be scalar or list of length {n}")
        return list(val)

    @staticmethod
    def _broadcast_float(val, n, name):
        """Broadcast a scalar float to a list of length n."""
        if isinstance(val, (int, float)):
            return [float(val)] * n
        if len(val) != n:
            raise ValueError(f"{name} must be scalar or list of length {n}")
        return [float(v) for v in val]

    def get_config(self):
        """
        Return configuration dictionary with all parameters.

        Returns:
            dict: Configuration parameters for simulation
        """
        return {
            "number_of_states": self.number_of_states,
            "number_of_categories": self.number_of_categories,
            "category_prior": self.category_prior.copy(),
            "transition_matrices": [P.copy() for P in self.transition_matrices],
            "desired_distribution": self.desired_distribution.copy(),
            "alpha_types": list(self.alpha_types),
            "alpha_means": list(self.alpha_means),
            "alpha_decay_rates": list(self.alpha_decay_rates),
            "number_of_steps": self.number_of_steps,
            "initial_distributions": [
                dist.copy() for dist in self.initial_distributions
            ],
            "state_names": list(self.state_names),
        }

    def __repr__(self):
        return (
            f"Configurator(number_of_states={self.number_of_states}, "
            f"number_of_categories={self.number_of_categories}, number_of_steps={self.number_of_steps}, "
            f"alpha_means={self.alpha_means})"
        )
