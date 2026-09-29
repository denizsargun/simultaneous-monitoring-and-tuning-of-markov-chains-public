# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Core State class representing the state of the environment.
"""

import numpy as np


class State:
    """
    Represents the state of the environment with its own prior distribution
    and sampled current state.
    """

    def __init__(self, number_of_states, prior):
        """
        Initialize state by sampling from the given prior.

        Args:
            number_of_states: Number of possible states
            prior: Prior probability distribution over states (must sum to 1)
        """
        self.prior = np.asarray(prior, dtype=float)
        if self.prior.shape != (number_of_states,):
            raise ValueError(
                f"prior must have shape ({number_of_states},), got {self.prior.shape}"
            )
        if not np.isclose(self.prior.sum(), 1.0):
            raise ValueError(f"prior must sum to 1.0, got {self.prior.sum():.6f}")
        self.current_state = np.random.choice(number_of_states, p=self.prior)

    def __repr__(self):
        return f"State(current_state={self.current_state}, prior={self.prior})"
