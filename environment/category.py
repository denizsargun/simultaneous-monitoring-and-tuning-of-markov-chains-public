# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Core Category class for managing category state and transitions.
"""

import numpy as np


class Category:
    """
    Represents a category in the environment with its own prior distribution
    and sampled current state.
    """

    def __init__(self, number_of_categories, prior):
        """
        Initialize category by sampling from the given prior.

        Args:
            number_of_categories: Number of possible categories
            prior: Prior probability distribution over categories (must sum to 1)
        """
        self.prior = np.asarray(prior, dtype=float)
        if self.prior.shape != (number_of_categories,):
            raise ValueError(
                f"prior must have shape ({number_of_categories},), got {self.prior.shape}"
            )
        if not np.isclose(self.prior.sum(), 1.0):
            raise ValueError(f"prior must sum to 1.0, got {self.prior.sum():.6f}")
        self.true_category = np.random.choice(number_of_categories, p=self.prior)

    def __repr__(self):
        return f"Category(true_category={self.true_category}, prior={self.prior})"
