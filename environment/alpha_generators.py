# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Alpha sequence generation strategies for environment dynamics.

Alpha controls the blending rate: P = (1-α)P + αT

Each category has its own AlphaGenerator with potentially different parameters.
"""

import numpy as np


class AlphaGenerator:
    """Base class for single-category alpha sequence generators."""

    @staticmethod
    def create(alpha_type, alpha_mean, alpha_decay_rate=0.0):
        """Factory method to create a single-category alpha generator."""
        if alpha_type == "constant":
            return ConstantMeanExponentialAlpha(mean_alpha=alpha_mean)
        elif alpha_type == "decaying":
            return DecayingMeanExponentialAlpha(
                initial_mean=alpha_mean, decay_rate=alpha_decay_rate
            )
        else:
            raise ValueError(
                f"Unknown alpha_type: {alpha_type}. Use 'constant' or 'decaying'."
            )

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ConstantMeanExponentialAlpha(AlphaGenerator):
    """Alpha with exponentially distributed values and constant mean."""

    def __init__(self, mean_alpha):
        if mean_alpha <= 0:
            raise ValueError("mean_alpha must be positive")
        self.mean_alpha = mean_alpha

    def generate_sequence(self, number_of_steps):
        return np.clip(np.random.exponential(self.mean_alpha, number_of_steps), 0, 1)

    def expected_sequence(self, number_of_steps):
        return np.full(number_of_steps, self.mean_alpha)

    def __repr__(self):
        return f"ConstantMeanExponentialAlpha(mean={self.mean_alpha})"


class DecayingMeanExponentialAlpha(AlphaGenerator):
    """Alpha with exponentially distributed values and exponentially decaying mean."""

    def __init__(self, initial_mean, decay_rate):
        if initial_mean <= 0:
            raise ValueError("initial_mean must be positive")
        if decay_rate < 0:
            raise ValueError("decay_rate must be non-negative")
        self.initial_mean = initial_mean
        self.decay_rate = decay_rate

    def generate_sequence(self, number_of_steps):
        t = np.arange(number_of_steps)
        means_t = self.initial_mean * np.exp(-self.decay_rate * t)
        samples = np.array([np.random.exponential(m) for m in means_t])
        return np.clip(samples, 0, 1)

    def expected_sequence(self, number_of_steps):
        t = np.arange(number_of_steps)
        return self.initial_mean * np.exp(-self.decay_rate * t)

    def __repr__(self):
        return (
            f"DecayingMeanExponentialAlpha("
            f"initial_mean={self.initial_mean}, decay_rate={self.decay_rate})"
        )
