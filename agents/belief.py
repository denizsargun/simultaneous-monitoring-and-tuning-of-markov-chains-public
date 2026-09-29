# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Belief class for maintaining agent's internal belief state.
"""

import numpy as np
from scipy import stats


class Belief:
    """Internal belief state maintained by the agent"""

    def __init__(self, number_of_states, number_of_categories):
        """
        Initialize belief state

        Args:
            number_of_states: Number of states in the system
            number_of_categories: Number of categories
        """
        self.number_of_states = number_of_states
        self.number_of_categories = number_of_categories
        # Joint belief matrix over categories and states
        self.belief_matrix = np.ones((number_of_categories, number_of_states)) / (
            number_of_categories * number_of_states
        )

    def update_with_observation(self, observation, P_model):
        """
        Update joint belief when measurement is available using Bayesian inference

        Predicts one step with P_model, then conditions on the observation:
        b'(c, x') ∝ 1[x' = observation] · Σ_x b(c, x) P_model[c][x, x'].

        Args:
            observation: Observed post-transition state X_{k+1}
            P_model: List of transition matrices for each category (P_{k+1})
        """
        # Compute likelihood of observation under each category's dynamics
        new_belief_matrix = np.zeros((self.number_of_categories, self.number_of_states))

        for c in range(self.number_of_categories):
            # Likelihood of observing this state given category c
            # This is the probability of transitioning to the observed state
            # from the current belief distribution under category c's dynamics
            likelihood = np.sum(self.belief_matrix[c, :] * P_model[c][:, observation])

            # Update belief: only the observed state gets probability mass
            new_belief_matrix[c, observation] = likelihood
            # All other states get zero probability
            for s in range(self.number_of_states):
                if s != observation:
                    new_belief_matrix[c, s] = 0.0

        # Normalize to get valid probability distribution
        total_prob = np.sum(new_belief_matrix)
        if total_prob > 0:
            self.belief_matrix = new_belief_matrix / total_prob
        else:
            # Fallback: uniform distribution over observed state across categories
            self.belief_matrix = np.zeros(
                (self.number_of_categories, self.number_of_states)
            )
            for c in range(self.number_of_categories):
                self.belief_matrix[c, observation] = 1.0 / self.number_of_categories

    def update_without_observation(self, P_model):
        """
        Update joint belief using transition model when no measurement

        Args:
            P_model: List of transition matrices for each category
        """
        # Update joint belief matrix using the provided P_model
        new_belief_matrix = np.zeros_like(self.belief_matrix)
        for c in range(self.number_of_categories):
            for s in range(self.number_of_states):
                # Transition from all previous states to current state s
                new_belief_matrix[c, s] = np.sum(
                    self.belief_matrix[c, :] * P_model[c][:, s]
                )

        # Normalize to maintain valid probability distribution
        total = np.sum(new_belief_matrix)
        if total > 0:
            new_belief_matrix = new_belief_matrix / total

        self.belief_matrix = new_belief_matrix

    def category_state_joint_entropy(self):
        """
        Compute entropy of joint belief state

        Returns:
            float: Entropy value in bits
        """
        joint_belief = self.belief_matrix.flatten()
        # Normalize to ensure valid probability distribution
        if np.sum(joint_belief) > 0:
            joint_belief = joint_belief / np.sum(joint_belief)
        else:
            joint_belief = np.ones_like(joint_belief) / len(joint_belief)
        return stats.entropy(joint_belief, base=2)

    def state_entropy(self):
        """
        Compute entropy of state belief (marginalizing over categories)

        Returns:
            float: Entropy value in bits
        """
        # Get marginal state distribution by summing over categories
        state_belief = np.sum(self.belief_matrix, axis=0)  # Sum over categories
        # Normalize to ensure valid probability distribution
        if np.sum(state_belief) > 0:
            state_belief = state_belief / np.sum(state_belief)
        else:
            state_belief = np.ones_like(state_belief) / len(state_belief)
        return stats.entropy(state_belief, base=2)

    def state_belief(self):
        """
        Extract state beliefs for each category from joint belief matrix

        Returns:
            numpy.ndarray: Array of shape (number_of_categories, number_of_states) with state beliefs
        """
        pi = np.zeros((self.number_of_categories, self.number_of_states))
        for c in range(self.number_of_categories):
            category_sum = np.sum(self.belief_matrix[c, :])
            if category_sum > 0:
                pi[c, :] = self.belief_matrix[c, :] / category_sum
            else:
                pi[c, :] = np.ones(self.number_of_states) / self.number_of_states
        return pi

    def category_belief(self):
        """
        Extract category beliefs from joint belief matrix

        Returns:
            numpy.ndarray: Array of shape (number_of_categories,) with category beliefs
        """
        return np.sum(self.belief_matrix, axis=1)
