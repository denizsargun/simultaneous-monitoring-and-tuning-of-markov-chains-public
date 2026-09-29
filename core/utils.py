# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Core utility functions for Markov chain simulations.

- compute_steady_state: steady state distribution via eigenvalue decomposition
- get_timestamp: ISO 8601 UTC timestamp generation
- mix_tuning_basis: weighted mix of one category's tuning basis row
"""

import numpy as np
from datetime import datetime, timezone


def compute_steady_state(P: np.ndarray) -> np.ndarray:
    """
    Compute steady state distribution of transition matrix P using eigenvalue decomposition.

    Finds the left eigenvector corresponding to eigenvalue 1 (or closest to 1),
    which represents the stationary distribution: π P = π.

    For ergodic (irreducible, aperiodic) chains, the steady state is unique.
    For chains with absorbing states (e.g., Death), this returns the absorbing
    distribution.

    Args:
        P: Transition matrix (numpy array, rows must sum to 1, entries non-negative)

    Returns:
        Steady state distribution (normalized probability vector)

    Raises:
        ValueError: If P is not a valid stochastic matrix
        np.linalg.LinAlgError: If eigenvalue decomposition fails
    """
    # Validate P is stochastic
    if not np.all(P >= 0):
        raise ValueError("P contains negative entries — not a valid stochastic matrix")
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(f"P rows do not sum to 1.0: {row_sums}")

    # Eigenvalue decomposition — let LinAlgError propagate
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    steady_state = np.real(eigenvectors[:, idx])

    # Fix numerical noise (tiny negative entries from floating point)
    steady_state = np.abs(steady_state)

    # Normalize to valid probability distribution
    total = steady_state.sum()
    if total == 0:
        raise ValueError(
            "Eigenvector for λ=1 is zero — this should never happen "
            "for a valid stochastic matrix"
        )
    return steady_state / total


def get_timestamp() -> str:
    """
    Generate ISO 8601 UTC timestamp with milliseconds: 20260220T101514.123Z

    Used for:
    - Simulation log directory names
    - Output file naming
    - Episode timestamps

    Returns:
        str: Timestamp in format YYYYMMDDTHHMMSS.mmmZ
    """
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S") + f".{now.microsecond // 1000:03d}Z"


def mix_tuning_basis(
    tuning_basis_row, tuning_weights, tuning_matrix_names, category_name
):
    """
    Tuning matrix of one category: T^(c) = sum_a w_a B^(c,a).

    Per Tenet 3, an action with positive weight whose matrix is missing for
    this category (no fit in the data, stored as None) raises. It is never
    replaced by another matrix.

    Args:
        tuning_basis_row: list over actions of matrices B^(c,a), or None where
            the category has no fit for the action
        tuning_weights: weight vector over actions (sums to 1)
        tuning_matrix_names: action names, aligned with tuning_basis_row
        category_name: name of category c (for error messages)

    Returns:
        np.ndarray: the mixed tuning matrix T^(c)
    """
    mixed_tuning_matrix = None
    for action_index, weight in enumerate(tuning_weights):
        if weight == 0:
            continue
        basis_matrix = tuning_basis_row[action_index]
        if basis_matrix is None:
            raise ValueError(
                f"No fitted tuning matrix for action "
                f"{tuning_matrix_names[action_index]!r} in category "
                f"{category_name!r}; it cannot be applied (Tenet 3: no fallback)."
            )
        contribution = weight * basis_matrix
        mixed_tuning_matrix = (
            contribution
            if mixed_tuning_matrix is None
            else mixed_tuning_matrix + contribution
        )
    if mixed_tuning_matrix is None:
        raise ValueError(f"tuning_weights has no positive entry: {tuning_weights!r}")
    return mixed_tuning_matrix


def metropolis_hastings_matrix(
    target_distribution,
    entropy_threshold=None,
    max_trials=100,
    initial_matrix=None,
):
    """
    Generate a transition matrix with target stationary distribution π*
    using the Metropolis-Hastings acceptance ratio.

    Off-diagonal: P(i,j) = min(1, π*(j)/π*(i)) * P_prop(i,j)
    Diagonal:     P(i,i) = 1 - Σ_{j≠i} P(i,j)

    Args:
        target_distribution: Desired stationary distribution π*
        entropy_threshold: Max allowed steady-state entropy (None = no constraint)
        max_trials: Number of attempts before raising ValueError
        initial_matrix: Proposal matrix (None = random Dirichlet)

    Returns:
        np.ndarray: Transition matrix whose stationary distribution is π*

    Raises:
        ValueError: If no valid matrix found within max_trials
    """
    from scipy import stats as _stats

    pi_star = np.asarray(target_distribution)
    n = len(pi_star)

    for _ in range(max_trials):
        P_prop = (
            initial_matrix.copy()
            if initial_matrix is not None
            else _stats.dirichlet.rvs(np.ones(n), size=n)
        )

        P = np.zeros((n, n))
        eps = 1e-10
        for i in range(n):
            for j in range(n):
                if i != j:
                    ratio = (pi_star[j] + eps) / (pi_star[i] + eps)
                    P[i, j] = min(1, ratio) * P_prop[i, j]
            P[i, i] = 1 - np.sum(P[i, :])

        P = np.maximum(P, 0)
        P = P / P.sum(axis=1)[:, np.newaxis]

        if entropy_threshold is not None:
            ss = compute_steady_state(P)
            if _stats.entropy(ss, base=2) <= entropy_threshold:
                return P
        else:
            return P

    raise ValueError(
        f"Could not generate matrix with entropy <= {entropy_threshold} "
        f"after {max_trials} trials"
    )
