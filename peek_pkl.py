# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Peek at a T_additive_complete_*.pkl file without running a simulation.

Prints a summary:
    - number_of_states, state_names, state_names_short
    - number_of_categories, category_names
    - number_of_actions, action_names (all unique actions in the pkl)
    - Per-category available actions
    - Simulation-interval-days metadata (if present)

Usage:
    python3 peek_pkl.py <path-to-pkl>

Example:
    python3 peek_pkl.py data/T_additive_complete_nsclc_targ_pdl1.pkl
"""

from typing import Any, Dict, List
import pickle
import sys


def peek_pkl(pkl_path: str) -> Dict[str, Any]:
    """Load a T_additive_complete pkl and return a summary dict.

    The pkl is expected to be a list of dicts with keys such as:
        category, action, P_0, T, states, states_short, simulation_interval_days,
        stratification_mode

    Returns:
        Summary dict with number_of_states, number_of_categories,
        number_of_actions, state_names, category_names, action_names,
        per_category_actions, simulation_interval_days.
    """
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    if not isinstance(results, list) or not results:
        raise ValueError(
            f"pkl at {pkl_path} is not a non-empty list; got {type(results).__name__}"
        )

    first_entry = results[0]
    if "P_0" not in first_entry:
        raise ValueError(
            f"First entry in {pkl_path} has no 'P_0' key. Keys: {list(first_entry)}"
        )

    P_0 = first_entry["P_0"]
    number_of_states = P_0.shape[0]

    state_names = first_entry.get("states")
    state_names_short = first_entry.get("states_short")

    # Collect unique categories and actions in order of appearance
    category_names: List[str] = []
    for entry in results:
        category_name = entry["category"]
        if category_name not in category_names:
            category_names.append(category_name)
    number_of_categories = len(category_names)

    action_names: List[str] = []
    for entry in results:
        action_name = entry["action"]
        if action_name not in action_names:
            action_names.append(action_name)
    number_of_actions = len(action_names)

    # Per-category available actions (map: category → list of actions)
    per_category_actions: Dict[str, List[str]] = {c: [] for c in category_names}
    for entry in results:
        per_category_actions[entry["category"]].append(entry["action"])

    simulation_interval_days = first_entry.get("simulation_interval_days")

    return {
        "pkl_path": pkl_path,
        "number_of_states": number_of_states,
        "state_names": state_names,
        "state_names_short": state_names_short,
        "number_of_categories": number_of_categories,
        "category_names": category_names,
        "number_of_actions": number_of_actions,
        "action_names": action_names,
        "per_category_actions": per_category_actions,
        "simulation_interval_days": simulation_interval_days,
        "number_of_entries": len(results),
    }


def print_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print the peek_pkl summary."""
    print("=" * 60)
    print(f"pkl summary — {summary['pkl_path']}")
    print("=" * 60)
    print(f"  number_of_entries        = {summary['number_of_entries']}")
    print(f"  number_of_states         = {summary['number_of_states']}")
    print(f"  state_names              = {summary['state_names']}")
    print(f"  state_names_short        = {summary['state_names_short']}")
    print(f"  number_of_categories     = {summary['number_of_categories']}")
    print(f"  category_names           = {summary['category_names']}")
    print(f"  number_of_actions        = {summary['number_of_actions']}")
    print(f"  action_names             = {summary['action_names']}")
    print(f"  simulation_interval_days = {summary['simulation_interval_days']}")
    print()
    print("  Per-category available actions:")
    for category_name, actions in summary["per_category_actions"].items():
        print(f"    {category_name}: {actions}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 peek_pkl.py <path-to-pkl>")
        sys.exit(1)
    summary = peek_pkl(sys.argv[1])
    print_summary(summary)
