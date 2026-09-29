# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Time-to-reach-pattern utilities.

Two public entry points:

1. `transition_matrix_expected_time_to_absorption(transition_matrix)` —
   closed-form expected hitting time to the (unique) absorbing state of a
   transition matrix.

2. `simulate_time_to_reach_pattern(yaml_path, number_of_runs_override)` —
   wrapper around `simulation.runner.run_from_yaml` that runs the YAML N times
   and reports the empirical distribution of `time_to_reach.pattern` from the
   YAML's metrics.time_to_reach block. Only episodes whose state trace matches
   the regex pattern (with exactly one capture group) are counted; others are
   reported as excluded.

Per Tenet 3, all params are required — no defaults, no silent fallbacks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import copy
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml


# ============================================================
# CLOSED-FORM: expected time to absorption of a fixed transition matrix
# ============================================================


def transition_matrix_expected_time_to_absorption(transition_matrix):
    """Expected remaining steps from each state until the (unique) absorbing state.

    Solves the linear system `(I - P) v = r` where
        r_i = 1 for transient states, 0 for the absorbing state.

    Per Tenet 3, requires EXACTLY one absorbing state — zero or multiple absorbing
    states raise `ValueError` (no silent fallback to "last state = terminal").

    Args:
        transition_matrix: n × n row-stochastic matrix.

    Returns:
        Tuple `(expected_hitting_time_vector, absorbing_state_index)` where:
          - expected_hitting_time_vector: length-n array of expected steps
          - absorbing_state_index: int identifying the absorbing row
    """
    number_of_states = transition_matrix.shape[0]
    absorbing_state_indices = [
        i for i in range(number_of_states) if np.isclose(transition_matrix[i, i], 1.0)
    ]
    if len(absorbing_state_indices) == 0:
        raise ValueError(
            "No absorbing state found (no row with P[i,i]==1). Per Tenet 3, "
            "no silent fallback to 'last state = terminal'."
        )
    if len(absorbing_state_indices) > 1:
        raise ValueError(
            "Multiple absorbing states found at indices "
            f"{absorbing_state_indices}. Expected exactly one."
        )
    absorbing_state_index = absorbing_state_indices[0]

    reward_vector = np.ones(number_of_states)
    reward_vector[absorbing_state_index] = 0.0

    linear_system_matrix = np.eye(number_of_states) - transition_matrix
    linear_system_matrix[absorbing_state_index, :] = 0.0
    linear_system_matrix[absorbing_state_index, absorbing_state_index] = 1.0

    expected_hitting_time_vector = np.linalg.solve(linear_system_matrix, reward_vector)
    return expected_hitting_time_vector, absorbing_state_index


# ============================================================
# EMPIRICAL: pattern-based arrival times over N Monte Carlo runs
# ============================================================


def _plot_arrival_times_cdf(
    arrival_times: List[Optional[int]],
    number_of_steps: int,
    plot_label: str,
    output_dir: Path,
) -> Optional[Path]:
    """Plot CDF + survival of matched arrival times.

    Excluded episodes (None) are not counted (they don't appear in either curve).
    Returns the saved PNG path (or None if no matched episodes).
    """
    from core.utils import get_timestamp

    matched_times = np.array(
        sorted(t for t in arrival_times if t is not None), dtype=float
    )
    number_matched = matched_times.size
    if number_matched == 0:
        return None

    cumulative_distribution_function = np.arange(1, number_matched + 1) / number_matched

    # Area under each curve from time step 0 to `number_of_steps`, above y=0.
    #   AUC_CDF      = ∫_0^N F(t) dt      = mean(N - t_i)
    #   AUC_survival = ∫_0^N (1 - F(t)) dt = N - AUC_CDF   (= mean matched time)
    # Computed over MATCHED episodes only — the same set the curves are drawn from.
    area_under_cdf = float(np.sum(number_of_steps - matched_times) / number_matched)
    area_under_survival = float(number_of_steps - area_under_cdf)

    # Extend the step curves across the full [0, number_of_steps] window so the
    # shaded region matches the reported AUC: F=0 before the first arrival and
    # stays flat (at its last value) after the last arrival.
    area_x = np.concatenate(([0.0], matched_times, [float(number_of_steps)]))
    cdf_y = np.concatenate(
        (
            [0.0],
            cumulative_distribution_function,
            [cumulative_distribution_function[-1]],
        )
    )
    survival_y = np.concatenate(
        (
            [1.0],
            1.0 - cumulative_distribution_function,
            [1.0 - cumulative_distribution_function[-1]],
        )
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.step(
        matched_times,
        cumulative_distribution_function,
        where="post",
        linewidth=2,
        color="#1f77b4",
    )
    ax1.axvline(
        x=number_of_steps,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Horizon ({number_of_steps})",
    )
    ax1.set_xlabel("Time Step", fontsize=11)
    ax1.set_ylabel("P(reached)", fontsize=11)
    ax1.set_title(
        f"{plot_label} — CDF (matched n={number_matched})",
        fontsize=12,
    )
    ax1.set_xlim(0, number_of_steps)
    ax1.set_ylim(0, 1.05)
    ax1.fill_between(area_x, cdf_y, step="post", alpha=0.15, color="#1f77b4")
    ax1.text(
        0.97,
        0.05,
        f"AUC (0–{number_of_steps}) = {area_under_cdf:.1f}",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="#1f77b4", alpha=0.85),
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.step(
        matched_times,
        1.0 - cumulative_distribution_function,
        where="post",
        linewidth=2,
        color="#d62728",
    )
    ax2.axvline(
        x=number_of_steps,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Horizon ({number_of_steps})",
    )
    ax2.set_xlabel("Time Step", fontsize=11)
    ax2.set_ylabel("P(not yet reached)", fontsize=11)
    ax2.set_title(f"{plot_label} — Survival", fontsize=12)
    ax2.set_xlim(0, number_of_steps)
    ax2.set_ylim(-0.05, 1.05)
    ax2.fill_between(area_x, survival_y, step="post", alpha=0.15, color="#d62728")
    ax2.text(
        0.97,
        0.95,
        f"AUC (0–{number_of_steps}) = {area_under_survival:.1f}",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="#d62728", alpha=0.85),
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    filename = output_dir / f"time_to_reach_pattern_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved CDF plot to: {filename}")
    return filename


def simulate_time_to_reach_pattern(
    yaml_path: str,
    number_of_runs_override: int,
) -> Dict[str, Any]:
    """Run a YAML config N times and report pattern-based arrival times.

    Loads the YAML, overrides `environment.number_of_runs` to
    `number_of_runs_override`, invokes `simulation.runner.run_from_yaml`, and
    collects the per-episode `time_to_reach.pattern` value from each run.

    Only episodes whose state trace matches the regex pattern are included.
    Others are counted as excluded (never included in the mean/CDF).

    All arguments required (Tenet 3).

    Args:
        yaml_path: Path to a YAML config file that includes a
            `metrics.time_to_reach.pattern` regex.
        number_of_runs_override: Number of Monte Carlo runs.

    Returns:
        Dict with:
          - arrival_times: list length N; entries are step indices or None
          - user_pattern: the regex string from the YAML
          - number_matched: episodes that matched the pattern
          - number_excluded: episodes that did not
          - mean/median/min/max: over matched episodes
          - output_dir: pathlib.Path to the Recorder log directory
          - cdf_plot_path: pathlib.Path to the saved CDF PNG (or None)
    """
    import tempfile
    from simulation.runner import _load_yaml, run_from_yaml

    top_level_config = copy.deepcopy(_load_yaml(yaml_path))
    top_level_config["environment"]["number_of_runs"] = number_of_runs_override

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_yaml:
        yaml.safe_dump(top_level_config, temp_yaml)
        temp_yaml_path = temp_yaml.name

    run_output = run_from_yaml(temp_yaml_path)
    output_dir = Path(run_output["output_dir"])

    # Extract per-episode pattern-based times from the requested_metrics that
    # each run stored (see simulation.runner._extract_requested_metrics).
    arrival_times: List[Optional[int]] = []
    for run_result in run_output["runs"]:
        requested = run_result.get("requested_metrics", {})
        arrival_times.append(requested.get("time_to_reach"))

    user_pattern = top_level_config["metrics"]["time_to_reach"]["pattern"]
    number_of_steps = top_level_config["environment"]["number_of_steps"]
    plot_label = f"time_to_reach.pattern: {user_pattern}"

    matched_times = [t for t in arrival_times if t is not None]
    number_matched = len(matched_times)
    number_excluded = len(arrival_times) - number_matched

    # Area under the CDF curve from step 0 to number_of_steps (above y=0), over
    # matched episodes — same quantity annotated on the plot.
    area_under_cdf = (
        float(np.sum(number_of_steps - np.array(matched_times)) / number_matched)
        if matched_times
        else None
    )
    area_under_survival = (
        float(number_of_steps - area_under_cdf) if matched_times else None
    )

    cdf_plot_path = _plot_arrival_times_cdf(
        arrival_times=arrival_times,
        number_of_steps=number_of_steps,
        plot_label=plot_label,
        output_dir=output_dir,
    )

    print(f"\n=== time_to_reach.pattern Summary ===")
    print(f"  pattern:               {user_pattern}")
    print(f"  number_of_runs:        {number_of_runs_override}")
    print(f"  number_matched:        {number_matched}")
    print(f"  number_excluded:       {number_excluded}")
    if matched_times:
        matched_array = np.array(matched_times)
        print(f"  mean arrival time:     {matched_array.mean():.2f}")
        print(f"  median arrival time:   {float(np.median(matched_array)):.2f}")
        print(f"  min arrival time:      {int(matched_array.min())}")
        print(f"  max arrival time:      {int(matched_array.max())}")
        print(f"  AUC under CDF (0-{number_of_steps}):  {area_under_cdf:.2f}")
        print(f"  AUC under survival:    {area_under_survival:.2f}")
    print(f"  Output directory:      {output_dir}")
    if cdf_plot_path is not None:
        print(f"  CDF plot:              {cdf_plot_path}")

    return {
        "arrival_times": arrival_times,
        "user_pattern": user_pattern,
        "number_matched": number_matched,
        "number_excluded": number_excluded,
        "mean_arrival_time": (float(np.mean(matched_times)) if matched_times else None),
        "median_arrival_time": (
            float(np.median(matched_times)) if matched_times else None
        ),
        "min_arrival_time": (int(np.min(matched_times)) if matched_times else None),
        "max_arrival_time": (int(np.max(matched_times)) if matched_times else None),
        "area_under_cdf": area_under_cdf,
        "area_under_survival": area_under_survival,
        "output_dir": output_dir,
        "cdf_plot_path": cdf_plot_path,
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python3 plot_time_to_reach_pattern.py <yaml_path> <number_of_runs>"
        )
        print(
            "  e.g., python3 plot_time_to_reach_pattern.py "
            "sample_configs/sample_run.yaml 100"
        )
        sys.exit(1)

    yaml_path_arg = sys.argv[1]
    number_of_runs_arg = int(sys.argv[2])
    simulate_time_to_reach_pattern(
        yaml_path=yaml_path_arg,
        number_of_runs_override=number_of_runs_arg,
    )
