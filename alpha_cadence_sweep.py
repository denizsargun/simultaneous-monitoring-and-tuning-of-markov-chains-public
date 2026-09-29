# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Alpha x tuning-cadence sweep.

Runs a base YAML config over the grid

    environment.alpha.mean              in sweep.alpha_means     (constant, no decay)
    agent.tuning_policy.tuning_cadence  in sweep.tuning_cadences

The selected tuning is applied every `tuning_cadence` steps; the 'None' basis
matrix (no tuning) is applied on the steps in between. Each grid cell is a
regular `simulation.runner.run_from_yaml` call, so every episode is logged as
usual, and every aggregate below is computed from those logged episodes
(Tenet 1).

Outputs, in results/sweep_results/alpha_cadence_sweep_<timestamp>/:
    cell_alpha_<a>_cadence_<n>.yaml   exact config run for the cell
    cell_alpha_<a>_cadence_<n>.log    captured console output of the cell
    sweep_summary.csv                 one row per cell (means + standard errors,
                                      plus the cell's episode log directory)
    alpha_cadence_heatmaps.png        heatmap per metric (x: alpha mean, y: cadence)
    alpha_cadence_pattern_metrics.png time_to_reach pattern metrics, same axes

The sweep directory lives outside results/simulation_results/ so that
`Recorder.get_latest_log_dir` keeps returning episode log directories.

Usage:
    python3 alpha_cadence_sweep.py sample_configs/alpha_cadence_sweep.yaml

Per Tenet 3, every parameter is required — no defaults, no silent fallbacks.
"""

import contextlib
import copy
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from core.utils import get_timestamp
from simulation.runner import _load_yaml, _require, run_from_yaml


SWEEP_OUTPUT_BASE_DIR = "results/sweep_results"

# Chart palette: sequential blue ramp (light -> dark) and chart chrome.
SEQUENTIAL_BLUE_RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
CHART_SURFACE_COLOR = "#fcfcfb"
PRIMARY_INK_COLOR = "#0b0b0b"
SECONDARY_INK_COLOR = "#52514e"
MUTED_INK_COLOR = "#898781"
BASELINE_COLOR = "#c3c2b7"


# ============================================================
# CONFIG VALIDATION (Tenet 3)
# ============================================================


def _validate_sweep_values(sweep_config: Dict[str, Any]):
    """Return (alpha_means, tuning_cadences) from the `sweep` block, validated.

    alpha_means: strictly increasing list of >= 2 numbers in (0, 1].
    tuning_cadences: strictly increasing list of >= 2 positive integers.
    (Two values per axis are the minimum for a 2-D alpha x cadence map;
    increasing order keeps the heatmap axes monotone.)
    """
    alpha_means = _require(sweep_config, "alpha_means", "sweep")
    tuning_cadences = _require(sweep_config, "tuning_cadences", "sweep")

    if not isinstance(alpha_means, list) or len(alpha_means) < 2:
        raise ValueError(
            f"sweep.alpha_means must be a list of >= 2 values, got {alpha_means!r}"
        )
    for alpha_mean in alpha_means:
        if isinstance(alpha_mean, bool) or not isinstance(alpha_mean, (int, float)):
            raise TypeError(
                f"sweep.alpha_means entries must be numbers, got {alpha_mean!r}"
            )
        if not 0 < alpha_mean <= 1:
            raise ValueError(
                f"sweep.alpha_means entries must be in (0, 1], got {alpha_mean!r}"
            )
    if any(later <= earlier for earlier, later in zip(alpha_means, alpha_means[1:])):
        raise ValueError(
            f"sweep.alpha_means must be strictly increasing, got {alpha_means!r}"
        )

    if not isinstance(tuning_cadences, list) or len(tuning_cadences) < 2:
        raise ValueError(
            "sweep.tuning_cadences must be a list of >= 2 values, got "
            f"{tuning_cadences!r}"
        )
    for tuning_cadence in tuning_cadences:
        if isinstance(tuning_cadence, bool) or not isinstance(tuning_cadence, int):
            raise TypeError(
                "sweep.tuning_cadences entries must be integers, got "
                f"{tuning_cadence!r}"
            )
        if tuning_cadence < 1:
            raise ValueError(
                "sweep.tuning_cadences entries must be >= 1, got " f"{tuning_cadence!r}"
            )
    if any(
        later <= earlier for earlier, later in zip(tuning_cadences, tuning_cadences[1:])
    ):
        raise ValueError(
            "sweep.tuning_cadences must be strictly increasing, got "
            f"{tuning_cadences!r}"
        )

    return [float(alpha_mean) for alpha_mean in alpha_means], list(tuning_cadences)


def _validate_base_config(base_config: Dict[str, Any]) -> None:
    """Check the base config is compatible with the sweep. Raises on mismatch.

    - alpha is constant with zero decay (the sweep studies alpha without decay)
    - alpha.mean and tuning_cadence are absent (the sweep owns them)
    - tuning_space is 'data' (cadence off-steps need the 'None' basis entry)
    - number_of_runs >= 2 (per-cell standard errors need two runs)
    - output.plot is 0 (per-episode plots open a blocking window)
    - metrics.state_rates is on and metrics.time_to_reach.pattern is present
      (both are summarized per cell)
    """
    environment_config = _require(base_config, "environment")
    alpha_config = _require(environment_config, "alpha", "environment")
    alpha_type = _require(alpha_config, "type", "environment.alpha")
    if alpha_type != "constant":
        raise ValueError(
            f"environment.alpha.type must be 'constant' for this sweep, got "
            f"{alpha_type!r}"
        )
    alpha_decay_rate = _require(alpha_config, "decay_rate", "environment.alpha")
    if alpha_decay_rate != 0.0:
        raise ValueError(
            "environment.alpha.decay_rate must be 0.0 for this sweep, got "
            f"{alpha_decay_rate!r}"
        )
    if "mean" in alpha_config:
        raise ValueError(
            "environment.alpha.mean must NOT be set in the base config; it is "
            "set per cell from sweep.alpha_means."
        )

    agent_config = _require(base_config, "agent")
    tuning_policy_config = _require(agent_config, "tuning_policy", "agent")
    tuning_space = _require(tuning_policy_config, "tuning_space", "agent.tuning_policy")
    if tuning_space != "data":
        raise ValueError(
            "agent.tuning_policy.tuning_space must be 'data' for this sweep (the "
            f"cadence off-steps apply the 'None' basis entry), got {tuning_space!r}"
        )
    if "tuning_cadence" in tuning_policy_config:
        raise ValueError(
            "agent.tuning_policy.tuning_cadence must NOT be set in the base "
            "config; it is set per cell from sweep.tuning_cadences."
        )

    number_of_runs = _require(environment_config, "number_of_runs", "environment")
    if isinstance(number_of_runs, bool) or not isinstance(number_of_runs, int):
        raise TypeError(
            f"environment.number_of_runs must be an integer, got {number_of_runs!r}"
        )
    if number_of_runs < 2:
        raise ValueError(
            "environment.number_of_runs must be >= 2 for per-cell standard "
            f"errors, got {number_of_runs!r}"
        )
    _require(environment_config, "number_of_steps", "environment")

    output_config = _require(base_config, "output")
    if _require(output_config, "plot", "output"):
        raise ValueError(
            "output.plot must be 0 for a sweep: per-episode plots open a "
            "blocking window for every cell."
        )

    metrics_config = _require(base_config, "metrics")
    if not _require(metrics_config, "state_rates", "metrics"):
        raise ValueError("metrics.state_rates must be 1: the sweep summarizes it.")
    time_to_reach_config = _require(metrics_config, "time_to_reach", "metrics")
    if not isinstance(time_to_reach_config, dict):
        raise TypeError(
            "metrics.time_to_reach must be a dict with a 'pattern' entry, got "
            f"{time_to_reach_config!r}"
        )
    _require(time_to_reach_config, "pattern", "metrics.time_to_reach")


def _build_cell_config(
    base_config: Dict[str, Any], alpha_mean: float, tuning_cadence: int
) -> Dict[str, Any]:
    """Return a copy of the base config with the cell's alpha mean and cadence."""
    cell_config = copy.deepcopy(base_config)
    cell_config["environment"]["alpha"]["mean"] = alpha_mean
    cell_config["agent"]["tuning_policy"]["tuning_cadence"] = tuning_cadence
    return cell_config


# ============================================================
# PER-CELL AGGREGATION (from logged episodes, Tenet 1)
# ============================================================


def _mean_and_standard_error(values: List[float]):
    """Sample mean and standard error of the mean (needs >= 2 values)."""
    values_array = np.asarray(values, dtype=float)
    if values_array.size < 2:
        raise ValueError(f"Standard error needs >= 2 values, got {values_array.size}.")
    mean_value = float(values_array.mean())
    standard_error = float(values_array.std(ddof=1) / np.sqrt(values_array.size))
    return mean_value, standard_error


def _summarize_cell(
    run_output: Dict[str, Any], alpha_mean: float, tuning_cadence: int
) -> Dict[str, Any]:
    """Aggregate one cell's logged episodes into a flat summary row."""
    recorder = run_output["logger"]
    episodes = recorder.episodes
    runs = run_output["runs"]
    logged_config = recorder.full_config["config"]
    state_names = logged_config["state_names"]
    tuning_matrix_names = logged_config["tuning_matrix_names"]
    no_tuning_index = tuning_matrix_names.index("None")
    number_of_steps = run_output["config"]["environment"]["number_of_steps"]

    realized_alpha_values = [
        step["alpha_value"] for episode in episodes for step in episode["steps"]
    ]
    applied_tuning_step_fractions = [
        float(
            np.mean(
                [
                    step["tuning_weights"][no_tuning_index] != 1.0
                    for step in episode["steps"]
                ]
            )
        )
        for episode in episodes
    ]

    summary_row = {
        "alpha_mean": alpha_mean,
        "tuning_cadence": tuning_cadence,
        "number_of_runs": len(runs),
        "realized_alpha_mean": float(np.mean(realized_alpha_values)),
        "applied_tuning_step_fraction": float(np.mean(applied_tuning_step_fractions)),
    }

    # Pattern arrival times: None = the run never matched within the horizon.
    arrival_times = [run["requested_metrics"]["time_to_reach"] for run in runs]
    matched_indicators = [arrival is not None for arrival in arrival_times]
    matched_fraction = float(np.mean(matched_indicators))
    summary_row["time_to_reach_matched_fraction"] = matched_fraction
    summary_row["time_to_reach_matched_fraction_standard_error"] = float(
        np.sqrt(matched_fraction * (1.0 - matched_fraction) / len(runs))
    )
    # Restricted mean: unmatched runs are censored at the horizon.
    censored_arrival_times = [
        arrival if arrival is not None else number_of_steps for arrival in arrival_times
    ]
    (
        summary_row["restricted_mean_time_to_reach"],
        summary_row["restricted_mean_time_to_reach_standard_error"],
    ) = _mean_and_standard_error(censored_arrival_times)

    for state_index, state_name in enumerate(state_names):
        (
            summary_row[f"state_rate_{state_name}_mean"],
            summary_row[f"state_rate_{state_name}_standard_error"],
        ) = _mean_and_standard_error(
            [run["requested_metrics"]["state_rates"][state_index] for run in runs]
        )

    scalar_metric_names = [
        metric_name
        for metric_name in runs[0]["requested_metrics"]
        if metric_name not in ("state_rates", "time_to_reach")
    ]
    for metric_name in scalar_metric_names:
        metric_values = [run["requested_metrics"][metric_name] for run in runs]
        if any(metric_value is None for metric_value in metric_values):
            raise ValueError(
                f"Requested metric {metric_name!r} is None for at least one run "
                f"(alpha_mean={alpha_mean}, tuning_cadence={tuning_cadence})."
            )
        (
            summary_row[f"{metric_name}_mean"],
            summary_row[f"{metric_name}_standard_error"],
        ) = _mean_and_standard_error(metric_values)

    summary_row["cell_output_dir"] = str(run_output["output_dir"])
    return summary_row


# ============================================================
# PLOTS
# ============================================================


def _style_axes(axes) -> None:
    """Recessive chrome: muted ticks, hairline spines, surface background."""
    axes.set_facecolor(CHART_SURFACE_COLOR)
    axes.tick_params(colors=MUTED_INK_COLOR, labelcolor=SECONDARY_INK_COLOR)
    for spine in axes.spines.values():
        spine.set_color(BASELINE_COLOR)


def _cell_edges(center_values: List[float]) -> np.ndarray:
    """Cell boundaries on a linear axis: midpoints between consecutive centers,
    extended by half the neighbouring gap at both ends (needs >= 2 centers)."""
    centers = np.asarray(center_values, dtype=float)
    midpoints = (centers[:-1] + centers[1:]) / 2
    first_edge = centers[0] - (midpoints[0] - centers[0])
    last_edge = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate(([first_edge], midpoints, [last_edge]))


def _heatmap_panel_columns(summary_rows: List[Dict[str, Any]]) -> List[str]:
    """Summary-row columns shown as heatmaps: every per-cell mean/fraction."""
    excluded_columns = {
        "alpha_mean",
        "tuning_cadence",
        "number_of_runs",
        "cell_output_dir",
    }
    return [
        column
        for column in summary_rows[0]
        if column not in excluded_columns and not column.endswith("_standard_error")
    ]


def _draw_metric_heatmap(
    figure,
    axes,
    summary_rows: List[Dict[str, Any]],
    alpha_means: List[float],
    tuning_cadences: List[int],
    column: str,
    title: str,
    title_fontsize: int,
) -> None:
    """Heatmap of one summary column: x = alpha mean (linear), y = tuning cadence.

    Each panel is scaled to its own min/max; the colorbar carries the values.
    """
    row_lookup = {
        (row["alpha_mean"], row["tuning_cadence"]): row for row in summary_rows
    }
    values = np.array(
        [
            [
                row_lookup[(alpha_mean, tuning_cadence)][column]
                for alpha_mean in alpha_means
            ]
            for tuning_cadence in tuning_cadences
        ],
        dtype=float,
    )
    mesh = axes.pcolormesh(
        _cell_edges(alpha_means),
        _cell_edges(tuning_cadences),
        values,
        cmap=LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE_RAMP),
        shading="flat",
    )
    colorbar = figure.colorbar(mesh, ax=axes)
    colorbar.outline.set_edgecolor(BASELINE_COLOR)
    colorbar.ax.tick_params(colors=MUTED_INK_COLOR, labelcolor=SECONDARY_INK_COLOR)
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_xlabel("alpha mean (constant)", fontsize=9, color=SECONDARY_INK_COLOR)
    axes.set_ylabel("tuning cadence (steps)", fontsize=9, color=SECONDARY_INK_COLOR)
    axes.set_title(title, fontsize=title_fontsize, color=PRIMARY_INK_COLOR)
    _style_axes(axes)


def _plot_heatmaps(
    summary_rows: List[Dict[str, Any]],
    alpha_means: List[float],
    tuning_cadences: List[int],
    output_path: Path,
) -> None:
    """One heatmap per summary metric (x = alpha mean, y = tuning cadence)."""
    panel_columns = _heatmap_panel_columns(summary_rows)
    number_of_columns = 4
    number_of_rows = int(np.ceil(len(panel_columns) / number_of_columns))

    figure, axes_grid = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(5.2 * number_of_columns, 3.8 * number_of_rows),
        squeeze=False,
    )
    figure.patch.set_facecolor(CHART_SURFACE_COLOR)
    for panel_index, column in enumerate(panel_columns):
        _draw_metric_heatmap(
            figure,
            axes_grid[panel_index // number_of_columns][
                panel_index % number_of_columns
            ],
            summary_rows,
            alpha_means,
            tuning_cadences,
            column=column,
            title=column.replace("_", " "),
            title_fontsize=10,
        )
    for unused_panel_index in range(
        len(panel_columns), number_of_rows * number_of_columns
    ):
        axes_grid[unused_panel_index // number_of_columns][
            unused_panel_index % number_of_columns
        ].axis("off")

    figure.suptitle(
        "Alpha x tuning cadence sweep — per-cell means (darker = larger)",
        fontsize=13,
        color=PRIMARY_INK_COLOR,
    )
    figure.tight_layout()
    figure.savefig(
        output_path, dpi=200, bbox_inches="tight", facecolor=CHART_SURFACE_COLOR
    )
    plt.close(figure)


def _plot_pattern_metrics(
    summary_rows: List[Dict[str, Any]],
    alpha_means: List[float],
    tuning_cadences: List[int],
    time_to_reach_pattern: str,
    number_of_steps: int,
    output_path: Path,
) -> None:
    """The two time_to_reach pattern metrics, x = alpha mean, y = tuning cadence."""
    panels = [
        (
            "restricted_mean_time_to_reach",
            f"Restricted mean time to {time_to_reach_pattern} "
            f"(steps, censored at {number_of_steps})",
        ),
        (
            "time_to_reach_matched_fraction",
            f"Fraction of runs matching {time_to_reach_pattern} "
            f"within states 0..{number_of_steps - 1}",
        ),
    ]
    figure, axes_list = plt.subplots(1, len(panels), figsize=(15, 5.5))
    figure.patch.set_facecolor(CHART_SURFACE_COLOR)
    for axes, (column, title) in zip(axes_list, panels):
        _draw_metric_heatmap(
            figure,
            axes,
            summary_rows,
            alpha_means,
            tuning_cadences,
            column=column,
            title=title,
            title_fontsize=11,
        )
    figure.tight_layout()
    figure.savefig(
        output_path, dpi=200, bbox_inches="tight", facecolor=CHART_SURFACE_COLOR
    )
    plt.close(figure)


# ============================================================
# PUBLIC API
# ============================================================


def run_alpha_cadence_sweep(yaml_path: str) -> Dict[str, Any]:
    """Run the alpha x tuning-cadence grid described by `yaml_path`.

    Returns:
        Dict with `summary_rows` (one dict per cell), `sweep_output_dir`,
        `summary_csv_path`, `heatmaps_path`, and `pattern_metrics_path`.
    """
    top_level_config = _load_yaml(yaml_path)
    sweep_config = _require(top_level_config, "sweep")
    alpha_means, tuning_cadences = _validate_sweep_values(sweep_config)
    base_config = {
        key: value for key, value in top_level_config.items() if key != "sweep"
    }
    _validate_base_config(base_config)

    sweep_output_dir = (
        Path(SWEEP_OUTPUT_BASE_DIR) / f"alpha_cadence_sweep_{get_timestamp()}"
    )
    sweep_output_dir.mkdir(parents=True, exist_ok=False)
    number_of_cells = len(alpha_means) * len(tuning_cadences)
    print(
        f"Alpha x cadence sweep: {len(alpha_means)} alpha means x "
        f"{len(tuning_cadences)} cadences = {number_of_cells} cells, "
        f"{base_config['environment']['number_of_runs']} runs each"
    )
    print(f"Sweep output: {sweep_output_dir}")

    summary_rows = []
    cell_counter = 0
    for tuning_cadence in tuning_cadences:
        for alpha_mean in alpha_means:
            cell_counter += 1
            cell_name = f"cell_alpha_{alpha_mean:g}_cadence_{tuning_cadence}"
            cell_yaml_path = sweep_output_dir / f"{cell_name}.yaml"
            with open(cell_yaml_path, "w") as cell_yaml_file:
                yaml.safe_dump(
                    _build_cell_config(base_config, alpha_mean, tuning_cadence),
                    cell_yaml_file,
                    sort_keys=False,
                )
            cell_log_path = sweep_output_dir / f"{cell_name}.log"
            with open(cell_log_path, "w") as cell_log_file, contextlib.redirect_stdout(
                cell_log_file
            ):
                run_output = run_from_yaml(str(cell_yaml_path))
            summary_row = _summarize_cell(run_output, alpha_mean, tuning_cadence)
            summary_rows.append(summary_row)
            print(
                f"  [{cell_counter}/{number_of_cells}] alpha_mean={alpha_mean:g} "
                f"cadence={tuning_cadence}: restricted mean time to reach = "
                f"{summary_row['restricted_mean_time_to_reach']:.1f}, "
                f"matched = {summary_row['time_to_reach_matched_fraction']:.2f}, "
                f"realized alpha = {summary_row['realized_alpha_mean']:.3f}"
            )

    summary_csv_path = sweep_output_dir / "sweep_summary.csv"
    with open(summary_csv_path, "w", newline="") as summary_csv_file:
        csv_writer = csv.DictWriter(summary_csv_file, fieldnames=list(summary_rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(summary_rows)

    heatmaps_path = sweep_output_dir / "alpha_cadence_heatmaps.png"
    _plot_heatmaps(summary_rows, alpha_means, tuning_cadences, heatmaps_path)
    pattern_metrics_path = sweep_output_dir / "alpha_cadence_pattern_metrics.png"
    _plot_pattern_metrics(
        summary_rows,
        alpha_means,
        tuning_cadences,
        time_to_reach_pattern=base_config["metrics"]["time_to_reach"]["pattern"],
        number_of_steps=base_config["environment"]["number_of_steps"],
        output_path=pattern_metrics_path,
    )

    print(f"\nSummary CSV:     {summary_csv_path}")
    print(f"Heatmaps:        {heatmaps_path}")
    print(f"Pattern metrics: {pattern_metrics_path}")
    return {
        "summary_rows": summary_rows,
        "sweep_output_dir": sweep_output_dir,
        "summary_csv_path": summary_csv_path,
        "heatmaps_path": heatmaps_path,
        "pattern_metrics_path": pattern_metrics_path,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 alpha_cadence_sweep.py <sweep_yaml_path>")
        print(
            "  e.g., python3 alpha_cadence_sweep.py "
            "sample_configs/alpha_cadence_sweep.yaml"
        )
        sys.exit(1)
    run_alpha_cadence_sweep(sys.argv[1])
