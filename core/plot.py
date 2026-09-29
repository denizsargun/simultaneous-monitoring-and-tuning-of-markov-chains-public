# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Plot simulation results from logged episode data (Tenet #1).

All data comes from the episode dict — no live objects needed.
Per Tenet 3, all parameters are required and no silent fallbacks are allowed.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.utils import get_timestamp, compute_steady_state


def plot_episode(episode, config, output_dir, show):
    """Generate 12-subplot visualization from a logged episode.

    All parameters required (Tenet 3).

    Args:
        episode: Episode dict from Recorder.
        config: Config dict. Required — must contain
            entropy_threshold, desired_distribution, state_names,
            category_names, tuning_matrix_names.
        output_dir: Directory to save plot (Path or str). Required.
        show: Whether to call plt.show() (bool). Required.

    Returns:
        matplotlib Figure
    """
    steps = episode["steps"]
    number_of_steps = len(steps)
    number_of_states = episode["number_of_states"]
    number_of_categories = episode["number_of_categories"]
    true_category = episode["true_category"]
    time_steps = range(number_of_steps)

    # Extract data from logged steps
    state_history = [step["true_state_before"] for step in steps]
    decisions = [step["measurement_decision"] for step in steps]
    entropies = [step["entropy"] for step in steps]
    # Per-step cost from log (already the sum of measurement + switching costs).
    step_costs = [step["cost"] for step in steps]

    belief_matrices = [
        step["belief_matrix"] for step in steps if step["belief_matrix"] is not None
    ]
    current_transition_matrices_before = [
        step["env_P_matrix_before"]
        for step in steps
        if step["env_P_matrix_before"] is not None
    ]
    current_transition_matrices_after = [
        step["env_P_matrix_after"]
        for step in steps
        if step["env_P_matrix_after"] is not None
    ]
    alpha_values = [step["alpha_value"] for step in steps]

    # Config params — required per Tenet 3.
    entropy_threshold = config["entropy_threshold"]
    desired_distribution = config["desired_distribution"]
    state_labels = config["state_names"]
    if state_labels is None:
        raise KeyError(
            "config['state_names'] is None — required to label plots. "
            "Per Tenet 3, no silent fallback."
        )
    if len(state_labels) != number_of_states:
        raise ValueError(
            f"state_names has {len(state_labels)} entries but episode has "
            f"{number_of_states} states. These must match."
        )

    # Visualization: map each state index to its y-axis display position.
    # Per Tenet 3, this must be provided (no default).
    state_y_axis_positions_by_index = config["state_y_axis_positions_by_index"]
    if state_y_axis_positions_by_index is None:
        raise KeyError(
            "config['state_y_axis_positions_by_index'] is None — required to "
            "arrange state y-axis positions. Per Tenet 3, no silent fallback."
        )
    if len(state_y_axis_positions_by_index) != number_of_states:
        raise ValueError(
            f"state_y_axis_positions_by_index has "
            f"{len(state_y_axis_positions_by_index)} entries but episode has "
            f"{number_of_states} states. These must match."
        )
    # Inverse mapping: y_position → state_index. Used to figure out which
    # state name to show at each y-axis tick.
    state_index_by_y_position = [0] * number_of_states
    for state_index, y_position in enumerate(state_y_axis_positions_by_index):
        state_index_by_y_position[y_position] = state_index
    # Y-axis tick labels — states in y-position order.
    y_axis_tick_labels = [
        state_labels[state_index_by_y_position[position]]
        for position in range(number_of_states)
    ]

    fig, axes = plt.subplots(4, 3, figsize=(26, 26))
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]
    ax7, ax8, ax9 = axes[2]
    ax10, ax11, ax12 = axes[3]

    # (1,1) True State — map each true_state_before through the y-position map
    state_history_y_positions = [
        state_y_axis_positions_by_index[state_index] for state_index in state_history
    ]
    ax1.plot(time_steps, state_history_y_positions, "b-", linewidth=1)
    ax1.plot(time_steps, state_history_y_positions, "k.", markersize=2)
    ax1.set_title("True State Over Time", fontsize=12)
    ax1.set_ylabel("State", fontsize=7)
    ax1.set_ylim(-0.5, number_of_states - 0.5)
    ax1.set_yticks(range(number_of_states))
    ax1.set_yticklabels(y_axis_tick_labels)
    ax1.set_xlim(0, number_of_steps)
    ax1.set_xlabel("Time Step", fontsize=7)
    ax1.grid(True)
    ax1.tick_params(axis="both", which="major", labelsize=6)

    # (1,2) Marginalized State Belief — reorder rows by y-position
    if belief_matrices:
        beliefs = np.array(belief_matrices)
        marginal_state_beliefs = beliefs.sum(axis=1)
        row_sums = marginal_state_beliefs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        marginal_state_beliefs = marginal_state_beliefs / row_sums
        # Reorder columns of marginal_state_beliefs by y-position so that when
        # we transpose for imshow, each row `y` shows the state at position y.
        marginal_state_beliefs_reordered = marginal_state_beliefs[
            :, state_index_by_y_position
        ]
        image_belief = ax2.imshow(
            marginal_state_beliefs_reordered.T,
            aspect="auto",
            interpolation="none",
            origin="lower",
            vmin=0,
            vmax=1,
            extent=[0, number_of_steps, -0.5, number_of_states - 0.5],
        )
        ax2.set_title("Belief State Evolution", fontsize=12)
        ax2.set_ylabel("State", fontsize=7)
        ax2.set_yticks(range(number_of_states))
        ax2.set_yticklabels(y_axis_tick_labels)
        ax2.set_xlim(0, number_of_steps)
        ax2.set_xlabel("Time Step", fontsize=7)
        ax2.tick_params(axis="both", which="major", labelsize=6)
        colorbar_belief = plt.colorbar(
            image_belief, ax=ax2, orientation="horizontal", pad=0.15
        )
        colorbar_belief.set_label("Probability", fontsize=7)
        colorbar_belief.ax.tick_params(labelsize=6)
    else:
        ax2.text(0.5, 0.5, "No belief data", transform=ax2.transAxes, ha="center")

    # (1,3) Category Belief
    # Per Tenet 3, category_names is required — no fallback to "Cat {c}".
    if belief_matrices:
        beliefs = np.array(belief_matrices)
        category_beliefs = beliefs.sum(axis=2)
        row_sums = category_beliefs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        category_beliefs = category_beliefs / row_sums

        category_names_config = config["category_names"]
        if category_names_config is None:
            raise KeyError(
                "config['category_names'] missing — required to label the "
                "category-belief panel. Per Tenet 3, silent fallbacks are forbidden."
            )
        if len(category_names_config) != number_of_categories:
            raise ValueError(
                f"category_names has {len(category_names_config)} entries but the "
                f"belief matrix has {number_of_categories} categories. Must match."
            )
        for category_index in range(number_of_categories):
            label = category_names_config[category_index] + (
                " (True)" if category_index == true_category else ""
            )
            ax3.plot(
                range(len(category_beliefs)),
                category_beliefs[:, category_index],
                linewidth=2,
                label=label,
            )
        ax3.set_title("Category Belief Evolution", fontsize=12)
        ax3.set_ylabel("Category Probability", fontsize=7)
        ax3.set_xlabel("Time Step", fontsize=7)
        ax3.set_xlim(0, number_of_steps)
        ax3.set_ylim(0, 1)
        ax3.grid(True)
        ax3.legend(fontsize=5, loc="best")
        ax3.tick_params(axis="both", which="major", labelsize=6)
    else:
        ax3.text(0.5, 0.5, "No belief data", transform=ax3.transAxes, ha="center")

    # (2,1) Entropy + Measurement Markers
    ax4.plot(time_steps, entropies, "g-", linewidth=1)
    if entropy_threshold is not None:
        ax4.axhline(
            y=entropy_threshold,
            color="r",
            linestyle="--",
            label=f"Threshold = {entropy_threshold:.4f}",
        )
    measurement_step_indices = [
        t for t, decision in zip(time_steps, decisions) if decision == 1
    ]
    measurement_entropies = [entropies[t] for t in measurement_step_indices]
    if measurement_step_indices:
        ax4.scatter(
            measurement_step_indices,
            measurement_entropies,
            c="red",
            s=15,
            zorder=5,
            label=f"Measurements ({len(measurement_step_indices)})",
        )
    ax4.set_title("Entropy & Measurement Decisions", fontsize=12)
    ax4.set_ylabel("Entropy", fontsize=7)
    ax4.set_xlabel("Time Step", fontsize=7)
    ax4.set_xlim(0, number_of_steps)
    ax4.set_ylim(bottom=0)
    ax4.grid(True)
    ax4.legend(fontsize=5)
    ax4.tick_params(axis="both", which="major", labelsize=6)

    # (2,2) L2: Tuning Matrix vs current transition matrix (before update)
    if current_transition_matrices_before:
        reference_tuning_matrix = steps[0]["tuning_matrix"]
        if reference_tuning_matrix is not None:
            l2_distances = [
                np.linalg.norm(reference_tuning_matrix - P, "fro")
                for P in current_transition_matrices_before
            ]
            ax5.plot(range(len(l2_distances)), l2_distances, "purple", linewidth=2)
        else:
            ax5.text(0.5, 0.5, "No tuning matrix", transform=ax5.transAxes, ha="center")
    else:
        ax5.text(0.5, 0.5, "No P matrix data", transform=ax5.transAxes, ha="center")
    ax5.set_xlim(0, number_of_steps)
    ax5.set_title("L2: Tuning Matrix vs Current P", fontsize=12)
    ax5.set_ylabel("L2 Distance", fontsize=7)
    ax5.set_xlabel("Time Step", fontsize=7)
    ax5.set_ylim(bottom=0)
    ax5.grid(True)
    ax5.tick_params(axis="both", which="major", labelsize=6)

    # (2,3) L2: Desired vs Actual Steady State
    if desired_distribution is not None and current_transition_matrices_after:
        desired_distribution = np.array(desired_distribution)
        steady_state_distances = [
            np.linalg.norm(compute_steady_state(P) - desired_distribution)
            for P in current_transition_matrices_after
        ]
        ax6.plot(
            range(len(steady_state_distances)),
            steady_state_distances,
            "orange",
            linewidth=2,
        )
    else:
        ax6.text(0.5, 0.5, "No steady state data", transform=ax6.transAxes, ha="center")
    ax6.set_xlim(0, number_of_steps)
    ax6.set_title("L2: Current vs Desired Steady State", fontsize=12)
    ax6.set_ylabel("L2 Distance", fontsize=7)
    ax6.set_xlabel("Time Step", fontsize=7)
    ax6.set_ylim(bottom=0)
    ax6.grid(True)
    ax6.tick_params(axis="both", which="major", labelsize=6)

    # (3,1) Alpha Over Time
    valid_alphas = [
        (t, alpha_value)
        for t, alpha_value in zip(time_steps, alpha_values)
        if alpha_value is not None
    ]
    if valid_alphas:
        alpha_time_indices, alpha_time_values = zip(*valid_alphas)
        ax7.plot(alpha_time_indices, alpha_time_values, "c-", linewidth=1, alpha=0.6)
        rolling_window_size = (
            min(50, len(alpha_time_values) // 5) if len(alpha_time_values) > 10 else 1
        )
        if rolling_window_size > 1:
            alpha_array = np.array(alpha_time_values)
            rolling_average = np.convolve(
                alpha_array,
                np.ones(rolling_window_size) / rolling_window_size,
                mode="valid",
            )
            ax7.plot(
                range(
                    rolling_window_size - 1,
                    rolling_window_size - 1 + len(rolling_average),
                ),
                rolling_average,
                "b-",
                linewidth=2,
                label=f"Rolling avg (w={rolling_window_size})",
            )
            ax7.legend(fontsize=5)
    else:
        ax7.text(0.5, 0.5, "No alpha data", transform=ax7.transAxes, ha="center")
    ax7.set_title("Realized Alpha Over Time", fontsize=12)
    ax7.set_ylabel("Alpha", fontsize=7)
    ax7.set_xlabel("Time Step", fontsize=7)
    ax7.set_xlim(0, number_of_steps)
    ax7.set_ylim(bottom=0)
    ax7.grid(True)
    ax7.tick_params(axis="both", which="major", labelsize=6)

    # (3,2) Measurement Decisions
    ax8.plot(time_steps, decisions, "r-", linewidth=1, alpha=0.7)
    cumulative_measurements = np.cumsum(decisions)
    ax8_twin = ax8.twinx()
    ax8_twin.plot(time_steps, cumulative_measurements, "b-", linewidth=2, alpha=0.8)
    ax8_twin.set_ylabel("Cumulative Count", fontsize=7, color="blue")
    ax8_twin.set_ylim(bottom=0)
    ax8_twin.tick_params(axis="y", labelcolor="blue", labelsize=6)
    ax8.set_title(f"Measurement Decisions (total: {sum(decisions)})", fontsize=12)
    ax8.set_ylabel("Measure (1/0)", fontsize=7, color="red")
    ax8.set_xlabel("Time Step", fontsize=7)
    ax8.set_xlim(0, number_of_steps)
    ax8.set_ylim(-0.1, 1.1)
    ax8.tick_params(axis="y", labelcolor="red", labelsize=6)
    ax8.tick_params(axis="x", labelsize=6)
    ax8.grid(True, alpha=0.3)

    # (3,3) Tuning Matrix Switching
    # Per Tenet 1, all values are derived from the log. We reconstruct switching
    # costs by counting steps where the recorded per-step cost implies a switch
    # (approach: switching_decisions from Metrics).
    from core.metrics import Metrics

    metrics = Metrics(episode, config)
    switching_decisions = metrics.switching_decisions or [0] * number_of_steps
    total_switches = sum(switching_decisions)
    # Per-step switch cost from the log: on a switching step it is the step
    # cost minus the measurement cost (if that step also measured).
    measurement_cost = episode["measurement_cost"]
    cumulative_switching_cost = np.cumsum(
        [
            (step_cost - (measurement_cost if measured else 0.0)) if switched else 0.0
            for step_cost, switched, measured in zip(
                step_costs, switching_decisions, decisions
            )
        ]
    )

    ax9.plot(time_steps, switching_decisions, "m-", linewidth=1, alpha=0.7)
    ax9_twin = ax9.twinx()
    ax9_twin.plot(
        time_steps, cumulative_switching_cost, "darkviolet", linewidth=2, alpha=0.8
    )
    ax9_twin.set_ylabel("Cumulative Switch Cost", fontsize=7, color="darkviolet")
    ax9_twin.set_ylim(bottom=0)
    ax9_twin.tick_params(axis="y", labelcolor="darkviolet", labelsize=6)
    ax9.set_title(f"Tuning Matrix Switching (total: {total_switches})", fontsize=12)
    ax9.set_ylabel("Switch (1/0)", fontsize=7, color="m")
    ax9.set_xlabel("Time Step", fontsize=7)
    ax9.set_xlim(0, number_of_steps)
    ax9.set_ylim(-0.1, 1.1)
    ax9.tick_params(axis="y", labelcolor="m", labelsize=6)
    ax9.tick_params(axis="x", labelsize=6)
    ax9.grid(True, alpha=0.3)

    # (4,1) Tuning Matrix Distribution Over Time (heatmap)
    # Uses recorded `tuning_weights` — one weight vector per step over the
    # agent's tuning-matrix basis. Rows are labeled with tuning_matrix_names
    # from config. Per Tenet 3, no fallbacks.
    tuning_weights_all = [
        step["tuning_weights"] for step in steps if step["tuning_weights"] is not None
    ]
    if tuning_weights_all:
        tuning_weights_array = np.array(tuning_weights_all)
        number_of_basis_tuning_matrices = tuning_weights_array.shape[1]

        tuning_matrix_names_config = config["tuning_matrix_names"]
        if tuning_matrix_names_config is None:
            raise KeyError(
                "config['tuning_matrix_names'] missing — required to label the "
                "tuning-weights heatmap. Per Tenet 3, silent fallbacks are forbidden."
            )
        if len(tuning_matrix_names_config) != number_of_basis_tuning_matrices:
            raise ValueError(
                f"tuning_matrix_names has {len(tuning_matrix_names_config)} entries "
                f"but the agent's basis has {number_of_basis_tuning_matrices}. Must match."
            )
        row_labels = list(tuning_matrix_names_config)

        image_tuning = ax10.imshow(
            tuning_weights_array.T,
            aspect="auto",
            interpolation="none",
            origin="lower",
            vmin=0,
            vmax=1,
            extent=[0, number_of_steps, -0.5, number_of_basis_tuning_matrices - 0.5],
        )
        ax10.set_title("Tuning Matrix Distribution Over Time", fontsize=12)
        ax10.set_ylabel("Tuning Matrix", fontsize=7)
        ax10.set_yticks(range(number_of_basis_tuning_matrices))
        ax10.set_yticklabels(row_labels)
        ax10.set_xlim(0, number_of_steps)
        ax10.set_xlabel("Time Step", fontsize=7)
        ax10.tick_params(axis="both", which="major", labelsize=6)
        colorbar_tuning = plt.colorbar(
            image_tuning, ax=ax10, orientation="horizontal", pad=0.15
        )
        colorbar_tuning.set_label("Tuning Weight", fontsize=7)
        colorbar_tuning.ax.tick_params(labelsize=6)
    else:
        ax10.text(
            0.5,
            0.5,
            "No tuning_weights logged",
            transform=ax10.transAxes,
            ha="center",
        )

    # (4,2) Total Cost (from log — Tenet 1)
    ax11.bar(
        time_steps, step_costs, color="salmon", alpha=0.5, width=1.0, label="Step cost"
    )
    cumulative_cost = np.cumsum(step_costs)
    ax11_twin = ax11.twinx()
    ax11_twin.plot(
        time_steps, cumulative_cost, "darkred", linewidth=2, label="Cumulative"
    )
    ax11_twin.set_ylabel("Cumulative Cost", fontsize=7, color="darkred")
    ax11_twin.set_ylim(bottom=0)
    ax11_twin.tick_params(axis="y", labelcolor="darkred", labelsize=6)
    ax11.set_title(f"Cost (total: {cumulative_cost[-1]:.2f})", fontsize=12)
    ax11.set_ylabel("Step Cost", fontsize=7, color="salmon")
    ax11.set_xlabel("Time Step", fontsize=7)
    ax11.set_xlim(0, number_of_steps)
    ax11.tick_params(axis="y", labelcolor="salmon", labelsize=6)
    ax11.tick_params(axis="x", labelsize=6)
    ax11.grid(True, alpha=0.3)

    # (4,3) reserved
    ax12.axis("off")

    plt.subplots_adjust(
        left=0.08, right=0.92, top=0.96, bottom=0.04, wspace=0.4, hspace=0.45
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    filename = output_dir / f"simulation_results_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {filename}")

    if show:
        plt.show()

    return fig


def plot_time_to_reach_cdf(
    episodes,
    target_state,
    number_of_steps,
    state_name,
    config,
    output_dir,
    show,
):
    """Plot empirical CDF of time-to-reach for a target state across episodes.

    All parameters required (Tenet 3).

    Censored episodes (never reached) count as number_of_steps.

    Args:
        episodes: list of episode dicts
        target_state: int, state index
        number_of_steps: simulation horizon
        state_name: label for the target state (string)
        config: config dict for Metrics
        output_dir: directory to save plot (Path or str)
        show: whether to call plt.show() (bool)

    Returns:
        matplotlib Figure
    """
    from core.metrics import Metrics, time_to_reach_auc

    times = []
    for episode in episodes:
        metrics = Metrics(episode, config)
        time_to_reach_map = metrics.time_to_reach
        time_reached = time_to_reach_map.get(target_state)
        times.append(
            time_reached
            if time_reached is not None and time_reached >= 0
            else number_of_steps
        )

    times = np.sort(times)
    total_episodes = len(times)
    cumulative_distribution_function = np.arange(1, total_episodes + 1) / total_episodes
    area_under_cdf = time_to_reach_auc(episodes, target_state, number_of_steps, config)

    label = state_name

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # CDF
    ax1.step(
        times,
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
        f"Time to Reach {label} — CDF  "
        f"(n={total_episodes}, AUC={area_under_cdf:.1f})",
        fontsize=12,
    )
    ax1.set_xlim(0, number_of_steps)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Survival = 1 - CDF
    ax2.step(
        times,
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
    ax2.set_title(f"Time to Reach {label} — Survival", fontsize=12)
    ax2.set_xlim(0, number_of_steps)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    filename = output_dir / f"time_to_reach_{label}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved CDF plot to: {filename}")

    if show:
        plt.show()

    return fig
