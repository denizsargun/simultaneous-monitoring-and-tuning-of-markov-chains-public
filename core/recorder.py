# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Simulation logger for recording all raw simulation data to disk.

Captures ALL raw simulation data without processing per Tenet #1:
- Full configuration and initial state
- Environment matrices (P, T) at each step
- All agent decisions and states
- Random outcomes and transition probabilities
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Any, Optional
from pathlib import Path
from core.utils import get_timestamp


# Default output directory for simulation logs
DEFAULT_OUTPUT_DIR = "results/simulation_results"


class Recorder:
    """
    Records and aggregates all relevant data from simulation runs including:
    - State transitions
    - Agent actions (tuning and measurement decisions)
    - Rewards/costs
    - Environment states
    - Agent beliefs
    """

    def __init__(
        self,
        simulation_id: Optional[str] = None,
        output_dir: str = None,
        _from_log: bool = False,
    ):
        """
        Initialize the simulation logger.

        Args:
            simulation_id: Unique identifier for this simulation run (ISO8601 timestamp if None)
            output_dir: Base directory for output (default: results/simulation_results)
            _from_log: If True, skip creating output directory (used by from_log_dir)
        """
        # Generate ISO8601 timestamp as simulation ID
        timestamp = get_timestamp()

        if simulation_id is None:
            self.simulation_id = timestamp
        else:
            self.simulation_id = simulation_id

        # Set output directory: results/simulation_results/<timestamp>/
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR
        self.output_dir = Path(output_dir) / self.simulation_id

        if not _from_log:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Episode-level data
        self.episodes = []
        self.current_episode = None
        self.episode_counter = 0

        # Full configuration storage
        self.full_config = None

        # Aggregate statistics
        self.aggregate_stats = {
            "total_episodes": 0,
            "total_steps": 0,
            "total_measurements": 0,
            "total_rewards": 0.0,
            "total_costs": 0.0,
        }

    def log_config(self, config: Dict[str, Any], agent_params: Dict[str, Any] = None):
        """
        Log the full configuration at simulation start (call once).

        Args:
            config: Configurator config dictionary with all matrices and parameters
            agent_params: Additional agent-specific parameters
        """
        self.full_config = {
            "simulation_id": self.simulation_id,
            "timestamp": get_timestamp(),
            "config": {
                "number_of_states": config.get("number_of_states"),
                "number_of_categories": config.get("number_of_categories"),
                "number_of_steps": config.get("number_of_steps"),
                "state_names": config.get("state_names"),
                "category_prior": (
                    np.array(config.get("category_prior")).copy()
                    if config.get("category_prior") is not None
                    else None
                ),
                "desired_distribution": (
                    np.array(config.get("desired_distribution")).copy()
                    if config.get("desired_distribution") is not None
                    else None
                ),
                "transition_matrices": [
                    np.array(P).copy() for P in config.get("transition_matrices", [])
                ],
                "alpha_sequences": (
                    np.array(config.get("alpha_sequences")).copy()
                    if config.get("alpha_sequences") is not None
                    else None
                ),
                "initial_distributions": [
                    np.array(d).copy() for d in config.get("initial_distributions", [])
                ],
                "category_names": config.get("category_names"),
                "tuning_matrix_names": config.get("tuning_matrix_names"),
                # Per-category tuning basis B^(c,a); None where no fit exists
                "tuning_basis_by_category": [
                    [M.copy() if M is not None else None for M in tuning_basis_row]
                    for tuning_basis_row in config["tuning_basis_by_category"]
                ],
                "state_y_axis_positions_by_index": config.get(
                    "state_y_axis_positions_by_index"
                ),
            },
            "agent_params": agent_params or {},
        }

        # Save config immediately
        config_path = self.output_dir / "config.pkl"
        with open(config_path, "wb") as f:
            pickle.dump(self.full_config, f)

        print(f"Configuration logged to {config_path}")

    def start_episode(
        self,
        agent_type: str,
        true_category: int,
        initial_state: int,
        config: Dict[str, Any],
        initial_env_P: List[np.ndarray] = None,
        initial_env_T: List[np.ndarray] = None,
        initial_agent_belief: np.ndarray = None,
        initial_agent_P_model: List[np.ndarray] = None,
    ):
        """
        Start collecting data for a new episode with full initial state.

        Args:
            agent_type: Agent.agent_type string, e.g.
                'Agent(meas=entropy_threshold,space=data,cat_known=True,select=given_constant)'
            true_category: True environment category
            initial_state: Initial state
            config: Configuration dictionary
            initial_env_P: Initial environment P matrices (all categories)
            initial_env_T: Initial environment T matrices (all categories)
            initial_agent_belief: Initial agent belief matrix
            initial_agent_P_model: Initial agent's model of P matrices
        """
        self.episode_counter += 1
        self.current_episode = {
            "episode_id": self.episode_counter,
            "start_timestamp": get_timestamp(),
            "agent_type": agent_type,
            "true_category": true_category,
            "initial_state": initial_state,
            # Per Tenet 3, these are required — no silent fake fallbacks.
            "number_of_states": config["number_of_states"],
            "number_of_categories": config["number_of_categories"],
            "number_of_steps": config["number_of_steps"],
            "measurement_cost": config["measurement_cost"],
            # Raw initial environment state
            "initial_env_P": (
                [P.copy() for P in initial_env_P] if initial_env_P else None
            ),
            "initial_env_T": (
                [T.copy() for T in initial_env_T] if initial_env_T else None
            ),
            # Raw initial agent state
            "initial_agent_belief": (
                initial_agent_belief.copy()
                if initial_agent_belief is not None
                else None
            ),
            "initial_agent_P_model": (
                [P.copy() for P in initial_agent_P_model]
                if initial_agent_P_model
                else None
            ),
            "steps": [],
        }

    def record_step(
        self,
        step_num: int,
        current_state: int,
        measurement_decision: int,
        measurement_result: Optional[int],
        tuning_matrix: np.ndarray,
        state_transition: Dict[str, int],
        belief_matrix: np.ndarray,
        entropy: float,
        reward: float = 0.0,
        cost: float = 0.0,
        true_category: Optional[int] = None,
        alpha_value: Optional[float] = None,
        env_P_matrix: Optional[np.ndarray] = None,
        env_T_matrix: Optional[np.ndarray] = None,
        env_P_matrix_after: Optional[np.ndarray] = None,
        transition_probabilities: Optional[np.ndarray] = None,
        agent_P_model: Optional[List[np.ndarray]] = None,
        tuning_matrix_index: Optional[int] = None,
        tuning_weights: Optional[np.ndarray] = None,
        *,
        agent_tuning_matrices_by_category: List[Optional[np.ndarray]],
    ):
        """
        Record ALL raw data for a single simulation step.

        Args:
            step_num: Step number
            current_state: Current true state (BEFORE transition)
            measurement_decision: 1 if measured, 0 otherwise
            measurement_result: Observed state AFTER transition (equals
                state_transition['to']) if measured, None otherwise
            tuning_matrix: Tuning matrix the environment applied to the true
                category, T^(C) = sum_a w_a B^(C,a)
            state_transition: Dict with 'from' and 'to' states
            belief_matrix: Full joint belief matrix (categories x states)
            entropy: Current entropy
            reward: Reward received
            cost: Cost incurred
            true_category: True category at this step
            alpha_value: Alpha adaptation rate at this step
            env_P_matrix: Environment's P matrix BEFORE step (for true category)
            env_T_matrix: Environment's T matrix BEFORE step (for true category)
            env_P_matrix_after: Environment's P matrix AFTER alpha update
            transition_probabilities: Row the next state was sampled from
                (blended P, clipped and renormalized)
            agent_P_model: Agent's full P_model list
            tuning_matrix_index: Index of selected tuning matrix (if discrete)
            tuning_weights: Weight vector over the agent's treatment basis.
                For tuning_space=data, indices correspond to `tuning_matrix_names`.
                For tuning_space=synthetic, indices correspond to `category_names`.
            agent_tuning_matrices_by_category: Agent's T^(c) for each category
                this step (None for categories the agent does not model)
        """

        if self.current_episode is None:
            raise ValueError("No episode started. Call start_episode() first.")

        # Store RAW data - no computed metrics
        step_data = {
            "step": step_num,
            # Environment state
            "true_state_before": current_state,
            "true_state_after": state_transition.get("to", current_state),
            "true_category": true_category,
            "alpha_value": alpha_value,
            # Raw environment matrices
            "env_P_matrix_before": (
                env_P_matrix.copy() if env_P_matrix is not None else None
            ),
            "env_T_matrix": env_T_matrix.copy() if env_T_matrix is not None else None,
            "env_P_matrix_after": (
                env_P_matrix_after.copy() if env_P_matrix_after is not None else None
            ),
            "transition_probabilities": (
                transition_probabilities.copy()
                if transition_probabilities is not None
                else None
            ),
            # Agent decisions
            "measurement_decision": measurement_decision,
            "measurement_result": measurement_result,
            "tuning_matrix": (
                tuning_matrix.copy() if tuning_matrix is not None else None
            ),
            "tuning_matrix_index": tuning_matrix_index,
            "tuning_weights": (
                np.asarray(tuning_weights).copy()
                if tuning_weights is not None
                else None
            ),
            "agent_tuning_matrices_by_category": [
                T.copy() if T is not None else None
                for T in agent_tuning_matrices_by_category
            ],
            # Agent state
            "belief_matrix": (
                belief_matrix.copy() if belief_matrix is not None else None
            ),
            "agent_P_model": (
                [P.copy() for P in agent_P_model] if agent_P_model is not None else None
            ),
            "entropy": entropy,
            # Costs (raw values, no accumulation)
            "reward": reward,
            "cost": cost,
        }

        self.current_episode["steps"].append(step_data)

        # Update aggregate stats
        self.aggregate_stats["total_steps"] += 1
        if measurement_decision:
            self.aggregate_stats["total_measurements"] += 1
        self.aggregate_stats["total_rewards"] += reward
        self.aggregate_stats["total_costs"] += cost

    def end_episode(
        self,
        final_metrics: Optional[Dict[str, float]] = None,
        final_env_P: List[np.ndarray] = None,
        final_agent_belief: np.ndarray = None,
    ):
        """
        End the current episode and save raw data to disk.

        Args:
            final_metrics: Optional dictionary of final metrics
            final_env_P: Final environment P matrices
            final_agent_belief: Final agent belief matrix
        """
        if self.current_episode is None:
            raise ValueError("No episode to end.")

        self.current_episode["end_timestamp"] = get_timestamp()

        # Store final states (raw)
        self.current_episode["final_env_P"] = (
            [P.copy() for P in final_env_P] if final_env_P else None
        )
        self.current_episode["final_agent_belief"] = (
            final_agent_belief.copy() if final_agent_belief is not None else None
        )

        # Compute episode-level statistics
        steps_data = self.current_episode["steps"]

        self.current_episode["summary"] = {
            "total_steps": len(steps_data),
            "total_measurements": sum(s["measurement_decision"] for s in steps_data),
            "measurement_frequency": (
                np.mean([s["measurement_decision"] for s in steps_data])
                if steps_data
                else 0
            ),
            "total_reward": sum(s["reward"] for s in steps_data),
            "total_cost": sum(s["cost"] for s in steps_data),
            "average_entropy": (
                np.mean([s["entropy"] for s in steps_data]) if steps_data else 0
            ),
            "state_visits": self._compute_state_visits(steps_data),
            "transition_counts": self._compute_transition_counts(steps_data),
        }

        # Add any final metrics provided
        if final_metrics:
            self.current_episode["summary"].update(final_metrics)

        # Save episode to disk immediately (raw pickle)
        episode_id = self.current_episode["episode_id"]
        episode_path = self.output_dir / f"episode_{episode_id:04d}.pkl"
        with open(episode_path, "wb") as f:
            pickle.dump(self.current_episode, f)
        print(f"Episode {episode_id} saved to {episode_path}")

        # Store episode and update aggregate stats
        self.episodes.append(self.current_episode)
        self.aggregate_stats["total_episodes"] += 1

        # Reset current episode
        self.current_episode = None

    def _get_state_labels(self, number_of_states: int) -> List[str]:
        """
        Get unique state labels (state names, spaces as underscores) for CSV columns,
        or numeric labels when the config has no state names.

        Args:
            number_of_states: Number of states

        Returns:
            List of state label strings
        """
        state_names = None
        if self.full_config and "config" in self.full_config:
            state_names = self.full_config["config"].get("state_names")

        if state_names is not None and len(state_names) == number_of_states:
            return [name.replace(" ", "_") for name in state_names]
        return [str(i) for i in range(number_of_states)]

    def _compute_state_visits(self, steps_data: List[Dict]) -> Dict[int, int]:
        """Compute frequency of visits to each state."""
        state_counts = {}
        for step in steps_data:
            state = step["true_state_before"]
            if state is not None:
                state_counts[state] = state_counts.get(state, 0) + 1
        return state_counts

    def _compute_transition_counts(self, steps_data: List[Dict]) -> Dict[str, int]:
        """Compute frequency of each state transition."""
        transition_counts = {}
        for step in steps_data:
            state_from = step["true_state_before"]
            state_to = step["true_state_after"]
            if state_from is not None and state_to is not None:
                transition = f"{state_from}->{state_to}"
                transition_counts[transition] = transition_counts.get(transition, 0) + 1
        return transition_counts

    def get_episode_dataframe(self, episode_id: Optional[int] = None) -> pd.DataFrame:
        """
        Get DataFrame for a specific episode or all episodes with flattened belief matrix.

        Args:
            episode_id: Specific episode ID, or None for all episodes

        Returns:
            DataFrame with episode data including flattened belief matrix
        """
        if episode_id is not None:
            episode = next(
                (e for e in self.episodes if e["episode_id"] == episode_id), None
            )
            if episode is None:
                raise ValueError(f"Episode {episode_id} not found")
            episodes = [episode]
        else:
            episodes = self.episodes

        rows = []
        for episode in episodes:
            number_of_categories = episode["number_of_categories"]
            number_of_states = episode["number_of_states"]
            state_labels = self._get_state_labels(number_of_states)

            for step in episode["steps"]:
                # Start with basic fields
                row = {
                    "episode_id": episode["episode_id"],
                    "agent_type": episode["agent_type"],
                }

                # Add scalar fields (skip numpy arrays for now)
                for k, v in step.items():
                    if not isinstance(v, np.ndarray):
                        row[k] = v

                # Flatten belief matrix with descriptive column names
                if "belief_matrix" in step and isinstance(
                    step["belief_matrix"], np.ndarray
                ):
                    belief_matrix = step["belief_matrix"]
                    for c in range(number_of_categories):
                        for s in range(number_of_states):
                            col_name = f"belief_category_{c}_state_{s}"
                            row[col_name] = belief_matrix[c, s]

                # Flatten env_P_matrix_before (P_ij values)
                if (
                    "env_P_matrix_before" in step
                    and step["env_P_matrix_before"] is not None
                ):
                    p_matrix = step["env_P_matrix_before"]
                    for i in range(p_matrix.shape[0]):
                        for j in range(p_matrix.shape[1]):
                            from_label = (
                                state_labels[i] if i < len(state_labels) else str(i)
                            )
                            to_label = (
                                state_labels[j] if j < len(state_labels) else str(j)
                            )
                            row[f"P_before_{from_label}_{to_label}"] = p_matrix[i, j]

                # Flatten env_P_matrix_after
                if (
                    "env_P_matrix_after" in step
                    and step["env_P_matrix_after"] is not None
                ):
                    p_after = step["env_P_matrix_after"]
                    for i in range(p_after.shape[0]):
                        for j in range(p_after.shape[1]):
                            from_label = (
                                state_labels[i] if i < len(state_labels) else str(i)
                            )
                            to_label = (
                                state_labels[j] if j < len(state_labels) else str(j)
                            )
                            row[f"P_after_{from_label}_{to_label}"] = p_after[i, j]

                rows.append(row)

        return pd.DataFrame(rows)

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Get DataFrame with episode-level summaries.

        Returns:
            DataFrame with episode summaries
        """
        rows = []
        for episode in self.episodes:
            row = {
                "episode_id": episode["episode_id"],
                "agent_type": episode["agent_type"],
                "true_category": episode["true_category"],
                "initial_state": episode["initial_state"],
                **episode["summary"],
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def save_to_csv(self, output_dir: Optional[Path] = None):
        """
        Save collected data to CSV files.

        Args:
            output_dir: Directory to save files (default: self.output_dir)
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Save episode data
        episode_df = self.get_episode_dataframe()
        episode_path = output_dir / f"{self.simulation_id}_episodes.csv"
        episode_df.to_csv(episode_path, index=False)

        # Save summary data
        summary_df = self.get_summary_dataframe()
        summary_path = output_dir / f"{self.simulation_id}_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Save aggregate statistics
        stats_path = output_dir / f"{self.simulation_id}_stats.txt"
        with open(stats_path, "w") as f:
            f.write(f"Simulation ID: {self.simulation_id}\n")
            f.write("=" * 50 + "\n")
            for key, value in self.aggregate_stats.items():
                f.write(f"{key}: {value}\n")

        print(f"Data saved to:")
        print(f"  - Episodes: {episode_path}")
        print(f"  - Summary: {summary_path}")
        print(f"  - Stats: {stats_path}")

        return episode_path, summary_path, stats_path

    def get_output_dir(self) -> Path:
        """Return the output directory path."""
        return self.output_dir

    @staticmethod
    def load_episode(episode_path: str) -> Dict[str, Any]:
        """
        Load a saved episode from disk.

        Args:
            episode_path: Path to .pkl file

        Returns:
            Episode data dictionary with all raw data
        """
        with open(episode_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load saved configuration from disk.

        Args:
            config_path: Path to config.pkl file

        Returns:
            Configuration dictionary
        """
        with open(config_path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def from_log_dir(cls, log_dir: str) -> "Recorder":
        """
        Create a Recorder by loading from an existing log directory.

        This allows using all existing methods (get_episode_dataframe, get_summary_dataframe, etc.)
        on previously logged simulation data. For plotting, use Simulator.plot_results().

        Args:
            log_dir: Path to simulation log folder (e.g., results/simulation_results/20260220T082221Z/)

        Returns:
            Recorder with episodes loaded from disk
        """
        log_path = Path(log_dir)
        if not log_path.exists():
            raise FileNotFoundError(f"Log directory not found: {log_path}")

        # Extract simulation_id from directory name
        simulation_id = log_path.name

        # Create instance using normal __init__ but skip mkdir
        instance = cls(
            simulation_id=simulation_id,
            output_dir=str(log_path.parent),
            _from_log=True,
        )

        # Load config if available
        config_path = log_path / "config.pkl"
        if config_path.exists():
            instance.full_config = cls.load_config(str(config_path))

        # Load all episodes
        episode_files = sorted(log_path.glob("episode_*.pkl"))
        for ep_file in episode_files:
            episode = cls.load_episode(str(ep_file))
            instance.episodes.append(episode)
            instance.episode_counter = max(
                instance.episode_counter, episode["episode_id"]
            )

            # Update aggregate stats
            instance.aggregate_stats["total_episodes"] += 1
            steps = episode.get("steps", [])
            instance.aggregate_stats["total_steps"] += len(steps)
            instance.aggregate_stats["total_measurements"] += sum(
                s.get("measurement_decision", 0) for s in steps
            )

        print(f"Loaded {len(instance.episodes)} episodes from {log_dir}")
        return instance

    @staticmethod
    def get_latest_log_dir(base_dir: str = DEFAULT_OUTPUT_DIR) -> str:
        """
        Get the path to the most recent simulation log directory.

        Args:
            base_dir: Base directory containing simulation logs

        Returns:
            Path to latest simulation log directory
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Base directory not found: {base_path}")

        sim_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        if not sim_dirs:
            raise FileNotFoundError(f"No simulation logs found in {base_path}")

        latest = max(sim_dirs, key=lambda d: d.name)
        return str(latest)

    def get_agent_comparison(self) -> pd.DataFrame:
        """
        Get comparison metrics across different agent types.

        Returns:
            DataFrame comparing agent performance
        """
        summary_df = self.get_summary_dataframe()

        if summary_df.empty:
            return pd.DataFrame()

        # Build agg dict with only columns that exist
        agg_dict = {
            "total_measurements": ["mean", "std"],
            "measurement_frequency": ["mean", "std"],
            "total_cost": ["mean", "std"],
            "total_reward": ["mean", "std"],
            "average_entropy": ["mean", "std"],
        }
        for col in ["control_error", "desired_distribution_distance"]:
            if col in summary_df.columns:
                agg_dict[col] = ["mean", "std"]

        comparison = summary_df.groupby("agent_type").agg(agg_dict)

        return comparison

    def __repr__(self):
        return (
            f"Recorder(id={self.simulation_id!r}, "
            f"episodes={len(self.episodes)}, "
            f"total_steps={self.aggregate_stats['total_steps']})"
        )
