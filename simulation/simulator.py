# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Simulator class for running simulations and computing metrics.
"""

import numpy as np
from typing import Optional
from core.metrics import compute_step_cost
from core.recorder import Recorder


class Simulator:
    """
    Main simulator that orchestrates interaction between environment and agent.
    Maintains clear separation between what environment and agent can access.
    Includes built-in data collection for all simulations.
    """

    def __init__(
        self,
        environment,
        agent,
        recorder: Recorder,
        measurement_cost: float,
        switching_cost: float,
        simulation_id: Optional[str] = None,
    ):
        """
        Initialize simulator with environment and agent.

        Args:
            environment: Environment instance
            agent: Agent instance
            simulation_id: Optional ID for the simulation
            recorder: Shared simulation logger for multi-episode runs.
                    If None, a new one is created per Tenet #1.
            measurement_cost: Cost per measurement taken
            switching_cost: Cost per treatment switch
        """
        self.environment = environment
        self.agent = agent
        self.measurement_cost = measurement_cost
        self.switching_cost = switching_cost

        self.recorder = recorder or Recorder(simulation_id)

        # Get agent type from agent's property (no fragile class name parsing)
        self.agent_type = getattr(
            self.agent, "agent_type", self.agent.__class__.__name__
        )

    def run(self, number_of_steps):
        """
        Run simulation for number_of_steps with integrated data collection

        Args:
            number_of_steps: Number of simulation steps
        """
        config = {
            "number_of_states": self.environment.number_of_states,
            "number_of_categories": self.environment.number_of_categories,
            "number_of_steps": number_of_steps,
            "measurement_cost": self.measurement_cost,
            "switching_cost": self.switching_cost,
        }

        # Log full config once (includes transition matrices, alpha sequences)
        if self.recorder.full_config is None:
            full_config = {
                "number_of_states": self.environment.number_of_states,
                "number_of_categories": self.environment.number_of_categories,
                "number_of_steps": number_of_steps,
                "transition_matrices": [
                    self.environment.current_transition_matrices[c].copy()
                    for c in range(self.environment.number_of_categories)
                ],
                "alpha_sequence": (
                    self.environment.alpha_sequence.copy()
                    if hasattr(self.environment, "alpha_sequence")
                    else None
                ),
                "category_names": getattr(self.environment, "category_names", None),
                "tuning_matrix_names": self.environment.tuning_matrix_names,
                "tuning_basis_by_category": self.environment.tuning_basis_by_category,
                "state_names": getattr(self.environment, "state_names", None),
                "state_y_axis_positions_by_index": getattr(
                    self.environment, "state_y_axis_positions_by_index", None
                ),
            }

            agent_params = {
                "agent_type": self.agent_type,
                "entropy_threshold": getattr(self.agent, "entropy_threshold", None),
                "desired_distribution": getattr(
                    self.agent, "desired_distribution", None
                ),
            }
            self.recorder.log_config(full_config, agent_params)

        # Get initial environment transition matrices and tuning matrices
        initial_env_P = [
            self.environment.current_transition_matrices[c].copy()
            for c in range(self.environment.number_of_categories)
        ]
        initial_env_T = [
            self.environment.tuning_matrices[c].copy()
            for c in range(self.environment.number_of_categories)
        ]

        # Get initial agent state
        initial_agent_belief = None
        if hasattr(self.agent, "belief") and hasattr(
            self.agent.belief, "belief_matrix"
        ):
            initial_agent_belief = self.agent.belief.belief_matrix.copy()

        initial_agent_P_model = None
        if hasattr(self.agent, "P_model"):
            initial_agent_P_model = [P.copy() for P in self.agent.P_model]

        self.recorder.start_episode(
            agent_type=self.agent_type,
            true_category=self.environment.category.true_category,
            initial_state=self.environment.state,
            config=config,
            initial_env_P=initial_env_P,
            initial_env_T=initial_env_T,
            initial_agent_belief=initial_agent_belief,
            initial_agent_P_model=initial_agent_P_model,
        )

        # Track previous tuning action (weights) for switching cost detection
        prev_tuning_weights = None

        for step in range(number_of_steps):
            # === CAPTURE PRE-STEP STATE ===
            # Record state and matrices BEFORE environment processes the step
            current_state = self.environment.state

            # Capture environment current transition matrix BEFORE alpha update
            true_cat_pre = self.environment.category.true_category
            env_P_before = self.environment.current_transition_matrices[
                true_cat_pre
            ].copy()
            env_T_before = self.environment.tuning_matrices[true_cat_pre].copy()

            # === AGENT DECISIONS ===
            # Agent decides whether to measure (based on entropy threshold)
            measurement_decision = self.agent.decide_measurement()

            # Agent decides the tuning action: weights over the basis actions
            tuning_weights = self.agent.decide_tuning()

            # === ENVIRONMENT STEP ===
            # Environment: applies the true category's T for these weights,
            # blends (P = (1-α)P + αT), transitions state, provides measurement
            measurement = self.environment.step(measurement_decision, tuning_weights)

            # Capture state AFTER transition
            next_state = self.environment.state

            # === RAW DATA LOGGING ===
            # Get agent's full belief matrix
            if hasattr(self.agent, "belief") and hasattr(
                self.agent.belief, "belief_matrix"
            ):
                belief_matrix = self.agent.belief.belief_matrix
            elif len(self.agent.history["belief_matrix"]) > 0:
                belief_matrix = self.agent.history["belief_matrix"][-1]
            else:
                # Create uniform belief matrix if not available
                belief_matrix = np.ones(
                    (
                        self.environment.number_of_categories,
                        self.environment.number_of_states,
                    )
                ) / (
                    self.environment.number_of_categories
                    * self.environment.number_of_states
                )

            # Get entropy
            if len(self.agent.history["entropy"]) > 0:
                entropy = self.agent.history["entropy"][-1]
            else:
                entropy = 0.0

            # Get current alpha value if available
            alpha_value = None
            if hasattr(self.environment, "alpha_sequence") and step < len(
                self.environment.alpha_sequence
            ):
                alpha_value = self.environment.alpha_sequence[step]

            # Environment matrices: P_before was captured before env.step(),
            # P_after is the current state after alpha blending
            true_cat = self.environment.category.true_category
            env_P_after = self.environment.current_transition_matrices[true_cat].copy()
            # Tuning matrix the environment applied to the true category
            applied_tuning_matrix = self.environment.tuning_matrices[true_cat].copy()

            # Transition probabilities: the row the environment sampled the next
            # state from (blended P, clipped and renormalized)
            transition_probs = self.environment.last_transition_probabilities.copy()

            # Get agent's P_model
            agent_P_model = None
            if hasattr(self.agent, "P_model"):
                agent_P_model = [P.copy() for P in self.agent.P_model]

            # Index of the chosen action when a single action is applied
            tuning_matrix_index = (
                int(np.argmax(tuning_weights))
                if np.count_nonzero(tuning_weights) == 1
                else None
            )

            # Agent's own T^(c) per category this step (None if not modeled)
            agent_tuning_matrices_by_category = self.agent.history[
                "tuning_matrices_by_category"
            ][-1]

            # Detect treatment switching: the chosen action (weights) changed.
            # Category independent, so it counts a regimen change even when two
            # actions happen to give the same matrix for the true category.
            tuning_switched = False
            if prev_tuning_weights is not None:
                weights_change = np.linalg.norm(tuning_weights - prev_tuning_weights)
                tuning_switched = weights_change > 1e-10  # Numerical tolerance
            prev_tuning_weights = tuning_weights.copy()

            # Calculate per-step cost
            cost = compute_step_cost(
                measurement_decision,
                tuning_switched,
                self.measurement_cost,
                self.switching_cost,
            )

            # Record step with ALL raw data
            self.recorder.record_step(
                step_num=step,
                current_state=current_state,
                measurement_decision=measurement_decision,
                measurement_result=measurement,
                tuning_matrix=applied_tuning_matrix,
                state_transition={"from": current_state, "to": next_state},
                belief_matrix=belief_matrix,
                entropy=entropy,
                reward=0.0,
                cost=cost,
                true_category=true_cat,
                alpha_value=alpha_value,
                # Raw environment matrices: before and after alpha blending
                env_P_matrix=env_P_before,
                env_T_matrix=env_T_before,
                env_P_matrix_after=env_P_after,
                transition_probabilities=transition_probs,
                agent_P_model=agent_P_model,
                tuning_matrix_index=tuning_matrix_index,
                tuning_weights=tuning_weights,
                agent_tuning_matrices_by_category=agent_tuning_matrices_by_category,
            )

            # Agent processes the measurement
            self.agent.process_measurement(measurement)

        # Get final states
        final_env_P = [
            self.environment.current_transition_matrices[c].copy()
            for c in range(self.environment.number_of_categories)
        ]

        final_agent_belief = None
        if hasattr(self.agent, "belief") and hasattr(
            self.agent.belief, "belief_matrix"
        ):
            final_agent_belief = self.agent.belief.belief_matrix.copy()

        self.recorder.end_episode(
            final_env_P=final_env_P,
            final_agent_belief=final_agent_belief,
        )

    def plot_results(self, output_dir):
        """Plot most recent episode. Delegates to core.plot.plot_episode.

        Per Tenet 3, output_dir is required (no auto-default).
        """
        from core.plot import plot_episode

        if self.recorder is None or len(self.recorder.episodes) == 0:
            print("No logged episodes available for plotting.")
            return

        episode = self.recorder.episodes[-1]

        # Build config for plot from recorder's full_config.
        # `tuning_matrix_names`, `category_names`, and `state_names` are stored
        # NESTED under the "config" key by Recorder.log_config.
        agent_params = self.recorder.full_config["agent_params"]
        inner_config = self.recorder.full_config["config"]
        config = {
            "entropy_threshold": agent_params.get("entropy_threshold"),
            "desired_distribution": agent_params.get("desired_distribution"),
            "tuning_matrix_names": inner_config["tuning_matrix_names"],
            "category_names": inner_config["category_names"],
            "state_names": inner_config["state_names"],
            "state_y_axis_positions_by_index": inner_config[
                "state_y_axis_positions_by_index"
            ],
        }

        plot_episode(
            episode=episode,
            config=config,
            output_dir=output_dir,
            show=True,
        )
