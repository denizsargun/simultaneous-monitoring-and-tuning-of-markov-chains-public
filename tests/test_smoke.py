# Copyright (c) 2025-2026 Deniz Sargun
# SPDX-License-Identifier: AGPL-3.0-only

"""Smoke and regression tests on the published data."""

from pathlib import Path

import numpy as np
import pytest
import yaml

import core.recorder as recorder_module
from agents.belief import Belief
from simulation.runner import run_from_yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONFIG_PATH = REPOSITORY_ROOT / "sample_configs" / "sample_run.yaml"


def _write_short_sample_config(directory, file_name, modify_config):
    """Write a shortened copy of the sample config, changed by modify_config."""
    with open(SAMPLE_CONFIG_PATH) as config_file:
        config = yaml.safe_load(config_file)
    config["environment"]["number_of_steps"] = 20
    config["environment"]["number_of_runs"] = 2
    config["output"]["plot"] = 0
    config["output"]["verbose"] = 0
    modify_config(config)
    config_path = directory / file_name
    config_path.write_text(yaml.safe_dump(config))
    return config, config_path


@pytest.fixture
def run_directory(tmp_path, monkeypatch):
    """Run from the repository root, writing results under tmp_path."""
    monkeypatch.chdir(REPOSITORY_ROOT)
    monkeypatch.setattr(
        recorder_module, "DEFAULT_OUTPUT_DIR", str(tmp_path / "results")
    )
    return tmp_path


def test_sample_run_produces_requested_metrics(run_directory):
    config, config_path = _write_short_sample_config(
        run_directory, "sample.yaml", lambda config: None
    )
    result = run_from_yaml(str(config_path))
    assert result["number_of_runs"] == 2
    requested_metrics = result["runs"][0]["requested_metrics"]
    expected_metric_names = {name for name, flag in config["metrics"].items() if flag}
    assert set(requested_metrics) == expected_metric_names


def test_action_without_fit_for_a_modeled_category_raises(run_directory):
    # 'Targeted' has no fit for the T-/PDL1-* cohorts. With the category
    # unknown the agent models every cohort, so start-up must fail.
    def use_targeted_with_unknown_category(config):
        config["agent"]["tuning_policy"]["is_category_known"] = False
        config["agent"]["tuning_policy"]["constant_tuning_selection"] = "Targeted"

    _, config_path = _write_short_sample_config(
        run_directory, "missing_fit.yaml", use_targeted_with_unknown_category
    )
    with pytest.raises(ValueError, match="no fitted tuning matrix"):
        run_from_yaml(str(config_path))


def test_measurement_reveals_the_next_state():
    # Deterministic flip chain: from a known X_k = 0 the next state is 1, so
    # observing it must put all belief mass on state 1.
    belief = Belief(number_of_states=2, number_of_categories=1)
    belief.belief_matrix = np.array([[1.0, 0.0]])
    flip_chain = [np.array([[0.0, 1.0], [1.0, 0.0]])]
    belief.update_with_observation(1, flip_chain)
    assert np.allclose(belief.belief_matrix, [[0.0, 1.0]])
