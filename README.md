# Simultaneous Monitoring and Tuning of Markov Chains

A simulation framework for controlling a Markov chain that can only be observed
at a cost. At every step an agent decides **whether to measure** the hidden state
and **how to tune** the chain's transition matrix, while keeping a Bayesian belief
over the hidden state and the hidden category of the system.

This repository accompanies

> D. Sargun, H. B. Tulay, C. E. Koksal, *Belief-Space Control for Personalized
> Cancer Treatment via Active Inference*, Allerton Conference on Communication,
> Control, and Computing, 2026.

**Data note.** The paper's experiments use AACR Project GENIE BPC **colorectal**
cancer data. This repository instead ships aggregate transition matrices fitted to
AACR Project GENIE BPC **NSCLC** (non-small-cell lung cancer) data: a baseline
matrix and additive treatment effects per patient cohort. It contains no
patient-level data. See [`data/README.md`](data/README.md).

**Research use only.** Nothing here is validated for clinical use or is medical
advice.

## Quick start

```bash
git clone https://github.com/denizsargun/simultaneous-monitoring-and-tuning-of-markov-chains-public.git
cd simultaneous-monitoring-and-tuning-of-markov-chains-public
pip install -r requirements.txt
python3 -m simulation.runner sample_configs/sample_run.yaml
```

Run commands from the repository root; paths in the configs are relative to it.
Results are written to `results/simulation_results/<timestamp>/`.

Or, from Python:

```python
from simulation.runner import run_from_yaml

result = run_from_yaml("sample_configs/sample_run.yaml")
print(result["runs"][0]["requested_metrics"])
```

## Model

Categories `c`, hidden states `X_k`, a desired state distribution `π*`, and a tuning
basis `B^(c,a)`: for each category `c` and action `a`, `B^(c,None) = P_0^(c)` (the
untreated baseline) and `B^(c,a) = rownorm(clip(P_0^(c) + T^(c,a)))`, with the last
(absorbing) state kept absorbing.

```
C, X_0 ─► b_0 ─► M_1 ─► T_0 ─► P_1 ─► X_1 ─► Y_1 ─► P̂_1 ─► b_1 ─► M_2 ─► T_1 ─► …
          └─ agent decides ─┘  └─ environment ───┘  └─ agent updates ─┘
```

At each step `k`:

1. **Agent decides** from its belief `b_k`: whether to measure the next state,
   `M_{k+1}`, and the action weights `w_k`.
2. **Environment** (true category `C`): `P_{k+1} = (1 − α_k) P_k + α_k T_k^(C)` with
   `T_k^(C) = Σ_a w_{k,a} B^(C,a)`; samples `X_{k+1} ~ P_{k+1}(X_k, ·)`; reveals
   `Y_{k+1} = X_{k+1}` if `M_{k+1} = 1`.
3. **Agent updates** its model of each category it has not ruled out,
   `P̂_{k+1}^(c) = (1 − ᾱ_k) P̂_k^(c) + ᾱ_k T_k^(c)`, and its joint belief with a
   Bayes filter (predict with `P̂_{k+1}`, then condition on `Y_{k+1}`).

## Configuration

Every run is driven by a YAML file; see the annotated
[`sample_configs/sample_run.yaml`](sample_configs/sample_run.yaml). Every parameter
is required and a missing one raises an error instead of falling back to a default;
the exceptions are the metric flags (absent means off) and the optional
`environment.state_names`.
Sections: `data`, `agent` (`measurement_policy`, `tuning_policy`), `environment`,
`costs`, `metrics`, `visualization`, `output`.

### Measurement policy — when to observe

| Parameter | Values |
|---|---|
| `state_measurement_decision_algorithm` | `entropy_threshold` (measure when the state-belief entropy ≥ `entropy_threshold`) \| `dynamic_programming` (a measurement-only value-iteration policy; requires `is_category_known: true`) |

### Tuning policy — which action to apply

| Parameter | Values |
|---|---|
| `tuning_space` | `data` (fitted matrices from `data/`) \| `synthetic` (one Metropolis–Hastings matrix per category) |
| `is_category_known` | `true` (prior one-hot on the true category) \| `false` (uniform prior, Bayesian inference) |
| `tuning_selection` | `given_constant` \| `random_constant` \| `convex` \| `argmax` |
| `constant_tuning_selection` | action name, required for `given_constant` |
| `tuning_cadence` | positive integer `N`: apply the selected action every `N` steps and `None` in between |

An action must be fitted for every category the agent models (only the true one
when the category is known, all of them otherwise); otherwise the run fails at
start-up. For example, `Targeted` is fitted only for `Targetable+` in the
`targ_pdl1` data. `random_constant`, `convex` and `argmax` can apply any action, so
with `is_category_known: false` they need a data file where every cohort has every
action (`pdl1_high` or `pdl1_pos`).

### Alpha (tuning rate)

The environment draws `α_k` from an exponential distribution clipped to `[0, 1]`
(so the realized mean is `m(1 − e^{−1/m})` for configured mean `m`). The agent uses
the configured, unclipped mean `m`, which overestimates the realized α when `m` is
large.

| Type | Environment | Agent estimate |
|---|---|---|
| `constant` | `α ~ Exp(m)` | `m` |
| `decaying` | `α ~ Exp(m·e^{−λk})` | `m·e^{−λk}` |

## Metrics

Computed from the logged episodes by `core.metrics` (the `Metrics` class, plus
`time_to_reach_regex_pattern` and `compute_overall_cost`):

| Metric | Description |
|---|---|
| `measurement_frequency` | fraction of steps with a measurement |
| `state_rates` | fraction of steps in each state |
| `time_to_reach` | step at which the single capture group of a regex `pattern` first matches the state trace |
| `tracking_error` | L2 distance between the true category's belief row and the true state |
| `control_error` | Jensen–Shannon distance between the belief and the steady state of the agent's model |
| `average_entropy` | mean state-belief entropy |
| `desired_distribution_distance` | L2 distance between the empirical state distribution and `π*` |
| `tuning_matrix_switches` | number of changes of the chosen action |
| `overall_cost` | Σ (`measurement_cost`·M + `switching_cost`·switch) |

Each run writes `config.pkl` (including the tuning basis), `episode_NNNN.pkl` (every
step's states, matrices, weights and belief), CSV summaries and a plot.

## Web app

A client-side port of the simulator runs in the browser on the same matrices:

- [`docs/index.html`](docs/index.html): served by GitHub Pages at
  `https://denizsargun.github.io/simultaneous-monitoring-and-tuning-of-markov-chains-public/`;
  loads `docs/data/matrices.json`.
- [`docs/app.html`](docs/app.html): self-contained variant with three data sets
  inlined, a data-set selector and an explicit Run button; opens directly from disk.

See [`docs/README.md`](docs/README.md).

## Repository layout

```
├── agents/               # Agent (measurement + tuning policies) and Belief
├── core/                 # config builder, recorder, metrics, plots, utilities
├── environment/          # true Markov chain, category, state, alpha generators
├── simulation/           # YAML runner and simulation loop
├── sample_configs/       # sample run and alpha × cadence sweep
├── data/                 # aggregate NSCLC matrices (see data/README.md)
├── docs/                 # web app
├── tests/                # smoke and regression tests
├── alpha_cadence_sweep.py
├── plot_time_to_reach_pattern.py
└── peek_pkl.py           # print a summary of a data pkl
```

## Design principles (tenets)

Code comments cite these as "Tenet N".

1. **Simulations are the source of truth.** Every step is logged; metrics are
   computed from the logs. (Life-expectancy curves in `plot_time_to_reach_pattern.py`
   and the web app are solved analytically.)
2. **PEP 8 / PEP 20**, formatted with Black.
3. **No hidden defaults.** Configuration fields are required and a missing value
   raises; a few internal helpers still carry defaults.
4. **Explicit names.**
5. **Data-independent core.** Generic code speaks of tuning, measurement and
   category rather than a particular domain.

## Known limitations

- `argmax` selection is degenerate on chains with an absorbing state: every
  steady state is the absorbing state, so it always picks `None`.
- `tracking_error` uses the true category's joint belief row without normalizing it.
- Runs are not seeded, so they are not reproducible run-for-run.
- The `dynamic_programming` measurement policy is solved once, on the untuned
  baseline, and ignores tuning.
- The synthetic Metropolis–Hastings basis uses a non-symmetric proposal without
  the Hastings correction, so its stationary distribution is not exactly `π*`.

## Development

```bash
pip install -r requirements-dev.txt
black --check .
pytest
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

The code is licensed under the GNU Affero General Public License v3.0 only
([`LICENSE`](LICENSE)). The matrices in `data/`, `docs/data/` and the data sets
inside `docs/app.html` are derived from AACR Project GENIE; see
[`data/README.md`](data/README.md).

## Acknowledgment

The authors would like to acknowledge the American Association for Cancer Research
and its material support in the development of the AACR Project GENIE registry, as
well as members of the consortium for their commitment to data sharing.
Interpretations are the responsibility of study authors.
