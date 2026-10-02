# Simultaneous Monitoring & Tuning — web app (`docs/`)

A **real, client-side** simulator of the monitoring-and-tuning Markov control system,
served as a GitHub Page. It is **not** a mockup: the environment, belief filter, agent
policies, and metrics all run in your browser on the actual fitted NSCLC transition
matrices. No backend, no build step, no dependencies.

```
docs/
├── index.html          # the app: UI + the entire simulation engine (vanilla JS + SVG)
├── app.html            # self-contained variant (three data sets inlined, dataset selector, explicit Run button); opens from disk
├── data/
│   └── matrices.json    # real fitted matrices exported from the .pkl (data only)
└── README.md            # this file
```

## How it works (data → engine → charts)

1. **Data.** `data/matrices.json` holds the real per-cohort baseline `P_0` and additive
   treatment shifts `T` (the same numbers the Python pipeline consumes), plus state
   labels and the 7-day step unit. It is exported once from the fitted pickle — see
   [Regenerating the data](#regenerating-the-data). Browsers can't read a Python
   `.pkl`, so the numbers are shipped as JSON.

2. **Engine.** On load, `index.html` fetches the JSON and reconstructs the model exactly
   as the Python runner does: effective treatment matrices are `clip(P_0 + T)` with
   non-absorbing rows renormalized and Death forced absorbing; the tuning basis is built
   per cohort from that cohort's own `P_0` and `T` (matching `simulation/runner.py`), and a
   drug a cohort has no fit for raises an error wherever it would be applied. It then runs a Monte-Carlo of
   full episodes. Each step, per the true cohort:
   - the **agent** decides whether to measure (entropy-threshold or dynamic-programming
     policy) and which treatment target to apply;
   - **tuning cadence** gates the treatment — the selected drug is applied every *N*
     steps and `None` (no treatment) in between;
   - the **environment** blends its transition matrix toward the target,
     `P ← (1−α)P + αT`, and samples the next state;
   - the **agent** blends its internal model and updates its Bayesian **belief** over
     `(cohort, state)`.

3. **Charts.** Two tabs. *Single patient run* simulates one patient from the selected
   initial state and shows the time-series panels (tuning schedule, state, belief,
   entropy + measurements, category belief). *Multi-patient averages* simulates the
   Monte Carlo runs, every patient starting in Progress, and shows the metric tiles and
   two time-to-reach CDFs: overall survival (weeks to Death) and progression-free
   survival (weeks to the first Progress after Disease Control, or to Death if Disease
   Control is never reached). Everything is inline SVG.

### Faithful port map

The JS in `index.html` follows these Python sources (differences below):

| Python | JS in `index.html` |
|---|---|
| `agents/belief.py` (`Belief`) | `beliefInit` / `beliefUpdateObs` / `beliefUpdateNoObs` / `stateMarginal` / `categoryBelief` |
| `agents/agent.py` (measurement, tuning weights, cadence, per-cohort P_model blend) | `runEpisode`, `weights`, `tuning`, `dpPolicy` |
| `environment/environment.py` (true cohort's tuning matrix, α-blend + sampling) | `runEpisode` env loop (`mixBasis`, `blend`, `sampleRow`) |
| `environment/alpha_generators.py` | `alphaExp` (agent, expected) + `expo` (environment, sampled exponential) |
| `simulation/runner.py` (`_load_matrices_from_yaml`) | `buildModel` |
| `core/metrics.py` | metric accumulation in `runEpisode` + `runAll` |
| `core/utils.py` (`compute_steady_state`, `mix_tuning_basis`) | `steadyState` (power iteration), `mixBasis` |
| `plot_time_to_reach_pattern.py` (regex AUC) | `regexArrival` |

**Differences from Python.** The effective matrices, tuning basis, belief updates and
cadence gate match the Python code. The rest differs in these ways:

- **Random numbers:** the browser seeds its own PRNG with a fresh random seed on every
  run, so runs are not reproducible and sampled trajectories don't match NumPy step for
  step; stochastic aggregates match distributionally.
- **Start state:** multi-patient averages start every patient in Progress; the single
  patient run starts in the selected initial state. Python uses `environment.initial_state`
  for every run.
- **Progression-free survival CDF:** browser only. Python's `time_to_reach` regex has no
  exact equivalent; its PFS pattern excludes patients who die before Disease Control.
- **Initial belief:** uniform over states in the browser; a random Dirichlet draw per
  category in Python.
- **Tracking error:** the browser normalizes the true category's belief row before
  taking the distance; Python uses the unnormalized joint row.
- **Steady state:** power iteration (400 steps) in the browser, an eigenvector in Python;
  they can differ on slowly mixing chains with an absorbing state.
- **Dynamic programming:** the browser forces a known category; Python requires
  `is_category_known: true` in the config.

## Viewing locally

`index.html` fetches `data/matrices.json`, which browsers block over `file://`, so serve
it over HTTP (or open `app.html`, which has its data inlined):

```bash
cd docs
python3 -m http.server 8000
# open http://localhost:8000/
```

On GitHub Pages it just works (served over HTTPS) — no server needed.

## Deploying on GitHub Pages

Settings → Pages → Source: **Deploy from a branch** → Branch **`main`**, folder **`/docs`**.
The site publishes at
`https://denizsargun.github.io/simultaneous-monitoring-and-tuning-of-markov-chains-public/`.

## Regenerating the data

`data/matrices.json` is derived from `data/T_additive_complete_nsclc_targ_pdl1.pkl` at the
repository root. To refresh it, run from the repository root:

```bash
python3 - <<'PY'
import pickle, json, numpy as np
from collections import OrderedDict
SRC='data/T_additive_complete_nsclc_targ_pdl1.pkl'
R=pickle.load(open(SRC,'rb'))
cohorts=OrderedDict()
for e in R:
    c=e['category']
    cohorts.setdefault(c,{'name':c,'P0':np.asarray(e['P_0']).round(6).tolist(),'treatments':{}})
    if e['action'] not in ('Unknown','None'):
        cohorts[c]['treatments'][e['action']]=np.asarray(e['T']).round(6).tolist()
json.dump({'source':SRC,'states':list(R[0]['states']),'states_short':list(R[0]['states_short']),
           'interval_days':int(R[0]['simulation_interval_days']),'death_index':len(R[0]['states'])-1,
           'cohort_order':list(cohorts),'cohorts':cohorts},
          open('docs/data/matrices.json','w'),indent=1)
PY
```

Only the derived transition probabilities are baked into the page.
