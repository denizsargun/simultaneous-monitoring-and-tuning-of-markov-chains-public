# Simultaneous Monitoring & Tuning — web app (`docs/`)

A **real, client-side** simulator of the monitoring-and-tuning Markov control system,
served as a GitHub Page. It is **not** a mockup: the environment, belief filter, agent
policies, and metrics all run in your browser on the actual fitted NSCLC transition
matrices. No backend, no build step, no dependencies.

```
docs/
├── index.html          # the app: UI + the entire simulation engine (vanilla JS + SVG)
├── app.html            # self-contained variant (four data sets inlined, dataset selector, explicit Run button); opens from disk
├── data/
│   └── matrices.json    # four real fitted data sets exported from the .pkls (data only)
└── README.md            # this file
```

## How it works (data → engine → charts)

1. **Data.** `data/matrices.json` holds four NSCLC data sets, chosen with the *Transition
   matrices* selector: one per stratification (`targ_pdl1`, `pdl1_high`, `pdl1_pos`) and
   `targ_pdl1_age65`, the four `targ_pdl1` cohorts each split at age 65. Each holds the real per-cohort baseline `P_0` and additive treatment shifts
   `T` (the same numbers the Python pipeline consumes), plus state labels and the 7-day
   step unit. It is exported from the fitted pickles — see
   [Regenerating the data](#regenerating-the-data). Browsers can't read a Python
   `.pkl`, so the numbers are shipped as JSON.

2. **Engine.** On load, `index.html` fetches the JSON and, for the selected data set,
   reconstructs the model exactly as the Python runner does: effective treatment matrices
   are `clip(P_0 + T)` with non-absorbing rows renormalized and Death forced absorbing;
   the tuning basis is built per cohort from that cohort's own `P_0` and `T` (matching
   `simulation/runner.py`), and a drug a cohort has no fit for raises an error wherever
   it would be applied. It then runs a Monte-Carlo of full episodes. Each step, per the
   true cohort:
   - the **agent** decides which treatment target to apply and whether to measure the
     next state. By default it knows the true cohort and measures every step (the
     entropy-threshold policy with threshold 0); ⚙ Settings can raise the threshold, switch
     to the dynamic-programming policy, or leave the cohort unknown (uniform prior,
     inferred by the belief filter);
   - **tuning cadence** gates the treatment — the selected drug is applied every *N*
     steps and `None` (no treatment) in between;
   - the **environment** blends its transition matrix toward the target,
     `P ← (1−α)P + αT`, and samples the next state;
   - the **agent** blends its internal model and updates its Bayesian **belief** over
     `(cohort, state)`.

3. **Layout.** The sidebar follows the tab. *Data*, *Simulation* (horizon and Monte Carlo
   runs) and *Treatment blend (α)* apply to every tab. *Patient* and *Tuning policy* drive
   the first two tabs; on *Cases compared* they give way to the case builder. **⚙ Settings**
   in the top bar holds the rest, for every tab: the random seed (new on every run by
   default, or fixed), the monitoring policy (algorithm, entropy threshold, the
   dynamic-programming parameters), whether the agent knows the cohort, the measurement and
   switching costs, and the desired distribution π*. The status shows the seed each run
   used, so fixing it in Settings repeats the run.

4. **Charts.** Three tabs. *Single patient run* simulates one patient from the selected
   initial state and shows the tuning schedule and the state trajectory; its *Additional
   metrics* sub-tab shows the belief over the states, its entropy with the measurements,
   and the cohort belief. *Multi-patient averages* simulates the Monte Carlo runs, every
   patient starting in Progress, and shows tiles for mean and median survival and mean PFS
   and two time-to-reach CDFs: overall survival (weeks to Death) and progression-free
   survival (weeks to the first Progress after Disease Control or to Death, whichever
   comes first). Its *Additional metrics* sub-tab has tiles for measurement frequency,
   tracking error, control error, JSD to Desired Distribution, tuning switches and overall
   cost. *Cases compared* runs up to eight (true category, tuning policy, initial state)
   cases with the sidebar's horizon, runs and α and the Settings, all on the same random
   seeds, and shows a table of their mean, P5, P50 and P95 survival and PFS times and the
   two CDFs overlaid.
   Every CDF marks P5, P50 and P95 under its x-axis: the first week at which the
   empirical CDF reaches 5%, 50% and 95%. A percentile not reached within the horizon is
   listed under the chart as not reached, and as "≥ horizon" in the tiles and the table.
   Everything is inline SVG.

### Faithful port map

The JS in `index.html` follows these Python sources (differences below):

| Python | JS in `index.html` |
|---|---|
| `agents/belief.py` (`Belief`) | `beliefInit` / `beliefUpdateObs` / `beliefUpdateNoObs` / `stateMarginal` / `categoryBelief` |
| `agents/agent.py` (entropy-threshold and dynamic-programming measurement, modeled cohorts, fitted-action check, tuning weights, cadence, per-cohort P_model blend) | `runEpisode`, `dpPolicy`, `modeledCohorts`, `checkFitted`, `weights`, `tuning` |
| `environment/environment.py` (true cohort's tuning matrix, α-blend + sampling) | `runEpisode` env loop (`mixBasis`, `blend`, `sampleRow`) |
| `environment/alpha_generators.py` | `alphaExp` (agent, expected) + `expo` (environment, sampled exponential) |
| `simulation/runner.py` (`_load_matrices_from_yaml`) | `buildModel` |
| `core/metrics.py` | metric accumulation in `runEpisode` + `aggregate` |
| `core/utils.py` (`compute_steady_state`, `mix_tuning_basis`) | `steadyState` (power iteration), `mixBasis` |

**Differences from Python.** The effective matrices, tuning basis, belief updates and
cadence gate match the Python code. The rest differs in these ways:

- **Defaults:** the browser opens measuring every step (Python's `entropy_threshold`
  policy with threshold 0) with the agent knowing the true cohort (Python's
  `is_category_known: true`). ⚙ Settings offers both of Python's measurement policies and
  an unknown cohort (uniform prior over the cohorts, inferred by the belief filter); then
  every action the selection can apply must be fitted for every cohort, so e.g.
  `Targeted` fails on `targ_pdl1`.
- **Random numbers:** the browser seeds its own PRNG, with a fresh random seed on every
  run unless ⚙ Settings fixes one. A fixed seed repeats a browser run exactly, but sampled
  trajectories never match NumPy step for step; stochastic aggregates match
  distributionally.
- **Tracking error:** the browser normalizes the true cohort's belief row before the L2
  distance; Python uses the unnormalized joint row.
- **Control error and distance to π\*:** both are Jensen–Shannon divergences (natural
  log, so at most ln 2 ≈ 0.69), averaged over the runs: control error between the belief
  and the model's steady state, *JSD to Desired Distribution* between each run's state
  rates and π*. Python's `control_error` is the Jensen–Shannon distance (the square root
  of the divergence) and its `desired_distribution_distance` the L2 distance. The argmax
  selection and the dynamic-programming policy use the distance, as in Python.
- **Start state:** multi-patient averages start every patient in Progress; the single
  patient run starts in the selected initial state, and each compared case in its own.
  Python uses `environment.initial_state` for every run.
- **Progression-free survival CDF:** browser only. Python's `time_to_reach` regex has no
  exact equivalent; its PFS pattern excludes patients who die before Disease Control.
- **Initial belief:** uniform over states in the browser; a random Dirichlet draw per
  category in Python.
- **Steady state:** power iteration (400 steps) in the browser, an eigenvector in Python;
  they can differ on slowly mixing chains with an absorbing state.

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

`docs/data/matrices.json` is derived from the four pickles in `data/` at the repository
root. To refresh it, run from the repository root:

```bash
python3 - <<'PY'
import pickle, json, numpy as np
from collections import OrderedDict
TABLES='data'
LABELS={'targ_pdl1':'Targetable × PD-L1 · 4 cohorts','pdl1_high':'PD-L1 High / Low',
        'pdl1_pos':'PD-L1 Positive / Negative',
        'targ_pdl1_age65':'Targetable × PD-L1 × age <65 / ≥65 · 8 cohorts'}
datasets=OrderedDict()
for key,label in LABELS.items():
    SRC=f'{TABLES}/T_additive_complete_nsclc_{key}.pkl'
    R=pickle.load(open(SRC,'rb'))
    cohorts=OrderedDict()
    for e in R:
        c=e['category']
        cohorts.setdefault(c,{'name':c,'P0':np.asarray(e['P_0']).round(6).tolist(),'treatments':{}})
        if e['action'] not in ('Unknown','None'):
            cohorts[c]['treatments'][e['action']]=np.asarray(e['T']).round(6).tolist()
    datasets[key]={'label':label,'source':SRC,'states':list(R[0]['states']),
                   'states_short':list(R[0]['states_short']),
                   'interval_days':int(R[0]['simulation_interval_days']),
                   'death_index':len(R[0]['states'])-1,'cohort_order':list(cohorts),'cohorts':cohorts}
json.dump({'default_dataset':'targ_pdl1','datasets':datasets},
          open('docs/data/matrices.json','w'),indent=1)
PY
```

`app.html` inlines the same four data sets in its `DATASETS` constant, without the
`source` field and with `label` last; update it to match.

Only the derived transition probabilities are baked into the page.
