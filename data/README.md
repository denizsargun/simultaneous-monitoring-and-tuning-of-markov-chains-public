# Data

Aggregate transition matrices for Stage IV non-small-cell lung cancer (NSCLC),
fitted to the AACR Project GENIE Biopharma Collaborative (BPC) NSCLC v2.0
registry. The files contain **only fitted matrices and labels**: no patient
records, identifiers, dates, counts or free text.

The paper this repository accompanies uses GENIE BPC colorectal cancer data;
these NSCLC matrices are provided so the code can be run on real-data-derived
dynamics.

## Files

| File | Cohorts (stratification) |
|---|---|
| `T_additive_complete_nsclc_targ_pdl1.pkl` | `Targetable+` (targetable driver), and driver-negative patients by PD-L1: `T-/PDL1-High` (≥50%), `T-/PDL1-Low` (<50%), `T-/PDL1-Unknown` |
| `T_additive_complete_nsclc_pdl1_high.pkl` | PD-L1 `High` (≥50%) vs `Low` (<50%), tested patients only |
| `T_additive_complete_nsclc_pdl1_pos.pkl` | PD-L1 `Positive` (≥1%) vs `Negative` (<1%), tested patients only |

`sample_configs/` use the `targ_pdl1` file. `docs/data/matrices.json` is the same
`targ_pdl1` data as JSON (rounded to 6 decimals), and `docs/app.html` inlines all
three.

## Contents

Each pkl is a list of dicts, one per (cohort, action):

| Key | Meaning |
|---|---|
| `category` | cohort name |
| `action` | the string `"None"` (untreated baseline) or a treatment class: `Chemo`, `IO`, `Targeted`, `Chemo+VEGF`, `Investigational` |
| `P_0` | 3×3 weekly baseline transition matrix of the cohort (untreated) |
| `T` | 3×3 additive weekly effect of the action (zero for `None`) |
| `states`, `states_short` | `Progress`, `Disease Control`, `Death` (`PD`, `DC`, `D`); `Death` is absorbing |
| `simulation_interval_days` | 7 (one step = one week) |
| `stratification_mode` | `targ_pdl1`, `pdl1_high` or `pdl1_pos` |

`Targeted` has no fit for the three `T-/…` cohorts (driver-negative patients);
the simulation refuses to apply it to them.

The simulation uses `B = rownorm(clip(P_0 + T))` with `Death` kept absorbing.

Inspect a file with `python3 peek_pkl.py data/<file>.pkl`. Only load pickles you
trust; `SHA256SUMS` lists the published checksums.

## How the matrices were estimated

Observations (imaging and oncologist assessments) were mapped to the three states
per patient. `P_0` was estimated from observation pairs with no active treatment
and rescaled to a 7-day step. For each action, `T` was fitted by maximum
likelihood so that `(P_0 + T)^k` matches the transitions observed while the
action was active, with `k` the average observation interval in weeks. The fitting pipeline
was written by H. Bugra Tulay and is not part of this repository.

**Caveats.** Some (cohort, action) cells rest on few observed transitions, and
entries of `P_0 + T` below zero are clipped. For example, `Chemo+VEGF` in
`T-/PDL1-High` has zero weekly probability of `Death`.

## Terms and acknowledgment

These matrices are derived from AACR Project GENIE data; use them in line with
GENIE's terms of access. The repository's code license (AGPL-3.0) covers the code,
not these data.

The authors would like to acknowledge the American Association for Cancer Research
and its material support in the development of the AACR Project GENIE registry, as
well as members of the consortium for their commitment to data sharing.
Interpretations are the responsibility of study authors.
