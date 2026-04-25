# pymatchit

`pymatchit` is a Python-native causal matching library inspired by R MatchIt and
optmatch. It focuses on the design stage of observational studies: choosing
matched or weighted comparison groups before estimating treatment effects.

## Quickstart

Install in editable mode:

```bash
pip install -e ".[dev]"
```

Run nearest-neighbor matching:

```python
import pandas as pd
import pymatchit

data = pd.DataFrame(
    {
        "treat": [1, 1, 0, 0, 0],
        "age": [29, 35, 31, 40, 46],
        "income": [18_000, 25_000, 19_000, 35_000, 50_000],
        "educ": [10, 12, 10, 14, 16],
    }
)

fit = pymatchit.matchit(
    "treat ~ age + income + educ",
    data,
    method="nearest",
    distance="mahalanobis",
    replace=False,
)

print(fit.matched_data())
print(fit.balance_summary())
print(fit.match_summary())
```

Plot common design-stage diagnostics:

```python
import matplotlib.pyplot as plt

pymatchit.love_plot(fit, threshold=0.1)
pymatchit.ecdf_plot(fit, "age")
pymatchit.qq_plot(fit, "income")
pymatchit.density_plot(fit, "educ")
pymatchit.jitter_plot(fit)
pymatchit.histogram_plot(fit)

plt.show()
```

The plots are intentionally MatchIt-style rather than pixel-perfect MatchIt
ports. They use matplotlib only and focus on quick diagnostics for balance,
propensity score overlap, and matched/discarded units.

Other implemented methods:

```python
# Exact matching
exact_fit = pymatchit.matchit(
    "treat ~ age + sex + region",
    data,
    method="exact",
    exact=["sex", "region"],
)

# Propensity score subclassification
subclass_fit = pymatchit.matchit(
    "treat ~ age + income + educ",
    data,
    method="subclass",
    n_subclasses=6,
)

# Optimal 1:1 pair matching
optimal_fit = pymatchit.matchit(
    "treat ~ age + income + educ",
    data,
    method="optimal",
    distance="logit",
    caliper=0.25,
)

# Experimental full matching prototype, not optmatch-equivalent
full_fit = pymatchit.matchit(
    "treat ~ age + income + educ",
    data,
    method="full",
    distance="logit",
)
```

## Design-Stage Matching

`pymatchit` deliberately separates design from outcome analysis.

During matching, the API uses treatment assignment and pretreatment covariates
only. It does not look at outcomes, fit outcome regressions, or estimate causal
effects. The output is a matched or weighted dataset plus diagnostics:

- `matched_data()` returns retained rows with `weights` and `subclass` columns.
- `weights_` stores matching or subclassification weights.
- `subclass_` stores matched-pair, exact-stratum, or subclass labels.
- `balance_summary()` reports standardized mean differences and variance ratios.
- `effective_sample_size()` reports weighted effective sample sizes.
- `match_summary()` gives a compact MatchIt-style overview.

After checking balance, estimate treatment effects in a separate analysis step
using the matched data and weights. This keeps modeling choices explicit and
avoids using outcomes to tune the design.

## Implemented Scope

- Formula parsing with `patsy`, for formulas such as
  `"treat ~ age + income + educ"`.
- Propensity score estimation with scikit-learn logistic regression.
- Distance options: `"propensity"`, `"logit"`, and `"mahalanobis"`.
- Exact matching with overlap-only strata.
- Greedy nearest-neighbor 1:1 matching for binary treatment and ATT.
- Optimal 1:1 pair matching with SciPy assignment.
- Propensity score subclassification with ATT and ATE-style weights.
- Experimental OR-Tools full matching prototype for binary treatment. This is
  not optmatch-equivalent and should not be treated as production full matching.
- Balance diagnostics for matched and weighted designs.

## Currently Unsupported

The project does not yet aim for exact parity with MatchIt or optmatch. Current
unsupported features include:

- Production-grade optmatch-equivalent full matching.
- Genetic matching.
- Cardinality and profile matching.
- Multi-category or continuous treatments.
- Survey weights.
- Sampling weights in propensity score estimation.
- Variable-ratio nearest-neighbor matching.
- Optmatch-equivalent optimal full matching and network-flow behavior.
- Exact replication of R object classes, print methods, and plotting APIs.
- Built-in treatment-effect estimation or outcome modeling.

## Examples

See `examples/lalonde_like_demo.py` for a small, self-contained script using a
Lalonde-like toy dataset.

## Development

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```
