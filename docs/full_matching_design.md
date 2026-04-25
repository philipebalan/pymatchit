# Optmatch-Style Full Matching Design

This document sketches an optmatch-style full matching design for `pymatchit`.
It is intentionally a design note, not an implementation plan with committed
solver behavior.

## Goal

Full matching partitions treated and control units into matched subclasses. Each
subclass must contain at least one treated and at least one control unit, while
allowing variable treated-to-control ratios. The design-stage objective is to
minimize total within-subclass matching distance subject to ratio and omission
constraints.

The API should continue to avoid outcome modeling. Full matching should return
subclass labels, matching weights, diagnostics, and matched data, leaving
treatment-effect estimation to downstream code.

## Min-Cost Flow Formulation

Let:

- `T` be the set of treated units.
- `C` be the set of control units.
- `d_ij` be the distance between treated unit `i in T` and control unit `j in C`.
- `E` be the feasible edge set after applying calipers and exact blocks.
- `x_ij >= 0` indicate assignment flow between treated `i` and control `j`.

A practical full-matching construction can be represented as a min-cost flow on
a bipartite network:

- Source connects to treated unit nodes.
- Treated unit nodes connect to feasible control unit nodes.
- Control unit nodes connect to sink.
- Edge costs from treated to control are `d_ij`.
- Capacities enforce allowable controls per treated-side cluster, or are
  expanded through template nodes to encode subclass ratio constraints.

Basic objective:

```text
minimize    sum_{(i,j) in E} d_ij * x_ij
```

Subject to:

```text
unit coverage constraints:
    each retained treated unit belongs to exactly one subclass
    each retained control unit belongs to at most one subclass

subclass overlap constraints:
    each nonempty subclass contains >= 1 treated and >= 1 control

ratio constraints:
    min_controls <= controls_per_treated_subclass
    controls_per_treated_subclass <= max_controls

feasibility constraints:
    x_ij = 0 for edges outside exact blocks
    x_ij = 0 for distances beyond caliper

omission constraints:
    number or fraction of omitted units <= omit_fraction
```

The exact integer formulation depends on the subclass representation. Two viable
approaches are:

1. Expanded template network: create candidate treated/control ratio templates
   and solve a flow problem over those templates.
2. Mixed-integer formulation: explicitly model subclass activation and unit
   membership with binary variables.

The template network is closer to a pure min-cost flow approach. The mixed
integer formulation is more direct but requires a MIP solver and is probably too
heavy for this package initially.

## Proposed Solver Options

### OR-Tools

OR-Tools provides mature min-cost flow and assignment tooling. It is the most
appropriate option if full matching is treated as a real network optimization
feature.

Pros:

- Purpose-built graph optimization.
- Handles large sparse networks better than dense assignment approaches.
- Clear support for capacities, supplies, and edge costs.

Cons:

- Adds a substantial optional dependency.
- Requires careful packaging and platform support.
- Integer edge costs may require distance scaling.

Recommendation: make OR-Tools an optional extra, for example
`pip install pymatchit[full]`.

### SciPy

SciPy already exists in the dependency set and includes optimization routines,
including linear programming via `scipy.optimize.linprog`.

Pros:

- No new dependency.
- Good fit for small to medium linear relaxations.
- Useful fallback for prototypes and tests.

Cons:

- General LP tools are not specialized for min-cost flow.
- Sparse graph construction is easy to get wrong.
- Integer feasibility and subclass structure may require extra work.

Recommendation: useful for prototypes, validation, and small problems, but not
the preferred production path for full matching.

### NetworkX

NetworkX includes graph algorithms and min-cost flow utilities.

Pros:

- Very readable graph construction.
- Good for reference implementations and debugging.
- Easy to inspect graph structure.

Cons:

- Pure-Python performance can be poor for large matching problems.
- May not scale to realistic observational datasets.
- Another dependency unless kept optional.

Recommendation: useful for an educational or reference backend, not the default
for large data.

## API Design

Proposed public API:

```python
fit = pymatchit.matchit(
    "treat ~ age + income + educ + race",
    data,
    method="full",
    distance="logit",
    min_controls=0.5,
    max_controls=4,
    omit_fraction=0.1,
    caliper=0.25,
    exact=["race"],
    solver="auto",
)
```

Proposed arguments:

- `method="full"`: selects full matching.
- `min_controls`: minimum allowed control-to-treated ratio within subclasses.
  Values below `1` allow subclasses with more treated than controls.
- `max_controls`: maximum allowed control-to-treated ratio within subclasses.
- `omit_fraction`: maximum fraction of units that may be omitted to restore
  feasibility or improve total cost.
- `caliper`: forbids treated-control edges with distance above the threshold.
- `exact`: exact blocking columns. Full matching should solve separately within
  blocks or forbid cross-block edges.
- `solver`: one of `"auto"`, `"ortools"`, `"scipy"`, or `"networkx"`.

Proposed result fields:

- `matched_data()`
- `weights_`
- `subclass_`
- `matches`
- `balance_summary()`
- `effective_sample_size()`
- `match_summary()`
- `diagnostics`, including solver status, objective value, omitted units, and
  infeasible blocks.

## Exact Blocks

Exact blocks should be handled by solving independent subproblems per block.
This has two advantages:

- It prevents accidental cross-block matching by construction.
- It reduces graph size and usually improves performance.

For each exact block:

1. Check whether both treated and controls are present.
2. Build feasible edges within the block.
3. Apply caliper constraints.
4. Solve the block-level full matching problem.
5. Offset subclass labels when combining block results.

Blocks without overlap should either be dropped or counted against
`omit_fraction`, depending on the chosen omission semantics.

## Pseudo-Code

```python
def full_match(formula, data, distance="logit", exact=None, **options):
    parsed = parse_formula(formula, data)
    scores = estimate_propensity_score(parsed.treatment, parsed.covariates)
    distances = compute_distance_matrix(
        parsed.treatment,
        parsed.covariates,
        distance=distance,
        propensity_score=scores,
    )

    if exact is None:
        blocks = [data.index]
    else:
        blocks = group_indices_by_exact_columns(data, exact)

    all_weights = zeros_like_units()
    all_subclasses = missing_subclass_labels()
    diagnostics = []

    for block in blocks:
        treated, controls = split_block_by_treatment(block)
        if no_overlap(treated, controls):
            mark_block_omitted(block)
            continue

        block_edges = feasible_edges(distances, treated, controls)
        block_edges = apply_caliper(block_edges, options.caliper)

        graph = build_full_matching_flow_graph(
            treated,
            controls,
            block_edges,
            min_controls=options.min_controls,
            max_controls=options.max_controls,
            omit_fraction=options.omit_fraction,
        )

        solution = solve_min_cost_flow(graph, solver=options.solver)
        subclasses = decode_solution_to_subclasses(solution)
        weights = compute_full_matching_weights(subclasses, estimand="ATT")

        merge_block_result(all_weights, all_subclasses, weights, subclasses)
        diagnostics.append(solution.summary)

    return MatchResult(
        method="full",
        weights_=all_weights,
        subclass_=all_subclasses,
        diagnostics=diagnostics,
    )
```

## Failure Modes

- No feasible treated-control edges after calipers.
- Exact blocks with only treated or only controls.
- `min_controls` and `max_controls` constraints that make matching infeasible.
- `omit_fraction` too small to permit a feasible solution.
- All distances identical, causing many equivalent optimal solutions.
- Very small blocks where ratio constraints cannot be satisfied.
- Missing values removed by formula parsing but not reflected in user
  expectations.
- Perfect separation in propensity score estimation.
- Numerical instability from very large or very small distance values.
- Solver unavailable, not installed, or incompatible with the platform.
- Solver reports infeasible, unbounded, or time-limit status.

## Performance Risks

- Dense distance matrices scale as `O(n_treated * n_control)` memory.
- Exact blocks reduce graph size, but a single large block may still dominate
  runtime.
- Calipers help only if represented sparsely; computing dense distances first
  can still be expensive.
- NetworkX may be too slow for large data.
- General LP solvers in SciPy may be slower than specialized min-cost flow.
- OR-Tools introduces dependency and installation complexity.
- Template expansion for ratio constraints can create many graph nodes.
- Omissions add decision complexity and may push the design toward MIP-like
  formulations.
- Deterministic tie-breaking requires stable ordering and possibly tiny
  lexicographic perturbations, which must not distort real distances.

## Open Design Questions

- Should full matching initially support ATT only, or also ATE weights?
- Should `omit_fraction` apply to all units, controls only, or separately by
  treatment group?
- Should calipers be measured on raw distance, standardized distance, or logit
  propensity score units?
- Should OR-Tools be an optional dependency or a hard dependency once full
  matching lands?
- What is the smallest useful subset of optmatch behavior that can be tested
  rigorously without overbuilding the solver abstraction?
