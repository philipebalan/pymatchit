# AGENTS.md

## Project goal

Build a Python-native causal matching library inspired by R MatchIt and optmatch.

## Priorities

1. Correctness over feature breadth.
2. Deterministic output.
3. Explicit tests for every matching method.
4. No hidden outcome modeling inside the design-stage API.
5. Keep matching separate from treatment-effect estimation.

## Initial supported methods

- nearest neighbor matching
- exact matching
- subclassification
- optimal pair matching
- balance diagnostics

## Deferred methods

- full matching
- genetic matching
- cardinality matching
- multi-category treatment
- survey weights
- exact parity with R object classes

## Coding standards

- Python 3.11+
- Use type hints
- Use pandas DataFrames for public APIs
- Use numpy arrays internally where performance matters
- Prefer simple readable implementations before optimization
- Add pytest tests for edge cases
- Avoid dense distance matrices when an implementation can block or stream

## Validation

Where feasible, tests should compare against known small examples with hand-computed matches and weights.