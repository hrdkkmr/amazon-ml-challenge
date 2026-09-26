# Experiment summary

All numbers are measured on held-out **validation S1 entities** (20% of training
S1, split by entity, seed 42; 441,364 S1) unless marked otherwise. F0.5 is
the official entity-level macro F0.5 (singletons included), computed against
the full truth, so blocking misses count as false negatives.
Raw records: `experiments.jsonl`, `model_runs.jsonl`, `blocking_runs.jsonl`.

## Blocking (20k validation-S1 sample, `scripts/exp_blocking.py`)

| ID | Keys | Candidate selection | Pair recall | Cands/S1 |
|---|---|---|---|---|
| B1 | unigrams + adjacent bigrams (df ≤ 3000) | top-10 total per source | 0.9354 | 20.0 |
| B1 | same | top-50 total | 0.9609 | 99.9 |
| B2 | + unordered name/addr/cross token pairs | top-10 total | 0.9512 | 20.0 |
| B3 | B2 | union top-10 total / 3 name / 3 addr | 0.9666 | 22.0 |
| B3 | B2 | union 15 / 5 / 5 | 0.9727 | 34.8 |
| B3 | B2 | union 30 / 15 / 15 | 0.9804 | 82.1 |
| B3 | B2 | union 50 / 50 / 50 | 0.9860 | 209.7 |
| B4 | B2 + 4-char-prefix name pairs | union 15 / 5 / 5 | 0.9731 | 34.5 (rejected: +0.0004) |

Failure modes behind B1→B2→B3: common single words (beacon, telecom, des
moines) exceed the df cap; reordered names break adjacent bigrams; matches with
an empty address lose to address-sharing distractors in a single ranking.

## Learned blocking filter (full validation, `pipeline.py filter_train`)

| Pool | Filter | Pair recall | Cands/S1 |
|---|---|---|---|
| union 15/5/5 | none | 0.9742 | 34.8 |
| union 15/5/5 | q ≥ 0.002 | 0.9740 | 5.34 |
| union 30/15/15 | none | 0.9813 | 82.2 |
| union 30/15/15 | q ≥ 0.002 | **0.9810** | **6.03** |
| (reference) fixed cut 10/3/3 | — | 0.9674 | 22.15 |

## Matching model (full validation)

| Version | Candidates | Model | Decision | F0.5 | P (micro) | R (micro) | Singleton acc |
|---|---|---|---|---|---|---|---|
| v1 | cut 10/3/3 (22.1/S1) | HGB 63 leaves, 200k S1 | thr 0.70 + 1:1 | 0.97430 | 0.9938 | 0.9388 | 0.968 |
| v2 | pool 15/5/5 + filter (5.3/S1) | HGB 255 leaves, 600k S1 | thr 0.65 + 1:1 | 0.97810 | 0.9939 | 0.9502 | 0.969 |
| v3 | pool 30/15/15 + filter (6.0/S1) | same | thr 0.65 + 1:1 | 0.97903 | 0.9939 | 0.9534 | 0.967 |
| **v4 (final)** | same as v3 | **two-stage** HGB | **expected-F0.5 + 1:1** | **0.98059** | **0.9958** | **0.9521** | **0.977** |

Model variants on 60k validation S1 (`scripts/exp_model.py`, v1 features / v3 features):

| Variant | v1 features | v3 features |
|---|---|---|
| Logistic regression (standardized) | 0.9343 | – |
| HGB 63 leaves, 400 it | 0.9744 | – |
| HGB 127 leaves, 800 it | 0.9753 | – |
| HGB 255 leaves, ≤1200 it | 0.9758 | 0.9790 |
| Two-stage (HGB-63 stage 1) | 0.9758 | – |
| Two-stage (HGB-255 both stages) | – | **0.9803** |

Feature iterations (small 20k-S1 smoke test, same candidates): base features
0.9583 → + house-number similarity/containment, name rarity, acronym,
phone-token features 0.9669.

One-to-one target assignment: +0.0002 at the best threshold (v1: 0.97430 vs
0.97415 without).

## Final validation breakdown (v4, `scripts/final_report.py`)

* F0.5 by country: US 0.9824, India 0.9779 (France absent from training).
* F0.5 by number of true matches: 0 → 0.977, 1 → 0.932, 2 → 0.978, ≥3 → ~0.985.
* 84.3% of validation S1 score a perfect 1.0.
* Of 1,528,629 true pairs: 29,132 missed by blocking, 44,089 rejected by the
  matcher; 6,068 false-positive pairs (657 of them on singleton S1).

## Test-set outputs (v4)

1,732,544 S1 rows; 5,850,285 matches; 97,123 S1 predicted as singletons;
12,870,155 candidates (7.43/S1; 93 S1 without candidates); official validator:
PASS.
