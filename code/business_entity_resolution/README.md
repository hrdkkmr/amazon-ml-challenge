# Business Entity Resolution — Amazon ML Challenge 2026

For every Source-1 (S1) business in the test set, find all matching records in
Source 2 (S2) and Source 3 (S3). Scored with entity-level macro F0.5.

Pipeline: **normalize → learned transliteration → sparse-key blocking (union of
three rankings) → learned blocking filter → pairwise + competition features →
two-stage gradient-boosted matcher → one-to-one target assignment +
expected-F0.5 decision**. Validation macro F0.5 = 0.9806 at 6 candidates/S1
(see `experiments/SUMMARY.md`). Only the provided
challenge data is used (no external data, APIs, or pretrained models).

See `METHODOLOGY.md` (the filled-in `Documentation_template.md`) for the full
write-up and results.

## Requirements

* Python 3.14 (tested with CPython 3.14.3 on Windows 11), ~8 GB free RAM,
  ~6 GB free disk for caches.
* `pip install -r requirements.txt` (numpy, pandas, scipy, scikit-learn,
  rapidfuzz, pyarrow, psutil — all pinned).

## Data layout

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

Paths are configured in `src/config.py`; tunables live in
`configs/default.json` (candidate cut, worker counts, sample sizes, model).

## Reproduce end to end

Each stage caches its result under `cache/` and is skipped when the result
exists, so an interrupted run can simply be restarted (candidate generation
also checkpoints per country/source partition).

```bash
# 1. normalize the raw TSVs (chunked, parallel)                         ~3 min
python src/pipeline.py preprocess   --config configs/default.json
# 2. learn Indic->Latin token dictionary + native state names from train pairs  <1 min
python src/pipeline.py translit     --config configs/default.json
# 3. training split: blocking pool, learned candidate filter, final candidates  ~55 min
python src/pipeline.py pool_train filter_train cands_train --config configs/default.json
# 4. featurize 600k training-split S1, fit the two-stage matcher           ~21 min
python src/pipeline.py train        --config configs/default.json
# 5. validation: score all training candidates, pick the decision rule    ~20 min
python src/pipeline.py score_train evaluate --config configs/default.json
# 6. test: blocking pool -> filtered candidates -> scoring -> both TSVs  ~53 min
python src/pipeline.py pool_test cands_test predict_test --config configs/default.json
# 7. official validator (--check-ids) + stricter own checks                ~1 min
python src/pipeline.py validate     --config configs/default.json
```

`python src/pipeline.py all --config configs/default.json` runs every stage in
order (~2.5 h on a 16-core laptop). Run heavy stages one at a time: each is
sized to fit ~8 GB of RAM. To only regenerate the submission from the shipped
models (`models/`: filter, matcher, decision rule; copy `translit_dict.pkl` and
`state_map.pkl` into `cache/`), run steps 1 and 6-7.

## Outputs

| File | Content |
|---|---|
| `output/matching_results.tsv` | final matches: `source1_entity_id`, `matched_entity_ids` (one row per test S1, empty = singleton) |
| `output/candidate_pairs.tsv` | the exact candidate set scored by the matcher: `source1_entity_id`, `candidate_entity_ids` |
| `models/filter.pkl` | learned blocking filter (stage 2 of blocking) |
| `models/matcher.pkl` | trained two-stage matcher + feature list |
| `models/decision.json` | decision rule chosen on validation (expected-F0.5, one-to-one) |
| `experiments/experiments.jsonl` | logged validation experiments (candidate stats, F0.5 per rule) |
| `experiments/*.log` | stage logs (timings, memory) |

`matching_results.tsv` is always a subset of `candidate_pairs.tsv`: both are
written by `predict_test` from the same saved candidate table
(`cache/cands_test.parquet`), and `validate` checks it.

## Code map (`src/`)

| Module | Role |
|---|---|
| `config.py` | paths + parameters (`Config.load(json)`) |
| `data_io.py` | chunked TSV reading, parallel normalization into Parquet caches |
| `normalization.py` | name/address normalization (accents, legal suffixes, abbreviations, states, numbers) |
| `translit.py` | Indic-script transliteration: learned token dictionary + rule fallback + phonetic key |
| `records.py` | enriched record access (`RecordStore`), native-script state mapping |
| `blocking.py` | blocking keys, sparse inverted indexes, top-K retrieval (serial + memory-mapped parallel) |
| `candidate_generation.py` | per-country/per-source candidate pool (checkpointed partitions) |
| `filtering.py` | learned blocking filter features (string-free) |
| `features.py` | pairwise similarity features (rapidfuzz, token/number/state/legal) + S1-side competition |
| `training.py` | candidate table, target-side competition, featurization loop, model fitting |
| `evaluation.py` | S1-level split, macro F0.5 (official definition), candidate recall stats |
| `inference.py` | one-to-one assignment, threshold / expected-F0.5 decisions, TSV writers |
| `validation.py` | official validator + streaming consistency checks |
| `pipeline.py` | CLI stages |

`scripts/` holds analysis helpers (`inspect_data.py` data profile,
`exp_blocking.py` blocking recall study, `error_analysis.py` FP/FN reports,
`exp_model.py` model variants, `final_report.py` validation breakdown,
`smoke_test.py` small end-to-end test, `package_submission.py` builds
FINAL_SUBMISSION/ and the zip). `utils/validate_submission.py` is the
official validator, copied unchanged.
