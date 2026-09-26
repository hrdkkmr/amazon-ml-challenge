# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We resolve every Source-1 business against Sources 2/3 with a three-stage
pipeline that runs on one laptop within an ~8 GB memory budget. First,
**sparse-key blocking** uses order-insensitive token-pair keys and a union of
three rankings. Second, a **learned blocking filter** uses only blocking
signals and cuts the pool from 82 to **6 candidates per S1 while keeping 98.1%
of true pairs**. Third, a **two-stage gradient-boosted matcher** is followed by
an expected-F0.5 decision rule with a one-to-one target constraint. Measured
on 441k held-out training S1, the final system reaches **macro F0.5 = 0.9806**
(precision 0.996, recall 0.952, singleton accuracy 0.977). It uses only the
provided data: no external data, APIs or pretrained models.

---

## 2. Methodology

### 2.1 Problem Analysis

We profiled all files with memory-safe chunked reads
(`scripts/inspect_data.py`, `experiments/data_profile.json`):

* **Sizes.** Train has 2.21M S1, 5.03M S2 and 5.29M S3 records; test has
  1.73M S1, 4.89M S2 and 5.08M S3. All IDs are unique, and only
  `business_address` has nulls (about 3%).
* **Ground truth.** There are 7.64M true pairs, **3.46 matches per S1** on
  average, up to 11 per S1, and 5.6% singletons. **No S2/S3 record matches
  more than one S1.** About 26% of training S2/S3 records match no S1 at all.
  Test has relatively more of these unmatched distractor records (1.73M S1 vs
  9.97M targets), plus a new country, France (15% of test S1).
* **Name noise.** Case and accent differences, and legal-suffix variants (Pvt
  Ltd / Private Limited / L.L.C. / [LLC]). Word reordering and typos,
  including digits used as letters ("0f", "5ervices"). Honorifics (Mr, M/s,
  Shri) and junk prefixes (">>", "--"). Appended filler words (Services,
  Center) and appended phone numbers. Aliases ("X d/b/a Y", "X formerly: Y"),
  web domains ("bethchapel.com") and acronyms. About 9% of Indian S2/S3 names
  are written entirely in Devanagari, Bengali, Tamil, Telugu, Kannada or
  Gujarati script.
* **Address noise.** Abbreviations, reordered components and partial
  addresses (only city and state, or empty). State names vs codes, including
  state names in native script. **Perturbed house numbers** (39 ↔ 3941,
  254 ↔ 25, 158 ↔ 159), ordinal noise ("18st") and city typos.
* **Records matching on one field only.** Some true S2/S3 records carry a
  random alias name and match only by address. Others have an empty address
  and match only by name.
* **Hard negatives exist.** Some records have the same name at a neighbouring
  house number (905 vs 903 Wiehl Road), or a legal-suffix sibling (LLC vs
  PLLC), and are *not* matches.

### 2.2 Solution Strategy

**Approach Type:** Blocking (sparse retrieval + learned filter) → pairwise
classifier with competition context → constrained F0.5 decision.

**Core Innovations:**
1. Rare **combination keys**, with a union of name-only, address-only and
   total rankings. Individual words are very common in this data, but their
   pairs are not.
2. A **blocking filter learned on blocking signals only**. It makes the
   candidate set 13× smaller than the pool at a cost of 0.03 points of recall.
3. **Transliteration learned from the training pairs themselves**.
4. **Exploiting that each target belongs to at most one S1**: target-side
   competition features plus a one-to-one assignment.
5. An **expected-F0.5 subset decision** per S1, which lets each S1 get zero,
   one or many matches.

---

## 3. Candidate Generation (Blocking)

### Normalization

Implemented in `normalization.py`, `translit.py` and `records.py`:

* NFKC normalization, lower-casing and Latin diacritic folding (Indic vowel
  signs are preserved), then punctuation removal.
* Legal suffixes are mapped to canonical forms. Honorifics and filler words
  are removed into a separate `name_core`, and the original normalized name
  is kept alongside it.
* Aliases are split (d/b/a, formerly, aka), and web domains are reduced to
  their root.
* Address abbreviations are expanded, US and Indian state names are mapped to
  codes, and leading zeros and ordinals are normalized.
* **Transliteration:** non-Latin names are translated token by token with a
  dictionary learned from 551k aligned training pairs (1,312 entries, 93–95%
  token coverage on test). Unseen tokens fall back to a rule-based Brahmi
  transliterator, which uses one offset table for all Indic Unicode blocks.
  Native-script state names are mapped to codes with a dictionary voted from
  training pairs.

### Stage 1: sparse retrieval (`blocking.py`)

Each record emits order-insensitive keys:

| Family | Keys |
|---|---|
| name | core tokens, concatenated core (matches domain-style names), unordered token pairs |
| address | informative tokens (generic words removed; ordinals → digits; `b239` also emits `239`), unordered token pairs |
| cross | name token × address token |

* Keys are hashed to stable 64-bit integers. For each (country, target
  source, key family) we build one CSR postings matrix, dropping keys with
  document frequency above 3,000.
* A query's score for a target is the sum of IDF weights of the keys they
  share. Scores come from sparse matrix products over batches of 1,000
  queries, spread across 5 worker processes that memory-map the same index.
* Countries are treated as an open set of labels: France is simply one more
  partition. No Cartesian product is ever formed.
* **Pool per S1 and source:** the union of the top-30 by total score, the
  top-15 by name score and the top-15 by address score. The name ranking
  recovers matches with empty addresses; the address ranking recovers
  alias-named records.

### Stage 2: learned filter (`filtering.py`)

* A HistGradientBoosting model sees only 15 cheap, string-free signals of each
  pool pair:
  * the three IDF scores and their sum, and the three ranks;
  * per-(S1, source) gaps to the best total, name and address scores, and the
    pool size;
  * target-side competition: how many S1 retrieved this target, and the
    margin and rank of this S1 among them. These are exact per partition,
    because a target competes only with S1 records of the same country.
* It is trained on the pool pairs of 200k training S1 (sample A).
* Pairs with q < 0.002 are dropped. The survivors are **exactly**
  `candidate_pairs.tsv` and exactly what the matcher scores.

| Blocking design (validation S1) | Pair recall | Candidates / S1 |
|---|---|---|
| unigram + bigram keys, top-10 | 0.9354 | 20.0 |
| + token-pair keys, top-10 | 0.9512 | 20.0 |
| + union of 3 rankings (10/3/3) | 0.9674 | 22.1 |
| pool 30/15/15 (unfiltered) | 0.9813 | 82.2 |
| **pool 30/15/15 + learned filter (final)** | **0.9810** | **6.03** |

- **Blocking keys used:** name tokens, token pairs and concatenation; address
  tokens and token pairs; name×address pairs. All are IDF-weighted, with
  df ≤ 3000.
- **Candidate pairs generated:** 12,870,155 test pairs (**7.43 per S1**, median
  7; only 93 of 1.73M S1 have none). On training, 13.3M pairs (6.0 per S1).
- **How true matches were not lost:** recall was measured on held-out S1 after
  every design change (table above). Missed pairs were inspected, which led to
  the pair keys, the ranking union and the ordinal/alphanumeric number
  normalization. The pool depth was then chosen at the point where the
  filtered recall stops improving.

---

## 4. Matching Model

**Features used** (computed only for candidate pairs; `features.py`,
`training.py`; 86 features in total):

- **Name:** rapidfuzz ratio, partial ratio, token-sort, token-set and WRatio;
  Jaro-Winkler; concatenated-name ratio and partial ratio (for domains); a
  token-set score on the alias part; token Jaccard; fuzzy token coverage in
  both directions; Jaccard of phonetic skeletons; same bag of tokens; first
  token equal; legal-form agreement and conflict; acronym match; phone-token
  flag; script and domain flags. Phone-like digit runs are stripped before
  comparison.
- **Address:** ratio, token-set, token-sort, partial and partial-token-set;
  fuzzy token coverage in both directions; empty-address flag; number Jaccard;
  numbers unseen on the S1 side; house-number equality; **house-number fuzzy
  similarity and containment**; ZIP/PIN agreement; state agreement.
- **Rarity:** how often the core name occurs among targets and among S1
  records. Common names need address evidence.
- **Blocking:** the three IDF scores, the three ranks, the filter
  probability q (out-of-sample, since the filter and matcher use disjoint
  training S1), and the filter's gap and competition signals.
- **Competition:** per S1 and per S1×source, gaps, margins and ranks of
  several similarities, plus the number of candidates. Per target, the number
  of competing S1 and this S1's margin and rank among them.

**Model type:** scikit-learn `HistGradientBoostingClassifier` (BSD license,
CPU-only, with early stopping), used in **two stages**:

* **Stage 1:** 255 leaves, learning rate 0.05, up to 1,200 iterations. It is
  trained on the candidates of 600k training-split S1 (3.61M pairs, 56%
  positive). Those S1 are disjoint from the filter's sample; the negatives
  are exactly the hard negatives that survive blocking.
* **Stage 2:** the same model type, trained on all features plus context from
  stage-1 probabilities over all candidates of the same S1: rank, gap to the
  best, margin, sum, and the number above 0.5 and above 0.8. The stage-1
  probabilities for this are 2-fold cross-fitted, out of fold.

We compared against a standardized logistic regression (0.934 F0.5) and
single-stage HGB variants (0.9744 → 0.9790).

**Decision rule / threshold selection:**

1. **One-to-one:** each target is kept only for the S1 that gives it the
   highest probability.
2. **Per S1:** choose the top-j candidates that maximize the approximate
   expected F0.5, `1.25·Σp_top-j / (0.25·(Σp_all + 0.1) + j)`, against
   P(no match) = Π(1−p). Zero, one or many matches are allowed.

This rule was selected on validation against a threshold sweep from 0.10 to
0.95. The best fixed threshold (0.70) gives 0.97999, and the expected-F0.5
rule gives 0.98059. The rule's parameters change the score by less than
0.00002, so the choice is not overfitted.

---

## 5. Results & Error Analysis

All results are on held-out validation S1 (441,364 entities, split by entity).
Recall counts blocking misses.

| Version | Candidates / S1 | Model | Macro F0.5 | Precision | Recall |
|---|---|---|---|---|---|
| v1 | 22.1 (rank cut) | HGB-63, single stage | 0.9743 | 0.9938 | 0.9388 |
| v2 | 5.3 (filter, pool 15/5/5) | HGB-255 | 0.9781 | 0.9939 | 0.9502 |
| v3 | 6.0 (filter, pool 30/15/15) | HGB-255 | 0.9790 | 0.9939 | 0.9534 |
| **v4 (final)** | **6.0** | **two-stage HGB-255 + expected-F0.5** | **0.9806** | **0.9958** | **0.9521** |

- **F_0.5 Score (macro, validation):** **0.9806**. By country: US 0.9824,
  India 0.9779. Singleton accuracy is 0.977, and 84.3% of S1 score a perfect
  1.0.
- **Error budget:** of 1.53M true validation pairs, 29.1k (1.9%) are lost in
  blocking and 44.1k (2.9%) are rejected by the matcher. There are only 6.1k
  false-positive pairs.
- **Common false positives (wrong merges):**
  * sibling entities with the same name and a different legal suffix
    (LLC vs PLLC);
  * the same name at a neighbouring house number;
  * an exact name match whose target has an empty address;
  * alias-named records that share the S1's exact address.
- **Common false negatives (missed matches):**
  * S1 with exactly one true match (F0.5 0.932 in that segment), where the
    only candidate resembles the known hard-negative patterns;
  * heavily perturbed house numbers combined with a name typo;
  * random-alias names with only a partial address;
  * name-only records (empty address) with common names.

---

## 6. Conclusion

Most of the achievable accuracy came from blocking engineered against the
observed noise, rather than from a larger model:

* Combination keys and a ranking union lifted blocking recall from 93.5% to
  98.1%.
* A learned filter cut candidates from 82 to 6 per S1 without losing recall.
  That made the downstream matcher both cheaper and more accurate.
* A two-stage gradient-boosted matcher with competition context, combined
  with an F0.5-aware, one-to-one decision rule, gives validation macro
  F0.5 0.9806 at 99.6% precision.

The whole pipeline, from raw TSVs to validated submission files, runs in
about 2.5 hours on a laptop.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` contains:

* `src/`: the full pipeline;
* `configs/default.json`: the final configuration;
* `models/`: the trained filter and matcher, the decision rule, and the
  learned transliteration and state dictionaries;
* `scripts/`: profiling, experiment and error-analysis scripts;
* `utils/`: the official validator, unchanged;
* `experiments/`: logged runs;
* `README.md`: exact commands;
* `requirements.txt`: pinned dependencies.

The entry point is `python src/pipeline.py <stage …>`, and
`python src/pipeline.py all --config configs/default.json` regenerates
`output/candidate_pairs.tsv` and `output/matching_results.tsv` from the raw
data. Stage order: `preprocess → translit → pool_train → filter_train →
cands_train → train → score_train → evaluate → pool_test → cands_test →
predict_test → validate`.

Runtimes on a 16-core laptop, run one stage at a time:

| Stage | Time | Peak memory |
|---|---|---|
| preprocess | 3 min | – |
| pool_train | 38 min | 3.1 GB |
| filter_train | 7 min | – |
| cands_train | 9 min | – |
| train | 21 min | – |
| score_train | 12 min | – |
| evaluate | 8 min | – |
| pool_test | 34 min | 2.6 GB |
| cands_test | 6 min | – |
| predict_test | 13 min | – |
| validate | 1 min | – |

### B. Additional Results

See `experiments/SUMMARY.md` for all blocking, filter and model experiments,
including rejected ideas (4-character-prefix name keys: +0.0004 recall; the
two-stage model with small stage-1 trees: +0.0014). The raw logs are in
`experiments/*.jsonl`.
