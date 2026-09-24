# Business Entity Resolution

Matching 2.2M deduplicated reference entities (S1) against ~10.3M noisy business
records (S2 + S3).

Naive comparison is `2,206,821 x 10,320,219 = 22,774,876,013,799` pairs (~22.8
trillion), so the pipeline is built around **blocking**: only pairs that share a
key are ever compared. This repository currently implements the infrastructure
and the first blocker (exact normalized name) plus full blocking evaluation.

**Status: milestone 1 of 3.** See [Roadmap](#roadmap).

---

## Headline finding (read this before planning the model)

Measured against the real normalized names across all 10.3M records, on a
100,000-entity S1 sample:

| Metric | Value |
|---|---|
| Pair recall (normalized name exact match) | **25.79%** |
| Pair recall gained by also dropping separators (`name_key`) | +1.15% |
| **S1 entities for which ALL true matches are retrieved** | **3.74%** |
| S1 entities with at least one true match retrieved | 60.76% |
| Candidate precision of the resulting pairs | ~7.7% |

Two conclusions that shape the rest of the work:

1. **Exact-name blocking alone cannot produce a competitive score.** It recovers
   about a quarter of true matches and leaves 96% of S1 entities with an
   incomplete candidate set. Under a per-entity macro F0.5, an S1 missing even
   one of its matches has a hard ceiling on its own score. This blocker is a
   foundation, not a solution.
2. **Precision is weak even where it matches.** ~7.7% of exact-name candidates
   are true matches: many distinct businesses share a name. Precision work in the
   matcher is not optional, and `beta = 0.5` makes a false positive cost 4x a
   false negative.

Next blockers (token, character n-gram, then multilingual dense retrieval) are
what move recall. Re-run the blocking evaluation after each one.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore                      # excludes the dataset and generated artifacts
├── configs/
│   └── config.yaml                 # all paths, thresholds and switches
├── notebooks/
│   └── 1.ipynb                     # exploration ONLY - no production logic
├── src/                            # importable, tested, stable code
│   ├── data_loader.py              # config, streaming TSV IO, ground truth
│   ├── normalization.py            # Unicode-safe multilingual normalization
│   ├── blocking.py                 # inverted index + blocker union
│   ├── evaluation.py               # blocking metrics + per-entity F0.5
│   ├── utils.py                    # logging, progress, hashing, id codec, device
│   ├── features.py                 # milestone 2 (stub)
│   └── matching_model.py           # milestone 2 (stub)
├── scripts/                        # CLI entry points for heavy work
│   ├── prepare_data.py             # stage 1
│   ├── build_indexes.py            # stage 2
│   ├── generate_candidates.py      # stage 3
│   ├── evaluate_blocking.py        # stage 4
│   ├── train_model.py              # stage 5 (milestone 2 - stub)
│   └── predict.py                  # stage 6 (milestone 2 - stub)
├── outputs/                        # generated (gitignored)
└── logs/                           # run logs (gitignored)
```

**Architecture rule:** `notebooks/1.ipynb` is for exploration only. Anything
stable lives in `src/`; anything heavy is runnable from `scripts/`.

> `scripts/evaluate_blocking.py` and `src/utils.py` are additions to the file list
> in the brief. Evaluation had to be command-line runnable and belonged in
> neither `train_model.py` nor the library modules; `utils.py` holds the shared
> primitives (logging, progress, id codec, device detection).

---

## Environment setup

The dataset is **not** in this repository. It lives on the HPC.

### Local / HPC (CPU only is enough)

```bash
git clone <repo-url> entity-resolution
cd entity-resolution

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Only `numpy`, `pandas` and `PyYAML` are required to run stages 1-4.
`tqdm`/`psutil` add nicer progress and RSS logging. Everything GPU/model related
in `requirements.txt` is commented out and optional - **no stage requires a GPU**.

Verify the install:

```bash
python -m src.normalization        # multilingual normalization self-check
python -c "from src.utils import describe_device; print(describe_device())"
```

---

## Configuration

All paths live in `configs/config.yaml`. **No path is hardcoded in Python.**
Defaults work for a fresh clone with data in `./train` and `./test`.

Override without editing the file - environment variables take precedence over
`config.yaml`:

```bash
export ER_DATA_ROOT=/scratch/challenge/data/train
export ER_TEST_DATA_ROOT=/scratch/challenge/data/test
export ER_WORK_DIR=/scratch/$USER/er_outputs
```

or per command:

```bash
python scripts/prepare_data.py --data-root /scratch/challenge/data/train --work-dir /scratch/$USER/er_outputs
```

Precedence: CLI flag > environment variable > `config.yaml` > built-in default.

Every script also accepts `--config /path/to/other.yaml`.

---

## Running the pipeline

### Local smoke test (~30 seconds, <400 MB)

```bash
python scripts/prepare_data.py      --splits train --limit 100000 --overwrite
python scripts/build_indexes.py     --limit 100000 --overwrite
python scripts/generate_candidates.py --limit-s1 100000
python scripts/evaluate_blocking.py --split val --no-save
```

Smoke-test recall looks near zero **by design**: it indexes only the first 100k
target rows, so most true matches are not in the index. Use it to check the
plumbing, not the quality.

### Full run

```bash
# 1. Normalize all sources (~10 min, streaming, ~150 MB RSS)
python scripts/prepare_data.py

# 2. Build the inverted indexes (~2-4 min, ~250 MB peak per index)
python scripts/build_indexes.py

# 3. Generate candidate pairs for all 2.2M S1 entities
python scripts/generate_candidates.py

# 4. Evaluate against ground truth (validation split)
python scripts/evaluate_blocking.py --split val
python scripts/evaluate_blocking.py --split all      # val + train + all
```

Stage-by-stage reference:

| Command | Reads | Writes | Peak RAM |
|---|---|---|---|
| `prepare_data.py` | `train_source{1,2,3}.tsv` | `outputs/prepared/train_source{1,2,3}_norm.tsv` | ~150 MB |
| `build_indexes.py` | prepared S2/S3 | `outputs/indexes/train_source{2,3}_exact_name/` | ~250 MB/index |
| `generate_candidates.py` | prepared S1 + indexes | `outputs/candidates/candidate_pairs.tsv` | ~300 MB |
| `evaluate_blocking.py` | candidates + ground truth | `outputs/candidates/blocking_metrics_*.json` | ~600 MB |

`prepare_data.py` also writes `{split}_source1_norm.tsv` with a `split` column
(`train`/`val`) per S1 entity.

### HPC notes

* Everything is a plain CLI command - wrap it in your scheduler's batch script.
  Slurm directives are not included because the cluster's configuration was not
  specified; add `#SBATCH` lines appropriate to your site.
* Long stages log progress at a fixed interval and stream their output, so
  `tail -f logs/prepare_data.log` works.
* Set `ER_WORK_DIR` to scratch: `outputs/prepared/` is ~1.5 GB and
  `candidate_pairs.tsv` grows with candidate volume.
* Lower `io.chunksize` in `config.yaml` if a node has little RAM; peak memory
  tracks the chunk size, not the dataset size.
* `--overwrite` is off by default, so re-running a completed stage is a no-op
  (it verifies the existing artifact and skips).

---

## How blocking works

`normalized_name -> entity ids`, built per target source.

**Exact-name index.** The textbook implementation is `dict[str, list[int]]`, which
would hold ~4.0M string keys for S2 and cost ~0.7-1 GB before postings. Instead
the index is flat numpy:

| Array | dtype | Size (S2) | Purpose |
|---|---|---|---|
| `key_hashes` | uint64, sorted | ~32 MB | `searchsorted` lookup |
| `key_offsets` | int64 | ~32 MB | slice into `keys_blob` |
| `keys_blob` | concatenated UTF-8 | ~100 MB | exact verification |
| `postings` | int64 | ~40 MB | entity codes, grouped by key |
| `postings_offsets` | int64 | ~32 MB | slice into `postings` |

Queries are answered with `np.searchsorted`; **every hash hit is verified against
the stored string**, so the index cannot emit a spurious pair. A 64-bit hash
collision would cause a missed pair (~4e-7 probability at 4M keys) - blocking
fails toward "miss", never toward "wrong candidate". Keys are hashed with
blake2b, not python's `hash()`, because the latter is salted per process and
would make a persisted index unreadable in the next run.

**Union.** Multiple blockers each return postings for the same S1; the candidate
set is their union. Pairs are packed as `s1_position * 10**11 + entity_code`,
which lets a single `np.unique` do union + dedupe + sort at once. Each pair keeps
a `blockers` provenance column, so you can later see which blocker actually
earns its keep.

**Candidate cap.** `blocking.max_candidates_per_source` (or `--max-candidates`)
limits candidates per S1. This is a recall/precision trade-off, not a detail:
capping silently deletes true matches before the matcher sees them. Choose it
from the `recall_at_k_file_order` numbers in the evaluation report.

---

## Normalization

Multilingual and Unicode-safe, in `src/normalization.py`. The design constraint
that matters:

> **Never drop combining marks.**

India-sourced records appear in Devanagari, Kannada and Bengali. A naive
"strip accents" routine (NFD, then remove all `Mn` characters) is correct for
French and destroys Devanagari - vowel signs and the virama are combining marks,
so `कंस्ट्रक्शंस` would collapse into noise and every Indian business name would
start colliding with every other one.

Pipeline: `NFKC -> per-character table -> case -> whitespace collapse -> truncate`

Accent folding is applied **only to Latin-script characters**, one character at a
time. The table is built once per process over the whole code space and applied
with `str.translate` (C speed). Unassigned code points are excluded, keeping the
table at ~12k entries instead of ~900k.

```
"Orelee's Barbershop"                  -> "orelee s barbershop"
"Café Béque"                           -> "cafe beque"
"B+ Retail Inc"                        -> "b retail inc"
"राम मार्केटिंग प्राइवेट लिमिटेड"          -> "राम मार्केटिंग प्राइवेट लिमिटेड"   (marks preserved)
"ಶಿವಶಕ್ತಿ ವಿದ್ಯಾಲಯ"                       -> "ಶಿವಶಕ್ತಿ ವಿದ್ಯಾಲಯ"                  (marks preserved)
```

Original columns are never overwritten - `business_name` stays exactly as it came
in, because character-level features need the raw text later. `name_key` is an
extra separator-free variant for equality-only blocking.

---

## Validation protocol

**Split by S1 entity, never by pair.** All candidate pairs of one S1 stay in the
same split. Splitting pairs would leak: the same S1 (often with near-identical
address text) would appear in both train and val.

The split is a pure function of the S1 id (blake2b hash -> train/val), so the
candidate generator, the preparer and the evaluator agree on the held-out set
with no shared state to drift. Default 20% val, recorded in the `split` column of
`train_source1_norm.tsv`.

Blocking evaluation reports recall, candidate volume **and** the F0.5 you would
get if every candidate were accepted - that last number is the ceiling the
current blockers impose on any downstream matcher, and it says how much precision
work is left.

**Open question - zero-match entities.** 123,247 S1 entities (5.6%) have an empty
ground-truth list. The challenge's macro-average presumably includes them, which
means predicting *anything* for such an entity scores 0. Both policies are
computed and reported (`f05_accept_all_macro` averaging over entities that have
matches, and `f05_accept_all_macro_score_zero`); which one the leaderboard uses
should be confirmed. `evaluation.zero_match_policy` in `config.yaml` selects the
labelled primary.

`recall_at_k_file_order` is reported for K in 10/25/50/100/200. For an unranked
blocker "first K" is file order, i.e. arbitrary - it bounds what a future
re-ranker can achieve and makes the cost of capping volume explicit. It becomes
meaningful once blockers emit scores.

---

## Data facts (measured, not assumed)

| Fact | Value |
|---|---|
| S1 / S2 / S3 rows | 2,206,821 / 5,034,616 / 5,285,603 |
| Ground-truth rows | 2,206,821 (one per S1, no duplicates) |
| S1 with zero matches | 123,247 (5.6%) |
| Total true matches | 7,638,365 (S2: 3,693,619 / S3: 3,944,746) |
| Matches per S1 | min 0, max 11 (mode 3) |
| S1 with >= 1 S2 / S3 match | 1,919,076 / 1,940,545 |
| S2 / S3 missing `business_address` | 168,967 / 175,916 |
| Missing `business_name` | 0 |
| Entity ids | unique per source, numeric part 2-9 digits, **no leading zeros** |
| Unique `name_norm` (S1 / S2 / S3) | 1,520,684 / 3,949,779 / 4,191,008 |

Two consequences:

* **Address cannot be the primary blocking signal** - ~3.4% of target records have
  none, and ~170k records would be unreachable.
* **Ids pack into int64 losslessly** (no leading zeros), which is what lets the
  ground truth live in ~61 MB instead of ~400 MB of id strings and lets the
  evaluator do set membership with `searchsorted` instead of python sets.

Note: this normalization yields 3,949,779 unique S2 names, while an earlier
figure of ~4,028,180 was reported (~2% more). The difference is explained by the
`punctuation_to_space` and case settings here collapsing slightly more variants
(e.g. `heassociates.com` -> `heassociates com`, `Pvt.` -> `pvt`). The setting is
configurable; if the earlier normalization was validated against something
specific, it can be reproduced by flipping `normalization.punctuation_to_space`.

---

## Outputs

```
outputs/
├── prepared/
│   ├── train_source1_norm.tsv        entity_id, business_name, business_address,
│   ├── train_source2_norm.tsv        country, name_norm, name_key, address_norm,
│   ├── train_source3_norm.tsv        country_norm [, split]
│   └── prepare_manifest.json         row counts + settings (provenance)
├── indexes/
│   ├── train_source2_exact_name/     flat .npy arrays + keys.bin + meta.json
│   ├── train_source3_exact_name/
│   └── index_summary.json
└── candidates/
    ├── candidate_pairs.tsv           source1_entity_id, matched_entity_id,
    │                                 source, blockers
    ├── candidate_pairs_stats.json
    └── blocking_metrics_*.json
```

`candidate_pairs.tsv` is an **intermediate** artifact, not a submission. Its
format is defined by this project; matched ids are always valid S2/S3 ids and
deduplicated per S1.

`matching_results.tsv` (the graded artifact) is **not produced yet** - see
`scripts/predict.py`. It is blocked on the challenge's exact submission format,
which was not provided. The only format evidence available is the training ground
truth (`source1_entity_id`, comma-separated `matched_entity_ids`), which is the
likely shape, but guessing the format of the graded file would be worse than
asking. Once confirmed, `predict.py` becomes a small merge-join; the constraints
it must satisfy are already documented in that file (every S1 exactly once,
including the 123,247 with no match; deduplicated ids; deterministic ordering).

---

## Roadmap

**Milestone 1 (done): infrastructure + exact-name blocker + evaluation**

* config-driven paths, streaming IO, compact ground truth
* Unicode-safe multilingual normalization
* flat numpy inverted index with a persisted on-disk format
* union of blockers with provenance
* blocking evaluation: recall, volume, reduction ratio, per-entity F0.5 ceiling
* S1-level validation split

**Milestone 2 (next): more blockers, then a matcher**

Order matters - each step should be measured before the next:

1. **Token / inverted-index blocking** - shares tokens, not the whole string.
   Expected to move recall far more than any modelling work. The generic index
   and union machinery are already in place (`BLOCKER_TOKEN` is registered and
   raises a clear "not implemented" error).
2. **Character n-gram retrieval** - catches typos and transliteration variance.
3. **TF-IDF / BM25 lexical retrieval**, memory-efficient.
4. **Dense multilingual embedding retrieval** - GPU when available, embeddings
   computed once per record (not per pair) and cached.
5. **Matcher** - threshold on one lexical score first, then gradient-boosted
   trees on lexical + address features, then semantic features. See
   `src/features.py` and `src/matching_model.py` for the planned contract.
6. **Cross-encoder re-ranking** on top candidates only, if it still pays off.

`src/features.py`, `src/matching_model.py`, `scripts/train_model.py` and
`scripts/predict.py` are deliberate stubs that raise `NotImplementedError` with an
explanation. Building a matcher now would mostly measure the blocker's recall.

---

## Constraints

* **No external data or internet augmentation.** Nothing in this repository
  performs a network lookup; embeddings must come from a locally-run model.
* **Model license and parameter-count constraints** apply to the final solution.
  Record the chosen encoder and its size in `src/features.py` before shipping.
* **CPU must remain sufficient** for the whole pipeline. `utils.resolve_device()`
  returns `cuda` when available and `cpu` otherwise; nothing is GPU-gated.
* **Never materialize the 22.8T cross product.** All stages are chunked and
  streamed; peak memory is governed by `io.chunksize`.

---

## Assumptions made

Listed explicitly since they were not specified in the brief:

1. **Repository root = the challenge folder.** The existing `train/`, `notebooks/`
   and `src/` sat directly in the working directory, so the project root is that
   directory rather than a nested `entity-resolution/`.
2. **Data at `./train` and `./test` by default**, matching the files already
   present. Override with `ER_DATA_ROOT` / `--data-root` for the HPC.
3. **`notebooks/1.ipynb` and `src/normalization.py` were both empty (0 bytes)**, so
   there was no prior normalization logic to reuse despite the brief referring to
   one. Normalization was implemented from scratch and validated against
   Devanagari and Kannada records.
4. **The ground truth covers all 2.2M S1 entities**, including 123,247 with an
   empty match list.
5. **`train_source1.tsv` row order is the canonical S1 order**; candidates are
   emitted in that order.
6. **No scheduler or GPU specifics were assumed** - no `#SBATCH` directives, no
   GPU model names, no HPC paths.
7. **`matching_results.tsv` format is unknown** and deliberately not guessed.
