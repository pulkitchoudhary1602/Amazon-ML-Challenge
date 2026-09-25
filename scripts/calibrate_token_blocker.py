"""Token-blocker calibration: the rarest-K token blocker, measured.

What this answers
-----------------
Phase 0.2-0.4 priced a token blocker **analytically** - "a pair is reachable if it
shares a name token whose document frequency is at most the cap" - and reported
42.3905% of true pairs, 209,762,533 estimated candidate rows and 52.3535%
structural-zero S1. Those numbers were never produced by running a blocker. This
script runs one, and the analytic figures are carried into every output as a
labelled reference so the two cannot be confused.

The question this script exists to answer is not "how good is token blocking". It is
**how complementary is token blocking to exact + char**, measured:

    true pairs retrieved by exact + char
    true pairs retrieved by exact + char + token
    delta                            = the token blocker's marginal contribution
    delta / (total_true - exact_char) = the share of previously missed pairs recovered

That ratio is the number the next milestone is decided on, so it is computed from
measured counts only and reported per cell.

The blocker
-----------
Per entity: tokenize ``name_norm``, look up each token's **target-corpus** document
frequency, keep tokens with ``df <= cap``, take the ``K`` rarest, probe their
postings, union, dedupe. The index is built the same way, so a pair is retrieved
when both sides share a token that is under the cap *and* inside both sides' rarest
K. That is strictly narrower than Phase 0's analytic rule, which required only the
cap and no rank limit - the same gap the char blocker has, and the reason the K
dimension is in the grid at all. At a K large enough to cover every eligible token of
both names the blocker degenerates to the analytic rule, which makes the K=5 end of
the grid the closest this run comes to a direct cross-check against Phase 0's
42.3905%.

There is **no verification stage**: the token signal is boolean, so the blocker's own
definition is the retrieval rule. That is the structural difference from the char
blocker, which retrieves on rare trigrams and then thresholds an exact Jaccard. It
also means this script needs no trigram-style name-key store for the token stage, and
that its only parallel stage is the reused char verification.

Token semantics
---------------
``str.split()`` on ``name_norm``, deduplicated per entity - byte-identical to Phase
0's ``analyze_blocking_statistics._token_set``, which is the definition the analytic
reference was measured under. ``name_norm``, not ``name_key``: ``name_key`` strips the
separators, so it has no token boundaries to split on.

Tokens get **dense integer codes in order of first appearance** in the target corpus,
not hashes, so the token->code map is a bijection and a key lookup needs no string
re-verification. An S1 token absent from a source's vocabulary keeps code ``-1``,
which ``_TrigramDf.lookup`` scores as ``df == 0`` - the same "absent from this
source" state the char blocker counts.

Reused, not reimplemented
-------------------------
This script **imports** ``scripts.calibrate_char_blocker`` and reuses, unchanged:

* ``_TrigramIndex`` / ``_TrigramDf`` - both are generic over int64 keys and carry no
  trigram-specific state, so a token index is the same two structures;
* ``_S1Selection``, ``sweep_volume``, ``_AccumulatorSet``, ``bind_accumulators``,
  ``extract_metrics``, ``exact_volume_rows``, the artifact fingerprint/resume
  helpers, and the CLI list parsers;
* the whole char path - ``build_trigram_index``, ``build_s1_selection``,
  ``_NameKeyStore``, ``verify_similarities`` - so the char candidates entering the
  unions are the shipped blocker's own output, not a reimplementation of it.

The char blocker's file is **not modified**. The two functions written fresh
(``build_token_index``, ``build_token_s1_selection``) are the char algorithms bound to
the token encoder; the encoder is the only real difference, and a different key type
is not duplication.

Candidate sets
--------------
Unions, never intersections::

    exact             the shipped exact-name index (the baseline the deltas use)
    char              the reused char blocker at the --char-* settings
    token             this blocker, per (df cap, rarest-K) cell
    token_plus_exact  token | exact
    exact_plus_char   exact | char
    exact_char_token  exact | char | token      <- the headline set

``exact``, ``char`` and ``exact_plus_char`` depend on no token cell, so they are
measured once and reported both in their own block and per cell. A pair proposed by
any blocker is a candidate; no blocker's output is intersected away.

Volume, memory, and honesty about truncation
--------------------------------------------
Every cell reports its retrieval volume as ``exact`` (distinct pairs, measured) or
``upper_bound_from_posting_expansion`` (postings expanded, an over-count when two of
an entity's keys reach the same target). A cell whose bound exceeds
``--max-candidate-rows`` keeps its true volume, reports ``"evaluated": false`` and its
reason, and never presents a truncated recall as measured. Nothing is silently capped.

The S1 side is one rank-sorted array per source built once at the loosest
``(cap, K)``, so every cell is a filter over it rather than a rebuild; the target side
is one index per source at the same bounds. Candidate pairs are materialized one chunk
at a time and folded into fixed-size counter arrays, so peak memory is bounded by the
chunk and the counters, not by the candidate volume. The exact and char candidate sets
are packed sorted int64 arrays, so a cell's slice of them is two ``searchsorted`` calls
rather than a re-retrieval.

What this run costs
-------------------
The char stage re-derives, for the single char cell the unions need, the same
artifacts the char calibration builds for its whole grid: two trigram df passes, two
trigram index builds, two S1 selections, plus retrieval and verification. On the
reference corpus the char calibration's artifact stage was ~9.5 min for one cell, so
expect the same here. It is a deliberate cost: the unions must contain the shipped
blocker's candidates, and re-deriving them locally is the only way to guarantee that
without depending on another run's output directory. ``--resume`` skips it on a
re-run.

Outputs (``--output-dir``, default ``<work_dir>/calibration``)
-------------------------------------------------------------
* ``token_blocker_calibration.json`` - meta, grid, per-cell metrics, incremental block
* ``token_blocker_calibration.csv`` - one row per (df cap, K, Jaccard, set)
* ``token_blocker_volume.csv``      - the volume curve per (df cap, K, scope, kind)
* ``token_blocker_top_tokens.csv``  - head of the token df table, as evidence
* ``token_blocker_incremental.csv`` - the marginal-contribution table
* ``token_blocker_calibration.md``  - human-readable summary
* ``_artifacts/token/``             - corpus artifacts reused by ``--resume``

Usage::

    # HPC, full grid
    python scripts/calibrate_token_blocker.py --config configs/config.yaml --workers 0 --resume --timings

    # volume first, when the grid may be too large to verify
    python scripts/calibrate_token_blocker.py --config configs/config.yaml --volume-only

    # local smoke test on a tiny synthetic fixture (never the real corpus)
    python scripts/calibrate_token_blocker.py --config /tmp/smoke.yaml --limit-s1 200

NOTE: the full run reads every target row of S2 and S3 and is an HPC job. Do not run
it on a laptop.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.calibrate_char_blocker import (  # noqa: E402
    ARTIFACT_DIRNAME,
    ARTIFACT_VERSION,
    BOUND_KIND,
    REPORTED_METRICS,
    SPLIT_VAL,
    TOKEN_ANALYTIC_REFERENCE,
    _AccumulatorSet,
    _artifact_ready,
    _float_list,
    _int_list,
    _NameKeyStore,
    _payload_budget,
    _S1Selection,
    _target_rows,
    _TrigramDf,
    _TrigramIndex,
    _write_artifact_marker,
    bind_accumulators,
    build_name_key_store,
    build_s1_selection,
    build_trigram_index,
    count_s1_rows,
    count_trigram_df,
    csr_offsets,
    empty_index,
    exact_volume_rows,
    extract_metrics,
    rank_within_runs,
    resolve_exact_packed,
    s1_chunks,
    sweep_volume,
    target_row_counts,
    verify_similarities,
    write_csv,
)
from src.blocking import (  # noqa: E402
    BLOCKER_EXACT_NAME,
    PAIR_MULTIPLIER,
    ExactNameIndex,
    index_dir_for,
    pack_pairs,
    unpack_pairs,
)
from src.data_loader import (  # noqa: E402
    TARGET_SOURCES,
    describe_environment,
    iter_prepared,
    load_config,
    load_ground_truth,
)
from src.evaluation import CandidateEvaluation, split_mask_for  # noqa: E402
from src.normalization import NAME_KEY, NAME_NORM  # noqa: E402
from src.utils import (  # noqa: E402
    current_rss_bytes,
    detect_hardware,
    encode_entity_ids,
    ensure_dir,
    fmt_int,
    format_hardware_report,
    human_bytes,
    plan_inflight_window,
    read_json,
    resolve_workers,
    set_seed,
    setup_logging,
    write_json,
)

LOG_NAME = "calibrate_token_blocker"

# Grid defaults. Wider on the low end than the char grid because token document
# frequencies are far larger than trigram ones: a name carries ~19 trigrams but only
# a handful of tokens, so a cap that is tight for trigrams is extremely tight for
# tokens. These are NOT operating points.
DEFAULT_DF_CAPS = (100, 500, 1000, 5000)
DEFAULT_RAREST_KS = (1, 3, 5)

# The char cell the unions reuse. df cap 1000 / K 1 is the cell the last char run
# measured; 0.3 and 0.5 are its two reported thresholds. One verification pass serves
# both thresholds, because they are filters over the same similarity vector.
DEFAULT_CHAR_DF_CAP = 1000
DEFAULT_CHAR_RAREST_K = 1
DEFAULT_CHAR_JACCARDS = (0.3, 0.5)

DEFAULT_MAX_CANDIDATE_ROWS = 250_000_000
DEFAULT_CHUNK_ROWS = 200_000
DEFAULT_VERIFY_CHUNK_PAIRS = 200_000

# Same estimate the char stage uses: one verification pair holds an S1 key string, its
# position, and a target store row.
VERIFY_BYTES_PER_PAIR = 96

# Candidate-set names. ``exact`` is carried as the baseline the increments are measured
# against; the rest are the union sets the question is asked about.
SET_EXACT = "exact"
SET_CHAR = "char"
SET_TOKEN = "token"
SET_EXACT_PLUS_CHAR = "exact_plus_char"
SET_TOKEN_PLUS_EXACT = "token_plus_exact"
SET_EXACT_CHAR_TOKEN = "exact_char_token"

# The three sets that change with the token cell, and the three that do not.
PER_CELL_SETS = (SET_TOKEN, SET_TOKEN_PLUS_EXACT, SET_EXACT_CHAR_TOKEN)
GRID_FREE_SETS = (SET_EXACT, SET_CHAR, SET_EXACT_PLUS_CHAR)

# Every accumulator holds six int64 arrays of n_entities, which is what the per-cell
# peak is estimated from.
ACCUMULATOR_ARRAYS = 6

_MISSING_CODE = -1
_EMPTY_INT64 = np.empty(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Tokenization: Phase 0's definition, and a dense vocabulary to key on
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Tokens of an already-normalized field, Phase 0's semantics exactly.

    ``analyze_blocking_statistics._token_set`` is ``set(text.split())``: ``str.split()``
    with no argument splits on any Unicode whitespace run and drops empty fields, and
    the dedup is what makes the frequency a DOCUMENT frequency rather than a term
    frequency. A list is returned here so the caller keeps the ordering decisions
    explicit; the dedup happens in :meth:`_TokenVocabulary.codes_for_text`.

    Kept as a named function so the semantics have exactly one home and the tests can
    point at it rather than at ``str.split``.
    """
    return text.split() if text else []


class _TokenVocabulary:
    """Bijective token -> dense int64 code map, per target source.

    Dense codes in order of first appearance, deliberately not hashes. A hash-keyed
    index must re-verify the string after a lookup because two tokens can collide;
    with a bijection there is nothing to verify, which is exactly the property that
    lets the char blocker's ``_TrigramIndex`` be reused here unchanged - it documents
    that it needs no string comparison *because* its keys are injective.

    Insertion order depends only on corpus order, not on the chunk size, so the
    vocabulary is reproducible run to run and ``--resume`` re-derives the same codes
    from disk instead of renumbering them.
    """

    __slots__ = ("source", "_codes", "_tokens")

    def __init__(self, source: str) -> None:
        self.source = source
        self._codes: dict[str, int] = {}
        self._tokens: list[str] = []

    def __len__(self) -> int:
        return len(self._tokens)

    @property
    def tokens(self) -> list[str]:
        return self._tokens

    def code_of(self, token: str) -> int:
        """Existing code, or ``-1`` when the token is not in this source's corpus."""
        return self._codes.get(token, _MISSING_CODE)

    def intern(self, token: str) -> int:
        code = self._codes.get(token)
        if code is None:
            code = len(self._tokens)
            self._codes[token] = code
            self._tokens.append(token)
        return code

    def codes_for_text(self, text: str) -> np.ndarray:
        """Distinct token codes of one text, ascending. Empty when it has no token."""
        tokens = tokenize(text)
        if not tokens:
            return _EMPTY_INT64
        if len(tokens) == 1:
            return np.asarray([self.intern(tokens[0])], dtype=np.int64)
        seen: set[int] = set()
        intern = self.intern
        for token in tokens:
            seen.add(intern(token))
        out = np.fromiter(seen, dtype=np.int64, count=len(seen))
        out.sort()
        return out

    def codes_for_texts(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(codes, owners)`` - every text's distinct token codes, and its row index.

        The token twin of ``trigram_codes_for_list``: same contract (codes ascending
        within a row, owners giving the row each code came from), so every downstream
        algorithm that consumes it is unchanged. Grows the vocabulary.
        """
        parts: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        for row, text in enumerate(texts):
            codes = self.codes_for_text(text)
            if codes.size == 0:
                continue
            parts.append(codes)
            owners.append(np.full(codes.size, row, dtype=np.int64))
        if not parts:
            return _EMPTY_INT64, _EMPTY_INT64
        return np.concatenate(parts), np.concatenate(owners)

    def lookup_texts(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray, int]:
        """``(codes, owners, n_texts_without_tokens)`` for tokens already known here.

        Read-only: a token this source's corpus never contained resolves to a negative
        code rather than being interned, which keeps an S1-only token in the "absent
        from this source" bucket (``df == 0``) instead of inventing a key for it. Those
        entries are what make the absent-token drop count measurable, exactly as
        ``df == 0`` is on the char side.
        """
        parts: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        empty = 0
        codes_of = self._codes
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                empty += 1
                continue
            # Two absent tokens are two distinct tokens, not one. Giving each its own
            # negative code (real codes are >= 0) stops the dedup below from collapsing
            # them, so ``name_norm`` "acme private limited" against a corpus that knows
            # only "acme" reports 2 absent tokens rather than 1. Every negative code
            # scores 0 in ``_TrigramDf.lookup``, so all of them still land in the absent
            # bucket no matter which sentinel they got.
            seen: set[int] = set()
            missing: dict[str, int] = {}
            for token in tokens:
                code = codes_of.get(token)
                if code is None:
                    code = missing.get(token)
                    if code is None:
                        code = _MISSING_CODE - len(missing)
                        missing[token] = code
                seen.add(code)
            codes = np.fromiter(seen, dtype=np.int64, count=len(seen))
            codes.sort()
            parts.append(codes)
            owners.append(np.full(codes.size, row, dtype=np.int64))
        if not parts:
            return _EMPTY_INT64, _EMPTY_INT64, empty
        return np.concatenate(parts), np.concatenate(owners), empty

    # -- persistence --------------------------------------------------------
    def save(self, directory: Path) -> None:
        ensure_dir(directory)
        blob = b"\x00".join(token.encode("utf-8") for token in self._tokens)
        (directory / "vocab.bin").write_bytes(blob)
        # The byte count is recorded so ``load`` can detect a truncated blob: a blob cut
        # mid-token still splits into the right number of NUL-separated fields (one short
        # final field), so the token count alone cannot see the damage.
        write_json(
            directory / "vocab_meta.json",
            {"source": self.source, "n_tokens": len(self._tokens), "blob_bytes": len(blob)},
        )

    @classmethod
    def load(cls, directory: Path, source: str) -> "_TokenVocabulary":
        meta = read_json(directory / "vocab_meta.json")
        blob = (directory / "vocab.bin").read_bytes()
        vocab = cls(source)
        n_tokens = int(meta["n_tokens"])
        if n_tokens == 0:
            return vocab
        expected_bytes = int(meta["blob_bytes"])
        if len(blob) != expected_bytes:
            raise ValueError(
                f"{directory}: vocabulary blob is {len(blob)} bytes but the meta says "
                f"{expected_bytes}"
            )
        # One split rather than a million slices: the blob is NUL-joined and no token
        # can contain NUL (normalization folds punctuation to spaces), so the split is
        # exact.
        tokens = blob.split(b"\x00")
        if len(tokens) != n_tokens:
            raise ValueError(
                f"{directory}: vocabulary blob holds {len(tokens)} tokens but the meta says "
                f"{n_tokens}"
            )
        vocab._tokens = [token.decode("utf-8") for token in tokens]
        vocab._codes = {token: code for code, token in enumerate(vocab._tokens)}
        return vocab

    def top(self, df: _TrigramDf, limit: int) -> list[tuple[str, int]]:
        """The ``limit`` most frequent tokens as (text, df). Reporting only."""
        if len(df.values) == 0 or not self._tokens:
            return []
        order = np.argsort(df.values)[::-1][:limit]
        return [
            (self._tokens[int(df.codes[i])], int(df.values[i]))
            for i in order
            if 0 <= int(df.codes[i]) < len(self._tokens)
        ]

    def describe(self) -> dict:
        lengths = [len(token) for token in self._tokens] if self._tokens else [0]
        return {
            "source": self.source,
            "n_tokens": int(len(self._tokens)),
            "avg_chars_per_token": round(float(np.mean(lengths)), 3),
            "max_chars_per_token": int(np.max(lengths)),
        }


def describe_df(df: _TrigramDf, caps: Sequence[int]) -> dict:
    """``_TrigramDf.describe`` with the trigram wording corrected for tokens.

    The structure is reused verbatim; only the key that names a trigram is renamed, so
    nothing in the report claims a token count is a trigram count.
    """
    out = df.describe(caps)
    if "n_distinct_trigrams" in out:
        out["n_distinct_tokens"] = out.pop("n_distinct_trigrams")
    return out


# ---------------------------------------------------------------------------
# Corpus artifacts: df, index, S1 selection
# ---------------------------------------------------------------------------
def count_token_df(
    chunks: Iterable[pd.DataFrame],
    field: str,
    vocab: _TokenVocabulary,
    log: logging.Logger,
    label: str,
) -> _TrigramDf:
    """Token document frequency over a streamed target table.

    Same reduction as the char blocker's df count: dedup per chunk, then one global
    sort and an ``add.reduceat`` fold. Per-chunk dedup keeps the merge proportional to
    the distinct token count rather than to the ~50M total occurrences.

    The vocabulary grows here and only here. Codes are assigned on first sight, so the
    df table and the index that follows it share one numbering and ``--resume`` can
    restore it from disk.
    """
    started = time.time()
    rows = 0
    parts_codes: list[np.ndarray] = []
    parts_counts: list[np.ndarray] = []

    for frame in chunks:
        texts = frame[field].to_numpy(dtype=object)
        rows += len(texts)
        codes, _ = vocab.codes_for_texts(texts)
        if codes.size:
            unique, counts = np.unique(codes, return_counts=True)
            parts_codes.append(unique)
            parts_counts.append(counts.astype(np.int64))
        log.info("  %s: %s rows scanned for token df", label, fmt_int(rows))

    if not parts_codes:
        log.warning("%s: no tokens found in %s rows", label, fmt_int(rows))
        return _TrigramDf(_EMPTY_INT64, _EMPTY_INT64)

    all_codes = np.concatenate(parts_codes)
    all_counts = np.concatenate(parts_counts)
    del parts_codes, parts_counts
    gc.collect()

    order = np.argsort(all_codes, kind="stable")
    all_codes, all_counts = all_codes[order], all_counts[order]
    del order
    gc.collect()

    starts = np.flatnonzero(np.concatenate(([True], all_codes[1:] != all_codes[:-1])))
    table = _TrigramDf(all_codes[starts], np.add.reduceat(all_counts, starts))
    log.info(
        "%s: %s distinct tokens over %s rows in %.1f s",
        label,
        fmt_int(len(table)),
        fmt_int(rows),
        time.time() - started,
    )
    return table


def build_token_index(
    chunks: Iterable[pd.DataFrame],
    df: _TrigramDf,
    vocab: _TokenVocabulary,
    entity_column: str,
    field: str,
    max_df_cap: int,
    max_rank: int,
    source: str,
    log: logging.Logger,
    label: str,
) -> _TrigramIndex:
    """Build the shared token index for one target source.

    The char blocker's index build with the token encoder substituted; both structural
    properties the grid depends on are established identically:

    * each entity's surviving tokens are ordered by **df ascending** and truncated at
      ``max_rank``, so a tighter ``(cap, K)`` cell is a filter of this index;
    * postings are grouped by key and rank-ordered inside a key, so the rank filter is
      a posting prefix and the filtered offsets are a cumulative sum.
    """
    started = time.time()
    rows = 0
    key_parts: list[np.ndarray] = []
    entity_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    indexed_entities = 0
    structural_zero = 0

    for frame in chunks:
        texts = frame[field].to_numpy(dtype=object)
        entity_codes = encode_entity_ids(frame[entity_column])
        all_codes, owners = vocab.codes_for_texts(texts)
        rows += len(texts)

        if all_codes.size == 0:
            structural_zero += len(texts)
            continue

        dfs = df.lookup(all_codes)
        # ``df > 0`` drops a token the vocabulary knows but this source's table does
        # not. That cannot happen - the vocabulary is built by this source's own df
        # pass - and keeping the guard means a mismatched pairing fails closed.
        keep = (dfs > 0) & (dfs <= max_df_cap)
        if not keep.any():
            structural_zero += len(texts)
            continue
        codes, row_of, row_dfs = all_codes[keep], owners[keep], dfs[keep]

        # Row first, then df ascending, then code ascending. Row-major is what makes
        # rank_within_runs valid; the df/code tail is a total order, so the selected
        # keys and their ranks are reproducible run to run.
        order = np.lexsort((codes, row_dfs, row_of))
        codes, row_of, row_dfs = codes[order], row_of[order], row_dfs[order]

        ranks = rank_within_runs(row_of)
        keep_rank = ranks < max_rank
        if not keep_rank.any():
            structural_zero += len(texts)
            continue
        survivors = np.unique(row_of[keep_rank])
        indexed_entities += int(survivors.size)
        structural_zero += len(texts) - int(survivors.size)

        key_parts.append(codes[keep_rank])
        entity_parts.append(entity_codes[row_of[keep_rank]])
        rank_parts.append(ranks[keep_rank].astype(np.uint8))
        log.info("  %s: %s rows indexed for token keys", label, fmt_int(rows))

    if not key_parts:
        log.warning(
            "%s: no indexable tokens; the df cap of %s removed every key", label, max_df_cap
        )
        return empty_index(source)

    all_keys = np.concatenate(key_parts)
    all_entities = np.concatenate(entity_parts)
    all_ranks = np.concatenate(rank_parts)
    del key_parts, entity_parts, rank_parts
    gc.collect()

    # Key-major, then rank ascending within a key. lexsort is stable, so equal
    # (key, rank) pairs keep emission order and the build is reproducible.
    order = np.lexsort((all_ranks, all_keys))
    all_keys, all_entities, all_ranks = all_keys[order], all_entities[order], all_ranks[order]
    del order
    gc.collect()

    starts = np.flatnonzero(np.concatenate(([True], all_keys[1:] != all_keys[:-1])))
    index = _TrigramIndex(
        source=source,
        keys=all_keys[starts],
        key_df=df.lookup(all_keys[starts]).astype(np.int32),
        postings_offsets=np.concatenate((starts, [len(all_keys)])),
        postings=all_entities,
        ranks=all_ranks,
    )
    log.info(
        "%s: token index built in %.1f s (%s); %s entities indexed, %s rows contributed no "
        "surviving key (%.2f%% of rows)",
        label,
        time.time() - started,
        index.describe(),
        fmt_int(indexed_entities),
        fmt_int(structural_zero),
        100.0 * structural_zero / max(1, rows),
    )
    return index


def build_token_s1_selection(
    chunks: Iterable[pd.DataFrame],
    df: _TrigramDf,
    vocab: _TokenVocabulary,
    ground_truth,
    entity_column: str,
    field: str,
    max_df_cap: int,
    max_rank: int,
    n_entities: int,
    log: logging.Logger,
    label: str,
) -> tuple[_S1Selection, dict]:
    """Rank every S1 entity's tokens once, at the loosest grid setting.

    The char blocker's selection with the token encoder and read-only vocabulary
    lookup, so a token an S1 name has but this source's corpus does not stays "absent"
    (code ``-1``, ``df == 0``) instead of being interned into the vocabulary. Ordering
    is position, then df ascending, then code ascending - the index's ordering read
    from the other side - which is what makes a cell filter exact rather than
    approximate.

    The per-row drop reasons are counted separately because a row that keeps no key can
    have lost them three different ways and the fixes differ: no token at all, a token
    this source never had, too common here, or simply not among the entity's K rarest.
    """
    started = time.time()
    code_parts: list[np.ndarray] = []
    df_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    rows = 0
    unknown = 0
    no_tokens = 0
    dropped_absent = 0
    dropped_above_cap = 0
    dropped_beyond_k = 0

    for frame in chunks:
        texts = frame[field].to_numpy(dtype=object)
        rows += len(frame)

        positions = ground_truth.positions_of(frame[entity_column])
        known = positions >= 0
        unknown += int((~known).sum())
        if not known.any():
            continue
        positions, texts = positions[known], texts[known]

        all_codes, owners, empty_rows = vocab.lookup_texts(texts)
        no_tokens += empty_rows
        if all_codes.size == 0:
            continue

        entry_positions = positions[owners]
        dfs = df.lookup(all_codes)
        absent = dfs == 0
        above_cap = dfs > max_df_cap
        dropped_absent += int(absent.sum())
        dropped_above_cap += int(above_cap.sum())

        keep = ~absent & ~above_cap
        if not keep.any():
            continue
        codes, entry_positions, row_dfs = all_codes[keep], entry_positions[keep], dfs[keep]

        order = np.lexsort((codes, row_dfs, entry_positions))
        codes, entry_positions, row_dfs = codes[order], entry_positions[order], row_dfs[order]

        ranks = rank_within_runs(entry_positions)
        keep_rank = ranks < max_rank
        dropped_beyond_k += int((~keep_rank).sum())
        if not keep_rank.any():
            continue

        code_parts.append(codes[keep_rank])
        df_parts.append(row_dfs[keep_rank])
        rank_parts.append(ranks[keep_rank].astype(np.uint8))
        position_parts.append(entry_positions[keep_rank])
        log.info("  %s: %s S1 rows coded", label, fmt_int(rows))

    if code_parts:
        codes = np.concatenate(code_parts)
        dfs = np.concatenate(df_parts)
        ranks = np.concatenate(rank_parts)
        positions = np.concatenate(position_parts)
        del code_parts, df_parts, rank_parts, position_parts
        gc.collect()

        # Chunks arrive in prepared-table order, so the concatenation is not yet
        # position-sorted and the CSR below would be invalid. A stable sort restores it
        # while preserving each row's df order, hence its ranks.
        order = np.argsort(positions, kind="stable")
        codes, dfs, ranks, positions = codes[order], dfs[order], ranks[order], positions[order]
        del order
        gc.collect()
        _, offsets = csr_offsets(positions, n_entities)
    else:
        codes, dfs, ranks = _EMPTY_INT64, _EMPTY_INT64, np.empty(0, dtype=np.uint8)
        offsets = np.zeros(n_entities + 1, dtype=np.int64)

    selection = _S1Selection(codes=codes, ranks=ranks, dfs=dfs, offsets=offsets, n_rows=n_entities)
    per_row = np.diff(offsets)
    breakdown = {
        "n_s1_rows_read": int(rows),
        "n_s1_positions": int(n_entities),
        "s1_unknown_to_ground_truth": int(unknown),
        "s1_name_with_no_token": int(no_tokens),
        "s1_with_no_key_in_this_source": int(n_entities - int(np.count_nonzero(per_row))),
        "tokens_dropped_absent_from_this_source": int(dropped_absent),
        "tokens_dropped_above_df_cap": int(dropped_above_cap),
        "tokens_dropped_beyond_rarest_k": int(dropped_beyond_k),
        "entries_per_row": round(float(codes.size) / max(1, n_entities), 3),
    }
    log.info(
        "%s: S1 token selection built in %.1f s (%s); %s",
        label,
        time.time() - started,
        selection.describe(),
        breakdown,
    )
    return selection, breakdown


# ---------------------------------------------------------------------------
# Retrieval: one cell's candidate pairs
# ---------------------------------------------------------------------------
def retrieve_token_chunk(
    index: _TrigramIndex,
    selection: _S1Selection,
    start: int,
    stop: int,
    cap: int,
    rarest_k: int,
) -> np.ndarray:
    """Packed, unique, sorted ``(position, entity_code)`` pairs one cell retrieves.

    The char blocker's retrieval with no verification appended, because the token
    signal has no threshold: sharing an eligible token *is* the blocker's decision.
    Sorting by packed value is what lets a chunk's slice of the result be two
    ``searchsorted`` calls later.
    """
    codes, positions, ranks, dfs = selection.slice_full(start, stop)
    if codes.size == 0:
        return _EMPTY_INT64
    keep = selection.cell_filter(ranks, dfs, cap, rarest_k)
    if not keep.any():
        return _EMPTY_INT64
    key_positions, counts = index.lookup_many(codes[keep])
    owners, target_codes = index.expand(key_positions, counts)
    if owners.size == 0:
        return _EMPTY_INT64
    return np.unique(pack_pairs(positions[keep][owners], target_codes))


def slice_packed(packed: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Pairs whose S1 position lies in ``[start, stop)``.

    Packed pairs sort by (position, entity code), so this is two ``searchsorted`` calls
    on the globally sorted array rather than a per-chunk rebuild. That is what lets the
    exact and char candidate sets be computed once and sliced per cell.
    """
    if packed.size == 0:
        return _EMPTY_INT64
    low = int(np.searchsorted(packed, start * PAIR_MULTIPLIER))
    high = int(np.searchsorted(packed, stop * PAIR_MULTIPLIER))
    return packed[low:high]


def union_packed(*arrays: np.ndarray) -> np.ndarray:
    """Union of packed pair arrays: one ``np.unique`` does dedupe and sort at once.

    Union, never intersection - a pair proposed by any blocker is a candidate.
    """
    live = [array for array in arrays if array is not None and array.size]
    if not live:
        return _EMPTY_INT64
    if len(live) == 1:
        return live[0]
    return np.unique(np.concatenate(live))


# ---------------------------------------------------------------------------
# The reused char stage: char candidates at one cell, computed once
# ---------------------------------------------------------------------------
def build_char_artifacts(
    args: argparse.Namespace,
    config: dict,
    sources: Sequence[str],
    ground_truth,
    n_entities: int,
    artifact_root: Path,
    fingerprint: str,
    log: logging.Logger,
    timings: dict[str, float],
) -> tuple[dict[str, _TrigramIndex], dict[str, _S1Selection], dict[str, _TrigramDf], dict]:
    """Build the char blocker's corpus artifacts at one ``(cap, K)`` cell.

    Every build goes through the char calibration module's own functions
    (``count_trigram_df``, ``build_trigram_index``, ``build_s1_selection``) over
    ``name_key``, so the char candidates later entering the unions are the shipped
    blocker's output rather than a reimplementation. Only one cell is built, so the
    index is built directly at that cell and needs no per-cell filtering.

    Returns the indexes, selections, the df tables, and a dict carrying the cell's
    retrieval bound - which is what the char verification's worker count and in-flight
    window are sized against, and which costs about a second because ``sweep_volume``
    counts posting-list prefixes rather than expanding them.
    """
    cap, rarest_k = args.char_df_cap, args.char_rarest_k
    root = artifact_root / "char_stage"
    indexes: dict[str, _TrigramIndex] = {}
    selections: dict[str, _S1Selection] = {}
    dfs: dict[str, _TrigramDf] = {}
    stage = time.time()

    for source in sources:
        df_dir = root / source / "df"
        index_dir = root / source / "index"
        select_dir = root / source / "s1_selection"

        if args.resume and _artifact_ready(df_dir, fingerprint):
            log.info("[%s] resuming the char-stage trigram df", source)
            dfs[source] = _TrigramDf.load(df_dir)
        else:
            dfs[source] = count_trigram_df(
                iter_prepared(
                    config, args.split, source, columns=[NAME_KEY], chunksize=args.chunk_rows
                ),
                NAME_KEY,
                log,
                f"[{source}] char-stage df",
            )
            dfs[source].save(df_dir)
            _write_artifact_marker(df_dir, fingerprint, dfs[source].describe([cap]))

        if args.resume and _artifact_ready(index_dir, fingerprint):
            log.info("[%s] resuming the char-stage index", source)
            indexes[source] = _TrigramIndex.load(index_dir, source)
        else:
            indexes[source] = build_trigram_index(
                iter_prepared(
                    config,
                    args.split,
                    source,
                    columns=["entity_id", NAME_KEY],
                    chunksize=args.chunk_rows,
                ),
                dfs[source],
                "entity_id",
                NAME_KEY,
                cap,
                rarest_k,
                source,
                log,
                f"[{source}] char-stage index",
            )
            indexes[source].save(index_dir)
            _write_artifact_marker(index_dir, fingerprint, indexes[source].describe())

        if args.resume and _artifact_ready(select_dir, fingerprint):
            log.info("[%s] resuming the char-stage S1 selection", source)
            selections[source] = _S1Selection.load(select_dir)
        else:
            selections[source], breakdown = build_s1_selection(
                s1_chunks(config, args, ["entity_id", NAME_KEY]),
                dfs[source],
                ground_truth,
                "entity_id",
                NAME_KEY,
                cap,
                rarest_k,
                n_entities,
                log,
                f"[{source}] char-stage s1",
            )
            selections[source].save(select_dir)
            _write_artifact_marker(select_dir, fingerprint, {"breakdown": breakdown})

    _, totals = sweep_volume(
        indexes, selections, [(cap, rarest_k)], n_entities, args.chunk_rows, log
    )
    bound = int(totals.get((cap, rarest_k), 0))
    timings["char_artifacts"] = round(time.time() - stage, 1)
    log.info(
        "char stage artifacts at (cap=%s, K=%s) ready in %.1f min; retrieval bound %s pairs",
        cap,
        rarest_k,
        (time.time() - stage) / 60.0,
        fmt_int(bound),
    )
    return indexes, selections, dfs, {
        "df_cap": cap,
        "rarest_k": rarest_k,
        "retrieval_bound": bound,
        "df": {source: dfs[source].describe([cap]) for source in sources},
        "indexes": {source: indexes[source].describe() for source in sources},
        "s1_selection": {source: selections[source].describe() for source in sources},
    }


def retrieve_char_candidates(
    args: argparse.Namespace,
    sources: Sequence[str],
    char_indexes: dict[str, _TrigramIndex],
    char_selections: dict[str, _S1Selection],
    s1_store: _NameKeyStore,
    store_dirs: dict[str, str],
    artifact_root: Path,
    fingerprint: str,
    workers: int,
    window: int,
    verify_chunk_pairs: int,
    log: logging.Logger,
    timings: dict[str, float],
) -> dict[float, np.ndarray]:
    """Retrieve the char cell and verify it once, then filter per threshold.

    ``verify_similarities`` is the char calibration's, unchanged, so these thresholds
    are the same signal the char calibration and Phase 0.1 report coverage for. One
    verification pass serves every threshold because a threshold is a filter over the
    same similarity vector; the result is stored sorted and packed so a later union is
    a ``searchsorted`` slice rather than a re-retrieval.
    """
    cap, rarest_k = args.char_df_cap, args.char_rarest_k
    thresholds = list(args.char_jaccards)
    root = artifact_root / "char_stage"
    paths = {t: root / f"packed_{_jaccard_key(t)}.npy" for t in thresholds}

    if (
        args.resume
        and _artifact_ready(root, fingerprint)
        and all(path.is_file() for path in paths.values())
    ):
        log.info("resuming the char-stage candidate sets")
        return {t: np.load(path) for t, path in paths.items()}

    stage = time.time()
    collected: dict[float, list[np.ndarray]] = {t: [] for t in thresholds}
    retrieved = 0
    verified = 0

    for source in sources:
        index = char_indexes[source]
        selection = char_selections[source]
        for start in range(0, selection.n_rows, args.chunk_rows):
            stop = min(start + args.chunk_rows, selection.n_rows)
            codes, positions, ranks, dfs = selection.slice_full(start, stop)
            if codes.size == 0:
                continue
            keep = selection.cell_filter(ranks, dfs, cap, rarest_k)
            if not keep.any():
                continue
            key_positions, counts = index.lookup_many(codes[keep])
            owners, target_codes = index.expand(key_positions, counts)
            if owners.size == 0:
                continue

            packed = np.unique(pack_pairs(positions[keep][owners], target_codes))
            retrieved += int(packed.size)
            s1_positions, pair_targets = unpack_pairs(packed)
            similarities = verify_similarities(
                source,
                s1_positions,
                _target_rows(store_dirs, source, pair_targets),
                s1_store.key_at_position,
                workers=workers,
                window=window,
                chunk_pairs=verify_chunk_pairs,
                store_dirs=store_dirs,
            )
            verified += int(similarities.size)
            for threshold in thresholds:
                kept = similarities >= threshold
                if kept.any():
                    collected[threshold].append(packed[kept])

    out: dict[float, np.ndarray] = {}
    for threshold in thresholds:
        merged = union_packed(*collected[threshold]) if collected[threshold] else _EMPTY_INT64
        out[threshold] = merged
        np.save(paths[threshold], merged)
        log.info(
            "char stage (cap=%s, K=%s, J=%.2f): %s candidate pairs",
            cap,
            rarest_k,
            threshold,
            fmt_int(merged.size),
        )
    _write_artifact_marker(
        root, fingerprint, {"df_cap": cap, "rarest_k": rarest_k, "thresholds": thresholds}
    )
    timings["char_retrieval"] = round(time.time() - stage, 1)
    log.info(
        "char stage: %s retrieved, %s verified, %.1f s",
        fmt_int(retrieved),
        fmt_int(verified),
        time.time() - stage,
    )
    return out


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------
def evaluate_token_grid(
    args: argparse.Namespace,
    config: dict,
    indexes: dict[str, _TrigramIndex],
    selections: dict[str, _S1Selection],
    exact_packed: np.ndarray,
    char_packed: dict[float, np.ndarray],
    ground_truth,
    cells: Sequence[tuple[int, int]],
    cell_volume: dict[tuple[int, int], int],
    n_entities: int,
    n_target_records: Optional[int],
    workers: int,
    window: int,
    verify_chunk_pairs: int,
    log: logging.Logger,
    timings: dict[str, float],
) -> tuple[list[dict], dict]:
    """Retrieve, unite and score every token cell inside the candidate-row budget.

    The three grid-free sets (``exact``, ``char``, ``exact_plus_char``) are measured
    once here; the three per-cell sets (``token``, ``token_plus_exact``,
    ``exact_char_token``) are measured per cell. Accumulators are created per cell and
    freed at the end of it, so peak memory is a handful of counters rather than one set
    per cell in the grid.
    """
    stage = time.time()
    source_codes = (2, 3)
    thresholds = list(args.char_jaccards)
    needs_split = bool(config.get("evaluation", {}).get("split", {}).get("enabled", False))
    val_mask = split_mask_for(ground_truth, config, SPLIT_VAL) if needs_split else None

    # One evaluator, its truth structures shared by every accumulator set. k_values=()
    # on purpose: file-order recall@K is arbitrary for an unranked blocker, so reporting
    # it would put a meaningless number in the output.
    evaluation = CandidateEvaluation(
        ground_truth, n_target_records=n_target_records, k_values=(), log=log
    )
    truth = evaluation.true_pair_codes

    grid: list[dict] = []
    exact_rows: list[dict] = []

    def scored(
        accumulator: _AccumulatorSet, cell: dict, name: str, threshold: Optional[float]
    ) -> dict:
        """Bind, extract, and keep the exact-volume rows for one accumulator."""
        bind_accumulators(evaluation, accumulator)
        entry: dict[str, Any] = {
            "set": name,
            "threshold": threshold,
            "metrics": extract_metrics(evaluation),
        }
        if val_mask is not None:
            entry["metrics_val"] = extract_metrics(evaluation, val_mask, SPLIT_VAL)
        exact_rows.extend(exact_volume_rows(cell, entry, accumulator, n_entities))
        return entry

    accumulators_per_cell = ACCUMULATOR_ARRAYS * n_entities * 8 * (2 + len(thresholds))
    log.info(
        "evaluation: %d cells; per-cell accumulator peak about %s (token, token+exact and "
        "%d exact_char_token sets)",
        len(cells),
        human_bytes(accumulators_per_cell),
        len(thresholds),
    )

    # ---- grid-free sets: measured once ------------------------------------
    exact_acc = _AccumulatorSet(n_entities, source_codes)
    exact_acc.add(*unpack_pairs(exact_packed), truth)
    exact_metrics = scored(exact_acc, {"df_cap": None, "rarest_k": None}, SET_EXACT, None)["metrics"]
    del exact_acc
    log.info(
        "exact-name blocking alone: pair recall %.4f, %s candidates",
        exact_metrics["blocking_recall_pair"] or 0.0,
        fmt_int(exact_metrics["n_candidate_pairs"] or 0),
    )

    char_metrics: dict[str, dict] = {}
    exact_plus_char_metrics: dict[str, dict] = {}
    for threshold in thresholds:
        char_acc = _AccumulatorSet(n_entities, source_codes)
        char_acc.add(*unpack_pairs(char_packed[threshold]), truth)
        char_cell = {"df_cap": args.char_df_cap, "rarest_k": args.char_rarest_k}
        char_metrics[_jaccard_key(threshold)] = scored(char_acc, char_cell, SET_CHAR, threshold)["metrics"]

        union_acc = _AccumulatorSet(n_entities, source_codes)
        union_acc.add(*unpack_pairs(union_packed(exact_packed, char_packed[threshold])), truth)
        exact_plus_char_metrics[_jaccard_key(threshold)] = scored(
            union_acc, char_cell, SET_EXACT_PLUS_CHAR, threshold
        )["metrics"]
        log.info(
            "char alone (J=%.2f): pair recall %.4f; exact+char: pair recall %.4f (%s candidates)",
            threshold,
            char_metrics[_jaccard_key(threshold)]["blocking_recall_pair"] or 0.0,
            exact_plus_char_metrics[_jaccard_key(threshold)]["blocking_recall_pair"] or 0.0,
            fmt_int(exact_plus_char_metrics[_jaccard_key(threshold)]["n_candidate_pairs"] or 0),
        )

    # ---- per-cell sets ----------------------------------------------------
    incremental_rows: list[dict] = []
    n_true_pairs = int(ground_truth.n_matches)

    for cap, rarest_k in cells:
        volume = cell_volume[(cap, rarest_k)]
        if args.max_candidate_rows and volume > args.max_candidate_rows:
            log.warning(
                "cell (cap=%s, K=%s): expansion bound %s exceeds the %s budget - bound "
                "reported, recall NOT evaluated",
                cap,
                rarest_k,
                fmt_int(volume),
                fmt_int(args.max_candidate_rows),
            )
            grid.append(
                {
                    "df_cap": cap,
                    "rarest_k": rarest_k,
                    "volume": volume,
                    "volume_kind": BOUND_KIND,
                    "evaluated": False,
                    "skip_reason": "expansion_bound_above_max_candidate_rows",
                    "max_candidate_rows": args.max_candidate_rows,
                    "sets": [],
                }
            )
            continue

        cell_started = time.time()
        filtered = {source: index.filtered(cap, rarest_k) for source, index in indexes.items()}
        cell_accs = {
            SET_TOKEN: _AccumulatorSet(n_entities, source_codes),
            SET_TOKEN_PLUS_EXACT: _AccumulatorSet(n_entities, source_codes),
        }
        for threshold in thresholds:
            cell_accs[_threshold_key(threshold)] = _AccumulatorSet(n_entities, source_codes)
        retrieved = 0

        for start in range(0, n_entities, args.chunk_rows):
            stop = min(start + args.chunk_rows, n_entities)
            token_packed = _EMPTY_INT64
            for source, filtered_index in filtered.items():
                chunk = retrieve_token_chunk(
                    filtered_index, selections[source], start, stop, cap, rarest_k
                )
                if chunk.size:
                    token_packed = union_packed(token_packed, chunk)
            if token_packed.size:
                retrieved += int(token_packed.size)
                cell_accs[SET_TOKEN].add(*unpack_pairs(token_packed), truth)

            exact_slab = slice_packed(exact_packed, start, stop)
            token_exact = union_packed(token_packed, exact_slab)
            if token_exact.size:
                cell_accs[SET_TOKEN_PLUS_EXACT].add(*unpack_pairs(token_exact), truth)

            for threshold in thresholds:
                char_slab = slice_packed(char_packed[threshold], start, stop)
                everything = union_packed(token_packed, exact_slab, char_slab)
                if everything.size:
                    cell_accs[_threshold_key(threshold)].add(*unpack_pairs(everything), truth)

        cell = {"df_cap": cap, "rarest_k": rarest_k}
        entries = [
            scored(cell_accs[SET_TOKEN], cell, SET_TOKEN, None),
            scored(cell_accs[SET_TOKEN_PLUS_EXACT], cell, SET_TOKEN_PLUS_EXACT, None),
        ]
        for threshold in thresholds:
            entries.append(
                scored(cell_accs[_threshold_key(threshold)], cell, SET_EXACT_CHAR_TOKEN, threshold)
            )
        for entry in entries:
            entry["elapsed_seconds"] = round(time.time() - cell_started, 1)

        for threshold in thresholds:
            incremental_rows.append(
                {
                    "df_cap": cap,
                    "rarest_k": rarest_k,
                    "char_jaccard_threshold": threshold,
                    **_incremental_counts(
                        n_true_pairs,
                        exact_metrics,
                        char_metrics[_jaccard_key(threshold)],
                        exact_plus_char_metrics[_jaccard_key(threshold)],
                        entries[0]["metrics"],
                        entries[1]["metrics"],
                        _threshold_entry(entries, SET_EXACT_CHAR_TOKEN, threshold),
                    ),
                }
            )

        del cell_accs
        gc.collect()
        log.info(
            "cell (cap=%s, K=%s): %s token pairs, evaluated in %.1f s",
            cap,
            rarest_k,
            fmt_int(retrieved),
            time.time() - cell_started,
        )
        grid.append(
            {
                "df_cap": cap,
                "rarest_k": rarest_k,
                "volume": volume,
                "volume_kind": BOUND_KIND,
                "evaluated": True,
                "pairs_retrieved": retrieved,
                "elapsed_seconds": round(time.time() - cell_started, 1),
                "sets": entries,
            }
        )

    timings["evaluation_stage"] = round(time.time() - stage, 1)
    extras = {
        "workers": workers,
        "verify_window": window,
        "verify_chunk_pairs": verify_chunk_pairs,
        "sets": list(GRID_FREE_SETS) + list(PER_CELL_SETS),
        "split_enabled": needs_split,
        "exact": exact_metrics,
        "char": char_metrics,
        "exact_plus_char": exact_plus_char_metrics,
        "incremental": incremental_rows,
        "exact_volume_rows": exact_rows,
    }
    return grid, extras


def _jaccard_key(threshold: float) -> str:
    """One spelling for one threshold, used by report keys and artifact filenames.

    ``:g`` rather than ``str``: 1.0 renders as ``1``, so the key, the CSV filename and
    the accumulator key all agree and a reader joining them by threshold finds them.
    """
    return f"{threshold:g}"


def _threshold_key(threshold: float) -> str:
    """Accumulator key for one threshold's ``exact_char_token`` set."""
    return f"{SET_EXACT_CHAR_TOKEN}|{_jaccard_key(threshold)}"


def _threshold_entry(entries: Sequence[dict], name: str, threshold: float) -> dict:
    """The metrics of one (set, threshold) entry among a cell's entries."""
    for entry in entries:
        if entry["set"] == name and entry["threshold"] == threshold:
            return entry["metrics"]
    return {}


def _incremental_counts(
    n_true_pairs: int,
    exact_metrics: dict,
    char_metrics: dict,
    exact_char_metrics: dict,
    token_metrics: dict,
    token_exact_metrics: dict,
    everything_metrics: dict,
) -> dict:
    """The marginal-contribution numbers for one (cell, threshold).

    ``additional_recall_on_previous_misses`` is the number the design decision turns
    on: ``(Y - X) / (total_true - X)`` with X = exact+char and Y = exact+char+token.
    Every term is a measured count from this run; nothing here is estimated.
    """
    x = int(exact_char_metrics.get("true_pairs_retrieved") or 0)
    y = int(everything_metrics.get("true_pairs_retrieved") or 0)
    additional = y - x
    missed = n_true_pairs - x
    return {
        "n_true_pairs": n_true_pairs,
        "exact_true_pairs": int(exact_metrics.get("true_pairs_retrieved") or 0),
        "char_true_pairs": int(char_metrics.get("true_pairs_retrieved") or 0),
        "token_true_pairs": int(token_metrics.get("true_pairs_retrieved") or 0),
        "token_plus_exact_true_pairs": int(token_exact_metrics.get("true_pairs_retrieved") or 0),
        "exact_plus_char_true_pairs": x,
        "exact_char_token_true_pairs": y,
        "additional_true_pairs_from_token": additional,
        "additional_recall_points_of_total": (
            100.0 * additional / n_true_pairs if n_true_pairs else 0.0
        ),
        "previous_misses": missed,
        "additional_recall_on_previous_misses": (additional / missed) if missed > 0 else None,
        "exact_recall": exact_metrics.get("blocking_recall_pair"),
        "char_recall": char_metrics.get("blocking_recall_pair"),
        "exact_plus_char_recall": exact_char_metrics.get("blocking_recall_pair"),
        "token_recall": token_metrics.get("blocking_recall_pair"),
        "token_plus_exact_recall": token_exact_metrics.get("blocking_recall_pair"),
        "exact_char_token_recall": everything_metrics.get("blocking_recall_pair"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def build_markdown(report: dict) -> str:
    meta = report["meta"]
    lines: list[str] = []
    add = lines.append

    add("# token-blocker calibration")
    add("")
    add(f"* generated: {meta['generated_at']}")
    add(f"* prepared corpus: `{meta['prepared_dir']}`")
    add(
        f"* S1 entities: {fmt_int(meta['n_s1_entities'])}  |  true pairs: "
        f"{fmt_int(meta['n_true_pairs'])}"
    )
    add(f"* workers: {meta['workers'] if meta['workers'] else 'none (no verification stage ran)'}"
        f"  |  chunk rows: {fmt_int(meta['chunk_rows'])}")
    add(f"* candidate-row budget per cell: {fmt_int(meta['max_candidate_rows'])} (explicit guard)")
    add(
        f"* cells: {meta['grid_cells']} total, {meta['cells_evaluated']} evaluated, "
        f"{meta['cells_skipped']} not evaluated"
    )
    add(f"* elapsed: {meta['elapsed_minutes']:.1f} min")
    add("")
    add("## Grid")
    add("")
    add(f"* token df caps: {meta['df_caps']}")
    add(f"* token rarest-K: {meta['rarest_ks']}")
    add(
        f"* char cell reused for the unions: df cap {meta['char_df_cap']}, rarest-K "
        f"{meta['char_rarest_k']}, Jaccard {meta['char_jaccards']}"
    )
    add(f"* candidate sets: {meta['candidate_sets'] or 'not evaluated (--volume-only)'}")
    add("")
    add("**Token retrieval has no verification stage** - the token signal is boolean, so")
    add("sharing an eligible token is the blocker's decision. Every Jaccard column below is")
    add("the *char* threshold inside a union, not a token parameter.")
    add("")
    add(
        "The token blocker keeps an entity's `rarest-K` eligible tokens, so at a K large "
        "enough to cover every eligible token of both names it degenerates to Phase 0.2's "
        "analytic rule. The `K=5` rows are therefore the closest this grid comes to a "
        "direct cross-check against Phase 0's `42.3905%`."
    )
    add("")

    add("## Incremental contribution (the headline table)")
    add("")
    add(
        "`additional` is the **measured** count of true pairs that exact+char misses and "
        "exact+char+token finds. `share of misses recovered` is "
        "`additional / (true pairs - exact+char true pairs)`."
    )
    add("")
    add(
        "| df cap | K | char J | exact+char true | +token true | additional | additional pts "
        "| previous misses | share of misses recovered |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for row in report.get("incremental") or []:
        share = row["additional_recall_on_previous_misses"]
        add(
            f"| {row['df_cap']} | {row['rarest_k']} | {row['char_jaccard_threshold']:.2f} | "
            f"{fmt_int(row['exact_plus_char_true_pairs'])} | "
            f"{fmt_int(row['exact_char_token_true_pairs'])} | "
            f"{fmt_int(row['additional_true_pairs_from_token'])} | "
            f"{row['additional_recall_points_of_total']:.2f} | "
            f"{fmt_int(row['previous_misses'])} | "
            f"{'n/a' if share is None else f'{100.0 * share:.2f}%'} |"
        )
    add("")

    add("## Recall x volume, evaluated cells")
    add("")
    add(
        "| df cap | K | set | char J | pair recall | macro entity recall | full-recall S1 | "
        "partial-recall S1 | candidates | precision | reduction | F0.5 ceiling |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in report["grid"]:
        if not cell.get("evaluated"):
            continue
        for entry in cell["sets"]:
            metrics = entry["metrics"]
            add(
                f"| {cell['df_cap']} | {cell['rarest_k']} | {entry['set']} | "
                f"{_threshold_cell(entry['threshold'])} | "
                f"{_pct(metrics.get('blocking_recall_pair'))} | "
                f"{_pct(metrics.get('macro_recall_entity'))} | "
                f"{_pct(metrics.get('s1_full_recall_rate'))} | "
                f"{_pct(metrics.get('s1_partial_recall_rate'))} | "
                f"{fmt_int(metrics.get('n_candidate_pairs') or 0)} | "
                f"{_pct(metrics.get('candidate_precision'))} | "
                f"{_ratio(metrics.get('reduction_ratio'))} | "
                f"{_float(metrics.get('f05_ceiling_from_macro_recall'), 4)} |"
            )
    add("")

    add("### Grid-free sets, measured once")
    add("")
    add(
        "These do not depend on the token cell, so they are measured once and are the "
        "reference the deltas above are taken against."
    )
    add("")
    add(
        "| set | char J | pair recall | macro entity recall | candidates | precision | "
        "F0.5 ceiling |"
    )
    add("|---|---|---|---|---|---|---|")
    exact = report.get("exact") or {}
    add(
        f"| `{SET_EXACT}` | - | {_pct(exact.get('blocking_recall_pair'))} | "
        f"{_pct(exact.get('macro_recall_entity'))} | "
        f"{fmt_int(exact.get('n_candidate_pairs') or 0)} | "
        f"{_pct(exact.get('candidate_precision'))} | "
        f"{_float(exact.get('f05_ceiling_from_macro_recall'), 4)} |"
    )
    for group, name in (
        (report.get("char") or {}, SET_CHAR),
        (report.get("exact_plus_char") or {}, SET_EXACT_PLUS_CHAR),
    ):
        for key, block in group.items():
            add(
                f"| `{name}` | {key} | {_pct(block.get('blocking_recall_pair'))} | "
                f"{_pct(block.get('macro_recall_entity'))} | "
                f"{fmt_int(block.get('n_candidate_pairs') or 0)} | "
                f"{_pct(block.get('candidate_precision'))} | "
                f"{_float(block.get('f05_ceiling_from_macro_recall'), 4)} |"
            )
    add("")

    add("### Candidates per S1")
    add("")
    add(
        "| df cap | K | set | char J | candidates | avg | median | p90 | p99 | max | "
        "zero-candidate S1 |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in report["grid"]:
        if not cell.get("evaluated"):
            continue
        for entry in cell["sets"]:
            metrics = entry["metrics"]
            add(
                f"| {cell['df_cap']} | {cell['rarest_k']} | {entry['set']} | "
                f"{_threshold_cell(entry['threshold'])} | "
                f"{fmt_int(metrics.get('n_candidate_pairs') or 0)} | "
                f"{_float(metrics.get('avg_candidates_per_s1'), 1)} | "
                f"{_float(metrics.get('median_candidates_per_s1'))} | "
                f"{_float(metrics.get('p90_candidates_per_s1'))} | "
                f"{_float(metrics.get('p99_candidates_per_s1'))} | "
                f"{fmt_int(metrics.get('max_candidates_per_s1') or 0)} | "
                f"{fmt_int(metrics.get('n_s1_with_zero_candidates') or 0)} "
                f"({_pct(metrics.get('fraction_s1_with_zero_candidates'))}) |"
            )
    add("")

    add("## Per-source (S2 / S3) breakdown")
    add("")
    add(
        "| df cap | K | set | char J | source | true pairs | candidates | true retrieved | "
        "pair recall | macro recall | full-recall S1 | partial-recall S1 | precision |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in report["grid"]:
        if not cell.get("evaluated"):
            continue
        for entry in cell["sets"]:
            for prefix, block in (entry["metrics"].get("per_source") or {}).items():
                add(
                    f"| {cell['df_cap']} | {cell['rarest_k']} | {entry['set']} | "
                    f"{_threshold_cell(entry['threshold'])} | {prefix} | "
                    f"{fmt_int(block.get('n_true_pairs') or 0)} | "
                    f"{fmt_int(block.get('n_candidates') or 0)} | "
                    f"{fmt_int(block.get('true_pairs_retrieved') or 0)} | "
                    f"{_pct(block.get('blocking_recall_pair'))} | "
                    f"{_pct(block.get('macro_recall_entity'))} | "
                    f"{_pct(block.get('s1_full_recall_rate'))} | "
                    f"{_pct(block.get('s1_partial_recall_rate'))} | "
                    f"{_pct(block.get('candidate_precision'))} |"
                )
    add("")

    add("## Validation split")
    add("")
    add(
        "The same metrics restricted to the held-out S1 entities, using the existing "
        "S1-level `evaluation.split` config. Splitting is by S1 entity, so no pair of a "
        "validation entity appears in training."
    )
    add("")
    add(
        "| df cap | K | set | char J | val pair recall | val macro recall | val candidates | "
        "val zero-candidate S1 |"
    )
    add("|---|---|---|---|---|---|---|---|")
    any_val = False
    for cell in report["grid"]:
        if not cell.get("evaluated"):
            continue
        for entry in cell["sets"]:
            val = entry.get("metrics_val")
            if not val:
                continue
            any_val = True
            add(
                f"| {cell['df_cap']} | {cell['rarest_k']} | {entry['set']} | "
                f"{_threshold_cell(entry['threshold'])} | "
                f"{_pct(val.get('blocking_recall_pair'))} | "
                f"{_pct(val.get('macro_recall_entity'))} | "
                f"{fmt_int(val.get('n_candidate_pairs') or 0)} | "
                f"{fmt_int(val.get('n_s1_with_zero_candidates') or 0)} |"
            )
    if not any_val:
        add("")
        add("*The split is not enabled in this config, so no validation figures were computed.*")
    add("")

    add("## Candidate volume")
    add("")
    add(
        "`upper_bound_from_posting_expansion` counts the postings a cell's queries expand "
        "to and is an **upper bound** on the distinct candidate pairs (a target reached via "
        "two of an S1 entity's keys is counted twice there). `exact` is the distinct count, "
        "measured for evaluated sets. Both are reported for every evaluated cell; only "
        "`upper_bound_from_posting_expansion` exists for the rest."
    )
    add("")
    add(
        "| df cap | K | scope | kind | set | char J | candidates | avg/S1 | median | p90 | p99 "
        "| max | zero-candidate S1 |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in report.get("volume_curve") or []:
        add(
            f"| {_blank(row.get('df_cap'))} | {_blank(row.get('rarest_k'))} | "
            f"{row.get('scope', '')} | {row['kind']} | {row.get('set', '')} | "
            f"{_threshold_cell(row.get('jaccard_threshold'))} | "
            f"{fmt_int(row['n_candidate_pairs'])} | {row['avg_candidates_per_s1']:.1f} | "
            f"{_float(row['median_candidates_per_s1'])} | "
            f"{_float(row['p90_candidates_per_s1'])} | {_float(row['p99_candidates_per_s1'])} | "
            f"{fmt_int(row['max_candidates_per_s1'])} | "
            f"{fmt_int(row['n_s1_with_zero_candidates'])} "
            f"({100.0 * row['fraction_s1_with_zero_candidates']:.2f}%) |"
        )
    add("")
    add("Cells the candidate-row guard kept out of evaluation are listed with their true")
    add("volume; their recall is deliberately absent rather than truncated.")
    add("")
    skipped = [cell for cell in report["grid"] if not cell.get("evaluated")]
    if not skipped:
        add("* none")
    for cell in skipped:
        add(
            f"* **not evaluated** (cap={cell['df_cap']}, K={cell['rarest_k']}): true bound "
            f"{fmt_int(cell['volume'])} pairs, reason `{cell.get('skip_reason')}`"
        )
    add("")

    add("## Token-level drop reasons, per source")
    add("")
    add(
        "Counted at the token level. `above cap` is the df filter, `beyond K` the rank "
        "filter - separating them is what shows which one binds. `no token` counts names "
        "with no token at all; `no key here` counts S1 entities that kept no key for this "
        "source, which any of the three drops above can cause."
    )
    add("")
    add(
        "| source | rows read | no key here | no token | dropped: absent from this source | "
        "dropped: above cap | dropped: beyond K | entries/row |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for source, block in (report.get("structural") or {}).items():
        add(
            f"| {source} | {fmt_int(block.get('n_s1_rows_read', 0))} | "
            f"{fmt_int(block.get('s1_with_no_key_in_this_source', 0))} | "
            f"{fmt_int(block.get('s1_name_with_no_token', 0))} | "
            f"{fmt_int(block.get('tokens_dropped_absent_from_this_source', 0))} | "
            f"{fmt_int(block.get('tokens_dropped_above_df_cap', 0))} | "
            f"{fmt_int(block.get('tokens_dropped_beyond_rarest_k', 0))} | "
            f"{block.get('entries_per_row', 0)} |"
        )
    add("")

    add("## Most frequent tokens (evidence for the cap)")
    add("")
    add("| source | token | df | postings |")
    add("|---|---|---|---|")
    for row in (report.get("top_tokens") or [])[:40]:
        add(
            f"| {row['source']} | `{row['token']}` | {fmt_int(row['df'])} | "
            f"{fmt_int(row['postings'])} |"
        )
    add("")

    add("## Phase 0 analytic reference (NOT measured here)")
    add("")
    add(
        "These are Phase 0.2-0.4's **analytic estimates**: a pair counted as reachable if it "
        "shares a token with `df <= cap`, with no rarest-K limit and no retrieval. They were "
        "not produced by running a blocker and must not be read as measured retrieval "
        "results. Every other table in this report is measured."
    )
    add("")
    for key, value in TOKEN_ANALYTIC_REFERENCE.items():
        add(f"* `{key}`: {value}")
    add("")

    add("## Caveats")
    add("")
    add(
        "* No operating threshold is chosen here, for either blocker. The point of the run "
        "is the curve."
    )
    add(
        "* The token blocker has no verification stage: its candidate set IS its decision. "
        "Any Jaccard number in this report belongs to the reused char stage."
    )
    add(
        "* Token df is corpus-relative, counted over the prepared target tables of this run. "
        "No train-learned vocabulary is used, and no S1-side information enters it."
    )
    add(
        "* `exact` comes from the shipped `src.blocking.ExactNameIndex`; `char` from "
        "`scripts/calibrate_char_blocker`'s own df, index, selection, name-key store and "
        "Jaccard verification, imported unchanged. Only the token retrieval and the unions "
        "are new."
    )
    add(
        "* All sets are **unions**. A pair proposed by any blocker is a candidate; no "
        "blocker's output is intersected away."
    )
    add(
        "* Cells above `--max-candidate-rows` report their true volume and "
        "`evaluated: false`. Nothing is silently truncated."
    )
    add(
        "* Recall is measured against the training ground truth; the validation figures are "
        "the same measurement restricted to held-out S1 entities."
    )
    add("")
    return "\n".join(lines) + "\n"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _float(value: Any, places: int = 0) -> str:
    return "n/a" if value is None else f"{float(value):.{places}f}"


def _ratio(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f}x"


def _threshold_cell(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _blank(value: Any) -> str:
    return "" if value is None else str(value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="calibrate_token_blocker.py",
        description=(
            "Measure the rarest-K token blocker, and how much it adds on top of exact + char. "
            "Token df is corpus-relative; retrieval is an inverted index; there is no "
            "verification stage because the token signal is boolean. No operating threshold "
            "is chosen here."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="config YAML path")
    parser.add_argument("--data-root", default=None, help="override paths.data_root")
    parser.add_argument("--work-dir", default=None, help="override paths.work_dir")
    parser.add_argument(
        "--split", default="train", choices=["train", "test"], help="prepared split to read"
    )
    parser.add_argument(
        "--sources", default=",".join(TARGET_SOURCES), help="target sources, indexed separately"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="where reports and artifacts go; default <work_dir>/calibration",
    )
    parser.add_argument(
        "--df-caps",
        type=_int_list,
        default=list(DEFAULT_DF_CAPS),
        help="token document-frequency caps to sweep; the maximum is what gets indexed",
    )
    parser.add_argument(
        "--rarest-ks",
        type=_int_list,
        default=list(DEFAULT_RAREST_KS),
        help="how many of an entity's rarest surviving tokens act as its keys",
    )
    parser.add_argument(
        "--char-df-cap",
        type=int,
        default=DEFAULT_CHAR_DF_CAP,
        help="df cap of the char cell whose candidates the unions reuse",
    )
    parser.add_argument(
        "--char-rarest-k",
        type=int,
        default=DEFAULT_CHAR_RAREST_K,
        help="rarest-K of the char cell whose candidates the unions reuse",
    )
    parser.add_argument(
        "--char-jaccards",
        type=_float_list,
        default=list(DEFAULT_CHAR_JACCARDS),
        help="char Jaccard thresholds entering the unions (one verification pass serves all)",
    )
    parser.add_argument(
        "--max-candidate-rows",
        type=int,
        default=DEFAULT_MAX_CANDIDATE_ROWS,
        help=(
            "explicit, reported guard: a cell whose retrieval bound exceeds this is measured "
            "but not evaluated. Never silently capped - the cell reports its true volume and "
            "'evaluated: false'. 0 disables the guard."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="processes for the reused char verification (the only parallel stage); 0 = auto",
    )
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS, help="rows per chunk")
    parser.add_argument(
        "--verify-chunk-pairs",
        type=int,
        default=DEFAULT_VERIFY_CHUNK_PAIRS,
        help="pairs per char verification payload",
    )
    parser.add_argument(
        "--limit-s1", type=int, default=None, help="max S1 rows to process (smoke tests)"
    )
    parser.add_argument(
        "--volume-only", action="store_true", help="measure the volume curve only; skip evaluation"
    )
    parser.add_argument(
        "--top-tokens", type=int, default=50, help="rows per source in the evidence CSV"
    )
    parser.add_argument("--resume", action="store_true", help="reuse completed artifacts")
    parser.add_argument(
        "--timings", action="store_true", help="record per-stage wall clock in the JSON"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Artifact fingerprinting
# ---------------------------------------------------------------------------
def artifact_fingerprint(
    args: argparse.Namespace,
    config: dict,
    sources: Sequence[str],
    n_entities: int,
    target_rows: dict,
) -> str:
    """Fingerprint every input the corpus artifacts depend on.

    Deliberately broader than the char blocker's fingerprint: this run's artifacts live
    in their own namespace, but the char stage's index bounds and thresholds are inputs
    too, and a stale char artifact silently reused would corrupt every union containing
    it. The literal ``"token"`` also separates these artifacts from any other
    calibration's, so a shared artifact root can never hand this script a trigram table
    under a token name.
    """
    material = "|".join(
        str(part)
        for part in (
            ARTIFACT_VERSION,
            "token",
            args.split,
            ",".join(sorted(sources)),
            max(args.df_caps),
            max(args.rarest_ks),
            args.char_df_cap,
            args.char_rarest_k,
            ",".join(f"{t:g}" for t in sorted(args.char_jaccards)),
            args.chunk_rows,
            args.limit_s1,
            n_entities,
            config["resolved"]["prepared_dir"],
            ";".join(f"{key}={value}" for key, value in sorted(target_rows.items())),
        )
    )
    return hashlib.blake2b(material.encode("utf-8"), digest_size=16).hexdigest()


def _marker_payload(directory: Path) -> dict:
    """An artifact marker's payload, minus the two keys the marker protocol owns.

    Used instead of re-describing a resumed structure: a name-key store's ``describe``
    is cheap, but reading its blob back for one line of the report is not.
    """
    return {
        key: value
        for key, value in read_json(directory / "artifact.done.json").items()
        if key not in ("version", "fingerprint")
    }


def top_token_keys(
    index: _TrigramIndex, vocab: _TokenVocabulary, limit: int
) -> list[tuple[str, int, int]]:
    """Most-populated token keys as (text, df, postings). Evidence for the cap."""
    counts = index.counts_per_key()
    if counts.size == 0:
        return []
    order = np.argsort(counts)[::-1][:limit]
    out: list[tuple[str, int, int]] = []
    for position in order:
        code = int(index.keys[position])
        text = vocab.tokens[code] if 0 <= code < len(vocab.tokens) else f"<code {code}>"
        out.append((text, int(index.key_df[position]), int(counts[position])))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.time()
    log = setup_logging(LOG_NAME, level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        config = load_config(
            args.config,
            overrides={
                key: value
                for key, value in (("data_root", args.data_root), ("work_dir", args.work_dir))
                if value
            },
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2

    set_seed(config.get("project", {}).get("seed", 42))

    sources = [part.strip() for part in args.sources.split(",") if part.strip()]
    unknown = [source for source in sources if source not in TARGET_SOURCES]
    if unknown or not sources:
        log.error(
            "invalid --sources %r; expected a non-empty subset of %s",
            unknown or args.sources,
            list(TARGET_SOURCES),
        )
        return 2
    if not args.df_caps or any(cap <= 0 for cap in args.df_caps):
        log.error("--df-caps must be a non-empty list of positive ints: %s", args.df_caps)
        return 2
    if not args.rarest_ks or any(k <= 0 or k > 255 for k in args.rarest_ks):
        log.error("--rarest-ks must be a non-empty list in [1, 255]: %s", args.rarest_ks)
        return 2
    if args.char_df_cap <= 0 or not 1 <= args.char_rarest_k <= 255:
        log.error("--char-df-cap must be positive and --char-rarest-k in [1, 255]")
        return 2
    if not args.char_jaccards or any(not 0.0 < t <= 1.0 for t in args.char_jaccards):
        log.error("--char-jaccards must be a non-empty list in (0, 1]: %s", args.char_jaccards)
        return 2
    if args.chunk_rows <= 0 or args.verify_chunk_pairs <= 0:
        log.error("--chunk-rows and --verify-chunk-pairs must be positive")
        return 2
    if args.max_candidate_rows < 0:
        log.error("--max-candidate-rows must be >= 0 (0 disables the guard)")
        return 2

    log.info(describe_environment(config))
    hardware = detect_hardware()
    log.info("hardware:\n%s", format_hardware_report(hardware))
    if args.split != "train":
        log.warning(
            "--split %s: the ground truth exists only for the train split, so recall here is "
            "measured against the TRAIN ground truth. Use this only for candidate generation.",
            args.split,
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["resolved"]["work_dir"]) / "calibration"
    )
    ensure_dir(output_dir)
    # A dedicated namespace: this script and the char calibration share the default
    # output directory, and two artifact trees that look alike must never mix.
    artifact_root = output_dir / ARTIFACT_DIRNAME / "token"

    ground_truth = load_ground_truth(config, log=log)
    n_entities = ground_truth.n_entities
    s1_rows = count_s1_rows(config, args)
    if s1_rows <= 0:
        log.error("no source1 rows found; run scripts/prepare_data.py first")
        return 2
    log.info(
        "S1 rows to read: %s | ground-truth entities: %s | true pairs: %s",
        fmt_int(s1_rows),
        fmt_int(n_entities),
        fmt_int(ground_truth.n_matches),
    )

    target_rows = target_row_counts(config, args.split, sources, log)
    n_target_records = sum(target_rows.values()) if target_rows else None
    fingerprint = artifact_fingerprint(args, config, sources, n_entities, target_rows)

    cells = [(cap, k) for cap in args.df_caps for k in args.rarest_ks]
    log.info(
        "grid: %d df caps x %d rarest-K = %d cells; %d per-cell sets; token index bound "
        "(cap=%s, K=%s); char cell (cap=%s, K=%s, J=%s)",
        len(args.df_caps),
        len(args.rarest_ks),
        len(cells),
        len(PER_CELL_SETS),
        max(args.df_caps),
        max(args.rarest_ks),
        args.char_df_cap,
        args.char_rarest_k,
        args.char_jaccards,
    )

    timings: dict[str, float] = {}
    indexes: dict[str, _TrigramIndex] = {}
    selections: dict[str, _S1Selection] = {}
    dfs: dict[str, _TrigramDf] = {}
    vocabs: dict[str, _TokenVocabulary] = {}
    structural: dict[str, dict] = {}
    top_tokens: list[dict] = []

    # ---- per-source token artifacts ----------------------------------------
    for source in sources:
        stage = time.time()
        vocab_dir = artifact_root / source / "vocab"
        df_dir = artifact_root / source / "df"
        index_dir = artifact_root / source / "index"
        select_dir = artifact_root / source / "s1_selection"

        if args.resume and _artifact_ready(vocab_dir, fingerprint):
            log.info("[%s] resuming the token vocabulary", source)
            vocab = _TokenVocabulary.load(vocab_dir, source)
        else:
            vocab = _TokenVocabulary(source)
        vocabs[source] = vocab

        if args.resume and _artifact_ready(df_dir, fingerprint):
            log.info("[%s] resuming the token df table", source)
            df = _TrigramDf.load(df_dir)
        else:
            df = count_token_df(
                iter_prepared(
                    config, args.split, source, columns=[NAME_NORM], chunksize=args.chunk_rows
                ),
                NAME_NORM,
                vocab,
                log,
                f"[{source}] token df",
            )
            df.save(df_dir)
            vocab.save(vocab_dir)
            _write_artifact_marker(
                df_dir, fingerprint, {**describe_df(df, args.df_caps), **vocab.describe()}
            )
            _write_artifact_marker(vocab_dir, fingerprint, vocab.describe())
        dfs[source] = df

        if args.resume and _artifact_ready(index_dir, fingerprint):
            log.info("[%s] resuming the token index", source)
            index = _TrigramIndex.load(index_dir, source)
        else:
            index = build_token_index(
                iter_prepared(
                    config,
                    args.split,
                    source,
                    columns=["entity_id", NAME_NORM],
                    chunksize=args.chunk_rows,
                ),
                df,
                vocab,
                "entity_id",
                NAME_NORM,
                max(args.df_caps),
                max(args.rarest_ks),
                source,
                log,
                f"[{source}] token index",
            )
            index.save(index_dir)
            _write_artifact_marker(index_dir, fingerprint, index.describe())
        indexes[source] = index

        if args.resume and _artifact_ready(select_dir, fingerprint):
            log.info("[%s] resuming the S1 token selection", source)
            selection = _S1Selection.load(select_dir)
            structural[source] = read_json(select_dir / "artifact.done.json").get("breakdown", {})
        else:
            selection, breakdown = build_token_s1_selection(
                s1_chunks(config, args, ["entity_id", NAME_NORM]),
                df,
                vocab,
                ground_truth,
                "entity_id",
                NAME_NORM,
                max(args.df_caps),
                max(args.rarest_ks),
                n_entities,
                log,
                f"[{source}] token s1",
            )
            selection.save(select_dir)
            _write_artifact_marker(select_dir, fingerprint, {"breakdown": breakdown})
            structural[source] = breakdown
        selections[source] = selection

        top_tokens.extend(
            {"source": source, "token": text, "df": df_value, "postings": postings}
            for text, df_value, postings in top_token_keys(index, vocab, args.top_tokens)
        )
        timings[f"{source}_artifacts"] = round(time.time() - stage, 1)
        log.info(
            "[%s] token artifacts ready in %.1f min (rss %s)",
            source,
            (time.time() - stage) / 60.0,
            human_bytes(current_rss_bytes() or 0),
        )

    log.info("token df tables: %s", {s: describe_df(dfs[s], args.df_caps) for s in sources})
    log.info("token vocabularies: %s", {s: vocabs[s].describe() for s in sources})
    log.info("token indexes: %s", {s: indexes[s].describe() for s in sources})
    log.info("S1 token selections: %s", {s: selections[s].describe() for s in sources})

    # ---- volume sweep: every cell ------------------------------------------
    stage = time.time()
    volume_rows, cell_volume = sweep_volume(
        indexes, selections, cells, n_entities, args.chunk_rows, log
    )
    timings["volume_sweep"] = round(time.time() - stage, 1)

    # ---- exact-name index availability (read-only) -------------------------
    exact_indexes: dict[str, Optional[ExactNameIndex]] = {}
    for source in sources:
        directory = index_dir_for(config, args.split, source, BLOCKER_EXACT_NAME)
        try:
            exact_indexes[source] = ExactNameIndex.load(directory, log=None)
            log.info("[%s] exact-name index: %s", source, exact_indexes[source].describe())
        except (FileNotFoundError, ValueError) as exc:
            exact_indexes[source] = None
            log.warning(
                "[%s] exact-name index unavailable (%s); every set containing exact cannot be "
                "reported. Run: python scripts/build_indexes.py",
                source,
                exc,
            )

    extras: dict[str, Any] = {}
    exact_by_source: dict[str, np.ndarray] = {}
    exact_packed_merged = _EMPTY_INT64
    char_stage_info: Optional[dict] = None
    char_packed: dict[float, np.ndarray] = {}
    store_dirs: dict[str, str] = {}
    s1_store_describe: Optional[dict] = None
    workers: Optional[int] = None

    if args.volume_only:
        log.info("--volume-only: skipping the char stage, the exact set and the evaluation")
        grid = [
            {
                "df_cap": cap,
                "rarest_k": rarest_k,
                "volume": cell_volume[(cap, rarest_k)],
                "volume_kind": BOUND_KIND,
                "evaluated": False,
                "skip_reason": "volume_only_requested",
                "sets": [],
            }
            for cap, rarest_k in cells
        ]
    elif any(index is None for index in exact_indexes.values()):
        log.error(
            "the exact-name index is unavailable for at least one source, so the union sets "
            "cannot be built. Run: python scripts/build_indexes.py"
        )
        return 2
    else:
        # ---- the S1 name-key store the char verification needs -------------
        stage = time.time()
        s1_store_dir = artifact_root / "source1_namekeys"
        if args.resume and _artifact_ready(s1_store_dir, fingerprint):
            log.info("resuming the source1 name-key store")
            s1_store = _NameKeyStore.load(s1_store_dir)
            s1_store_describe = _marker_payload(s1_store_dir)
        else:
            s1_store = build_name_key_store(
                s1_chunks(config, args, ["entity_id", NAME_KEY]),
                "entity_id",
                NAME_KEY,
                log,
                "[source1] namekeys",
                ground_truth=ground_truth,
                n_entities=n_entities,
            )
            s1_store.save(s1_store_dir)
            s1_store_describe = s1_store.describe()
            _write_artifact_marker(s1_store_dir, fingerprint, s1_store_describe)
        timings["source1_namekey_store"] = round(time.time() - stage, 1)

        # ---- the exact-name candidate set (grid-independent) --------------
        stage = time.time()
        exact_by_source = resolve_exact_packed(
            config,
            args,
            exact_indexes,
            s1_store,
            artifact_root,
            ground_truth,
            n_entities,
            fingerprint,
            log,
        )
        exact_packed_merged = union_packed(*exact_by_source.values())
        timings["exact_pairs"] = round(time.time() - stage, 1)

        # ---- the reused char stage ----------------------------------------
        char_indexes, char_selections, _char_dfs, char_stage_info = build_char_artifacts(
            args,
            config,
            sources,
            ground_truth,
            n_entities,
            artifact_root,
            fingerprint,
            log,
            timings,
        )

        # Workers are clamped to the work available, so the clamp needs a real chunk
        # count - the char cell's retrieval bound is the honest one.
        n_verify_chunks = max(
            1, int(char_stage_info["retrieval_bound"] // max(1, args.verify_chunk_pairs))
        )
        workers = resolve_workers(
            args.workers,
            config.get("compute", {}).get("num_workers", 0),
            n_verify_chunks,
            log,
            "char verify workers",
        )
        window, verify_chunk_pairs = plan_inflight_window(
            workers,
            args.verify_chunk_pairs,
            VERIFY_BYTES_PER_PAIR,
            _payload_budget(config),
            log,
            "char verification payload",
        )
        log.info(
            "char verification: %d workers, in-flight window %d, %s pairs per payload",
            workers,
            window,
            fmt_int(verify_chunk_pairs),
        )

        # ---- the target name-key stores the char verification reads --------
        store_dirs = {source: str(artifact_root / source / "namekeys") for source in sources}
        for source in sources:
            store_dir = artifact_root / source / "namekeys"
            if args.resume and _artifact_ready(store_dir, fingerprint):
                log.info("[%s] resuming the target name-key store", source)
                continue
            store = build_name_key_store(
                iter_prepared(
                    config,
                    args.split,
                    source,
                    columns=["entity_id", NAME_KEY],
                    chunksize=args.chunk_rows,
                ),
                "entity_id",
                NAME_KEY,
                log,
                f"[{source}] namekeys",
            )
            store.save(store_dir)
            _write_artifact_marker(store_dir, fingerprint, store.describe())

        char_packed = retrieve_char_candidates(
            args,
            sources,
            char_indexes,
            char_selections,
            s1_store,
            store_dirs,
            artifact_root,
            fingerprint,
            workers,
            window,
            verify_chunk_pairs,
            log,
            timings,
        )

        grid, extras = evaluate_token_grid(
            args,
            config,
            indexes,
            selections,
            exact_packed_merged,
            char_packed,
            ground_truth,
            cells,
            cell_volume,
            n_entities,
            n_target_records,
            workers,
            window,
            verify_chunk_pairs,
            log,
            timings,
        )

    elapsed = time.time() - started
    timings["total"] = round(elapsed, 1)
    volume_rows.extend(extras.get("exact_volume_rows", []))

    report: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "log_name": LOG_NAME,
            "split": args.split,
            "ground_truth_split": "train",
            "sources": sources,
            "workers": extras.get("workers"),
            "verify_window": extras.get("verify_window"),
            "verify_chunk_pairs": extras.get("verify_chunk_pairs"),
            "chunk_rows": args.chunk_rows,
            "limit_s1": args.limit_s1,
            "s1_rows_read": int(s1_rows),
            "df_caps": list(args.df_caps),
            "rarest_ks": list(args.rarest_ks),
            "char_df_cap": args.char_df_cap,
            "char_rarest_k": args.char_rarest_k,
            "char_jaccards": list(args.char_jaccards),
            "max_candidate_rows": args.max_candidate_rows,
            "volume_only": bool(args.volume_only),
            "candidate_sets": extras.get("sets"),
            "n_s1_entities": int(n_entities),
            "n_true_pairs": int(ground_truth.n_matches),
            "n_target_records": n_target_records,
            "prepared_dir": str(config["resolved"]["prepared_dir"]),
            "hardware": hardware,
            "elapsed_minutes": round(elapsed / 60.0, 2),
            "timings_seconds": timings if args.timings else None,
            "artifact_fingerprint": fingerprint,
            "artifact_root": str(artifact_root),
            "resume_requested": bool(args.resume),
            "token_semantics": (
                "name_norm, str.split() distinct tokens (identical to Phase 0's "
                "analyze_blocking_statistics._token_set)"
            ),
            "verification": (
                "none for token (boolean signal, no threshold); the char unions reuse "
                "scripts/analyze_name_differences._trigram_jaccard unchanged, through the "
                "char calibration's verify_similarities"
            ),
            "grid_cells": len(cells),
            "cells_evaluated": sum(1 for cell in grid if cell.get("evaluated")),
            "cells_skipped": sum(1 for cell in grid if not cell.get("evaluated")),
            "split_metrics_enabled": extras.get("split_enabled"),
            "exact_index_available": {
                source: index is not None for source, index in exact_indexes.items()
            },
            "notes": [
                "No operating threshold is chosen by this script; it produces the evidence.",
                "Token candidate sets are UNIONS with the other blockers, never intersections.",
                "Cells above --max-candidate-rows report their true volume and evaluated=false.",
                "The exact-name blocker is read from the existing index; it is not modified.",
                "The char blocker is imported and reused unchanged; it is not modified.",
                "Token df is corpus-relative, counted from the prepared target tables.",
                "The token blocker has no verification stage; its candidate set is its decision.",
                "The Phase 0 token figures are analytic references, not measured here.",
            ],
        },
        "grid": grid,
        "volume_curve": volume_rows,
        "structural": structural,
        "top_tokens": top_tokens,
        "exact": extras.get("exact"),
        "char": extras.get("char"),
        "exact_plus_char": extras.get("exact_plus_char"),
        "incremental": extras.get("incremental"),
        "artifacts": {
            "indexes": {source: indexes[source].describe() for source in sources},
            "df": {source: describe_df(dfs[source], args.df_caps) for source in sources},
            "vocabularies": {source: vocabs[source].describe() for source in sources},
            "s1_selection": {source: selections[source].describe() for source in sources},
            "s1_namekey_store": s1_store_describe,
            "target_row_counts": target_rows,
            "exact_candidate_pairs": {
                source: int(packed.size) for source, packed in exact_by_source.items()
            },
            "exact_candidate_pairs_merged": int(exact_packed_merged.size),
            "char_candidate_pairs": {
                f"{_jaccard_key(threshold)}": int(packed.size)
                for threshold, packed in char_packed.items()
            },
            "char_stage": char_stage_info,
            "target_namekey_dirs": store_dirs,
        },
        "token_analytic_reference": TOKEN_ANALYTIC_REFERENCE,
    }

    json_path = output_dir / "token_blocker_calibration.json"
    write_json(json_path, report)
    log.info("wrote %s", json_path)

    csv_rows: list[dict] = []
    for cell in grid:
        if not cell.get("evaluated"):
            continue
        for entry in cell["sets"]:
            row: dict[str, Any] = {
                "df_cap": cell["df_cap"],
                "rarest_k": cell["rarest_k"],
                "jaccard_threshold": entry["threshold"],
                "set": entry["set"],
                "volume": cell["volume"],
                "volume_kind": cell["volume_kind"],
                "pairs_retrieved": cell.get("pairs_retrieved"),
                "elapsed_seconds": entry.get("elapsed_seconds"),
            }
            for key in REPORTED_METRICS:
                row[key] = entry["metrics"].get(key)
            for prefix, block in (entry["metrics"].get("per_source") or {}).items():
                for key, value in block.items():
                    row[f"{prefix}_{key}"] = value
            if entry.get("metrics_val"):
                for key in (
                    "blocking_recall_pair",
                    "macro_recall_entity",
                    "n_candidate_pairs",
                    "true_pairs_retrieved",
                    "n_s1_with_zero_candidates",
                    "fraction_s1_with_zero_candidates",
                ):
                    row[f"val_{key}"] = entry["metrics_val"].get(key)
            csv_rows.append(row)

    write_csv(output_dir / "token_blocker_calibration.csv", csv_rows)
    write_csv(output_dir / "token_blocker_volume.csv", volume_rows)
    write_csv(output_dir / "token_blocker_top_tokens.csv", top_tokens)
    write_csv(output_dir / "token_blocker_incremental.csv", extras.get("incremental") or [])

    markdown_path = output_dir / "token_blocker_calibration.md"
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    log.info(
        "wrote token_blocker_calibration.{json,csv,md}, token_blocker_volume.csv, "
        "token_blocker_top_tokens.csv, token_blocker_incremental.csv"
    )
    log.info(
        "done: %d cells, %d evaluated, %d not evaluated; elapsed %.1f min; rss %s",
        len(cells),
        report["meta"]["cells_evaluated"],
        report["meta"]["cells_skipped"],
        elapsed / 60.0,
        human_bytes(current_rss_bytes() or 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
