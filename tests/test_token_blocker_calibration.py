"""Fixture tests for ``scripts/calibrate_token_blocker.py`` (Phase 1 token calibration).

Two layers, both synthetic and both local-only:

* **unit tests** over the module's own internals - the tokenizer, the dense
  vocabulary, the token df count, the rarest-K index build, the S1 selection and its
  drop accounting, one cell's retrieval, and the packed-union helpers that build
  ``exact+char+token``;
* **a fixture end-to-end run** over the same 14 S1 / 9 S2 / 6 S3 / 11-pair fixture
  ``tests/test_blocking_statistics.py`` and ``tests/test_char_blocker_calibration.py``
  use, normalized by the real ``scripts/prepare_data.py`` and blocked by the real
  ``ExactNameIndex`` this file builds. That is where the claims that need a whole
  pipeline - the union sets, the incremental formulas, the volume accounting, the
  candidate-row guard, determinism, ``--resume`` - are actually checked.

The reused char stage is exercised for real here (it is part of every union), so
these tests also cover the char calibration module's index/selection/verification path
as this script consumes it - without modifying that script.

What this file deliberately does NOT assert
-------------------------------------------
That the token blocker recovers true pairs beyond ``exact + char`` *on this fixture*.
The fixture's exact-name pairs have identical ``name_norm``, hence identical
``name_key``, hence a Jaccard of exactly 1.0 - so the char stage structurally contains
every exact-name pair here and the token delta over ``exact + char`` can legitimately
be zero. The delta is asserted for shape and arithmetic instead
(``additional == y - x``, ``additional / previous_misses``), and the token blocker's
own contribution beyond ``exact`` alone IS asserted to be positive, because that
separation the fixture does support.

Nothing here reads the real dataset; the corpus is 29 rows. Runs standalone
(``python tests/test_token_blocker_calibration.py``).
"""

from __future__ import annotations

import ast
import atexit
import logging
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import calibrate_token_blocker as ct  # noqa: E402
from scripts.calibrate_char_blocker import BOUND_KIND, EXACT_KIND, VOLUME_ALL  # noqa: E402
from scripts import prepare_data  # noqa: E402
from src.blocking import BLOCKER_EXACT_NAME, build_index, pack_pairs, unpack_pairs  # noqa: E402
from src.data_loader import load_config  # noqa: E402
from src.normalization import NAME_KEY, NAME_NORM  # noqa: E402
from src.utils import read_json  # noqa: E402

# ---------------------------------------------------------------------------
# The fixture, copied from tests/test_blocking_statistics.py so this file is
# self-contained (the three are checked against each other by
# test_fixture_matches_blocking_statistics_fixture).
# ---------------------------------------------------------------------------
HEADER = ["entity_id", "business_name", "business_address", "country"]

S1 = [
    ("S1-1", "Sunrise Traders", "1 Main St, Austin, TX", "US"),
    ("S1-2", "Blue Sky Exports", "2 Dock Rd, Seattle, WA", "US"),
    ("S1-3", "Global Tech Solutions", "3 High St, London", "United Kingdom"),
    ("S1-4", "Acme Industries Private Limited", "", "India"),
    ("S1-5", "Meridian Logistics", "5 Freight Way, Denver, CO", "US"),
    ("S1-6", "Zenith Foods", "7 Rue de Rivoli, Paris", "France"),
    ("S1-7", "Delta Freight", "9 Depot Ave, Memphis, TN", "US"),
    ("S1-8", "Sunrise Traders", "1 Main St, Austin, TX", "US"),
    ("S1-9", "Blue Sky Exports", "2 Dock Rd, Seattle, WA", "US"),
    ("S1-10", "Unique Widgets", "", "US"),
    ("S1-11", "Acme Industries", "4 MG Road, Mumbai", "India"),
    ("S1-12", "Aurora Wholesale", "20 Elm St, Dallas, TX", "US"),
    ("S1-13", "Freightways", "30 Canyon Rd, Tucson, AZ", "US"),
    ("S1-14", "राम मार्केटिंग", "5 Karol Bagh, New Delhi", "India"),
]
S2 = [
    ("S2-201", "Sunrise Traders", "1 Main St, Austin, TX", "US"),
    ("S2-202", "BlueSky Exports", "2 Dock Rd, Seattle, WA", "US"),
    ("S2-203", "Global Tech Solutions", "3 High Street, London", "United Kingdom"),
    ("S2-204", "Acme Industries", "4 MG Road, Mumbai", "India"),
    ("S2-205", "Acme Industries", "4 MG Road, Mumbai", "India"),
    ("S2-206", "Quasar Holdings", "99 Elsewhere, Lyon", "France"),
    ("S2-207", "Northern Lights Trading", "77 Harbour Road, Vancouver", "Canada"),
    ("S2-208", "Freightwayss", "31 Canyon Road, Phoenix, AZ", "US"),
    ("S2-209", "Cascade Textiles", "88 Harbour Way, Reno, NV", "US"),
]
S3 = [
    ("S3-301", "Acme Private Limited", "4 MG Road, Mumbai", "India"),
    ("S3-302", "Meridian Logistics", "5 Freight Way, Denver, CO", "US"),
    ("S3-303", "Zenith Foods", "1 Rue de Rivoli, Paris", "France"),
    ("S3-304", "Acme Industries", "4 MG Road, Mumbai", "India"),
    ("S3-305", "Delta Freight", "9 Depot Ave, Memphis, TN", "US"),
    ("S3-306", "Ram Marketing", "5 Karol Bagh, New Delhi", "India"),
]
GROUND_TRUTH = [
    ("S1-1", "S2-201"),
    ("S1-2", "S2-202"),
    ("S1-3", "S2-203"),
    ("S1-4", "S3-301"),
    ("S1-5", "S3-302"),
    ("S1-6", "S3-303"),
    ("S1-7", "S3-305"),
    ("S1-8", ""),
    ("S1-9", ""),
    ("S1-10", ""),
    ("S1-11", ""),
    ("S1-12", "S2-207"),
    ("S1-13", "S2-208,S2-209"),
    ("S1-14", "S3-306"),
]

# Cap 10000 / K 10 is "no cap, no rank limit" for a 29-row corpus: it makes the token
# blocker coincide with Phase 0.2's analytic rule, which is the cross-check this grid
# is for. Cap 1 / K 1 is the tight end, where most keys are dropped.
_GRID = {"df_caps": [1, 10000], "rarest_ks": [1, 10], "char_jaccards": [0.3, 1.0]}
_LOOSE = (10000, 10)
_TIGHT = (1, 1)

_CACHE: dict[str, object] = {}


# ---------------------------------------------------------------------------
# unit-level helpers
# ---------------------------------------------------------------------------
def _make_vocab(texts: list[str], source: str = "source2") -> ct._TokenVocabulary:
    """A vocabulary interned from ``texts``, in order."""
    vocab = ct._TokenVocabulary(source)
    for text in texts:
        vocab.codes_for_text(text)
    return vocab


def _make_df(texts: list[str], vocab: ct._TokenVocabulary) -> ct._TrigramDf:
    """A token df table over ``texts``, one document each, interned into ``vocab``."""
    parts = [vocab.codes_for_text(text) for text in texts]
    live = [part for part in parts if part.size]
    if not live:
        return ct._TrigramDf(ct._EMPTY_INT64, ct._EMPTY_INT64)
    unique, counts = np.unique(np.concatenate(live), return_counts=True)
    return ct._TrigramDf(unique, counts.astype(np.int64))


def _frame(rows: list[tuple[str, str]], field: str = NAME_NORM) -> pd.DataFrame:
    """A prepared-shaped chunk: entity_id + one normalized name column."""
    return pd.DataFrame({"entity_id": [row[0] for row in rows], field: [row[1] for row in rows]})


def _codes_by_entity(
    rows: list[tuple[str, str]], field: str = NAME_NORM
) -> dict[int, str]:
    """``entity code -> name``, because postings carry codes, not row indices."""
    from src.utils import encode_entity_ids

    codes = encode_entity_ids(pd.Series([row[0] for row in rows]))
    return {int(code): row[1] for code, row in zip(codes, rows)}


def _code_for(rows: list[tuple[str, str]], entity_id: str, field: str = NAME_NORM) -> int:
    from src.utils import encode_entity_ids

    codes = encode_entity_ids(pd.Series([row[0] for row in rows]))
    for code, row in zip(codes, rows):
        if row[0] == entity_id:
            return int(code)
    raise KeyError(entity_id)


def _build_index(
    rows: list[tuple[str, str]],
    vocab: ct._TokenVocabulary,
    df: ct._TrigramDf,
    cap: int,
    k: int,
    source: str = "source2",
) -> ct._TrigramIndex:
    return ct.build_token_index(
        iter([_frame(rows)]),
        df,
        vocab,
        "entity_id",
        NAME_NORM,
        cap,
        k,
        source,
        logging.getLogger("test"),
        "test",
    )


class _Truth:
    """The two attributes ``build_token_s1_selection`` needs from the ground truth."""

    def __init__(self, positions: dict[str, int]) -> None:
        self._positions = positions

    def positions_of(self, entity_ids) -> np.ndarray:
        return np.array(
            [self._positions.get(str(entity_id), -1) for entity_id in entity_ids], dtype=np.int64
        )


def _truth_for(rows: list[tuple[str, str]]) -> _Truth:
    """One ground-truth position per row, in row order."""
    return _Truth({row[0]: index for index, row in enumerate(rows)})


def _s1_selection(
    rows: list[tuple[str, str]],
    vocab: ct._TokenVocabulary,
    df: ct._TrigramDf,
    cap: int,
    k: int,
    truth: _Truth,
) -> tuple[ct._S1Selection, dict]:
    return ct.build_token_s1_selection(
        iter([_frame(rows)]),
        df,
        vocab,
        truth,
        "entity_id",
        NAME_NORM,
        cap,
        k,
        len(rows),
        logging.getLogger("test"),
        "test",
    )


def _pack(*pairs: tuple[int, int]) -> np.ndarray:
    """A packed, sorted pair array from explicit (position, entity code) tuples."""
    if not pairs:
        return ct._EMPTY_INT64
    positions, codes = zip(*pairs)
    return np.unique(
        pack_pairs(np.asarray(positions, dtype=np.int64), np.asarray(codes, dtype=np.int64))
    ).astype(np.int64)


# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------
def test_tokenize_is_phase_0_s_token_set():
    """``tokenize`` must be ``text.split()`` and nothing else.

    Phase 0.2-0.4's analytic reference was measured under
    ``analyze_blocking_statistics._token_set`` = ``set(text.split())``. Any other
    splitting rule - punctuation, case folding, a regex - would make this run's
    numbers incomparable with the reference it exists to cross-check, so the rule is
    pinned here rather than left implicit at the call sites.
    """
    assert ct.tokenize("Sunrise Traders") == ["Sunrise", "Traders"]
    # Whitespace runs collapse, and leading/trailing whitespace yields no empty token.
    assert ct.tokenize("  acme   industries  ") == ["acme", "industries"]
    assert ct.tokenize("") == []
    assert ct.tokenize("   ") == []
    # Case and punctuation are NOT touched: normalization did that already, and
    # re-doing it here would diverge from Phase 0.
    assert ct.tokenize("Acme, Inc.") == ["Acme,", "Inc."]
    # A tab or a newline separates, because str.split() with no argument does.
    assert ct.tokenize("a\tb\nc") == ["a", "b", "c"]
    # Unicode whitespace too, which is what makes this correct on non-latin names.
    assert ct.tokenize("राम मार्केटिंग") == ["राम", "मार्केटिंग"]


def test_token_vocabulary_codes_are_dense_and_assigned_on_first_sight():
    """Codes must be 0..n-1 in order of first appearance, not hashes.

    Denseness is not cosmetic: it is why a lookup can be one ``searchsorted`` with no
    string re-verification, and therefore why the char blocker's ``_TrigramIndex`` can
    be reused for tokens unchanged.
    """
    vocab = _make_vocab(["acme industries", "acme trading"])
    assert vocab.code_of("acme") == 0, "the first token seen must take code 0"
    assert vocab.code_of("industries") == 1
    assert vocab.code_of("trading") == 2
    assert len(vocab) == 3
    assert vocab.tokens == ["acme", "industries", "trading"]
    # Unknown tokens are reported, never interned by a read-only lookup.
    assert vocab.code_of("missing") == -1


def test_codes_for_text_is_deduplicated_and_sorted():
    """A name is a *set* of tokens: that is what makes the count a document frequency."""
    vocab = _make_vocab([])
    codes = vocab.codes_for_text("acme acme industries acme")
    assert codes.tolist() == [0, 1], "repeats must collapse to one code each"
    assert np.all(np.diff(codes) > 0), "codes must be strictly ascending"
    assert vocab.codes_for_text("").size == 0
    assert vocab.codes_for_text("").dtype == np.int64


def test_codes_for_texts_returns_codes_and_owners():
    """Same contract as the trigram encoder, so downstream code needs no change."""
    vocab = _make_vocab([])
    codes, owners = vocab.codes_for_texts(["acme industries", "", "acme"])
    assert codes.tolist() == [0, 1, 0], "row 1 contributes nothing; row 2 reuses code 0"
    assert owners.tolist() == [0, 0, 2]
    assert ct._TokenVocabulary("s").codes_for_texts([""])[0].size == 0


def test_lookup_texts_keeps_unknown_tokens_as_minus_one():
    """An S1 token this source never saw must stay countable, not vanish.

    The char blocker counts the same state as ``df == 0``; if this dropped the token
    instead, ``tokens_dropped_absent_from_this_source`` could never be measured and a
    structural zero would look like a threshold that is too tight.
    """
    vocab = _make_vocab(["acme industries"])
    codes, owners, empty = vocab.lookup_texts(["acme foreign", "", "acme"])
    assert empty == 1, "one name had no token at all"
    assert codes.tolist() == [-1, 0, 0], "the unknown token must be present as -1"
    assert owners.tolist() == [0, 0, 2]
    assert len(vocab) == 2, "a read-only lookup must not grow the vocabulary"


def test_absent_tokens_are_counted_distinctly_not_collapsed():
    """Two unknown tokens are two drops, so each absent token needs its own code.

    Sharing one sentinel across every unknown token looks harmless and is not: the
    per-row dedup would then merge ``private`` and ``limited`` into a single entry and
    under-report ``tokens_dropped_absent_from_this_source`` against the identical
    ``name_norm``, which is precisely the diagnostic that decides whether a tight cap
    is a threshold problem or a corpus problem. Distinctness is asserted through the
    count the caller actually reads, not through the sentinel values themselves.
    """
    vocab = _make_vocab(["acme industries"])
    codes, _, _ = vocab.lookup_texts(["acme private limited"])
    assert codes.size == 3, "acme plus two distinct absent tokens"
    assert codes.tolist() == [-2, -1, 0], "absent codes stay below every real code"
    df = _make_df(["acme"], vocab)
    assert df.lookup(codes).tolist() == [0, 0, 1], "every absent code must score df 0"


def test_unknown_token_scores_df_zero_through_the_reused_table():
    """The ``-1`` convention has to be *scored* as df 0 by the reused df table.

    This is the join between the two conventions. ``_TrigramDf.lookup`` clips its
    ``searchsorted`` position and then compares, so ``-1`` lands on position 0 and is
    rejected unless code 0 really is -1 - which it cannot be. Asserted rather than
    assumed, because a table that returned a real df for -1 would silently credit
    absent tokens and inflate every reach number.
    """
    vocab = _make_vocab(["acme industries"])
    df = _make_df(["acme industries"], vocab)
    assert df.lookup(np.array([-1], dtype=np.int64))[0] == 0
    assert df.lookup(np.array([-1, 0], dtype=np.int64)).tolist() == [0, 1]
    # And it must hold when code 0 is the only row in the table, too.
    single = ct._TrigramDf(np.array([0], dtype=np.int64), np.array([7], dtype=np.int64))
    assert single.lookup(np.array([-1], dtype=np.int64))[0] == 0


def test_vocabulary_round_trips_through_disk_with_identical_codes():
    """``--resume`` re-derives codes from disk, so the numbering must survive exactly."""
    vocab = _make_vocab(["acme industries private", "meridian logistics"])
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "vocab"
        vocab.save(directory)
        restored = ct._TokenVocabulary.load(directory, "source2")
    assert restored.tokens == vocab.tokens, "token order must be preserved"
    assert len(restored) == len(vocab)
    for token in vocab.tokens:
        assert restored.code_of(token) == vocab.code_of(token)
    assert restored.code_of("absent") == -1


def test_vocabulary_round_trips_when_empty():
    """An all-capped-out corpus must not crash the persistence path."""
    vocab = ct._TokenVocabulary("source3")
    empty_df = ct._TrigramDf(ct._EMPTY_INT64, ct._EMPTY_INT64)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "vocab"
        vocab.save(directory)
        restored = ct._TokenVocabulary.load(directory, "source3")
    assert len(restored) == 0
    assert restored.top(empty_df, 5) == []
    assert restored.code_of("anything") == -1
    # ``codes_for_text`` is the write path, so it mints code 0 for a token the empty
    # corpus never saw. Pinned because the read-only lookup path must not behave this
    # way, and the two are easy to confuse.
    assert restored.codes_for_text("anything").tolist() == [0]
    assert restored.code_of("anything") == 0


def test_vocabulary_load_rejects_a_truncated_blob():
    """A half-written vocabulary must fail loudly, not renumber the corpus.

    The blob is cut mid-token rather than between tokens, so its NUL-separated field
    count is still one short instead of obviously wrong: ``acme\\x00industries`` becomes
    ``acme\\x00indus``. Only the recorded byte count catches that, which is why the
    meta carries one.
    """
    vocab = _make_vocab(["acme industries"])
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "vocab"
        vocab.save(directory)
        (directory / "vocab.bin").write_bytes(b"acme\x00indus")
        try:
            ct._TokenVocabulary.load(directory, "source2")
        except ValueError as exc:
            assert "vocabulary blob" in str(exc), f"unexpected message: {exc}"
        else:
            raise AssertionError("a truncated vocabulary blob was accepted")


def test_vocabulary_load_rejects_a_token_count_mismatch():
    """The field count is checked too, for a blob damaged without changing its length."""
    vocab = _make_vocab(["acme industries"])
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "vocab"
        vocab.save(directory)
        meta = json.loads((directory / "vocab_meta.json").read_text(encoding="utf-8"))
        meta["n_tokens"] = 3
        meta["blob_bytes"] = len(b"acme\x00industries")
        (directory / "vocab_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        try:
            ct._TokenVocabulary.load(directory, "source2")
        except ValueError as exc:
            assert "tokens" in str(exc), f"unexpected message: {exc}"
        else:
            raise AssertionError("a token-count mismatch was accepted")


# ---------------------------------------------------------------------------
# document frequency
# ---------------------------------------------------------------------------
def test_count_token_df_counts_documents_not_occurrences():
    """A token repeated inside one name must count once for that document.

    The whole cap is a *document*-frequency cap; counting occurrences would make a
    name that repeats a token look like several documents and drop a token the rest of
    the corpus finds perfectly rare.
    """
    vocab = ct._TokenVocabulary("source2")
    rows = [
        ("S2-1", "acme acme acme industries"),
        ("S2-2", "acme trading"),
        ("S2-3", "meridian logistics"),
    ]
    df = ct.count_token_df(
        iter([_frame(rows)]), NAME_NORM, vocab, logging.getLogger("test"), "test"
    )
    acme, industries, trading = (vocab.code_of(token) for token in ("acme", "industries", "trading"))
    assert df.lookup(np.array([acme]))[0] == 2, "acme is in two documents, not four occurrences"
    assert df.lookup(np.array([industries]))[0] == 1
    assert df.lookup(np.array([trading]))[0] == 1
    assert df.codes.tolist() == sorted(df.codes.tolist()), "codes must be sorted"
    assert len(df) == 5, "acme industries trading meridian logistics = 5 distinct tokens"


def test_count_token_df_is_chunk_invariant():
    """df must not depend on how the corpus was split into chunks.

    ``--chunk-rows`` is a memory knob, and a reader comparing two runs at different
    chunk sizes must get the same cap behaviour.
    """
    rows = [("S2-1", "acme industries"), ("S2-2", "acme trading"), ("S2-3", "acme holdings")]
    one = ct._TokenVocabulary("s")
    three = ct._TokenVocabulary("s")
    whole = ct.count_token_df(iter([_frame(rows)]), NAME_NORM, one, logging.getLogger("t"), "t")
    split = ct.count_token_df(
        iter([_frame([row]) for row in rows]), NAME_NORM, three, logging.getLogger("t"), "t"
    )
    assert one.tokens == three.tokens, "chunking must not renumber the vocabulary"
    assert whole.codes.tolist() == split.codes.tolist()
    assert whole.values.tolist() == split.values.tolist()


def test_count_token_df_on_an_empty_corpus_is_empty():
    vocab = ct._TokenVocabulary("s")
    df = ct.count_token_df(iter([]), NAME_NORM, vocab, logging.getLogger("t"), "t")
    assert len(df) == 0
    assert len(vocab) == 0
    assert df.lookup(np.array([0], dtype=np.int64)).tolist() == [0]


def test_describe_df_renames_the_key_so_no_token_count_is_a_trigram_count():
    """The reused structure's key names a trigram; the report must not inherit that."""
    vocab = ct._TokenVocabulary("s")
    df = ct.count_token_df(
        iter([_frame([("S2-1", "acme industries")])]),
        NAME_NORM,
        vocab,
        logging.getLogger("t"),
        "t",
    )
    described = ct.describe_df(df, [1, 10])
    assert "n_distinct_tokens" in described
    assert "n_distinct_trigrams" not in described, "a token count was labelled a trigram count"
    assert described["n_distinct_tokens"] == len(df)
    assert described["survivors_by_cap"] == {"1": df.cap_survivors(1), "10": df.cap_survivors(10)}
    assert ct.describe_df(ct._TrigramDf(ct._EMPTY_INT64, ct._EMPTY_INT64), [1]) == {
        "n_distinct_tokens": 0
    }


# ---------------------------------------------------------------------------
# index build: the rarest-K rule and the posting layout
# ---------------------------------------------------------------------------
def test_index_keeps_the_rarest_k_surviving_tokens_per_entity():
    """K must select the *rarest* tokens under the cap, and nothing else.

    This is the one filter that separates the token blocker from Phase 0.2's analytic
    rule, so it is checked against a hand-computed df ordering rather than against the
    builder's own output.
    """
    rows = [
        ("S2-1", "acme industries private"),
        ("S2-2", "acme trading private"),
        ("S2-3", "acme holdings private"),
    ]
    vocab = ct._TokenVocabulary("source2")
    df = ct.count_token_df(iter([_frame(rows)]), NAME_NORM, vocab, logging.getLogger("t"), "t")
    assert df.lookup(np.array([vocab.code_of("acme")]))[0] == 3
    assert df.lookup(np.array([vocab.code_of("private")]))[0] == 3
    assert df.lookup(np.array([vocab.code_of("trading")]))[0] == 1

    index = _build_index(rows, vocab, df, cap=10_000, k=1)
    owner = _code_for(rows, "S2-1")
    kept = {
        int(index.keys[position])
        for position in range(index.n_keys)
        if owner
        in index.postings[
            index.postings_offsets[position] : index.postings_offsets[position + 1]
        ].tolist()
    }
    # S2-1's three tokens all tie on df except none do: industries=1, trading=1,
    # holdings=1, acme=3, private=3. So the rarest is a df-1 token, picked by code.
    kept_tokens = [vocab.tokens[code] for code in kept]
    assert len(kept) == 1, f"K=1 must keep exactly one token, kept {kept_tokens}"
    assert df.lookup(np.array(sorted(kept), dtype=np.int64))[0] == 1, "K=1 kept a common token"
    assert "acme" not in kept_tokens and "private" not in kept_tokens


def test_index_ties_break_on_code_so_the_selection_is_reproducible():
    """Equal df must order deterministically, or two runs could keep different keys."""
    rows = [("S2-1", "alpha beta"), ("S2-2", "gamma delta")]
    vocab = ct._TokenVocabulary("source2")
    df = ct.count_token_df(iter([_frame(rows)]), NAME_NORM, vocab, logging.getLogger("t"), "t")
    codes = vocab.codes_for_text("alpha beta")
    dfs = df.lookup(codes)
    assert dfs.tolist() == [1, 1], "the fixture is meant to be a df tie"
    expected = int(codes[np.lexsort((codes, dfs))[:1]][0])
    index = _build_index(rows, vocab, df, cap=10_000, k=1)
    owner = _code_for(rows, "S2-1")
    assert owner in index.postings.tolist(), "S2-1 contributed no key"
    owned = {
        int(index.keys[position])
        for position in range(index.n_keys)
        if owner
        in index.postings[
            index.postings_offsets[position] : index.postings_offsets[position + 1]
        ].tolist()
    }
    assert owned == {expected}, "the tie must break on the lower code"


def test_index_postings_are_rank_ascending_within_a_key():
    """The property the whole ``(cap, K)`` grid filter depends on.

    A rank filter is only a posting *prefix* if each key's postings are rank-ascending;
    otherwise ``filtered()`` would silently return the wrong subset and every cell but
    the loosest would report the wrong recall.
    """
    rows = [
        ("S2-1", "acme industries private"),
        ("S2-2", "acme trading private"),
        ("S2-3", "acme holdings private"),
    ]
    vocab = ct._TokenVocabulary("source2")
    df = ct.count_token_df(iter([_frame(rows)]), NAME_NORM, vocab, logging.getLogger("t"), "t")
    index = _build_index(rows, vocab, df, cap=10_000, k=3)
    assert np.all(np.diff(index.keys) > 0), "keys must be sorted and unique"
    assert index.postings_offsets[-1] == index.n_postings
    assert len(index.postings_offsets) == index.n_keys + 1
    by_code = _codes_by_entity(rows)
    for position in range(index.n_keys):
        start = int(index.postings_offsets[position])
        stop = int(index.postings_offsets[position + 1])
        ranks = index.ranks[start:stop].astype(np.int64)
        assert np.all(np.diff(ranks) >= 0), "postings must be rank-ascending within a key"
        token = vocab.tokens[int(index.keys[position])]
        for code in index.postings[start:stop]:
            assert token in ct.tokenize(by_code[int(code)]), f"{token!r} posted to a row lacking it"


def test_index_is_empty_when_the_cap_removes_every_token():
    """A cap below every df must yield an empty index, not an exception."""
    rows = [("S2-1", "acme industries"), ("S2-2", "acme trading")]
    vocab = ct._TokenVocabulary("source2")
    df = ct.count_token_df(iter([_frame(rows)]), NAME_NORM, vocab, logging.getLogger("t"), "t")
    index = _build_index(rows, vocab, df, cap=0, k=1)
    assert index.n_keys == 0 and index.n_postings == 0
    assert index.filtered(0, 1).n_keys == 0, "filtering an empty index must stay empty"
    assert ct.retrieve_token_chunk(
        index,
        ct._S1Selection(
            ct._EMPTY_INT64,
            np.empty(0, dtype=np.uint8),
            ct._EMPTY_INT64,
            np.zeros(3, dtype=np.int64),
            2,
        ),
        0,
        2,
        0,
        1,
    ).size == 0


def test_filtered_cell_equals_a_directly_built_cell_for_tokens():
    """The shared-index proof, re-run on token codes rather than trigram codes.

    The index is built once at the loosest grid setting and every cell is a filter of
    it; if that were not exact, the calibration would be measuring the filter.
    """
    rows = [
        ("S2-1", "acme industries private"),
        ("S2-2", "acme trading private"),
        ("S2-3", "meridian logistics"),
        ("S2-4", "acme holdings"),
        ("S2-5", "freightways"),
    ]
    vocab = ct._TokenVocabulary("source2")
    df = ct.count_token_df(iter([_frame(rows)]), NAME_NORM, vocab, logging.getLogger("t"), "t")
    shared = _build_index(rows, vocab, df, cap=10_000, k=5)
    for cap in (1, 2, 3, 10_000):
        for k in (1, 2, 3, 5):
            filtered = shared.filtered(cap, k)
            direct = _build_index(rows, vocab, df, cap=cap, k=k)
            assert np.array_equal(filtered.keys, direct.keys), f"keys differ at (cap={cap}, K={k})"
            assert np.array_equal(filtered.postings_offsets, direct.postings_offsets)
            assert np.array_equal(filtered.postings, direct.postings)
            assert np.array_equal(filtered.ranks, direct.ranks)
            kept = shared.kept_counts_per_key(cap, k)
            assert int(kept[shared.key_df <= cap].sum()) == direct.n_postings


# ---------------------------------------------------------------------------
# S1 selection: drop accounting
# ---------------------------------------------------------------------------
def test_s1_selection_ranks_by_df_then_code_within_a_position():
    rows = [("S1-1", "acme industries private"), ("S1-2", "meridian logistics")]
    vocab = ct._TokenVocabulary("source2")
    df = _make_df(["acme industries private", "meridian logistics"], vocab)
    truth = _truth_for(rows)
    selection, breakdown = _s1_selection(rows, vocab, df, 10_000, 10, truth)
    assert selection.n_rows == 2
    codes, positions, ranks, dfs = selection.slice_full(0, 2)
    for position in np.unique(positions):
        rows_here = positions == position
        assert ranks[rows_here].tolist() == list(range(int(rows_here.sum())))
        assert np.all(np.diff(dfs[rows_here].astype(np.int64)) >= 0), "df must ascend within a row"
    assert breakdown["n_s1_positions"] == 2
    assert breakdown["s1_unknown_to_ground_truth"] == 0
    assert breakdown["entries_per_row"] == 2.5


def test_s1_selection_counts_each_drop_reason_separately():
    """The three ways a key can be lost must be counted apart, not lumped together.

    ``above cap`` is the df filter and ``beyond K`` the rank filter; separating them is
    what shows which one binds, and both are different from a token this source's
    corpus never contained. A single "dropped" number would make a fix
    unattributable.
    """
    rows = [("S1-1", "acme industries private limited")]
    vocab = ct._TokenVocabulary("source2")
    # "acme" and "industries" are in two documents; "private" in one; "limited" in none.
    df = _make_df(["acme industries", "acme industries"], vocab)
    truth = _truth_for(rows)

    _, breakdown = _s1_selection(rows, vocab, df, cap=1, k=10, truth=truth)
    assert breakdown["tokens_dropped_above_df_cap"] == 2, "acme and industries are both df 2"
    assert breakdown["tokens_dropped_absent_from_this_source"] == 2, (
        "private and limited are unknown to this corpus"
    )
    assert breakdown["tokens_dropped_beyond_rarest_k"] == 0, "K=10 truncates nothing"

    _, tight = _s1_selection(rows, vocab, df, cap=10_000, k=1, truth=truth)
    assert tight["tokens_dropped_above_df_cap"] == 0, "the cap is loose here"
    assert tight["tokens_dropped_absent_from_this_source"] == 2, "the -1 entries are counted too"
    assert tight["tokens_dropped_beyond_rarest_k"] == 1, "K=1 keeps one of the two survivors"

    _, capped = _s1_selection(rows, vocab, df, cap=0, k=1, truth=truth)
    assert capped["s1_with_no_key_in_this_source"] == 1, "every key was dropped"


def test_s1_selection_counts_names_with_no_token_at_all():
    """An empty name is a different failure from a capped-out one."""
    rows = [("S1-1", ""), ("S1-2", "acme")]
    vocab = ct._TokenVocabulary("source2")
    df = _make_df(["acme"], vocab)
    _, breakdown = _s1_selection(rows, vocab, df, 10_000, 10, _truth_for(rows))
    assert breakdown["s1_name_with_no_token"] == 1
    assert breakdown["tokens_dropped_absent_from_this_source"] == 0
    assert breakdown["s1_with_no_key_in_this_source"] == 1, (
        "S1-2 kept its one token; the empty name is the row with no key"
    )


def test_s1_selection_counts_rows_unknown_to_the_ground_truth():
    """A row the ground truth cannot place has no position to accumulate into."""
    rows = [("S1-1", "acme"), ("S1-9", "acme")]
    truth = _Truth({"S1-1": 0})
    vocab = ct._TokenVocabulary("source2")
    df = _make_df(["acme"], vocab)
    selection, breakdown = _s1_selection(rows, vocab, df, 10_000, 10, truth)
    assert breakdown["s1_unknown_to_ground_truth"] == 1
    assert breakdown["n_s1_positions"] == 2, "the position space is the caller's, not the rows'"
    assert int(selection.offsets[-1]) == 1, "only the known row contributed an entry"


def test_s1_selection_cell_filter_keeps_the_documented_prefix():
    """``cell_filter`` must be exactly ``df <= cap`` and ``rank < K``."""
    ranks = np.array([0, 1, 2, 0, 1], dtype=np.uint8)
    dfs = np.array([5, 5, 5, 50, 50], dtype=np.int64)
    keep = ct._S1Selection.cell_filter(ranks, dfs, cap=5, rarest_k=2)
    assert keep.tolist() == [True, True, False, False, False]
    assert ct._S1Selection.cell_filter(ranks, dfs, cap=100, rarest_k=1).tolist() == [
        True,
        False,
        False,
        True,
        False,
    ]


# ---------------------------------------------------------------------------
# retrieval and the packed unions
# ---------------------------------------------------------------------------
def test_retrieve_returns_the_pairs_sharing_an_eligible_token():
    """One cell's retrieval, against a hand-worked expectation."""
    rows = [("S1-1", "acme industries"), ("S1-2", "meridian logistics"), ("S1-3", "acme")]
    targets = [("S2-1", "acme trading"), ("S2-2", "meridian logistics")]
    vocab = ct._TokenVocabulary("source2")
    df = _make_df([text for _, text in targets] + [text for _, text in rows], vocab)
    index = _build_index(targets, vocab, df, 10_000, 10)
    truth = _truth_for(rows)
    selection, _ = _s1_selection(rows, vocab, df, 10_000, 10, truth)

    packed = ct.retrieve_token_chunk(index, selection, 0, len(rows), 10_000, 10)
    found = set(zip(*[part.tolist() for part in unpack_pairs(packed)]))
    assert found == {(0, _code_for(targets, "S2-1")), (1, _code_for(targets, "S2-2")), (2, _code_for(targets, "S2-1"))}
    # ...and it is sorted, which is what makes a chunk slice two searchsorted calls.
    assert np.all(np.diff(packed) > 0)


def test_retrieve_deduplicates_pairs_reachable_through_several_tokens():
    """Two shared tokens must still yield one candidate pair.

    Without the dedup the volume would be inflated by exactly the key-overlap factor,
    and the precision would be understated - the failure mode Phase 0.2 could only
    bound analytically.
    """
    rows = [("S1-1", "acme industries")]
    targets = [("S2-1", "acme industries")]
    vocab = ct._TokenVocabulary("source2")
    df = _make_df(["acme industries"], vocab)
    index = _build_index(targets, vocab, df, 10_000, 10)
    selection, _ = _s1_selection(rows, vocab, df, 10_000, 10, _truth_for(rows))
    packed = ct.retrieve_token_chunk(index, selection, 0, 1, 10_000, 10)
    assert packed.size == 1, "acme and industries reach the same target; one pair, not two"
    # The bound the sweep reports counts postings, so it is twice as large here - which
    # is exactly why the two volume kinds are labelled differently.
    assert index.kept_counts_per_key(10_000, 10).sum() == 2


def test_retrieve_returns_nothing_when_every_token_is_above_the_cap():
    """A zero-candidate cell must return an empty array, not raise."""
    rows = [("S1-1", "acme industries")]
    targets = [("S2-1", "acme industries")]
    vocab = ct._TokenVocabulary("source2")
    df = _make_df(["acme industries"], vocab)
    index = _build_index(targets, vocab, df, 10_000, 10)
    selection, _ = _s1_selection(rows, vocab, df, 10_000, 10, _truth_for(rows))
    assert ct.retrieve_token_chunk(index, selection, 0, 1, cap=0, rarest_k=10).size == 0
    assert ct.retrieve_token_chunk(index, selection, 0, 1, cap=10_000, rarest_k=0).size == 0


def _pairs_of(packed: np.ndarray) -> set[tuple[int, int]]:
    """A packed pair array as ``{(s1_position, target_code)}``, for readable assertions."""
    if packed.size == 0:
        return set()
    positions, codes = unpack_pairs(packed)
    return set(zip(positions.tolist(), codes.tolist()))


def test_retrieve_respects_the_cell_filter_not_just_the_index():
    """The S1 side is filtered per cell too, so a tight K must narrow retrieval.

    The index is built at the loosest cell; the S1 selection is filtered at query time.
    If the query side ignored its own rank limit, a cell would retrieve every pair whose
    shared token merely survived the cap, which is Phase 0.2's rule rather than this
    blocker's.

    Retrieval is keyed on the *shared* token, so a query row reaches every target row
    that shares one - not just the row it happens to match. ``loose`` therefore holds
    four pairs, two of them wrong, and that is the point: the rank limit is what makes
    the two rows' rarest tokens (``industries``, ``holdings``) narrow it back to the
    diagonal.
    """
    rows = [("S1-1", "acme industries private"), ("S1-2", "acme holdings private")]
    targets = [("S2-1", "acme industries private"), ("S2-2", "acme holdings private")]
    vocab = ct._TokenVocabulary("source2")
    df = _make_df([text for _, text in targets], vocab)
    index = _build_index(targets, vocab, df, 10_000, 10)
    selection, _ = _s1_selection(rows, vocab, df, 10_000, 10, _truth_for(rows))
    # Target codes are the blocking module's source-prefixed entity codes, not row
    # numbers, so read them off the index rather than guessing at the prefix.
    first, second = sorted(set(index.postings.tolist()))
    loose = _pairs_of(ct.retrieve_token_chunk(index, selection, 0, 2, 10_000, 10))
    assert loose == {(0, first), (0, second), (1, first), (1, second)}, (
        "every shared token retrieves its postings"
    )
    # K=1 keeps each row's rarest token, and the two rows' rarest tokens differ, so the
    # limit collapses the cross pairs and leaves the diagonal.
    tight = _pairs_of(ct.retrieve_token_chunk(index, selection, 0, 2, 10_000, 1))
    assert tight == {(0, first), (1, second)}, "the rarest token alone reaches the true pair"
    assert tight <= loose


def test_slice_packed_selects_by_s1_position():
    """A chunk's slice of a grid-free set must be two searchsorted calls, exactly."""
    packed = _pack((0, 2001), (0, 2002), (2, 2003), (5, 3001), (5, 3002))
    assert ct.slice_packed(packed, 0, 1).size == 2
    assert ct.slice_packed(packed, 1, 5).size == 1
    assert ct.slice_packed(packed, 5, 6).size == 2
    assert ct.slice_packed(packed, 6, 9).size == 0
    assert ct.slice_packed(ct._EMPTY_INT64, 0, 10).size == 0
    # Slicing the union of every chunk must reproduce the whole array.
    rejoined = np.concatenate([ct.slice_packed(packed, start, start + 2) for start in (0, 2, 4, 6)])
    assert np.array_equal(rejoined, packed)


def test_union_packed_dedupes_sorts_and_keeps_every_proposer():
    """The union is the only set operation this script performs - never an intersection."""
    exact = _pack((0, 2001), (1, 2001))
    char = _pack((1, 2001), (2, 3001))
    token = _pack((3, 3001))
    union = ct.union_packed(exact, char, token)
    assert union.tolist() == sorted(union.tolist()), "the union must be sorted"
    assert union.size == 4, "the (1, 2001) duplicate must collapse exactly once"
    for proposed in (exact, char, token):
        assert np.all(np.isin(proposed, union)), "a proposer's pair was intersected away"
    assert ct.union_packed().size == 0
    assert ct.union_packed(ct._EMPTY_INT64, exact).tolist() == exact.tolist()
    # And it must not mutate its inputs.
    assert np.array_equal(ct.union_packed(exact, char), np.unique(np.concatenate([exact, char])))


def test_exact_plus_char_union_contains_both_exact_and_char_pairs():
    """``exact+char`` must be built from BOTH sets, not from char alone.

    On the fixture this is impossible to separate end to end - identical ``name_norm``
    implies identical ``name_key`` implies a Jaccard of 1.0, so char structurally
    contains exact there - so the guard lives here, on the operation itself. Building
    the union from char only would report ``exact+char`` as equal to ``char``, which is
    precisely the claim the Step 0 analysis established to be false at scale.
    """
    exact_only = _pack((0, 2001))
    char_only = _pack((1, 3001))
    union = ct.union_packed(exact_only, char_only)
    assert union.tolist() == _pack((0, 2001), (1, 3001)).tolist()
    char_alone = ct.union_packed(char_only)
    assert not np.array_equal(char_alone, union), "char alone is not the union"
    assert exact_only.tolist()[0] not in char_alone.tolist()


def test_union_survives_when_one_side_is_empty():
    """A blocker that retrieves nothing must leave the union equal to the other side."""
    exact = _pack((0, 2001), (3, 3001))
    assert ct.union_packed(exact, ct._EMPTY_INT64).tolist() == exact.tolist()
    assert ct.union_packed(ct._EMPTY_INT64, ct._EMPTY_INT64).size == 0


def test_packed_pairs_round_trip_through_the_reused_codec():
    """The packing convention must be the shipped blocker's, unchanged."""
    positions = np.array([0, 7, 12345], dtype=np.int64)
    codes = np.array([2 * 10**10 + 11, 3 * 10**10 + 12, 2 * 10**10 + 13], dtype=np.int64)
    unpacked_positions, unpacked_codes = unpack_pairs(pack_pairs(positions, codes))
    assert np.array_equal(unpacked_positions, positions)
    assert np.array_equal(unpacked_codes, codes)
    assert ct._EMPTY_INT64.dtype == np.int64


# ---------------------------------------------------------------------------
# volume accounting
# ---------------------------------------------------------------------------
def test_top_token_keys_reports_df_and_postings():
    rows = [("S2-1", "acme industries"), ("S2-2", "acme trading"), ("S2-3", "acme")]
    vocab = ct._TokenVocabulary("source2")
    df = ct.count_token_df(iter([_frame(rows)]), NAME_NORM, vocab, logging.getLogger("t"), "t")
    index = _build_index(rows, vocab, df, 10_000, 10)
    top = ct.top_token_keys(index, vocab, 5)
    assert top and top[0][0] == "acme", f"acme is in all three rows: {top}"
    assert top[0][1] == 3 and top[0][2] == 3, "df 3 and three postings"
    assert ct.top_token_keys(index, vocab, 0) == []
    assert ct.top_token_keys(ct.empty_index("source2"), vocab, 5) == []


def test_sweep_volume_bounds_the_retrieval_and_totals_per_cell():
    """The sweep must price every cell without expanding a single posting, and its
    numbers must be internally consistent.

    The bound is an upper bound, so it can exceed the distinct count (two of an
    entity's keys can reach the same target); it must never be smaller, or the
    candidate-row guard would let an over-budget cell through.
    """
    rows = [("S1-1", "acme industries"), ("S1-2", "meridian logistics")]
    targets = [("S2-1", "acme industries"), ("S2-2", "meridian logistics")]
    vocab = ct._TokenVocabulary("source2")
    df = _make_df([text for _, text in targets], vocab)
    index = _build_index(targets, vocab, df, 10_000, 10)
    selection, _ = _s1_selection(rows, vocab, df, 10_000, 10, _truth_for(rows))
    cells = [(10_000, 10), (0, 1)]
    volume_rows, totals = ct.sweep_volume(
        {"source2": index}, {"source2": selection}, cells, len(rows), 200_000, logging.getLogger("t")
    )
    assert totals[(10_000, 10)] == 4, "two pairs, each reachable through two tokens"
    assert totals[(0, 1)] == 0, "a cap of 0 keeps nothing"
    scopes = {row["scope"] for row in volume_rows}
    assert scopes == {VOLUME_ALL, "S2"}
    for row in volume_rows:
        assert row["kind"] == BOUND_KIND
        assert row["n_candidate_pairs"] >= 0
        assert 0.0 <= row["fraction_s1_with_zero_candidates"] <= 1.0

    # The bound must be at least the true distinct count the retriever produces.
    distinct = ct.retrieve_token_chunk(index, selection, 0, len(rows), 10_000, 10).size
    assert totals[(10_000, 10)] >= distinct, "the bound under-counted the retrieval"


def test_incremental_counts_uses_measured_values_and_the_stated_formulas():
    """``additional = Y - X`` and ``additional / (total - X)``, on numbers I control."""
    n_true = 100
    row = ct._incremental_counts(
        n_true,
        {"true_pairs_retrieved": 30, "blocking_recall_pair": 0.30},
        {"true_pairs_retrieved": 20, "blocking_recall_pair": 0.20},
        {"true_pairs_retrieved": 40, "blocking_recall_pair": 0.40},
        {"true_pairs_retrieved": 25, "blocking_recall_pair": 0.25},
        {"true_pairs_retrieved": 45, "blocking_recall_pair": 0.45},
        {"true_pairs_retrieved": 55, "blocking_recall_pair": 0.55},
    )
    assert row["exact_plus_char_true_pairs"] == 40
    assert row["exact_char_token_true_pairs"] == 55
    assert row["additional_true_pairs_from_token"] == 15, "Y - X, measured"
    assert row["previous_misses"] == 60, "total - X"
    assert row["additional_recall_on_previous_misses"] == 15 / 60
    assert row["additional_recall_points_of_total"] == 15.0
    # The other four recalls must be carried through untouched.
    assert row["exact_true_pairs"] == 30
    assert row["char_true_pairs"] == 20
    assert row["token_true_pairs"] == 25
    assert row["token_plus_exact_true_pairs"] == 45
    assert row["exact_plus_char_recall"] == 0.40
    assert row["exact_char_token_recall"] == 0.55


def test_incremental_counts_handles_a_cell_with_no_misses_left():
    """When exact+char already has everything, the denominator is 0 and must be None.

    Reported as ``None`` rather than 0 or a division error: the honest statement is
    "there was nothing left to recover", not "it recovered 0% of nothing".
    """
    row = ct._incremental_counts(
        10,
        {"true_pairs_retrieved": 10},
        {"true_pairs_retrieved": 0},
        {"true_pairs_retrieved": 10},
        {"true_pairs_retrieved": 0},
        {"true_pairs_retrieved": 0},
        {"true_pairs_retrieved": 10},
    )
    assert row["previous_misses"] == 0
    assert row["additional_true_pairs_from_token"] == 0
    assert row["additional_recall_on_previous_misses"] is None
    assert row["additional_recall_points_of_total"] == 0.0


def test_incremental_counts_never_goes_negative_with_a_smaller_union():
    """A union cannot lose a pair, so a negative delta would mean a wiring bug."""
    row = ct._incremental_counts(
        10,
        {"true_pairs_retrieved": 10},
        {"true_pairs_retrieved": 10},
        {"true_pairs_retrieved": 10},
        {"true_pairs_retrieved": 10},
        {"true_pairs_retrieved": 10},
        {"true_pairs_retrieved": 10},
    )
    assert row["additional_true_pairs_from_token"] == 0
    assert row["additional_recall_on_previous_misses"] is None


# ---------------------------------------------------------------------------
# fixture end-to-end
# ---------------------------------------------------------------------------
def _write_tsv(path: Path, rows, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def _build_fixture(base: Path) -> Path:
    data = base / "data"
    work = base / "work"
    _write_tsv(data / "train_source1.tsv", S1, HEADER)
    _write_tsv(data / "train_source2.tsv", S2, HEADER)
    _write_tsv(data / "train_source3.tsv", S3, HEADER)
    _write_tsv(
        data / "train_ground_truth.tsv",
        GROUND_TRUTH,
        ["source1_entity_id", "matched_entity_ids"],
    )

    source = (REPO / "configs" / "config.yaml").read_text(encoding="utf-8")
    replacements = {
        '  data_root: "train"': f'  data_root: "{data.as_posix()}"',
        '  work_dir: "outputs"': f'  work_dir: "{work.as_posix()}"',
        '  prepared_dir: "outputs/prepared"': f'  prepared_dir: "{(work / "prepared").as_posix()}"',
        '  index_dir: "outputs/indexes"': f'  index_dir: "{(work / "indexes").as_posix()}"',
        '  log_dir: "logs"': f'  log_dir: "{(work / "logs").as_posix()}"',
        "  chunksize: 500000": "  chunksize: 3",  # force multi-chunk streaming
    }
    for old, new in replacements.items():
        assert old in source, f"config line not found: {old!r}"
        source = source.replace(old, new)
    assert not re.search(
        r'^\s*(work_dir|prepared_dir|index_dir|log_dir): "(?![/A-Za-z]:)', source, re.M
    )

    config_path = base / "config_token_calibration_fixture.yaml"
    config_path.write_text(source, encoding="utf-8")
    return config_path


def _prepare(config_path: Path) -> None:
    code = prepare_data.main(
        [
            "--config", str(config_path),
            "--splits", "train",
            "--sources", "source1,source2,source3",
            "--overwrite",
        ]
    )
    assert code == 0


def _fixture() -> tuple[Path, Path]:
    """(config_path, root) for the shared fixture, built once per process."""
    if "root" not in _CACHE:
        root = Path(tempfile.mkdtemp(prefix="er_token_calibration_"))
        config_path = _build_fixture(root)
        _prepare(config_path)
        config = load_config(str(config_path))
        for source in ("source2", "source3"):
            build_index(config, "train", source, BLOCKER_EXACT_NAME, overwrite=True)
        # The token blocker keys on name_norm, so assert the prepared tables carry it -
        # the whole script is silently empty without it.
        frame = pd.read_csv(
            Path(config["resolved"]["prepared_dir"]) / "train_source2_norm.tsv",
            sep="\t",
            encoding="utf-8",
        )
        assert NAME_NORM in frame.columns, "prepare_data did not write name_norm"
        assert NAME_KEY in frame.columns
        _CACHE["root"] = root
        _CACHE["config_path"] = config_path
        if not os.environ.get("ER_TEST_KEEP_FIXTURE"):
            atexit.register(shutil.rmtree, root, ignore_errors=True)
    return _CACHE["config_path"], _CACHE["root"]  # type: ignore[return-value]


def _grid_params() -> list[str]:
    return [
        "--df-caps", ",".join(str(cap) for cap in _GRID["df_caps"]),
        "--rarest-ks", ",".join(str(k) for k in _GRID["rarest_ks"]),
        "--char-jaccards", ",".join(str(t) for t in _GRID["char_jaccards"]),
    ]


def _run(config_path: Path, output_dir: Path, extra: list[str]) -> dict:
    """Run the calibration script on the fixture and return its JSON report."""
    argv = [
        "--config", str(config_path),
        "--output-dir", str(output_dir),
        "--chunk-rows", "3",
        "--verify-chunk-pairs", "2",
        "--workers", "1",
        "--top-tokens", "5",
        "--log-level", "WARNING",
        *_grid_params(),
        *extra,
    ]
    code = ct.main(argv)
    assert code == 0, f"token calibration exited {code}"
    return read_json(output_dir / "token_blocker_calibration.json")


def _report() -> tuple[dict, Path]:
    if "report" not in _CACHE:
        config_path, root = _fixture()
        output_dir = root / "calibration"
        _CACHE["report"] = _run(config_path, output_dir, [])
        _CACHE["output_dir"] = output_dir
    return _CACHE["report"], _CACHE["output_dir"]  # type: ignore[return-value]


def _variant_dir(key: str) -> Path:
    return _fixture()[1] / f"calibration_{key}"


def _report_variant(key: str, extra: list[str]) -> dict:
    if key not in _CACHE:
        config_path, _ = _fixture()
        _CACHE[key] = _run(config_path, _variant_dir(key), extra)
    return _CACHE[key]  # type: ignore[return-value]


_VOLATILE_META = (
    "generated_at",
    "elapsed_minutes",
    "timings_seconds",
    "hardware",
    "workers",
    "verify_window",
    "verify_chunk_pairs",
    "chunk_rows",
    "resume_requested",
    "artifact_fingerprint",
    "artifact_root",
    "volume_only",
)


def _strip_volatile(report: dict) -> dict:
    """Keep the results, drop the fields that legitimately differ between equal runs."""
    meta = {key: value for key, value in report["meta"].items() if key not in _VOLATILE_META}
    grid = [
        {key: value for key, value in cell.items() if key != "elapsed_seconds"}
        for cell in report["grid"]
    ]
    # Two equal runs into different --output-dir values record different absolute
    # artifact paths; that is provenance, not a result.
    artifacts = {
        key: value
        for key, value in report["artifacts"].items()
        if key != "target_namekey_dirs"
    }
    return {**report, "meta": meta, "grid": grid, "artifacts": artifacts}


def _entry(report: dict, cell: tuple[int, int], name: str, threshold=None) -> dict:
    """One (cell, set, threshold) metrics block, or a clear failure if it is absent."""
    for candidate in report["grid"]:
        if (candidate["df_cap"], candidate["rarest_k"]) != cell:
            continue
        for entry in candidate["sets"]:
            if entry["set"] == name and entry["threshold"] == threshold:
                return entry
    raise AssertionError(f"no entry for cell={cell} set={name} threshold={threshold}")


def test_fixture_matches_blocking_statistics_fixture():
    """The copied fixture must still be the other test's fixture, row for row.

    Compared through the AST rather than as text, so quoting or layout in the other
    file cannot make this pass vacuously.
    """
    for other_path in ("test_blocking_statistics.py", "test_char_blocker_calibration.py"):
        tree = ast.parse((REPO / "tests" / other_path).read_text(encoding="utf-8"))
        other: dict[str, object] = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                try:
                    other[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue  # not a literal (a call, a subscript, ...) - not a fixture
        for name, rows in (("S1", S1), ("S2", S2), ("S3", S3), ("GROUND_TRUTH", GROUND_TRUTH)):
            assert name in other, f"{name} is not defined in {other_path}"
            assert [tuple(row) for row in other[name]] == rows, f"{name} has drifted from {other_path}"


def test_end_to_end_report_shape():
    report, output_dir = _report()
    meta = report["meta"]
    assert meta["n_s1_entities"] == len(S1) == 14
    assert meta["n_true_pairs"] == 11
    assert meta["df_caps"] == _GRID["df_caps"]
    assert meta["rarest_ks"] == _GRID["rarest_ks"]
    assert meta["char_jaccards"] == _GRID["char_jaccards"]
    assert meta["grid_cells"] == 4, "2 df caps x 2 rarest-K"
    assert meta["cells_evaluated"] == 4
    assert meta["cells_skipped"] == 0
    assert meta["exact_index_available"] == {"source2": True, "source3": True}
    assert "no verification stage" in meta["verification"] or "none for token" in meta["verification"]
    assert meta["token_semantics"].startswith("name_norm")
    # The Phase 0 reference must be carried as a labelled reference, never as a result.
    assert report["token_analytic_reference"]["measured_here"] is False
    assert report["token_analytic_reference"]["pct_of_true_pairs"] == 42.3905
    assert set(report["structural"]) == {"source2", "source3"}
    assert report["incremental"], "the headline table is missing"
    assert report["exact"]["true_pairs_retrieved"] > 0, "the exact blocker found nothing"
    for name in (
        "token_blocker_calibration.json",
        "token_blocker_calibration.csv",
        "token_blocker_volume.csv",
        "token_blocker_top_tokens.csv",
        "token_blocker_incremental.csv",
        "token_blocker_calibration.md",
    ):
        assert (output_dir / name).is_file(), f"{name} was not written"
    text = (output_dir / "token_blocker_calibration.md").read_text(encoding="utf-8")
    assert "Phase 0 analytic reference (NOT measured here)" in text
    assert "## Incremental contribution" in text


def test_the_report_separates_measured_from_the_analytic_reference():
    """The 42.3905% figure must be presented as an analytic reference, never as retrieval.

    Phase 0.2-0.4 priced the token blocker without running it. Reproducing that number
    as if it were measured here is the one reporting error this script must not make.
    """
    _, output_dir = _report()
    text = (output_dir / "token_blocker_calibration.md").read_text(encoding="utf-8")
    assert "Phase 0 analytic reference (NOT measured here)" in text
    assert "42.3905" in text
    # The reference block must come after the measured tables, and say what it is.
    assert text.index("## Incremental contribution") < text.index("NOT measured here")
    assert "not produced by running a blocker" in text
    assert "measured_here" in str(_report()[0]["token_analytic_reference"])


def test_every_cell_reports_the_requested_metrics():
    report, _ = _report()
    required = {
        "blocking_recall_pair",
        "macro_recall_entity",
        "s1_full_recall_rate",
        "s1_partial_recall_rate",
        "n_candidate_pairs",
        "avg_candidates_per_s1",
        "median_candidates_per_s1",
        "p90_candidates_per_s1",
        "p99_candidates_per_s1",
        "max_candidates_per_s1",
        "n_s1_with_zero_candidates",
        "fraction_s1_with_zero_candidates",
        "candidate_precision",
        "reduction_ratio",
        "f05_ceiling_from_macro_recall",
    }
    report_sets = set()
    for cell in report["grid"]:
        assert cell["evaluated"], "the fixture is tiny; every cell must be evaluated"
        assert cell["sets"], "a cell reported no candidate sets"
        for entry in cell["sets"]:
            report_sets.add(entry["set"])
            metrics = entry["metrics"]
            missing = required - set(metrics)
            assert not missing, f"cell {cell['df_cap']}/{cell['rarest_k']} missing {missing}"
            assert set(metrics["per_source"]) == {"S2", "S3"}
            for source_metrics in metrics["per_source"].values():
                for key in ("n_true_pairs", "n_candidates", "true_pairs_retrieved"):
                    assert key in source_metrics, f"per-source metric {key} missing"
            assert 0.0 <= metrics["candidate_precision"] <= 1.0
            assert 0.0 <= metrics["blocking_recall_pair"] <= 1.0
            assert 0.0 <= metrics["macro_recall_entity"] <= 1.0
            assert 0 <= metrics["n_s1_with_zero_candidates"] <= metrics["n_s1_entities"]
            assert metrics["n_candidate_pairs"] >= 0
            assert metrics["reduction_ratio"] > 0
            assert entry["elapsed_seconds"] >= 0
            assert "reach" in metrics
    assert report_sets == set(ct.PER_CELL_SETS), f"unexpected sets reported: {report_sets}"
    # The grid-free sets must be measured exactly once, not per cell.
    assert set(report["char"]) == {f"{threshold:g}" for threshold in _GRID["char_jaccards"]}
    assert set(report["exact_plus_char"]) == set(report["char"])


def test_all_candidate_sets_are_unions_and_never_shrink():
    """Monotonicity: adding a blocker can only add candidates and true pairs.

    This is the structural claim that makes the incremental block meaningful. If a
    union ever retrieved *fewer* true pairs than one of its inputs, the sets would be
    intersecting somewhere and every delta would be suspect.
    """
    report, _ = _report()
    n_true = report["meta"]["n_true_pairs"]
    for threshold in _GRID["char_jaccards"]:
        for cell in _GRID["df_caps"]:
            for k in _GRID["rarest_ks"]:
                token = _entry(report, (cell, k), ct.SET_TOKEN)["metrics"]
                token_exact = _entry(report, (cell, k), ct.SET_TOKEN_PLUS_EXACT)["metrics"]
                everything = _entry(report, (cell, k), ct.SET_EXACT_CHAR_TOKEN, threshold)["metrics"]
                exact = report["exact"]
                char = report["char"][f"{threshold:g}"]
                both = report["exact_plus_char"][f"{threshold:g}"]

                for name, block in (
                    ("token", token),
                    ("token_plus_exact", token_exact),
                    ("exact_char_token", everything),
                    ("exact", exact),
                    ("char", char),
                    ("exact_plus_char", both),
                ):
                    assert 0 <= block["n_candidate_pairs"], f"{name} has a negative volume"
                    assert 0 <= block["true_pairs_retrieved"] <= n_true

                label = f"cell ({cell}, {k}) J={threshold}"
                assert token_exact["true_pairs_retrieved"] >= exact["true_pairs_retrieved"], label
                assert token_exact["true_pairs_retrieved"] >= token["true_pairs_retrieved"], label
                assert everything["true_pairs_retrieved"] >= both["true_pairs_retrieved"], label
                assert everything["true_pairs_retrieved"] >= token_exact["true_pairs_retrieved"], label
                assert everything["true_pairs_retrieved"] >= char["true_pairs_retrieved"], label

                assert token_exact["n_candidate_pairs"] >= exact["n_candidate_pairs"], label
                assert everything["n_candidate_pairs"] >= both["n_candidate_pairs"], label
                assert everything["n_candidate_pairs"] >= token_exact["n_candidate_pairs"], label


def test_token_adds_true_pairs_beyond_exact_alone():
    """The token blocker must genuinely contribute, not merely re-find the exact pairs.

    On this fixture the token blocker has to reach at least one true pair the exact-name
    blocker cannot: the fixture contains a pair whose names differ in ``name_norm``
    ("Acme Industries Private Limited" vs "Acme Private Limited") but share tokens. A
    delta of zero here would mean the token index or the S1 lookup is broken, not that
    the blocker is redundant.
    """
    report, _ = _report()
    exact = report["exact"]["true_pairs_retrieved"]
    token = _entry(report, _LOOSE, ct.SET_TOKEN)["metrics"]["true_pairs_retrieved"]
    token_exact = _entry(report, _LOOSE, ct.SET_TOKEN_PLUS_EXACT)["metrics"]["true_pairs_retrieved"]
    assert token > 0, "the token blocker retrieved nothing at all"
    assert token == 7, f"7 of the 11 fixture pairs share an eligible token; got {token}"
    assert exact == 5, f"the fixture's identical-name pairs number 5; got {exact}"
    assert token_exact > exact, "the token blocker added nothing beyond the exact-name blocker"
    assert token_exact >= token


def test_exact_char_token_is_the_union_of_all_three():
    """The headline set must contain every pair its three inputs contain."""
    report, _ = _report()
    for threshold in _GRID["char_jaccards"]:
        everything = _entry(report, _LOOSE, ct.SET_EXACT_CHAR_TOKEN, threshold)
        token_exact = _entry(report, _LOOSE, ct.SET_TOKEN_PLUS_EXACT)
        both = report["exact_plus_char"][f"{threshold:g}"]
        everything_metrics = everything["metrics"]
        assert everything_metrics["n_candidate_pairs"] >= both["n_candidate_pairs"]
        assert everything_metrics["n_candidate_pairs"] >= token_exact["metrics"]["n_candidate_pairs"]
        assert (
            everything_metrics["true_pairs_retrieved"]
            >= max(both["true_pairs_retrieved"], token_exact["metrics"]["true_pairs_retrieved"])
        )
        # The incremental row for this exact combination must describe it.
        match = [
            row
            for row in report["incremental"]
            if row["df_cap"] == _LOOSE[0]
            and row["rarest_k"] == _LOOSE[1]
            and row["char_jaccard_threshold"] == threshold
        ]
        assert len(match) == 1, f"no incremental row for ({_LOOSE}, J={threshold})"
        assert match[0]["exact_char_token_true_pairs"] == everything_metrics["true_pairs_retrieved"]
        assert match[0]["exact_plus_char_true_pairs"] == both["true_pairs_retrieved"]


def test_incremental_block_is_the_headline_table_and_its_arithmetic_holds():
    """Every incremental row must be exactly the formula the request specified."""
    report, _ = _report()
    n_true = report["meta"]["n_true_pairs"]
    assert len(report["incremental"]) == len(_GRID["df_caps"]) * len(_GRID["rarest_ks"]) * len(
        _GRID["char_jaccards"]
    ), "one incremental row per (cell, threshold)"
    for row in report["incremental"]:
        x = row["exact_plus_char_true_pairs"]
        y = row["exact_char_token_true_pairs"]
        assert row["additional_true_pairs_from_token"] == y - x, "additional must be Y - X"
        assert row["previous_misses"] == n_true - x
        assert row["additional_recall_points_of_total"] == 100.0 * (y - x) / n_true
        if x < n_true:
            assert row["additional_recall_on_previous_misses"] == (y - x) / (n_true - x)
            assert 0.0 <= row["additional_recall_on_previous_misses"] <= 1.0
        else:
            assert row["additional_recall_on_previous_misses"] is None
        assert row["additional_true_pairs_from_token"] >= 0
        # The measured recalls in the row must be the same numbers the grid reports.
        assert row["token_recall"] == _entry(
            report, (row["df_cap"], row["rarest_k"]), ct.SET_TOKEN
        )["metrics"]["blocking_recall_pair"]
        assert row["exact_char_token_recall"] == _entry(
            report,
            (row["df_cap"], row["rarest_k"]),
            ct.SET_EXACT_CHAR_TOKEN,
            row["char_jaccard_threshold"],
        )["metrics"]["blocking_recall_pair"]


def test_token_contribution_is_positive_once_char_is_absent_from_the_union():
    """With the char cell crippled, the token blocker's marginal contribution must show.

    This is the arithmetic of the headline question exercised end to end rather than on
    synthetic counters: a char cell whose df cap keeps almost nothing leaves
    ``exact + char`` short, and the token stage has to close part of that gap. The
    fixture cannot show this at a healthy char cell - identical ``name_norm`` implies a
    Jaccard of exactly 1.0, so char contains every exact-name pair here - which is
    exactly why the crippled variant is run.
    """
    report = _report_variant("char_cap1", ["--char-df-cap", "1", "--char-jaccards", "1.0"])
    assert report["meta"]["char_df_cap"] == 1
    assert report["meta"]["char_rarest_k"] == 1
    assert set(report["char"]) == {"1"}, "the char threshold list must be the one requested"

    both = report["exact_plus_char"]["1"]
    exact = report["exact"]
    char = report["char"]["1"]
    everything = _entry(report, _LOOSE, ct.SET_EXACT_CHAR_TOKEN, 1.0)["metrics"]
    token_exact = _entry(report, _LOOSE, ct.SET_TOKEN_PLUS_EXACT)["metrics"]

    assert both["true_pairs_retrieved"] >= exact["true_pairs_retrieved"], (
        "exact+char dropped an exact-name pair, so it is not a union of both"
    )
    assert token_exact["true_pairs_retrieved"] > exact["true_pairs_retrieved"], (
        "the token blocker added nothing beyond exact even with char crippled"
    )
    row = [
        item
        for item in report["incremental"]
        if (item["df_cap"], item["rarest_k"]) == _LOOSE
    ]
    assert len(row) == 1
    assert row[0]["additional_true_pairs_from_token"] > 0, (
        f"no measured contribution: char={char['true_pairs_retrieved']}, "
        f"exact+char={both['true_pairs_retrieved']}, +token={everything['true_pairs_retrieved']}"
    )
    assert row[0]["additional_true_pairs_from_token"] == (
        row[0]["exact_char_token_true_pairs"] - row[0]["exact_plus_char_true_pairs"]
    )


def test_tight_cell_handles_zero_candidate_entities_without_dividing_by_zero():
    """Zero-candidate entities must be *counted*, not crash or produce a NaN.

    Some S1 names share no token with either target corpus - the fixture's Devanagari
    name is the example, since neither source has a Devanagari token. Those entities
    are structural zeros for this blocker, and the metrics have to stay finite.
    """
    report, _ = _report()
    cell = report["grid"][0]
    assert (cell["df_cap"], cell["rarest_k"]) == _TIGHT
    assert cell["evaluated"], "an empty cell is still evaluated"
    for entry in cell["sets"]:
        metrics = entry["metrics"]
        assert 0 <= metrics["n_s1_with_zero_candidates"] <= metrics["n_s1_entities"]
        assert 0 <= metrics["n_candidate_pairs"]
        if metrics["n_candidate_pairs"] == 0:
            assert metrics["blocking_recall_pair"] == 0.0
            assert metrics["candidate_precision"] == 0.0
            assert metrics["s1_full_recall_rate"] == 0.0
            assert metrics["n_s1_with_zero_candidates"] == metrics["n_s1_entities"]
        for key, value in metrics.items():
            if key in ("per_source", "reach"):
                continue
            assert not (isinstance(value, float) and np.isnan(value)), f"{key} is NaN"
    # At a cap of 1 most keys are gone, so most entities must be structural zeros.
    token_metrics = _entry(report, _TIGHT, ct.SET_TOKEN)["metrics"]
    assert token_metrics["n_s1_with_zero_candidates"] > 0, (
        "cap 1 should leave some S1 entity with no key at all"
    )
    assert token_metrics["fraction_s1_with_zero_candidates"] <= 1.0
    # The sets that contain exact must still retrieve, because exact does not depend on
    # the token cell at all.
    assert _entry(report, _TIGHT, ct.SET_TOKEN_PLUS_EXACT)["metrics"]["n_candidate_pairs"] >= (
        report["exact"]["n_candidate_pairs"]
    )
    assert _entry(report, _TIGHT, ct.SET_EXACT_CHAR_TOKEN, 0.3)["metrics"][
        "n_candidate_pairs"
    ] >= report["exact_plus_char"]["0.3"]["n_candidate_pairs"]


def test_exact_and_char_sets_do_not_depend_on_the_token_cell():
    """``exact`` and ``exact+char`` are measured once, so they must be cell-invariant.

    They are reported inside every cell for readability; if two cells disagreed about
    them, the grid-free measurement would have leaked into a cell and every delta taken
    against them would be wrong.
    """
    report, _ = _report()
    per_cell = {}
    for cell in report["grid"]:
        for entry in cell["sets"]:
            if entry["set"] != ct.SET_EXACT_CHAR_TOKEN:
                continue
            per_cell[(cell["df_cap"], cell["rarest_k"], entry["threshold"])] = entry["metrics"][
                "n_candidate_pairs"
            ]
    for threshold in _GRID["char_jaccards"]:
        values = {
            key[2]: value for key, value in per_cell.items() if key[2] == threshold
        }
        assert len(set(values.values())) == 1, (
            f"exact+char volume varies by token cell at J={threshold}: {values}"
        )


def test_volume_accounting_agrees_with_the_evaluator():
    """The volume curve and the grid must be the same measurement, not two that ought
    to agree, and the bound must never undercut the exact count."""
    report, _ = _report()
    rows = report["volume_curve"]
    assert rows, "the volume curve is empty"
    kinds = {row["kind"] for row in rows}
    assert kinds == {BOUND_KIND, EXACT_KIND}, f"unexpected kinds: {kinds}"

    bounds = {
        (row["df_cap"], row["rarest_k"]): row["n_candidate_pairs"]
        for row in rows
        if row["kind"] == BOUND_KIND and row["scope"] == VOLUME_ALL
    }
    exact_rows = {
        (row["df_cap"], row["rarest_k"], row["set"], row["jaccard_threshold"], row["scope"]): row
        for row in rows
        if row["kind"] == EXACT_KIND
    }
    assert len(bounds) == len(_GRID["df_caps"]) * len(_GRID["rarest_ks"])
    for cell in report["grid"]:
        key = (cell["df_cap"], cell["rarest_k"])
        assert cell["volume"] == bounds[key], "the grid's volume is not the sweep's bound"
        assert cell["volume_kind"] == BOUND_KIND
        for entry in cell["sets"]:
            base = key + (entry["set"], entry["threshold"])
            all_row = exact_rows[base + (VOLUME_ALL,)]
            assert all_row["n_candidate_pairs"] == entry["metrics"]["n_candidate_pairs"], (
                "the volume row and the evaluator disagree about the same set"
            )
            per_source = sum(
                exact_rows[base + (scope,)]["n_candidate_pairs"] for scope in ("S2", "S3")
            )
            assert per_source == all_row["n_candidate_pairs"], "per-source volumes do not sum"
            if entry["set"] == ct.SET_TOKEN:
                # Only the token set is priced by this bound: it is the sum of the kept
                # keys' posting lengths, an over-count of the pairs those postings
                # produce. The union sets also carry pairs that never went through a
                # token posting at all - a grid-free exact pair whose names share no kept
                # token, say - so for them the comparison is meaningless, not failing.
                assert all_row["n_candidate_pairs"] <= bounds[key], (
                    "the exact token volume exceeds the upper bound, so the bound under-counts"
                )


def test_token_volume_is_reported_even_for_a_cell_that_is_not_evaluated():
    """The candidate-row guard must report the true volume and refuse to evaluate.

    "Do NOT silently cap/truncate" is the requirement this pins: a cell over budget
    keeps a number and loses only its recall, and nothing may present a truncated
    retrieval as a measured result.
    """
    report = _report_variant("guard", ["--max-candidate-rows", "1"])
    assert report["meta"]["max_candidate_rows"] == 1
    skipped = [cell for cell in report["grid"] if not cell["evaluated"]]
    evaluated = [cell for cell in report["grid"] if cell["evaluated"]]
    assert skipped, "the guard skipped nothing, so this test proves nothing"
    for cell in skipped:
        assert cell["skip_reason"] == "expansion_bound_above_max_candidate_rows"
        assert cell["volume_kind"] == BOUND_KIND, "the volume must still be reported"
        assert cell["volume"] > 1, "a cell inside the budget should not have been skipped"
        assert cell["sets"] == [], "a guarded cell must not report metrics"
        assert "pairs_retrieved" not in cell
    for cell in evaluated:
        # Only a cell whose true bound is within the budget may have been evaluated.
        assert cell["volume"] <= 1, f"cell {cell['df_cap']}/{cell['rarest_k']} exceeded the guard"
    assert report["meta"]["cells_skipped"] == len(skipped)
    # The volume curve still covers every cell, so the guard costs nothing but recall.
    bounds = [row for row in report["volume_curve"] if row["kind"] == BOUND_KIND]
    assert len(bounds) == report["meta"]["grid_cells"] * 3, (
        "one bound row per cell per source, plus one all-scope row per cell"
    )
    # And the report says so, rather than quietly omitting the cells.
    text = (_variant_dir("guard") / "token_blocker_calibration.md").read_text(encoding="utf-8")
    assert "not evaluated" in text
    assert "expansion_bound_above_max_candidate_rows" in text
    # No cell was evaluated at all here, so nothing may claim a measured token delta.
    assert not report["incremental"], "a guarded run reported an incremental result"


def test_volume_only_skips_evaluation_but_still_prices_every_cell():
    """``--volume-only`` must produce the volume curve without any retrieval."""
    report = _report_variant("volume_only", ["--volume-only"])
    assert report["meta"]["volume_only"] is True
    assert report["meta"]["cells_evaluated"] == 0
    assert report["meta"]["workers"] is None, "no verification stage ran, so no workers"
    assert report["artifacts"]["char_stage"] is None, "the char stage must not have been built"
    assert report["artifacts"]["exact_candidate_pairs_merged"] == 0
    assert report["incremental"] is None or report["incremental"] == []
    for cell in report["grid"]:
        assert not cell["evaluated"]
        assert cell["skip_reason"] == "volume_only_requested"
        assert cell["volume"] >= 0
    bounds = [
        row
        for row in report["volume_curve"]
        if row["kind"] == BOUND_KIND and row["scope"] == VOLUME_ALL
    ]
    assert len(bounds) == report["meta"]["grid_cells"]
    assert len(report["top_tokens"]) > 0, "the token evidence must still be produced"
    assert (_variant_dir("volume_only") / "token_blocker_calibration.md").is_file()
    assert (_variant_dir("volume_only") / "token_blocker_volume.csv").is_file()


def test_per_source_breakdown_sums_to_the_total():
    """S2 and S3 are indexed and retrieved independently; the parts must sum."""
    report, _ = _report()
    assert report["artifacts"]["target_row_counts"] == {"source2": len(S2), "source3": len(S3)}
    for cell in report["grid"]:
        for entry in cell["sets"]:
            per_source = entry["metrics"]["per_source"]
            assert per_source["S2"]["n_true_pairs"] == 6, "6 of the 11 pairs point into source2"
            assert per_source["S3"]["n_true_pairs"] == 5
            assert (
                per_source["S2"]["n_candidates"] + per_source["S3"]["n_candidates"]
                == entry["metrics"]["n_candidate_pairs"]
            )
            assert (
                per_source["S2"]["true_pairs_retrieved"]
                + per_source["S3"]["true_pairs_retrieved"]
                == entry["metrics"]["true_pairs_retrieved"]
            )


def test_validation_split_metrics_are_reported():
    """The existing S1-level split must be reused, and its mask must restrict the counts."""
    report, _ = _report()
    for cell in report["grid"]:
        for entry in cell["sets"]:
            val = entry["metrics_val"]
            assert val is not None, "the validation split mask was not reported"
            assert 0.0 <= val["blocking_recall_pair"] <= 1.0
            assert val["n_candidate_pairs"] <= entry["metrics"]["n_candidate_pairs"]
            assert val["n_s1_entities"] <= entry["metrics"]["n_s1_entities"]
            assert val["n_true_pairs"] <= entry["metrics"]["n_true_pairs"]


def test_structural_breakdown_reports_every_drop_reason():
    """Each source's drop accounting must be reported per reason, with the keys the
    report claims."""
    report, _ = _report()
    for source, block in report["structural"].items():
        for key in (
            "n_s1_rows_read",
            "n_s1_positions",
            "s1_unknown_to_ground_truth",
            "s1_name_with_no_token",
            "s1_with_no_key_in_this_source",
            "tokens_dropped_absent_from_this_source",
            "tokens_dropped_above_df_cap",
            "tokens_dropped_beyond_rarest_k",
            "entries_per_row",
        ):
            assert key in block, f"{source} omits {key}"
        assert block["n_s1_rows_read"] == len(S1)
        assert block["tokens_dropped_absent_from_this_source"] > 0, (
            "the fixture has S1 tokens neither target corpus contains, so this must be counted"
        )
        assert block["s1_name_with_no_token"] == 0
        assert block["tokens_dropped_above_df_cap"] >= 0
        assert block["tokens_dropped_beyond_rarest_k"] >= 0
        assert block["entries_per_row"] > 0.0


def test_top_tokens_are_reported_as_evidence_for_the_cap():
    """The evidence table must come from the df table, not from a re-count.

    Ordering is per source, not across the table: each source's df is counted and sorted
    on its own, so the two blocks are concatenated and only each block is descending.
    Sorting the concatenation would interleave the sources and say nothing about either.
    """
    report, _ = _report()
    assert report["top_tokens"], "no token evidence was reported"
    for row in report["top_tokens"]:
        assert row["source"] in ("source2", "source3")
        assert row["df"] >= 1
        assert row["postings"] >= 1
        assert row["token"], "an empty token was reported"
    by_source: dict[str, list[int]] = {"source2": [], "source3": []}
    for row in report["top_tokens"]:
        by_source[row["source"]].append(row["df"])
    assert all(dfs for dfs in by_source.values()), "a source reported no token evidence"
    for source, dfs in by_source.items():
        assert dfs == sorted(dfs, reverse=True), f"{source}'s evidence table must be df-descending"


def test_artifact_namespace_is_separate_from_the_char_calibration():
    """Both scripts default to the same output directory, so the trees must not mix.

    A shared artifact directory with a colliding fingerprint could hand this script a
    trigram df table under a token name - a silent correctness failure that no metric
    would reveal.
    """
    report, _ = _report()
    root = Path(report["meta"]["artifact_root"])
    assert root.name == "token", f"unexpected artifact namespace: {root}"
    assert root.parent.name == ct.ARTIFACT_DIRNAME
    assert (root / "source2" / "df" / "artifact.done.json").is_file()
    assert (root / "source2" / "vocab" / "vocab_meta.json").is_file()
    assert (root / "source2" / "index" / "artifact.done.json").is_file()
    assert (root / "source2" / "s1_selection" / "artifact.done.json").is_file()
    # The char stage's own artifacts live under a clearly-labelled subdirectory.
    assert (root / "char_stage").is_dir()
    assert report["meta"]["artifact_fingerprint"] == _report()[0]["meta"]["artifact_fingerprint"]


def test_deterministic_output_across_a_resumed_rerun():
    """``--resume`` must reproduce the report, not merely skip work.

    A resumed run reads its vocabulary, df table, index and selections back from disk;
    if the codes were renumbered or an artifact were stale, the numbers would move.
    """
    report, output_dir = _report()
    config_path, _ = _fixture()
    resumed = _run(config_path, output_dir, ["--resume"])
    assert _strip_volatile(resumed) == _strip_volatile(report), (
        "a resumed run reproduced different numbers"
    )


def test_workers_do_not_change_the_result():
    """The only parallel stage is char verification, whose output must be exact.

    Verification is float32 Jaccard per pair, so a worker count must not shift a single
    threshold decision - otherwise the union would depend on the machine.
    """
    report, _ = _report()
    variant = _report_variant("workers2", ["--workers", "2"])
    assert _strip_volatile(variant) == _strip_volatile(report)


def test_chunk_rows_do_not_change_the_result():
    """``--chunk-rows`` is a memory knob; the numbers must not move with it."""
    report, _ = _report()
    variant = _report_variant("chunks", ["--chunk-rows", "1", "--verify-chunk-pairs", "1"])
    assert _strip_volatile(variant) == _strip_volatile(report)


def test_rejects_impossible_grids_before_reading_anything():
    """Bad arguments must fail with exit code 2 and no partial artifact.

    The negative cases use the ``--flag=value`` form on purpose: argparse itself
    rejects ``--df-caps -1`` as a malformed argument (a value may not begin with ``-``
    when passed separately), which is also an exit code 2 but never reaches this
    script's own validation. ``=`` is what makes the validation below reachable - and
    it is the form a user should be told to use.
    """
    config_path, root = _fixture()
    for argv in (
        ["--df-caps=0"],
        ["--df-caps=-1,10"],
        ["--rarest-ks", "0"],
        ["--rarest-ks", "300"],
        ["--char-df-cap", "0"],
        ["--char-rarest-k", "0"],
        ["--char-jaccards", "0"],
        ["--char-jaccards", "1.5"],
        ["--chunk-rows", "0"],
        ["--verify-chunk-pairs", "0"],
        ["--max-candidate-rows=-1"],
        ["--sources", "source9"],
        ["--sources", ""],
    ):
        import contextlib
        import io

        output_dir = root / "calibration_rejected"
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            try:
                code = ct.main(
                    ["--config", str(config_path), "--output-dir", str(output_dir), *argv]
                )
            except SystemExit as exc:  # argparse's own rejection is also exit 2
                code = exc.code if isinstance(exc.code, int) else 2
        assert code == 2, f"{argv} should have been rejected with exit code 2, got {code}"
        assert not (output_dir / "token_blocker_calibration.json").is_file(), (
            f"{argv} wrote a report before validating"
        )


def test_help_lists_every_documented_flag():
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            ct.parse_args(["--help"])
        except SystemExit:
            pass
    text = buffer.getvalue()
    for flag in (
        "--config",
        "--data-root",
        "--work-dir",
        "--split",
        "--sources",
        "--output-dir",
        "--df-caps",
        "--rarest-ks",
        "--char-df-cap",
        "--char-rarest-k",
        "--char-jaccards",
        "--max-candidate-rows",
        "--workers",
        "--chunk-rows",
        "--verify-chunk-pairs",
        "--limit-s1",
        "--volume-only",
        "--top-tokens",
        "--resume",
        "--timings",
        "--log-level",
    ):
        assert flag in text, f"--help omits {flag}"


def _main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        else:
            print(f"[PASS] {test.__name__}")
    print()
    if failures:
        print(f"{failures} of {len(tests)} tests failed")
        return 1
    print(f"all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
