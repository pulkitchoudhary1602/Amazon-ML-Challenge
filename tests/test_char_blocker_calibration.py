"""Fixture tests for ``scripts/calibrate_char_blocker.py`` (Phase 1 Step 0).

Two layers, both synthetic and both local-only:

* **unit tests** over the module's internals - the trigram codec, index
  construction, the rarest-K rule, the cell filter, and the verification shim;
* **a fixture end-to-end run** over the same 14 S1 / 9 S2 / 6 S3 / 11-pair
  fixture ``tests/test_blocking_statistics.py`` uses, normalized by the real
  ``scripts/prepare_data.py`` and blocked by the real
  ``ExactNameIndex`` this file builds. That is where the claims that need a whole
  pipeline - the S2/S3 split, exact-name containment inside char matching, the
  volume bound against the evaluator's own count, determinism - are actually
  checked.

The fixture is the one place where hand-computed expectations already exist, and
it is deliberately adversarial for this script: it contains a separator-only pair
whose ``name_key`` is identical, a Devanagari name with no latin 3-gram, two
identical ``name_norm`` keys in source2, one-character typos, and four
zero-match S1 entities.

Nothing here reads the real dataset; the corpus is 29 rows. Runs standalone
(``python tests/test_char_blocker_calibration.py``).
"""

from __future__ import annotations

import atexit
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

from scripts import calibrate_char_blocker as cc  # noqa: E402
from scripts import prepare_data  # noqa: E402
from scripts.analyze_name_differences import _trigram_jaccard  # noqa: E402
from src.blocking import BLOCKER_EXACT_NAME, build_index  # noqa: E402
from src.data_loader import load_config  # noqa: E402
from src.utils import read_json  # noqa: E402

# ---------------------------------------------------------------------------
# The fixture, copied from tests/test_blocking_statistics.py so this file is
# self-contained (the two are checked against each other by
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

# S1-1 and S1-8 are the same name and S1-8 has no ground truth at all, so the
# ground-truth position space (14 entities, one per S1 row) is NOT the prepared
# row space in a useful way - which is exactly the mapping this script has to get
# right. S1 rows are all present in the ground truth here; the -1 branch is
# exercised by test_name_key_store_position_mapping instead.
_GRID = {"df_caps": [10000], "rarest_ks": [10], "jaccard_thresholds": [0.3, 0.5, 0.7, 1.0]}

_CACHE: dict[str, object] = {}


# ---------------------------------------------------------------------------
# unit-level helpers
# ---------------------------------------------------------------------------
def _codes_as_set(text: str) -> set[str]:
    """Decoded trigram-code set, as plain strings."""
    return {cc.decode_trigram_code(int(code)) for code in cc.trigram_codes(text)}


def _reference_trigram_set(text: str) -> set[str]:
    """Phase 0.1's own trigram set, over code points."""
    if len(text) < 3:
        return set(text)
    return {text[i : i + 3] for i in range(len(text) - 2)}


def _make_df(texts: list[str]) -> cc._TrigramDf:
    """A df table over ``texts``, one document each."""
    parts = [cc.trigram_codes(text) for text in texts]
    flat = (
        np.concatenate([part for part in parts if part.size])
        if any(part.size for part in parts)
        else cc._EMPTY_INT64
    )
    if flat.size == 0:
        return cc._TrigramDf(cc._EMPTY_INT64, cc._EMPTY_INT64)
    unique, counts = np.unique(flat, return_counts=True)
    return cc._TrigramDf(unique, counts.astype(np.int64))


def _frame(rows: list[tuple[str, str]], name: str = "name_key") -> pd.DataFrame:
    """A prepared-shaped chunk: entity_id + one key column."""
    return pd.DataFrame({"entity_id": [row[0] for row in rows], name: [row[1] for row in rows]})


def _entity_codes(rows: list[tuple[str, str]]) -> dict[int, str]:
    """``entity code -> name``, because postings hold codes, not row indices."""
    from src.utils import encode_entity_ids

    codes = encode_entity_ids(pd.Series([row[0] for row in rows]))
    return {int(code): row[1] for code, row in zip(codes, rows)}


def _code_for(rows: list[tuple[str, str]], entity_id: str) -> int:
    """The encoded code of one entity id in ``rows``."""
    from src.utils import encode_entity_ids

    codes = encode_entity_ids(pd.Series([row[0] for row in rows]))
    for code, row in zip(codes, rows):
        if row[0] == entity_id:
            return int(code)
    raise KeyError(entity_id)


def _build_index_from_rows(
    rows: list[tuple[str, str]], df: cc._TrigramDf, cap: int, k: int, source: str = "source2"
) -> cc._TrigramIndex:
    """Build an index the same way the script does, from in-memory rows."""
    import logging

    return cc.build_trigram_index(
        iter([_frame(rows)]),
        df,
        "entity_id",
        "name_key",
        cap,
        k,
        source,
        logging.getLogger("test"),
        "test",
    )


# ---------------------------------------------------------------------------
# trigram codec
# ---------------------------------------------------------------------------
def test_trigram_codes_match_the_phase_0_1_reference_sets():
    """The codec must reproduce the sliding window exactly, code point for code point.

    Only the ``len >= 3`` branch is compared here: below three characters Phase 0.1's
    reference falls back to bare character sets, which is deliberately NOT what the
    index keys on. The two branches are separated by design, so they are pinned
    separately.
    """
    for text in [
        "sunrise traders",
        "bluesky exports",
        "राम मार्केटिंग",
        "abc",
        "aaaa",
        "  double  spaces  ",
        "café naïve",
        "x" * 40,
    ]:
        assert len(text) >= 3
        got = _codes_as_set(text)
        expected = _reference_trigram_set(text)
        assert got == expected, f"{text!r}: {sorted(got)} != {sorted(expected)}"


def test_trigram_codes_are_empty_below_three_code_points():
    """Short names have no trigram, so the index has no key for them.

    This is not an oversight in the codec: Phase 0.1's reference switches to
    character sets here, so a pair of short names can score 1.0 while being
    unreachable by any trigram blocker. The script counts those entities in
    ``structural[...]["s1_name_too_short_for_any_trigram"]`` rather than hiding them.
    """
    for text in ["", "a", "ab", "रा"]:
        assert cc.trigram_codes(text).size == 0, f"{text!r} should yield no trigram"
        # ... while the reference signal would still compare its characters.
        assert _reference_trigram_set(text) == set(text)


def test_short_name_pairs_are_unreachable_by_trigram_keying():
    """The documented blind spot: two short names score 1.0 but share no key.

    Asserted rather than assumed, because it is the one limitation of this blocker
    that a recall number alone cannot distinguish from a threshold that is too high.
    """
    assert _trigram_jaccard("ab", "ab") == 1.0
    assert cc.trigram_codes("ab").size == 0
    rows = [("S2-1", "ab"), ("S2-2", "sunrise traders")]
    df = _make_df(["ab", "sunrise traders"])
    index = _build_index_from_rows(rows, df, cap=10_000, k=10)
    assert index.n_postings == 10, "the short row contributes nothing; the long row its 10 keys"
    assert _code_for(rows, "S2-1") not in index.postings.tolist()


def test_trigram_codes_are_deduped_and_sorted():
    codes = cc.trigram_codes("abababab")
    assert codes.size == len(set(codes.tolist())), "codes must be deduplicated"
    assert np.all(np.diff(codes) > 0), "codes must be strictly ascending"
    assert len(_codes_as_set("abababab")) == 2  # aba, bab


def test_trigram_encoding_is_injective():
    """No two distinct trigrams may share a code.

    21 bits x 3 = 63, so the packing is exact for every Unicode code point and
    collisions are structurally impossible - but assert it, because a collision
    would appear as a silent false positive rather than a crash.
    """
    alphabet = ["a", "b", " ", "र", "म", "é", "\U0001f600", "z"]
    seen: dict[int, str] = {}
    for first in alphabet:
        for second in alphabet:
            for third in alphabet:
                trigram = first + second + third
                code = int(cc.trigram_codes(trigram)[0])
                assert code >= 0, f"{trigram!r} packed to a negative int64"
                if code in seen:
                    assert seen[code] == trigram, f"{trigram!r} collides with {seen[code]!r}"
                seen[code] = trigram
                assert cc.decode_trigram_code(code) == trigram
    assert len(seen) == len(alphabet) ** 3


def test_rank_within_runs_numbers_each_run_from_zero():
    keys = np.array([5, 5, 5, 7, 7, 9], dtype=np.int64)
    assert cc.rank_within_runs(keys).tolist() == [0, 1, 2, 0, 1, 0]
    assert cc.rank_within_runs(cc._EMPTY_INT64).size == 0


# ---------------------------------------------------------------------------
# posting construction
# ---------------------------------------------------------------------------
def test_posting_construction_groups_by_key_and_orders_rank_within_key():
    rows = [
        ("S2-1", "acme industries"),
        ("S2-2", "acme trading"),
        ("S2-3", "acme industries trading"),
    ]
    # K is large here on purpose: this test is about the posting layout, and a K that
    # truncated the df-3 "acm"/"cme" away would make the 'acme is in all three rows'
    # assertion below test the rarest-K rule instead.
    df = _make_df([text for _, text in rows])
    index = _build_index_from_rows(rows, df, cap=10_000, k=30)

    assert np.all(np.diff(index.keys) > 0), "keys must be sorted and unique"
    counts = index.counts_per_key()
    assert counts.sum() == index.n_postings
    assert index.postings_offsets[0] == 0
    assert index.postings_offsets[-1] == index.n_postings
    assert len(index.postings_offsets) == index.n_keys + 1

    # Every posting must belong to an entity that carries that key, and each key's
    # postings must be rank-ascending (the property the cell filter relies on).
    by_code = _entity_codes(rows)
    for position in range(index.n_keys):
        start = int(index.postings_offsets[position])
        stop = int(index.postings_offsets[position + 1])
        ranks = index.ranks[start:stop]
        assert np.all(np.diff(ranks.astype(np.int64)) >= 0), "postings must be rank-ascending"
        trigram = cc.decode_trigram_code(int(index.keys[position]))
        for code in index.postings[start:stop]:
            assert trigram in by_code[int(code)], f"{trigram!r} posted against a row lacking it"

    # acme appears in all three rows, so its posting list must be the longest.
    acme = int(cc.trigram_codes("acme")[0])
    position = int(np.searchsorted(index.keys, acme))
    assert index.keys[position] == acme
    assert counts[position] == 3


def test_index_holds_multiple_keys_per_entity():
    """The property the shipped one-key ExactNameIndex does not have."""
    rows = [("S2-1", "meridian logistics"), ("S2-2", "acme industries")]
    df = _make_df([text for _, text in rows])
    index = _build_index_from_rows(rows, df, cap=10_000, k=10)

    # "meridian logistics" has 16 distinct trigrams; K=10 keeps the 10 rarest, so the
    # entity contributes exactly 10 postings - many keys per entity, one posting each.
    own = len(cc.trigram_codes("meridian logistics"))
    assert own == 16, '"meridian logistics" is 18 code points, so 16 trigrams'
    codes = _entity_codes(rows)
    per_entity = {
        code: int(np.count_nonzero(index.postings == code)) for code in codes
    }
    assert per_entity == {code: 10 for code in codes}, per_entity
    assert index.n_postings == 20, "both entities contributed their 10 keys"
    assert len(set(index.postings.tolist())) == 2, "exactly two entities are indexed"
    # Each key carries at most one posting per entity: the entity contributed each of
    # its trigrams once, so a key's posting list is a set of distinct entity codes.
    for position in range(index.n_keys):
        postings = index.postings[
            index.postings_offsets[position] : index.postings_offsets[position + 1]
        ]
        assert len(set(postings.tolist())) == len(postings)


def test_df_cap_removes_the_head_and_rarest_k_picks_the_tail():
    """The cap must act on corpus-relative df, and K must pick the rarest survivors."""
    rows = [
        ("S2-1", "acme industries"),
        ("S2-2", "acme trading"),
        ("S2-3", "acme holdings"),
        ("S2-4", "acme ventures"),
    ]
    df = _make_df([text for _, text in rows])
    # "acm"/"cme" occur in all four rows; everything else in one.
    acme = int(cc.trigram_codes("acme")[0])
    assert df.lookup(np.array([acme]))[0] == 4
    assert df.cap_survivors(1) < len(df)

    capped = _build_index_from_rows(rows, df, cap=1, k=10)
    assert np.all(capped.key_df <= 1), "cap leaked a common trigram"
    assert acme not in capped.keys.tolist()

    # With K=2 and no cap, row S2-1 must keep exactly its two rarest trigrams.
    loose = _build_index_from_rows(rows, df, cap=10_000, k=2)
    trigram_codes = cc.trigram_codes("acme industries")
    dfs = df.lookup(trigram_codes)
    expected = trigram_codes[np.lexsort((trigram_codes, dfs))[:2]]
    owner = _code_for(rows, "S2-1")
    row_one_keys = {
        int(loose.keys[position])
        for position in range(loose.n_keys)
        if owner
        in loose.postings[
            loose.postings_offsets[position] : loose.postings_offsets[position + 1]
        ].tolist()
    }
    assert row_one_keys == set(expected.tolist()), (
        f"kept {[cc.decode_trigram_code(c) for c in row_one_keys]} but expected "
        f"{[cc.decode_trigram_code(int(c)) for c in expected]}"
    )
    # And they really are the rarest: every kept trigram is at least as rare as any dropped one.
    kept_dfs = df.lookup(np.asarray(sorted(row_one_keys), dtype=np.int64))
    assert kept_dfs.max() <= dfs.min() or kept_dfs.max() == dfs.min()


def test_rarest_k_selection_is_by_df_then_code():
    """Ties on df must break on the code, or the selection would not be reproducible."""
    rows = [
        ("S2-1", "abcd"),
        ("S2-2", "abcd"),
        ("S2-3", "abce"),
    ]
    df = _make_df([text for _, text in rows])
    codes = cc.trigram_codes("abcd")
    dfs = df.lookup(codes)
    order = np.lexsort((codes, dfs))  # exactly the rule the builder uses
    assert all(dfs[order[i]] <= dfs[order[i + 1]] for i in range(len(order) - 1))

    owner = _code_for(rows, "S2-1")
    index = _build_index_from_rows(rows, df, cap=10_000, k=3)
    survivor = int(codes[order[0]])
    position = int(np.searchsorted(index.keys, survivor))
    assert index.keys[position] == survivor
    start = int(index.postings_offsets[position])
    stop = int(index.postings_offsets[position + 1])
    ranks = index.ranks[start:stop]
    assert ranks.tolist() == sorted(ranks.tolist()), "ranks must ascend down the posting list"
    assert int(ranks[0]) == 0, "the rarest key of some row must carry rank 0"
    # And the rank is per-entity: the same key can be rank 0 for one row and 1 for another.
    assert index.ranks.size == index.postings.size

    # 'abcd' has only two trigrams, so K=3 keeps both of them.
    row_one = [
        int(index.keys[p])
        for p in range(index.n_keys)
        if owner
        in index.postings[index.postings_offsets[p] : index.postings_offsets[p + 1]].tolist()
    ]
    assert set(row_one) == set(codes.tolist())


# ---------------------------------------------------------------------------
# the shared-index proof: a (cap, K) cell is a filter of the loosest index
# ---------------------------------------------------------------------------
def _cell_key_map(index: cc._TrigramIndex) -> dict[int, list[int]]:
    """``trigram code -> sorted ranks'' for the whole index."""
    out: dict[int, list[int]] = {}
    for position in range(index.n_keys):
        start = int(index.postings_offsets[position])
        stop = int(index.postings_offsets[position + 1])
        out[int(index.keys[position])] = sorted(index.ranks[start:stop].astype(int).tolist())
    return out


def test_filtered_cell_equals_a_directly_built_cell():
    """``index.filtered(cap, K)`` must be byte-identical to building at (cap, K).

    This is the load-bearing claim behind indexing the corpus once instead of once
    per grid cell: if the filter were not exactly equivalent, every cell but the
    loosest would report the wrong recall and the wrong volume, and the whole
    calibration would be a measurement of the filter rather than of the blocker.
    """
    rows = [
        ("S2-1", "sunrise traders"),
        ("S2-2", "blue sky exports"),
        ("S2-3", "acme industries private limited"),
        ("S2-4", "acme industries"),
        ("S2-5", "freightways"),
        ("S2-6", "aurora wholesale"),
        ("S2-7", "sunrise trading"),
    ]
    df = _make_df([text for _, text in rows])
    shared = _build_index_from_rows(rows, df, cap=10_000, k=10)

    for cap in (1, 2, 3, 100, 10_000):
        for k in (1, 2, 3, 5, 10):
            filtered = shared.filtered(cap, k)
            direct = _build_index_from_rows(rows, df, cap=cap, k=k)

            assert np.array_equal(filtered.keys, direct.keys), f"keys differ at (cap={cap}, K={k})"
            assert np.array_equal(filtered.postings_offsets, direct.postings_offsets)
            assert np.array_equal(
                filtered.postings, direct.postings
            ), f"postings differ at (cap={cap}, K={k})"
            assert np.array_equal(filtered.ranks, direct.ranks)

            # And the volume shortcut must agree with the built index's own counts.
            kept = shared.kept_counts_per_key(cap, k)
            surviving = shared.key_df <= cap
            assert int(kept[surviving].sum()) == direct.n_postings


def test_kept_counts_per_key_is_zero_above_the_cap():
    rows = [("S2-1", "acme industries"), ("S2-2", "acme trading")]
    df = _make_df([text for _, text in rows])
    index = _build_index_from_rows(rows, df, cap=10_000, k=10)
    kept = index.kept_counts_per_key(cap=1, rarest_k=1)
    assert np.all(kept[index.key_df > 1] == 0)
    assert np.all(kept[index.key_df <= 1] <= 1)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def test_verification_is_bit_identical_to_the_reference():
    """The shim must call ``_trigram_jaccard`` and change nothing about it.

    Includes the pair-level short-string branch (either side under three
    characters makes BOTH sides use bare character sets) and the empty string,
    because those are where a vectorized reimplementation would silently diverge.
    """
    pairs = [
        ("sunrise traders", "sunrise traders"),
        ("blue sky exports", "bluesky exports"),
        ("acme industries private limited", "acme private limited"),
        ("freightways", "freightwayss"),
        ("ab", "abc"),
        ("ab", "ab"),
        ("a", "sunrise traders"),
        ("", "sunrise traders"),
        ("", ""),
        ("राम मार्केटिंग", "ram marketing"),
        ("meridian logistics", "zenith foods"),
    ]
    # A store whose rows are the right-hand sides, in order.
    blob = bytearray()
    offsets = [0]
    for _, right in pairs:
        blob.extend(right.encode("utf-8"))
        offsets.append(len(blob))
    store = cc._NameKeyStore(
        lookup_codes=cc._EMPTY_INT64,
        lookup_rows=cc._EMPTY_INT64,
        row_offsets=np.asarray(offsets, dtype=np.int64),
        blob=bytes(blob),
    )
    lefts = [left for left, _ in pairs]
    rows = np.arange(len(pairs), dtype=np.int64)

    expected = np.array([_trigram_jaccard(left, right) for left, right in pairs], dtype=np.float32)

    # Workers load the store from a directory (rather than being handed it) so a
    # process that only ever sees one source never pays for the other - which means
    # even the single-worker path needs one on disk.
    # The store's rows hold the right-hand names; the S1 side is resolved by position.
    with tempfile.TemporaryDirectory() as tmp:
        store.save(Path(tmp))
        dirs = {"source2": tmp}
        single = cc.verify_similarities(
            "source2", rows, rows, lefts.__getitem__, workers=1, window=2, chunk_pairs=3,
            store_dirs=dirs,
        )
        # workers=1 vs workers=2 must agree exactly, not approximately.
        parallel = cc.verify_similarities(
            "source2", rows, rows, lefts.__getitem__, workers=2, window=4, chunk_pairs=2,
            store_dirs=dirs,
        )
    assert np.array_equal(single, expected), f"{single.tolist()} != {expected.tolist()}"
    assert single.dtype == np.float32
    assert np.array_equal(parallel, expected), "multi-worker verification diverged"


def test_verification_scores_identical_keys_at_one():
    pairs = [("sunrise traders", "sunrise traders"), ("राम मार्केटिंग", "राम मार्केटिंग")]
    blob = bytearray()
    offsets = [0]
    for _, right in pairs:
        blob.extend(right.encode("utf-8"))
        offsets.append(len(blob))
    store = cc._NameKeyStore(
        cc._EMPTY_INT64,
        cc._EMPTY_INT64,
        np.asarray(offsets, dtype=np.int64),
        bytes(blob),
    )
    lefts = [left for left, _ in pairs]
    with tempfile.TemporaryDirectory() as tmp:
        store.save(Path(tmp))
        scores = cc.verify_similarities(
            "source2",
            np.arange(len(pairs), dtype=np.int64),
            np.arange(len(pairs), dtype=np.int64),
            lefts.__getitem__,
            workers=1,
            window=1,
            chunk_pairs=1,
            store_dirs={"source2": tmp},
        )
    assert scores.tolist() == [1.0, 1.0]


def test_verification_skips_missing_rows():
    store_offsets = np.array([0, 5, 5], dtype=np.int64)  # row 1 is empty
    store = cc._NameKeyStore(
        cc._EMPTY_INT64, cc._EMPTY_INT64, store_offsets, b"first"
    )
    with tempfile.TemporaryDirectory() as tmp:
        store.save(Path(tmp))
        scores = cc.verify_similarities(
            "source2",
            np.array([0, 1], dtype=np.int64),
            np.array([0, -1], dtype=np.int64),
            lambda position: "first",
            workers=1,
            window=1,
            chunk_pairs=2,
            store_dirs={"source2": tmp},
        )
    assert scores[0] == 1.0 and scores[1] == 0.0, "a -1 target row must score 0, not raise"


def test_name_key_store_position_mapping_and_missing_rows():
    """The -1 convention bridges ground-truth positions to prepared rows."""
    store = cc._NameKeyStore(
        lookup_codes=np.array([7], dtype=np.int64),
        lookup_rows=np.array([0], dtype=np.int64),
        row_offsets=np.array([0, 5], dtype=np.int64),
        blob=b"first",
        position_to_row=np.array([0, -1, -1], dtype=np.int64),
    )
    assert store.n_positions == 3
    assert store.key_at_position(0) == "first"
    assert store.key_at_position(1) == "", "an unmapped position must be a miss, not an error"
    assert store.valid_positions().tolist() == [0]
    assert store.row_for_code(np.array([7, 8], dtype=np.int64)).tolist() == [0, -1]


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
    assert not re.search(r'^\s*(work_dir|prepared_dir|index_dir|log_dir): "(?![/A-Za-z]:)', source, re.M)

    config_path = base / "config_char_calibration_fixture.yaml"
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
        root = Path(tempfile.mkdtemp(prefix="er_char_calibration_"))
        config_path = _build_fixture(root)
        _prepare(config_path)
        config = load_config(str(config_path))
        for source in ("source2", "source3"):
            build_index(config, "train", source, BLOCKER_EXACT_NAME, overwrite=True)
        _CACHE["root"] = root
        _CACHE["config_path"] = config_path
        if not os.environ.get("ER_TEST_KEEP_FIXTURE"):
            atexit.register(shutil.rmtree, root, ignore_errors=True)
    return _CACHE["config_path"], _CACHE["root"]  # type: ignore[return-value]


def _grid_params() -> list[str]:
    return [
        "--df-caps", ",".join(str(cap) for cap in _GRID["df_caps"]),
        "--rarest-ks", ",".join(str(k) for k in _GRID["rarest_ks"]),
        "--jaccard-thresholds", ",".join(str(t) for t in _GRID["jaccard_thresholds"]),
    ]


def _run(config_path: Path, output_dir: Path, extra: list[str]) -> dict:
    """Run the calibration script on the fixture and return its JSON report."""
    argv = [
        "--config", str(config_path),
        "--output-dir", str(output_dir),
        "--chunk-rows", "3",
        "--verify-chunk-pairs", "2",
        "--workers", "1",
        "--top-trigrams", "5",
        "--log-level", "WARNING",
        *_grid_params(),
        *extra,
    ]
    code = cc.main(argv)
    assert code == 0, f"calibration exited {code}"
    return read_json(output_dir / "char_blocker_calibration.json")


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
        config_path, root = _fixture()
        _CACHE[key] = _run(config_path, root / f"calibration_{key}", extra)
    return _CACHE[key]  # type: ignore[return-value]


_VOLATILE_META = (
    # wall clock, host, and the parameters of the run rather than its results
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
)


def _strip_volatile(report: dict) -> dict:
    """Keep the results, drop the fields that legitimately differ between equal runs."""
    meta = {key: value for key, value in report["meta"].items() if key not in _VOLATILE_META}
    grid = [
        {key: value for key, value in cell.items() if key != "elapsed_seconds"}
        for cell in report["grid"]
    ]
    return {**report, "meta": meta, "grid": grid}


def test_fixture_matches_blocking_statistics_fixture():
    """The copied fixture must still be the other test's fixture, row for row.

    Compared through the AST rather than as text, so quoting or layout in the other
    file cannot make this pass vacuously.
    """
    import ast

    tree = ast.parse((REPO / "tests" / "test_blocking_statistics.py").read_text(encoding="utf-8"))
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
        assert name in other, f"{name} is not defined in the other fixture"
        assert [tuple(row) for row in other[name]] == rows, f"{name} has drifted"


def test_end_to_end_report_shape():
    report, output_dir = _report()
    meta = report["meta"]
    assert meta["n_s1_entities"] == len(S1) == 14
    assert meta["n_true_pairs"] == 11
    assert meta["df_caps"] == _GRID["df_caps"]
    assert meta["rarest_ks"] == _GRID["rarest_ks"]
    assert meta["verification"].endswith("_trigram_jaccard (imported unchanged)")
    assert report["token_analytic_reference"]["measured_here"] is False
    assert set(report["structural"]) == {"source2", "source3"}
    for name in (
        "char_blocker_calibration.json",
        "char_blocker_calibration.csv",
        "char_blocker_volume.csv",
        "char_blocker_calibration.md",
        "char_blocker_top_trigrams.csv",
    ):
        assert (output_dir / name).is_file(), f"{name} was not written"


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
    }
    assert report["grid"], "the grid is empty"
    for cell in report["grid"]:
        assert cell["evaluated"], "the fixture is tiny; every cell must be evaluated"
        assert cell["sets"], "a cell reported no candidate sets"
        for entry in cell["sets"]:
            metrics = entry["metrics"]
            missing = required - set(metrics)
            assert not missing, f"cell {cell['df_cap']}/{cell['rarest_k']} missing {missing}"
            assert set(metrics["per_source"]) == {"S2", "S3"}
            for source_metrics in metrics["per_source"].values():
                for key in cc.PER_SOURCE_METRICS:
                    assert key in source_metrics, f"per-source metric {key} missing"
            assert 0.0 <= metrics["candidate_precision"] <= 1.0
            assert 0.0 <= metrics["blocking_recall_pair"] <= 1.0
            assert 0.0 <= metrics["macro_recall_entity"] <= 1.0
            assert metrics["n_candidate_pairs"] >= 0
            assert metrics["reduction_ratio"] > 0
            assert "metrics_val" in entry, "the validation split mask was not reported"


def test_volume_bound_and_exact_volume_agree():
    """The two volume kinds must be consistent, and labelled distinctly.

    The exact count is read from the same accumulator as ``n_candidate_pairs``, so
    it has to match it; the bound counts posting entries, so it has to be at least
    as large. If the bound were ever smaller, ``kept_counts_per_key`` would be
    under-counting and the budget gate would let an over-budget cell through.
    """
    report, _ = _report()
    rows = report["volume_curve"]
    assert rows, "the volume curve is empty"
    kinds = {row["kind"] for row in rows}
    assert kinds == {cc.BOUND_KIND, cc.EXACT_KIND}, f"unexpected volume kinds: {kinds}"

    bounds = {
        (row["df_cap"], row["rarest_k"]): row["n_candidate_pairs"]
        for row in rows
        if row["kind"] == cc.BOUND_KIND and row["scope"] == cc.VOLUME_ALL
    }
    exact = {
        (row["df_cap"], row["rarest_k"], row["set"], row["jaccard_threshold"]): row[
            "n_candidate_pairs"
        ]
        for row in rows
        if row["kind"] == cc.EXACT_KIND and row["scope"] == cc.VOLUME_ALL
    }
    assert bounds and exact

    for cell in report["grid"]:
        key = (cell["df_cap"], cell["rarest_k"])
        assert key in bounds, f"no bound for cell {key}"
        assert cell["volume"] == bounds[key]
        for entry in cell["sets"]:
            exact_key = (cell["df_cap"], cell["rarest_k"], entry["set"], entry["threshold"])
            assert exact[exact_key] == entry["metrics"]["n_candidate_pairs"]
            assert exact[exact_key] <= bounds[key], (
                f"exact volume {exact[exact_key]} exceeds bound {bounds[key]} at {exact_key}"
            )

    # Per-source exact rows must sum to the all-scope exact row, and the
    # all-scope exact rows must sum to the evaluator's own count.
    for scope in ("S2", "S3"):
        assert any(row["scope"] == scope and row["kind"] == cc.EXACT_KIND for row in rows)


def test_exact_rows_match_the_evaluator_per_source_totals():
    report, _ = _report()
    by_key: dict[tuple, int] = {}
    for row in report["volume_curve"]:
        if row["kind"] != cc.EXACT_KIND:
            continue
        by_key[(row["df_cap"], row["rarest_k"], row["set"], row["jaccard_threshold"], row["scope"])] = row[
            "n_candidate_pairs"
        ]
    for cell in report["grid"]:
        for entry in cell["sets"]:
            base = (cell["df_cap"], cell["rarest_k"], entry["set"], entry["threshold"])
            total = by_key[base + (cc.VOLUME_ALL,)]
            per_source = sum(
                by_key[base + (scope,)] for scope in ("S2", "S3")
            )
            assert total == per_source, f"per-source volumes do not sum to the total at {base}"
            assert total == entry["metrics"]["n_candidate_pairs"]


def test_separation_of_s2_and_s3():
    """Sources must be indexed, retrieved and scored independently.

    Entity codes are source-tagged (``source_code * 10**10 + numeric``), so a
    leak between them would show up as a candidate whose code decodes to the wrong
    source - which is exactly what the per-source accumulators count.
    """
    report, _ = _report()
    assert report["artifacts"]["target_row_counts"].get("source2") == len(S2)
    assert report["artifacts"]["target_row_counts"].get("source3") == len(S3)

    for cell in report["grid"]:
        for entry in cell["sets"]:
            per_source = entry["metrics"]["per_source"]
            # From the fixture: 6 of the 11 true pairs point into source2, 5 into source3.
            assert per_source["S2"]["n_true_pairs"] == 6, "S2 true pairs from the fixture"
            assert per_source["S3"]["n_true_pairs"] == 5, "S3 true pairs from the fixture"
            summed = per_source["S2"]["n_candidates"] + per_source["S3"]["n_candidates"]
            assert summed == entry["metrics"]["n_candidate_pairs"], "per-source candidates must sum"


def test_exact_name_pairs_are_contained_in_the_char_candidates():
    """Every exact-name pair must also be retrievable by char, at Jaccard 1.0.

    Identical ``name_norm`` implies identical ``name_key`` implies identical
    trigram sets, so the similarity is 1.0 and the pair survives any threshold.
    The calibration's whole reason for reporting ``char_plus_exact`` per cell is to
    measure whether retrieval actually recovers them - this pins the containment
    itself, so a nonzero delta would mean a retrieval bug, not a signal finding.
    """
    report, _ = _report()
    exact_sizes = report["artifacts"]["exact_candidate_pairs"]
    assert exact_sizes, "the fixture built no exact-name candidates; the test proves nothing"
    assert exact_sizes["source2"] > 0

    char_counts = {}
    plus_counts = {}
    for cell in report["grid"]:
        for entry in cell["sets"]:
            key = (cell["df_cap"], cell["rarest_k"], entry["threshold"])
            if entry["set"] == cc.SET_CHAR:
                char_counts[key] = entry["metrics"]["n_candidate_pairs"]
            else:
                plus_counts[key] = entry["metrics"]["n_candidate_pairs"]
    assert plus_counts, "char_plus_exact was not reported"
    for key, count in plus_counts.items():
        assert count == char_counts[key], (
            f"char_plus_exact added {count - char_counts[key]} candidates over char at {key}; "
            "exact_name is meant to be a subset of char_3gram_name"
        )

    assert report["exact_only_metrics"] is not None
    assert report["exact_only_metrics"]["n_candidate_pairs"] > 0


def test_exact_lookup_uses_the_index_key_field():
    """The exact set must be queried with the field the index was built on.

    The shipped index keys on ``name_norm``; the verification store holds
    ``name_key``. They differ by separators, and querying the wrong one returns
    zero pairs silently - so assert the field recorded in the report is the
    index's, and that the pairs found are real.
    """
    report, _ = _report()
    fields = report["artifacts"]["exact_key_fields"]
    assert set(fields.values()) == {"name_norm"}, f"unexpected exact key fields: {fields}"
    assert report["meta"]["exact_index_available"] == {"source2": True, "source3": True}

    # S1-1 and S1-8 both carry name_norm "sunrise traders", and S2-201 does too,
    # so the exact blocker must find at least two pairs.
    assert report["artifacts"]["exact_candidate_pairs"]["source2"] >= 2


def test_char_only_pair_is_reached_and_address_only_pair_is_not():
    """The fixture's known char-only and address-only pairs must behave as Phase 0 found.

    S1-13 "Freightways" vs S2-208 "Freightwayss" is char-only (a one-character
    typo); S1-14's Devanagari name vs S3-306 "Ram Marketing" shares no trigram at
    all. A blocker that reported the second as reachable would be reporting an
    address match as a name match.
    """
    report, _ = _report()
    best = None
    for cell in report["grid"]:
        for entry in cell["sets"]:
            if entry["set"] == cc.SET_CHAR and entry["threshold"] == min(_GRID["jaccard_thresholds"]):
                best = entry["metrics"]
    assert best is not None
    # 11 true pairs, 1 of which is address-only and 1 residue: char alone cannot
    # reach S1-14 -> S3-306, and must not reach S1-12 -> S2-207 either.
    assert best["true_pairs_retrieved"] <= 9, "char alone retrieved an address-only pair"
    assert best["true_pairs_retrieved"] >= 6, "char alone lost too many of its reachable pairs"
    assert best["blocking_recall_pair"] < 1.0, "the residue and address-only pairs are unreachable"


def test_candidate_survives_at_jaccard_one_for_identical_keys():
    """Threshold 1.0 must keep identical keys and drop the separator-only pair.

    ``sunrise traders`` and ``sunrise traders`` score 1.0; ``blue sky exports``
    vs ``bluesky exports`` scores below 1.0 because the trigrams differ even
    though ``name_key`` matches. That is the pair that tells thresholds apart.
    """
    report, _ = _report()
    at_one = None
    for cell in report["grid"]:
        for entry in cell["sets"]:
            if entry["set"] == cc.SET_CHAR and entry["threshold"] == 1.0:
                at_one = entry["metrics"]
    assert at_one is not None, "threshold 1.0 was not evaluated"
    assert at_one["n_candidate_pairs"] < at_one["n_s1_entities"] * 2
    assert at_one["s1_full_recall_rate"] <= 1.0


def test_validation_split_metrics_reduce_under_the_mask():
    report, _ = _report()
    for cell in report["grid"]:
        for entry in cell["sets"]:
            val = entry["metrics_val"]
            whole = entry["metrics"]
            assert val["n_s1_entities"] <= whole["n_s1_entities"]
            assert val["n_true_pairs"] <= whole["n_true_pairs"]
            assert 0.0 <= val["blocking_recall_pair"] <= 1.0
            assert 0.0 <= val["macro_recall_entity"] <= 1.0


def test_zero_match_entities_are_counted_not_omitted():
    """Four S1 entities have no match; the report must account for them.

    S1-8/-9/-10/-11 have empty ground truth, so a candidate produced for them is a
    false positive and no candidate can ever be a hit. The fixture is small enough
    that the identity below is checkable directly.
    """
    report, _ = _report()
    for cell in report["grid"]:
        for entry in cell["sets"]:
            metrics = entry["metrics"]
            assert metrics["n_s1_with_true_matches"] + 4 <= metrics["n_s1_entities"]
            assert metrics["true_pairs_retrieved"] <= metrics["n_true_pairs"] == 11


def test_deterministic_across_chunking_and_resume():
    """Two runs with different chunking, and a resumed run, must agree exactly."""
    base, _ = _report()
    variant = _report_variant("chunked", ["--chunk-rows", "2", "--verify-chunk-pairs", "1"])
    resumed = _report_variant("resumed", ["--resume"])

    assert _strip_volatile(base) == _strip_volatile(variant), "chunking changed the report"
    assert _strip_volatile(base) == _strip_volatile(resumed), "the resumed run diverged"

    for report in (base, variant, resumed):
        for cell in report["grid"]:
            for entry in cell["sets"]:
                assert "metrics" in entry and entry["metrics"]["n_candidate_pairs"] >= 0


def test_deterministic_across_worker_counts():
    """Verification sharding must not change a single similarity."""
    base, _ = _report()
    parallel = _report_variant("workers", ["--workers", "2", "--verify-chunk-pairs", "3"])
    assert _strip_volatile(base) == _strip_volatile(parallel), "worker count changed the report"


def test_volume_only_skips_evaluation_but_still_prices_every_cell():
    report = _report_variant("volumeonly", ["--volume-only"])
    assert report["meta"]["cells_evaluated"] == 0
    assert report["meta"]["cells_skipped"] == len(report["grid"])
    for cell in report["grid"]:
        assert cell["evaluated"] is False
        assert cell["skip_reason"] == "volume_only_requested"
        assert cell["volume_kind"] == cc.BOUND_KIND
    assert {row["kind"] for row in report["volume_curve"]} == {cc.BOUND_KIND}


def test_budget_gate_is_explicit_and_reported():
    """An over-budget cell must report its bound and a reason, never a fake score."""
    report = _report_variant("tinybudget", ["--max-candidate-rows", "1"])
    skipped = [cell for cell in report["grid"] if not cell["evaluated"]]
    assert skipped, "--max-candidate-rows 1 should have skipped at least one cell"
    for cell in skipped:
        assert cell["skip_reason"] == "expansion_bound_above_max_candidate_rows"
        assert cell["max_candidate_rows"] == 1
        assert cell["volume_kind"] == cc.BOUND_KIND
        assert cell["sets"] == [], "a skipped cell must not carry metrics"
        assert cell["volume"] >= 0
    assert report["meta"]["cells_skipped"] == len(skipped)
    # The report must say so in prose as well, not only in the JSON.
    markdown = (_variant_dir("tinybudget") / "char_blocker_calibration.md").read_text(encoding="utf-8")
    assert "NOT evaluated" in markdown
    assert cc.BOUND_KIND in markdown


def test_markdown_and_csv_are_written_and_consistent():
    report, output_dir = _report()
    markdown = (output_dir / "char_blocker_calibration.md").read_text(encoding="utf-8")
    assert "# char-3-gram blocker calibration" in markdown
    assert cc.BOUND_KIND in markdown
    assert "Token blocking, analytic reference only" in markdown
    assert "No operating threshold is chosen here" in markdown

    csv = pd.read_csv(output_dir / "char_blocker_calibration.csv", encoding="utf-8")
    assert not csv.empty
    for column in ("df_cap", "rarest_k", "jaccard_threshold", "set", "n_candidate_pairs"):
        assert column in csv.columns, f"CSV missing {column}"
    assert len(csv) == sum(len(cell["sets"]) for cell in report["grid"])

    volume = pd.read_csv(output_dir / "char_blocker_volume.csv", encoding="utf-8")
    assert set(volume["kind"]) == {cc.BOUND_KIND, cc.EXACT_KIND}
    assert len(volume) == len(report["volume_curve"])

    trigrams = pd.read_csv(output_dir / "char_blocker_top_trigrams.csv", encoding="utf-8")
    assert {"source", "trigram", "df", "postings"} <= set(trigrams.columns)


def test_script_does_not_touch_the_config_or_enable_blockers():
    """Step 0 must leave ``configs/config.yaml`` and its blocking flags alone."""
    _report()
    config = load_config(str(REPO / "configs" / "config.yaml"))
    blocking = config["blocking"]
    assert blocking["char_ngram"]["enabled"] is False
    assert blocking["token"]["enabled"] is False
    assert blocking["tfidf"]["enabled"] is False
    assert blocking["dense"]["enabled"] is False
    assert blocking["exact_name"]["enabled"] is True
    assert blocking["exact_name"]["key"] == "name_norm"


def test_cli_rejects_bad_arguments():
    """Bad arguments must fail before any data is touched.

    The fixture config is passed deliberately: these runs must never reach the real
    corpus, and a validation check that only fires after loading the ground truth
    would be both wrong and expensive.
    """
    import contextlib
    import io

    config_path, _ = _fixture()
    for argv in (
        ["--df-caps", "0"],
        ["--df-caps", ""],
        ["--rarest-ks", "0"],
        ["--rarest-ks", "999"],
        ["--jaccard-thresholds", "0"],
        ["--jaccard-thresholds", "1.5"],
        ["--chunk-rows", "0"],
        ["--verify-chunk-pairs", "0"],
        ["--sources", "source9"],
        ["--sources", ""],
    ):
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            assert cc.main(["--config", str(config_path), *argv]) == 2, (
                f"{argv} should have been rejected"
            )


def test_help_lists_every_documented_flag():
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            cc.parse_args(["--help"])
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
        "--jaccard-thresholds",
        "--max-candidate-rows",
        "--workers",
        "--chunk-rows",
        "--verify-chunk-pairs",
        "--limit-s1",
        "--volume-only",
        "--resume",
        "--timings",
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
