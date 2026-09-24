"""Unicode-safe, multilingual normalization for business names and addresses.

The dataset is a mix of English/Latin records and Indian records in Devanagari
(``राम मार्केटिंग प्राइवेट लिमिटेड``), Kannada, Bengali, and others. The design
constraint that drives everything in this module:

    **Never drop combining marks.**

A naive "strip accents" routine (NFD then remove all ``Mn`` characters) is
correct for French and catastrophic for Devanagari: the vowel signs and virama
are combining marks. ``कंस्ट्रक्शंस`` would collapse into garbage, and every
Indian business name would start colliding with every other one.

So accent folding is applied **only to Latin-script characters**, character by
character. Everything else - Devanagari, Kannada, Bengali, Arabic-Indic digits,
CJK - passes through untouched apart from case and separator handling.

Pipeline per field::

    NFKC  ->  per-character table  ->  case  ->  whitespace collapse  ->  truncate

The per-character table is built once per process (see ``_build_table``) and
applied with ``str.translate``, which runs at C speed. This matters: it is
invoked on ~10.3M S2/S3 records plus 2.2M S1 records.

Public API
----------
``Normalizer``            - configurable, the recommended entry point.
``normalize_text``        - module-level convenience using default settings.
``add_normalized_columns``- mutate a DataFrame chunk in place (used by scripts).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns produced by this module, appended to every prepared table.
NAME_NORM = "name_norm"
NAME_KEY = "name_key"
ADDRESS_NORM = "address_norm"
COUNTRY_NORM = "country_norm"

_WHITESPACE_RE = re.compile(r"\s+")

# Categories we keep verbatim:
#   L* letters, M* marks (this is the Indic-vowel-sign lifeline), N* numbers
# Everything in C* (control/format/private-use/surrogate), Z* (separators),
# P* (punctuation) and S* (symbols) is rewritten by the table.
_KEEP_CATEGORY_INITIALS = frozenset("LMN")

# Categories left OUT of the table entirely. Cn is "unassigned" - roughly
# 800k code points that never occur in real text. Excluding them keeps the
# translate dict at ~12k entries instead of ~900k (saves ~60MB of RSS).
_SKIP_CATEGORIES = frozenset({"Cn"})


def _fold_latin_character(char: str) -> str:
    """Strip diacritics from a single Latin character, in isolation.

    Decomposing one character at a time is what makes this safe: ``é`` becomes
    ``e``, while ``क`` is never decomposed and never loses its marks.
    """
    decomposed = unicodedata.normalize("NFD", char)
    folded = "".join(c for c in decomposed if not unicodedata.combining(c))
    return folded or char


@lru_cache(maxsize=8)
def _build_table(fold_latin_accents: bool, punctuation_to_space: bool) -> dict:
    """Build the ``str.translate`` table covering the whole code space.

    Cost: one pass over 0x110000 code points (~1s), cached per process. The
    resulting dict only holds entries that actually change a character.

    Args:
        fold_latin_accents: apply Latin-only diacritic folding.
        punctuation_to_space: map punctuation/symbols to a space rather than
            deleting them, so ``heassociates.com`` tokenizes to
            ``heassociates com`` instead of ``heassociatescom``.

    Returns:
        Mapping of code point -> replacement string (``""`` means delete).
    """
    table: dict = {}
    replacement_for_category = {
        "Cc": "",  # control
        "Cf": "",  # format: zero-width joiners, BOM, soft hyphen
        "Cs": "",  # surrogates (cannot be encoded anyway)
        "Co": " ",  # private use
        "Zs": " ",
        "Zl": " ",
        "Zp": " ",
        "Cn": "",
    }

    for code_point in range(0x110000):
        char = chr(code_point)
        category = unicodedata.category(char)

        if category in _SKIP_CATEGORIES:
            continue

        initial = category[0]
        if initial in _KEEP_CATEGORY_INITIALS:
            # Latin-only accent folding. unicodedata.name is only consulted for
            # letters, so this stays cheap.
            if fold_latin_accents and initial == "L" and unicodedata.name(char, "").startswith("LATIN"):
                folded = _fold_latin_character(char)
                if folded != char:
                    table[code_point] = folded
            continue

        if category in replacement_for_category:
            table[code_point] = replacement_for_category[category]
        elif initial in ("P", "S"):
            table[code_point] = " " if punctuation_to_space else ""
        elif initial == "Z":
            table[code_point] = " "
        else:  # pragma: no cover - defensive; C* not enumerated above
            table[code_point] = ""

    return table


class Normalizer:
    """Configurable normalizer. Build once, reuse across chunks.

    Example:
        >>> norm = Normalizer()
        >>> norm.name("Orelee's Barbershop")
        "orelee s barbershop"
        >>> norm.name("B+ Retail Inc")
        'b retail inc'
        >>> norm.name("राम मार्केटिंग प्राइवेट लिमिटेड")   # doctest: +SKIP
        'राम मार्केटिंग प्राइवेट लिमिटेड'                 # marks preserved
    """

    __slots__ = (
        "unicode_form",
        "fold_latin_accents",
        "case_mode",
        "punctuation_to_space",
        "max_length",
        "_table",
    )

    def __init__(
        self,
        unicode_form: str = "NFKC",
        fold_latin_accents: bool = True,
        case_mode: str = "lower",
        punctuation_to_space: bool = True,
        max_length: int = 512,
    ) -> None:
        form = (unicode_form or "NFKC").upper()
        if form not in ("NFC", "NFD", "NFKC", "NFKD"):
            raise ValueError(f"unsupported unicode_form: {unicode_form!r}")
        if form in ("NFD", "NFKD"):
            # NFD/NFKD would split Indic vowel signs off their base consonant,
            # which is exactly what we must avoid. Warn loudly, do not forbid:
            # a researcher may want to measure the damage.
            logger.warning(
                "unicode_form=%s decomposes scripts and will damage Indic text; NFKC is strongly recommended",
                form,
            )
        if case_mode not in ("lower", "casefold"):
            raise ValueError(f"unsupported case_mode: {case_mode!r}")

        self.unicode_form = form
        self.fold_latin_accents = bool(fold_latin_accents)
        self.case_mode = case_mode
        self.punctuation_to_space = bool(punctuation_to_space)
        self.max_length = int(max_length)
        self._table = _build_table(self.fold_latin_accents, self.punctuation_to_space)

    @classmethod
    def from_config(cls, config: dict) -> "Normalizer":
        """Build from the ``normalization:`` block of config.yaml."""
        section = (config or {}).get("normalization", {}) or {}
        return cls(
            unicode_form=section.get("unicode_form", "NFKC"),
            fold_latin_accents=section.get("fold_latin_accents", True),
            case_mode=section.get("case_mode", "lower"),
            punctuation_to_space=section.get("punctuation_to_space", True),
            max_length=section.get("max_length", 512),
        )

    # -- core ---------------------------------------------------------------
    def text(self, value: Any) -> str:
        """Normalize a single string. Returns ``""`` for null/blank input."""
        if value is None:
            return ""
        if not isinstance(value, str):
            # NaN, floats, ints - anything non-string is coerced, since pandas
            # hands us object columns with mixed types more often than one hopes.
            if isinstance(value, float) and value != value:  # NaN
                return ""
            value = str(value)
        if not value:
            return ""

        result = unicodedata.normalize(self.unicode_form, value)
        result = result.translate(self._table)
        result = result.lower() if self.case_mode == "lower" else result.casefold()
        result = _WHITESPACE_RE.sub(" ", result).strip()
        if self.max_length and len(result) > self.max_length:
            result = result[: self.max_length].rstrip()
        return result

    def name(self, value: Any) -> str:
        """Normalize a business name (token structure preserved)."""
        return self.text(value)

    def address(self, value: Any) -> str:
        """Normalize an address. Same rules; kept as a named entry point so the
        two fields can diverge later (e.g. dropping unit numbers) without
        touching call sites."""
        return self.text(value)

    def country(self, value: Any) -> str:
        """Normalize a country label to a lowercase token - no punctuation
        mapping, since country values are single tokens like ``US``/``India``."""
        if value is None:
            return ""
        text = str(value)
        if not text:
            return ""
        return unicodedata.normalize(self.unicode_form, text).translate(self._table).strip().lower()

    def key(self, value: Any) -> str:
        """Separator-free variant of :meth:`name`, for equality-only blocking.

        ``"orelee s barbershop"`` -> ``"oreleesbarbershop"``. Useful because the
        noisy sources sprinkle in stray punctuation; it makes ``A B C`` and
        ``ABC`` collide, which is usually what we want at the blocking stage.
        """
        return _WHITESPACE_RE.sub("", self.name(value))

    # -- vectorized ---------------------------------------------------------
    def series(self, values: pd.Series, method: str = "name") -> pd.Series:
        """Normalize a whole pandas Series efficiently.

        Deduplicates before normalizing: S2 has ~5.0M rows but only ~4.0M
        distinct normalized names, and addresses repeat even more. Normalizing
        the unique set and mapping back is the single biggest speedup available
        in the preprocessing stage, because :meth:`text` is a Python-level call.

        Args:
            values: the raw column.
            method: one of ``name``, ``address``, ``country``, ``key``, ``text``.

        Returns:
            A new Series of normalized strings, aligned to ``values``.
        """
        func = getattr(self, method)
        as_str = values.astype("object") if values.dtype != object else values
        uniques = pd.unique(as_str)
        mapping = {value: func(value) for value in uniques}
        return as_str.map(mapping).astype("string")


def normalize_text(value: Any, **kwargs: Any) -> str:
    """Module-level convenience wrapper. Prefer a shared ``Normalizer`` in loops."""
    return Normalizer(**kwargs).text(value)


def add_normalized_columns(
    frame: pd.DataFrame,
    config: dict,
    normalizer: Optional[Normalizer] = None,
    columns: Optional[dict] = None,
) -> pd.DataFrame:
    """Add ``name_norm`` / ``name_key`` / ``address_norm`` / ``country_norm``.

    The original columns are never overwritten - ``business_name`` stays exactly
    as it came out of the TSV, which we need later for character-level features
    and for eyeballing errors.

    Args:
        frame: a chunk with raw columns.
        config: the loaded config (for the ``normalization`` and ``columns`` blocks).
        normalizer: reuse an existing instance; built from config when omitted.
        columns: column-name mapping; defaults to ``config['columns']``.

    Returns:
        The same DataFrame, mutated (chunk-scale, so copying would be wasteful).
    """
    normalizer = normalizer or Normalizer.from_config(config)
    columns = columns if columns is not None else config.get("columns", {})

    name_column = columns.get("name", "business_name")
    address_column = columns.get("address", "business_address")
    country_column = columns.get("country", "country")

    if name_column in frame.columns:
        frame[NAME_NORM] = normalizer.series(frame[name_column], "name")
        frame[NAME_KEY] = normalizer.series(frame[name_column], "key")
    if address_column in frame.columns:
        frame[ADDRESS_NORM] = normalizer.series(frame[address_column], "address")
    if country_column in frame.columns:
        frame[COUNTRY_NORM] = normalizer.series(frame[country_column], "country")

    return frame


if __name__ == "__main__":  # manual sanity check: python -m src.normalization
    import sys

    # Windows consoles default to cp1252 and cannot encode Devanagari/Kannada.
    # Force UTF-8 so this check is runnable on the dev machine too.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    _norm = Normalizer()
    _cases = [
        ("Orelee's Barbershop", "ASCII apostrophe"),
        ("B+ Retail Inc", "symbol"),
        ("Café Béque", "Latin accents folded"),
        ("LLC Moncada Léarning Center", "mixed case + accent"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "Devanagari - marks MUST survive"),
        ("कंस्ट्रक्शंस", "Devanagari conjuncts with virama"),
        ("Pvt. EFS Print Ventures Ltd.", "abbreviation dots"),
        ("ಶಿವಶಕ್ತಿ ವಿದ್ಯಾಲಯ", "Kannada - marks MUST survive"),
        ("heassociates.com", "domain-like name"),
        ("FOUNDATION EXCEL AGENCY PRIVATE  LIMITED", "double space"),
        ("H.No.16-11-23/37/A, 2Nd Floor", "address"),
        ("", "empty"),
        (None, "null"),
    ]
    print(f"{'input':<44} {'-> normalized':<46} note")
    print("-" * 120)
    for _raw, _note in _cases:
        print(f"{str(_raw)[:42]:<44} {_norm.name(_raw)[:44]:<46} {_note}")
    print()
    print("key('Orelee's Barbershop') =", repr(_norm.key("Orelee's Barbershop")))
