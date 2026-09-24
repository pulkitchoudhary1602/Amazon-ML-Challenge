"""Pair features for the matching model.  **NOT IMPLEMENTED - milestone 2.**

This module is intentionally a contract, not an implementation. Milestone 1 is
the infrastructure plus the exact-name blocker; building features before the
candidate set has usable recall would be premature, because the measured ceiling
of exact-name blocking is far too low (see README) and any model trained on it
would learn to rank a candidate set that is missing most true matches.

Planned interface
-----------------
``extract_features(config, candidate_frame, ...) -> pandas.DataFrame``

The function receives a chunk of candidate pairs (``source1_entity_id``,
``matched_entity_id``, ``source``, ``blockers``) plus the normalized S1/S2/S3
tables, and returns one row of features per candidate pair. It must stay
chunkable: 22.8T pairs are never materialized, and even a 100M-pair candidate set
must be processed in slices.

Planned feature groups
----------------------
Lexical (CPU, cheap - do these first, they carry most of the signal):
    * exact match on ``name_norm`` / ``name_key``
    * token Jaccard and token-set ratio on the name
    * ``rapidfuzz`` ratios: ``ratio``, ``partial_ratio``, ``token_sort_ratio``
    * character 3-gram Jaccard
    * length ratio, token-count difference
    * first-token (usually the distinctive part of a business name) equality

Address (CPU):
    * the same lexical family on ``address_norm``
    * postal-code equality, city/state token overlap
    * NOTE: 168,967 S2 and 175,916 S3 records have no address at all, so address
      features must be nullable and the model must handle their absence. Address
      can never be the sole blocking signal for the same reason.

Categorical: source (S2/S3), country match, agreement across blockers.

Semantic (GPU when available - resolve the device with
``utils.resolve_device_from_config(config)``; never hardcode "cuda"):
    * cosine similarity of multilingual sentence embeddings for the name field.
      The corpus is mixed English/Devanagari/Kannada, so a multilingual encoder
      is required; embeddings are computed once per record and cached, not per
      pair, otherwise the cost is candidate-volume x encode-cost.
      Encoding is the clearest GPU win in the project: ~12.6M texts, one-time,
      embarrassingly parallel. Batch by ``compute.gpu_batch_size`` and store at
      ``compute.embedding_dtype`` (float16 roughly halves storage for a
      negligible effect on cosine ranking). A CPU path must exist and produce
      equivalent output - slower, not different.
    * Optional cross-encoder score on the top-N candidates only. This is the one
      stage where a GPU is close to mandatory in practice, but it runs on a small
      "uncertain" candidate band, so its cost stays bounded.

Provenance: the ``blockers`` column survives the union in ``blocking.py`` so
"which blocker produced this pair" is available as a feature and as an audit
signal for pruning blockers that never contribute true matches.

Constraints to respect when implementing
----------------------------------------
* External data / internet augmentation is NOT allowed: no pretrained-knowledge
  lookups against the open web, no geocoding services.
* Models used in the final solution must satisfy the challenge's license and
  parameter-count constraints; record the chosen encoder and its size here.
* Keep the feature computation deterministic and chunk-local so HPC jobs can be
  resumed.
* Every accelerator-dependent feature must degrade to CPU. When torch or CUDA is
  absent, ``resolve_device_from_config`` returns ``"cpu"`` and the feature is
  either computed on CPU or skipped - never a hard failure.
"""

from __future__ import annotations

from typing import Any

NOT_IMPLEMENTED_MESSAGE = (
    "src/features.py is a milestone-2 stub: pair features are not implemented yet.\n"
    "Milestone 1 delivers the pipeline plus the exact-name blocker, whose recall\n"
    "ceiling is too low to train a useful matcher on. Add token / character n-gram /\n"
    "dense blockers first, then implement this module."
)


def extract_features(*args: Any, **kwargs: Any):
    """Planned: candidate pairs -> feature matrix. See module docstring."""
    raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


FEATURE_GROUPS = ("lexical", "address", "categorical", "semantic")
