"""Match classifier / ranker.  **NOT IMPLEMENTED - milestone 2.**

Contract for the stage that turns candidate pairs into match decisions.

Why the metric shapes the design
--------------------------------
The challenge scores F0.5, computed **per S1 entity** and then macro-averaged.
Two consequences drive every choice here:

1. **Precision is worth 4x recall.** ``beta = 0.5`` means a false positive costs
   four times a false negative. With 7,638,365 true matches spread over
   22.8 trillion possible pairs, and 123,247 S1 entities that have no match at
   all, a classifier with mediocre precision destroys the macro average: one
   spurious match on an empty-GT entity takes that entity's F0.5 to zero.

2. **Per-entity aggregation, not per-pair.** Averaging over entities stops a
   single S1 with 11 matches from dominating, and it means a per-entity decision
   policy (e.g. a threshold tuned per candidate-count bucket) is legitimate and
   often better than one global threshold.

Planned interface
-----------------
``train(config, features_path, ground_truth, ...) -> ModelBundle``
``predict(config, ModelBundle, candidate_chunks) -> iterator of decisions``

Recommended progression (measure at each step, do not skip ahead):

1. **Threshold on a lexical score** - a single feature (e.g. token-set ratio)
   with a threshold tuned on the val split. Gives a real F0.5 number and a floor
   to beat. Cheap, CPU-only, interpretable.
2. **Gradient-boosted trees** on the lexical + address features. Still CPU-only;
   this is expected to be the bulk of the score. Handle missing addresses
   natively.
3. **Add semantic features** from multilingual embeddings, GPU when available.
4. **Re-ranking with a cross-encoder** on the top candidates only, if it still
   pays for itself after step 3.

The candidate-cap interaction
-----------------------------
``blocking.max_candidates_per_source`` controls how many candidates the matcher
sees per S1. This is a recall/precision trade-off, not a tuning detail: capping
too aggressively silently deletes true matches before the model ever sees them,
which shows up as a hard ceiling on per-entity F0.5. Choose the cap from
``recall_at_k_file_order`` once blockers produce ranked scores, and re-check the
blocking report after every blocker change.

Constraints
-----------
* CPU must remain sufficient for the whole pipeline; GPU is an optimization.
* Model license and parameter-count constraints apply to whatever is finally
  submitted - record them here before shipping.
* No external data or internet augmentation.
"""

from __future__ import annotations

from typing import Any

NOT_IMPLEMENTED_MESSAGE = (
    "src/matching_model.py is a milestone-2 stub: no match classifier is implemented yet.\n"
    "Suggested first step (cheap, CPU-only, gives a real F0.5 baseline): threshold the\n"
    "token-set ratio feature on the validation split."
)


def train(*args: Any, **kwargs: Any):
    """Planned: fit the match classifier on the training split."""
    raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


def predict(*args: Any, **kwargs: Any):
    """Planned: score candidate pairs and emit match decisions."""
    raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


def tune_threshold(*args: Any, **kwargs: Any):
    """Planned: choose the decision threshold by macro F0.5 on the val split.

    Optimizing the *macro per-entity* F0.5, not accuracy or pair-level F1, is the
    point - the two do not agree on this dataset.
    """
    raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)
