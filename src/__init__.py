"""Business Entity Resolution for the ML challenge.

Pipeline stages, each runnable from the command line:

    scripts/prepare_data.py        raw TSV -> normalized TSV (+ split column)
    scripts/build_indexes.py       normalized TSV -> inverted indexes
    scripts/generate_candidates.py indexes + S1 -> candidate pairs
    scripts/train_model.py         candidate pairs + features -> match model
    scripts/predict.py             candidates + model -> matching_results.tsv

Heavy stages are chunked and streamed so peak memory is governed by
``io.chunksize`` rather than by dataset size. The architecture is CPU-first:
every stage runs without a GPU, and the accelerator-beneficial ones (embedding
generation, dense retrieval, batched embedding similarity, transformer
re-ranking) resolve a device through ``utils.resolve_device`` rather than
hardcoding one. See README "Compute architecture".

Submodules are deliberately NOT imported here: ``import src`` stays cheap and
free of pandas/torch, so CLI scripts and tests only pay for what they use.
"""

__version__ = "0.1.0"
