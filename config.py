"""
Shared defaults for the qAIR pipeline.

Single source of truth for values that were previously hardcoded in
multiple places (e.g. embedding dim was 768 in training/dataset.py but
384 in evaluation/sample_inference.py after the encoder model changed).
Import from here instead of re-hardcoding.
"""

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768

N_QUBITS = 12
PERSISTENT_STEPS = 5

BATCH_SIZE = 8
EPOCHS = 20
PATIENCE = 5

BASE_LR = 5e-4
NEW_LR = 2e-4
WEIGHT_DECAY = 2e-2

CACHE_DIR = "./cache"
CKPT_DIR = "./ckpt"
