"""
Shared defaults for the qAIR pipeline.

Single source of truth for values that were previously hardcoded in
multiple places (e.g. embedding dim was 768 in training/dataset.py but
384 in evaluation/sample_inference.py after the encoder model changed).
Import from here instead of re-hardcoding.
"""

# v41: switched from all-mpnet-base-v2 (768-dim) to all-MiniLM-L6-v2
# (384-dim) and n_qubits 12 -> 6. The quantum circuit's simulation cost
# scales roughly O(n_qubits * 2^n_qubits) -- 12 qubits was the dominant
# per-epoch cost. Changing EMBEDDING_DIM invalidates any cache built
# under the old dim; training/dataset.py refuses to silently load a
# dim-mismatched cache (see its cache-loading dim check).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

N_QUBITS = 6
PERSISTENT_STEPS = 5

BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 5

# Batch size for offline hypothesis generation during cache building
# (models/generator.py's generate_batch). Independent of BATCH_SIZE
# above -- that one governs the training DataLoader, this one governs
# how many prompts are generated together by the LLM in one call.
GEN_BATCH_SIZE = 16

BASE_LR = 5e-4
NEW_LR = 2e-4
WEIGHT_DECAY = 2e-2

CACHE_DIR = "./cache"
CKPT_DIR = "./ckpt"
