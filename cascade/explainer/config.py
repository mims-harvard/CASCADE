"""
Configuration for donor-level and cell-level downstream prediction analysis
(Methods 9.8, 9.11) built on top of frozen CASCADE embeddings.

Paths are resolved relative to CASCADE_DATA_ROOT (see cascade/training/train_ddp.py)
unless overridden by CLI arguments.
"""
import os
from pathlib import Path

CASCADE_DATA_ROOT = Path(os.environ.get("CASCADE_DATA_ROOT", "/path/to/DATASET"))
CASCADE_CKPT_ROOT = Path(os.environ.get("CASCADE_CKPT_ROOT", "/path/to/CASCADE-checkpoints"))

# ============================================================================
# EMBEDDING EXTRACTION CONFIGURATION
# ============================================================================

MERGED_CONTEXTS = 'disease_cell_type'  # AUTISM uses 'disease_cell_type_tissue' instead

TEMPERATURE = 0.1
D_MODEL = 384
NHEAD = 6
NLAYERS = 12
PAD_TOKEN = "<pad>"
NCLASS = 2
CELL_EMB_STYLE = 'avg-pool'
DIM_EMBEDDING = 384
DROPOUT = 0.1
BATCH = 1024
# Must match the checkpoint's training-time architecture so load_state_dict succeeds
# on the context-projector weights. This default (False) matches the serial
# EmbeddingDataLoader extraction path in embeddings.py, which reads cell_emb directly
# (no projector unpacking needed). The parallel extraction path in
# get_embeddings_parallel.py trains/extracts against context_specific_projections=True
# checkpoints and unpacks (xcs, cell_emb) explicitly instead of using this default -
# embedding extraction always pulls the shared-encoder cell_emb (Methods 9.7), never
# the projected xcs.
CONTEXT_SPECIFIC_PROJECTIONS = False
DA = True

# ============================================================================
# TASK CONFIGURATION
# ============================================================================

DONOR_LEVEL_TASKS = ['disease']
CELL_LEVEL_TASKS = ['cell_type', 'S.Score', 'G2M.Score', 'Phase', 'state', 'lineage', 'pseudotime', 'pseudotime_ranks']

# Huntington's disease donor-level tasks (Methods 9.1: Vonsattel grade, CAG repeat length)
HH_DONOR_LEVEL_TASKS = ['VS_Grade', 'CAG_1', 'CAG_2', 'CAG_max', 'CAG_min', 'CAG_diff']

CLASSIFICATION_TASKS = ['disease', 'cell_type', 'Phase', 'state', 'lineage']
REGRESSION_TASKS = ['S.Score', 'G2M.Score', 'pseudotime', 'pseudotime_ranks'] + HH_DONOR_LEVEL_TASKS

# HH dataset uses a different attention architecture than the other datasets
HH_ATTENTION_HEADS = 4
HH_ATTENTION_LAYERS = 6
HH_ATTENTION_DROPOUT = 0.1

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

MAX_EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 0.001

DEBUG_MODE = False
DEBUG_MAX_BATCHES = 5

LOAD_EXISTING_EMBEDDINGS = False

ATTENTION_HEADS = 4
ATTENTION_LAYERS = 2
ATTENTION_DROPOUT = 0.1

EARLY_STOPPING_PATIENCE = 10
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4

# ============================================================================
# CONTEXT COMBINATIONS
# ============================================================================

CONTEXT_COMBINATIONS = [
    ['ALL'],  # No filtering - use all data as baseline
    ['CELLS'],
    ['TISSUE'],
    ['DISEASE'],
    ['CELLS', 'TISSUE'],
    ['CELLS', 'TISSUE', 'DISEASE'],
]

# Datasets whose embeddings are context-agnostic (single view per cell, no per-context augmentation)
CONTEXT_AGNOSTIC_DATASETS = {'HH'}

# ============================================================================
# PATH RESOLUTION
# ============================================================================

def get_default_paths(dataset_name: str):
    """Default paths for a dataset's tokenized data, model checkpoint, and tokenizer."""
    dataset_dir = CASCADE_DATA_ROOT / dataset_name

    if dataset_name.upper() == 'HH':
        # HH uses pre-extracted, context-agnostic embeddings; no transformer/tokenizer needed.
        return {
            'output_dir': dataset_dir,
            'model_path': None,
            'token_dictionary_file': None,
            'embeddings_path': dataset_dir / "embeddings_all_labels_complete_contextagnostic.pkl",
        }

    return {
        'output_dir': dataset_dir,
        'model_path': None,  # must be supplied by the caller (--model-path)
        'token_dictionary_file': dataset_dir / f"tokenizer_dictionary_{dataset_name}.pkl",
    }


# Datasets pretrained/extracted with all three contexts (disease, cell_type, tissue)
# rather than the 2-context default (see run_explain_new_parallel.py's DATASET_CONFIGS).
THREE_CONTEXT_DATASETS = {'AUTISM', 'SEATTLE'}


def get_collator_path(dataset_name: str):
    """Default collator pickle path for a dataset. Returns None for datasets that
    load tokenized data directly (e.g. via --list_data) instead of a single collator file."""
    if dataset_name.upper() == 'HH':
        return None
    merged_contexts = 'disease_cell_type_tissue' if dataset_name.upper() in THREE_CONTEXT_DATASETS else MERGED_CONTEXTS
    return CASCADE_DATA_ROOT / dataset_name / f"collator_{merged_contexts}_kempner_metadata_final_corrected.pkl"
