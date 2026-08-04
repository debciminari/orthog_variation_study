#!/usr/bin/env python3
DEFAULT_LABSE = "sentence-transformers/LaBSE"

SRC_COL = "nds"
TGT_COL = "de"

N_DISTRACTORS = 4          
OVERGEN_K = 8              
RETRY_BUDGET = 3           
SEED = 42                  
WORD_EDIT_MIN = 1
WORD_EDIT_MAX = 3

BASE_MODEL = DEFAULT_LABSE
EPOCHS = 3
BATCH_SIZE = 16            
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_SEQ_LEN = 128          
FINETUNE_SEEDS = [13, 42, 97, 7, 123]  
