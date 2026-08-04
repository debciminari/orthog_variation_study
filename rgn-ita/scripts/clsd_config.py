#!/usr/bin/env python3
"""
clsd_config.py
==============
Unica fonte di verita' per le costanti dell'esperimento CLSD. I tre script le
importano da qui, cosi' non possono divergere tra loro.

Due gruppi:
  * GENERAZIONE / FILTRO   (usati da generate_distractors.py + apply_threshold.py)
  * FINE-TUNING            (usati da finetune_labse.py; NON devono variare tra i
                            5 set ne' tra i seed, altrimenti la P@1 non e'
                            comparabile)
"""

# --- backend di generazione ---
DEFAULT_MODEL = "qwen3.5:9b"               # tag Ollama
DEFAULT_API_BASE = "http://localhost:11434"
DEFAULT_LABSE = "sentence-transformers/LaBSE"

# --- costruzione candidate set (frozen su dev/test) ---
N_DISTRACTORS = 4          # distrattori per item nel set finale
OVERGEN_K = 8              # candidati richiesti per item a ogni chiamata di generazione
RETRY_BUDGET = 3           # tentativi di rigenerazione per item corto
SEED = 42                  # seed per campionamento/riproducibilita' della costruzione dati

# --- filtro dei distrattori (edit-distance a livello di PAROLA, valore assoluto) ---
# Teniamo i distrattori che modificano tra WORD_EDIT_MIN e WORD_EDIT_MAX token
# rispetto al target: >=1 (altrimenti e' il target) e <= una soglia piccola (oltre,
# il distrattore diverge troppo per restare superficialmente simile). Interpretabile
# come "il distrattore cambia da 1 a N parole"; robusto anche su frasi corte.
WORD_EDIT_MIN = 1
WORD_EDIT_MAX = 3

# --- fine-tuning (fissi su TUTTI i run: solo il SET e il SEED variano) ---
BASE_MODEL = DEFAULT_LABSE
EPOCHS = 3
BATCH_SIZE = 16            # corpus piccolo; in-batch negatives => N-1 negativi effettivi
LR = 2e-5
WARMUP_RATIO = 0.1
MAX_SEQ_LEN = 128          # target mediano ~5 token; 128 e' abbondante
FINETUNE_SEEDS = [13, 42, 97, 7, 123]   # 5 seed/set per media +/- stdev
