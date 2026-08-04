# Repository Structure

This repository contains the experiments on orthographic variation for two language pairs, Romagnol-Italian and Low Saxon-German. Each language
pair lives in its own folder with an identical layout: a `scripts/`
folder holding the pipeline code and a `data/` folder holding the input data and additional resources created for the experiments.

```
.
├── rgn-ita/            # Romagnolo–Italian experiments
│   ├── scripts/        # pipeline code
│   └── data/           # datasets: train/dev/test splits, glossary, variants
│
└── nds-deu/            # Low Saxon–German experiments
    ├── scripts/        # same pipeline as rgn-ita, tuned for this pair
    └── data/           # datasets: train/dev/test splits, glossary, variants
```

The two `scripts/` folders share the same file names but are maintained
independently per language pair.

---

## rgn-ita/  (Romagnolo–Italian)

### rgn-ita/scripts/

| File | Description |
|------|-------------|
| `build_dataset.py` | Builds the parallel dataset and the orthographic-variation training sets (L0–L5). |
| `build_lexical.py` | Builds the additional resources. |
| `alignment.py` | Word alignment step. |
| `distance.py` | String-distance utilities (e.g., Levenshtein). |
| `generate_distractors.py` | Generates distractor candidates for the ranking task. |
| `finetune_labse.py` | Fine-tunes the LaBSE encoder on the training sets. |
| `extract_embeddings.py` | Extracts sentence embeddings. |
| `evaluate.py` | Aggregates CLSD runs: metrics (P@1, MRR). |
| `analyze_intrinsic.py` | Intrinsic evaluation of embeddings: intra-lemma cohesion, pair cosine, CKA. |
| `heldout.py` |  Held-out set construction handling. |
| `tokenization.py` | Tokenization utilities. |
| `clsd_common.py` | Shared helpers imported across scripts. |
| `clsd_config.py` | Shared configuration. |
| `intrinsic_common.py` | Shared helpers for the intrinsic evaluation scripts. |

### rgn-ita/data/

| File | Description |
|------|-------------|
| `train_L0.csv` … `train_L5.csv` | Training sets at increasing levels of orthographic variation. |
| `dev.csv` | Development split. |
| `test.csv` | Test split. |
| `dev_with_distractors.csv` | Development split with distractor candidates for the ranking task. |
| `test_with_distractors.csv` | Test split with distractor candidates. |
| `heldout_glossary.csv` | Held-out glossary of lemma variants for intrinsic evaluation. |
| `variants.csv` | Orthographic variant inventory. |

---

## nds-deu/  (Low Saxon–German)

### nds-deu/scripts/

Same pipeline as `rgn-ita/scripts/`, maintained independently for this
language pair. See the table above for per-script descriptions.

### nds-deu/data/

Same structure as `rgn-ita/data/`, maintained independently for this
language pair. See the table above for per-script descriptions.
