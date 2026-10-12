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

## Encoders: LaBSE and multilingual-e5-base

The experiments are run with two encoders, LaBSE and multilingual-e5-base.
Scripts with the `_e5` suffix are the multilingual-e5-base counterparts of
the LaBSE scripts. They use the same data, distractors, hyperparameters
(from `clsd_config.py`), loss, batch sampler, checkpoint selection and
metrics. The only differences are those required by the model, and they
are marked with `[E5]` in the code:

- the base model is `intfloat/multilingual-e5-base`;
- the `"query: "` prefix is added to both sides, in training and in
  evaluation, as recommended by the e5 model card for symmetric tasks;
- the pooling is checked to be mean pooling, the pooling e5 was trained with;
- outputs go to separate folders (`out/runs_e5`, `out/e5`,
  `embeddings_e5.npz`), so that LaBSE results are not overwritten.

The model name and the prefix are defined once in `finetune_e5.py` and
imported by the other `_e5` scripts, so they are guaranteed to be identical
across the pipeline.

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
| `finetune_e5.py` | Fine-tunes the multilingual-e5-base encoder on the training sets (counterpart of `finetune_labse.py`). |
| `extract_embeddings.py` | Extracts sentence embeddings (LaBSE). |
| `extract_embeddings_e5.py` | Extracts type-level embeddings of the held-out forms for intrinsic evaluation (multilingual-e5-base). Keys in the output `.npz` match the LaBSE ones, so `analyze_intrinsic.py` works unchanged. |
| `evaluate.py` | Aggregates CLSD runs: metrics (P@1, MRR). |
| `evaluate_e5.py` | TODO: add description. |
| `aggregate_e5.py` | Aggregates CLSD runs for multilingual-e5-base: metrics (P@1, MRR), paired bootstrap, entropy vs P@1 trend test, error analysis. Includes the multilingual-e5-base zero-shot baseline. |
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

For `aggregate_e5.py`, pass the column names of this pair with
`--src-col nds --tgt-col deu` (the defaults are `rgn` and `ita`).

### nds-deu/data/

Same structure as `rgn-ita/data/`, maintained independently for this
language pair. See the table above for per-file descriptions.

## License

- **Code** (all `.py` files under `nds-deu/scripts/` and `rgn-ita/scripts/`)
  is released under the MIT License — see [`LICENSE`](LICENSE).
- **Data** (all files under `nds-deu/data/` and `rgn-ita/data/`)
  is released under CC BY 4.0 — see [`LICENSE-DATA`](LICENSE-DATA).
