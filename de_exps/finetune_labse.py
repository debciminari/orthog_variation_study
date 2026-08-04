#!/usr/bin/env python3
"""
Finetune labse
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import clsd_common as C
from clsd_config import (
    BASE_MODEL, BATCH_SIZE, EPOCHS, LR, MAX_SEQ_LEN, WARMUP_RATIO,
    SRC_COL, TGT_COL,
)


def set_all_seeds(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def capture_env() -> dict:
    import platform
    import sys

    env = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
    }
    try:
        import datasets
        import sentence_transformers
        import transformers

        env["transformers"] = transformers.__version__
        env["sentence_transformers"] = sentence_transformers.__version__
        env["datasets"] = datasets.__version__
    except Exception:  
        pass
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["gpu_count"] = torch.cuda.device_count()
        env["gpu_total_mem_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    return env


def make_loss_callback():
    from transformers import TrainerCallback

    class LossHistory(TrainerCallback):
        def __init__(self):
            self.history = []

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.history.append({
                    "step": state.global_step,
                    "epoch": round(state.epoch, 3) if state.epoch else None,
                    "loss": logs["loss"],
                    "learning_rate": logs.get("learning_rate"),
                })

    return LossHistory()


def load_pairs_csv(path: str, src_col: str, tgt_col: str):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [(r[src_col].strip(), r[tgt_col].strip()) for r in csv.DictReader(f)]
    return [(s, t) for s, t in rows if s and t]


def load_cands_csv(path: str, src_col: str, tgt_col: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        dcols = sorted(
            [c for c in cols if c.startswith("distractor_") and not c.endswith("_sim")],
            key=lambda c: int(c.split("_")[1]),
        )
        items, n_short, n_skip_nodistr, n_skip_nodata = [], 0, 0, 0
        for i, r in enumerate(reader):
            src = (r.get(src_col) or "").strip()
            true = (r.get(tgt_col) or "").strip()
            if not src or not true:
                n_skip_nodata += 1
                continue
            distr = [(r.get(c) or "").strip() for c in dcols]
            distr = [d for d in distr if d and d != true]  
            if not distr:                     
                n_skip_nodistr += 1
                continue
            if len(distr) < 3: 
                n_short += 1
            items.append({"id": i, "src": src, "true": true, "distractors": distr})
    return items, n_short, n_skip_nodistr, n_skip_nodata


def clsd_p_at_1(model, items):
    correct, recs = 0, []
    for it in items:
        cands = [it["true"]] + it["distractors"]
        e_src = model.encode(it["src"], normalize_embeddings=True)
        e_c = model.encode(cands, normalize_embeddings=True)
        sims = e_c @ e_src
        rank_true = int((sims > sims[0]).sum())  
        hit = int(rank_true == 0)
        correct += hit
        recs.append({
            "id": it.get("id"), "src": it["src"], "true": it["true"],
            "distractors": it["distractors"], "sim_true": float(sims[0]),
            "sim_distr": [float(x) for x in sims[1:]],
            "n_distractors": len(it["distractors"]),
            "rank_true": rank_true, "hit": hit,
        })
    p1 = correct / max(1, len(items))
    return p1, recs


def run(args):
    SEED = args.seed
    set_all_seeds(SEED)
    t_start = time.time()
    env = capture_env()

    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import (
        BatchSamplers, SentenceTransformerTrainingArguments,
    )

    dev_items, dev_short, dev_nodistr, dev_nodata = load_cands_csv(
        args.dev_cands, args.src_col, args.tgt_col)
    test_items, test_short, test_nodistr, test_nodata = load_cands_csv(
        args.test_cands, args.src_col, args.tgt_col)
    print(f"[cands] dev : {len(dev_items)} item validi | "
          f"{dev_short} con <3 distrattori (tenuti) | "
          f"{dev_nodistr} ignorati (0 distrattori) | "
          f"{dev_nodata} ignorati (src/true mancanti)")
    print(f"[cands] test: {len(test_items)} item validi | "
          f"{test_short} con <3 distrattori (tenuti) | "
          f"{test_nodistr} ignorati (0 distrattori) | "
          f"{test_nodata} ignorati (src/true mancanti)")

    pairs = load_pairs_csv(args.train, args.src_col, args.tgt_col)

    train_ds = Dataset.from_dict({
        args.src_col: [p[0] for p in pairs],
        args.tgt_col: [p[1] for p in pairs],
    })

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = MAX_SEQ_LEN
    loss = MultipleNegativesRankingLoss(model)

    run_tag = f"{args.run_name}_seed{SEED}"
    run_out = Path(args.out_dir) / run_tag
    run_out.mkdir(parents=True, exist_ok=True)

    targs = SentenceTransformerTrainingArguments(
        output_dir=str(run_out / "ckpt"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        fp16=torch.cuda.is_available(),
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,   
        eval_strategy="no",       
        save_strategy="epoch",
        save_total_limit=EPOCHS,
        logging_steps=50,
        seed=SEED,
        run_name=args.run_name,
        report_to=[],
    )

    loss_cb = make_loss_callback()
    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=train_ds, loss=loss,
        callbacks=[loss_cb],
    )

   
    with C.Timer("train") as tr:
        trainer.train()
    train_seconds = tr.seconds

    (run_out / "loss_curve.json").write_text(
        json.dumps(loss_cb.history, indent=2, ensure_ascii=False), encoding="utf-8"
    )

   
    with C.Timer("checkpoint-selection (dev)") as te:
        ckpts = sorted(
            Path(run_out / "ckpt").glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        best = {"dev_p1": -1.0, "path": None}
        dev_curve = []
        for c in ckpts:
            m = SentenceTransformer(str(c))
            p1, _ = clsd_p_at_1(m, dev_items)
            dev_curve.append({"checkpoint": c.name, "dev_p1": p1})
            if p1 > best["dev_p1"]:
                best = {"dev_p1": p1, "path": str(c)}

        p1_final, _ = clsd_p_at_1(model, dev_items)
        dev_curve.append({"checkpoint": "final", "dev_p1": p1_final})
        if p1_final >= best["dev_p1"]:
            best = {"dev_p1": p1_final, "path": "final"}
    eval_seconds = te.seconds

    best_model = model if best["path"] == "final" else SentenceTransformer(best["path"])

    with C.Timer("test inference") as tt:
        test_p1, test_recs = clsd_p_at_1(best_model, test_items)
    test_infer_seconds = tt.seconds
    with C.Timer("dev inference") as td:
        dev_p1, _ = clsd_p_at_1(best_model, dev_items)
    dev_infer_seconds = td.seconds

    total_seconds = time.time() - t_start

    n_test = len(test_recs)
    n_dev = len(dev_items)
    timing = {
        "train_seconds": round(train_seconds, 2),
        "train_seconds_per_epoch": round(train_seconds / max(1, EPOCHS), 2),
        "eval_seconds": round(eval_seconds, 2),  
        "dev_inference_seconds": round(dev_infer_seconds, 3),
        "dev_inference_ms_per_item": round(1000 * dev_infer_seconds / max(1, n_dev), 2),
        "test_inference_seconds": round(test_infer_seconds, 3),
        "test_inference_ms_per_item": round(1000 * test_infer_seconds / max(1, n_test), 2),
        "total_seconds": round(total_seconds, 2),
    }

    C.write_jsonl(run_out / "test_predictions.jsonl", test_recs)
    summary = {
        "run_tag": run_tag, "set": args.run_name, "seed": SEED,
        "train_file": args.train, "n_train_pairs": len(pairs), "n_test_items": n_test,
        "n_dev_items": n_dev,
        "dev_items_short": dev_short, "test_items_short": test_short,
        "dev_p1": round(dev_p1, 4), "test_p1": round(test_p1, 4),
        "best_checkpoint": best["path"], "dev_curve": dev_curve,
        "timing": timing, "environment": env,
        "config": {
            "base": BASE_MODEL, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr": LR, "warmup_ratio": WARMUP_RATIO, "seed": SEED,
            "max_seq_len": MAX_SEQ_LEN, "loss": "MultipleNegativesRankingLoss",
        },
    }
    C.dump_json(run_out / "summary.json", summary)

    summary["_loss_curve"] = loss_cb.history
    print(json.dumps({
        "set": args.run_name, "seed": SEED, "dev_p1": summary["dev_p1"],
        "test_p1": summary["test_p1"], "train_s": timing["train_seconds"],
    }, ensure_ascii=False))
    return summary


def append_txt_log(log_path, summary: dict):
    import datetime as _dt

    s = summary
    t = s["timing"]
    env = s["environment"]
    cfg = s["config"]
    lines = []
    W = 78
    lines.append("=" * W)
    lines.append(f"RUN: {s['run_tag']}    (scritto: {_dt.datetime.now().isoformat(timespec='seconds')})")
    lines.append("=" * W)

    lines.append("-- risultati --")
    lines.append(f"  set            : {s['set']}")
    lines.append(f"  seed           : {s['seed']}")
    lines.append(f"  train file     : {s['train_file']}")
    lines.append(f"  n train pairs  : {s['n_train_pairs']}")
    lines.append(f"  n dev items    : {s['n_dev_items']}  (con <3 distrattori: {s['dev_items_short']})")
    lines.append(f"  n test items   : {s['n_test_items']}  (con <3 distrattori: {s['test_items_short']})")
    lines.append(f"  DEV  P@1       : {s['dev_p1']}")
    lines.append(f"  TEST P@1       : {s['test_p1']}")
    lines.append(f"  best checkpoint: {s['best_checkpoint']}")

    lines.append("-- timing (secondi) --")
    lines.append(f"  train total    : {t['train_seconds']}  ({t['train_seconds_per_epoch']}/epoca)")
    lines.append(f"  checkpoint sel : {t['eval_seconds']}")
    lines.append(f"  dev inference  : {t['dev_inference_seconds']}  ({t['dev_inference_ms_per_item']} ms/item)")
    lines.append(f"  test inference : {t['test_inference_seconds']}  ({t['test_inference_ms_per_item']} ms/item)")
    lines.append(f"  TOTALE         : {t['total_seconds']}")

    lines.append("-- config --")
    for k, v in cfg.items():
        lines.append(f"  {k:14s}: {v}")

    lines.append("-- environment --")
    for k, v in env.items():
        lines.append(f"  {k:14s}: {v}")

    lines.append("-- dev P@1 per checkpoint --")
    for row in s.get("dev_curve", []):
        lines.append(f"  {row['checkpoint']:16s}: {round(row['dev_p1'], 4)}")

    lc = s.get("_loss_curve", [])
    lines.append(f"-- training loss ({len(lc)} punti loggati) --")
    if lc:
        lines.append(f"  loss finale    : {lc[-1].get('loss')}  (step {lc[-1].get('step')}, epoca {lc[-1].get('epoch')})")
        for row in lc:
            lines.append(f"    step {row['step']:>6}  epoca {row['epoch']}  loss {row['loss']}  lr {row.get('learning_rate')}")
    lines.append("")  
    lines.append("")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_parser():
    ap = argparse.ArgumentParser(
        description="Fine-tune LaBSE + CLSD P@1 su set frozen. "
                    "Modalita' singola (--train + --seed) o multi (--all-sets + --seeds).")

    ap.add_argument("--dev-cands", required=True,
                    help="CSV wide dev (nds, de, distractor_1..N)")
    ap.add_argument("--test-cands", required=True,
                    help="CSV wide test (nds, de, distractor_1..N)")
    ap.add_argument("--out-dir", default="out/runs")
    ap.add_argument("--src-col", default=SRC_COL)
    ap.add_argument("--tgt-col", default=TGT_COL)
    ap.add_argument("--log-file", default=None,
                    help="log txt cumulativo (append). Default: <out-dir>/runs_log.txt")

    # ------
    ap.add_argument("--train", help="[singola] path del training set")
    ap.add_argument("--run-name", help="[singola] tag del set, es. l1")
    ap.add_argument("--seed", type=int, help="[singola] seed del run")

    # ------
    ap.add_argument("--all-sets", nargs="+", metavar="TAG=PATH",
                    help="[multi] elenco set come tag=path, es. l1=train_l1.csv l2=train_l2.csv")
    ap.add_argument("--seeds", nargs="+", type=int,
                    help="[multi] lista di seed, es. --seeds 13 42 97")
    return ap


def main():
    from types import SimpleNamespace

    args = build_parser().parse_args()
    log_file = args.log_file or str(Path(args.out_dir) / "runs_log.txt")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    multi = bool(args.all_sets)
    single = bool(args.train and args.run_name is not None and args.seed is not None)

    if multi:
        sets = []
        for spec in args.all_sets:
            if "=" not in spec:
                raise SystemExit(f"--all-sets vuole tag=path, ricevuto: {spec!r}")
            tag, path = spec.split("=", 1)
            sets.append((tag, path))
        seeds = args.seeds or [13, 42, 97]
        total = len(sets) * len(seeds)
        print(f"[multi] {len(sets)} set x {len(seeds)} seed = {total} run")
        done = 0
        board = []
        for tag, path in sets:
            for sd in seeds:
                done += 1
                print(f"\n===== RUN {done}/{total}: set={tag} seed={sd} =====")
                a = SimpleNamespace(
                    train=path, run_name=tag, seed=sd,
                    dev_cands=args.dev_cands, test_cands=args.test_cands,
                    out_dir=args.out_dir, src_col=args.src_col, tgt_col=args.tgt_col,
                )
                summary = run(a)
                append_txt_log(log_file, summary)
                board.append((tag, sd, summary["dev_p1"], summary["test_p1"]))

        print("\n===== RIEPILOGO =====")
        for tag, sd, dp1, tp1 in board:
            print(f"  {tag:6s} seed{sd:<4d}  dev={dp1}  test={tp1}")
        print(f"log cumulativo -> {log_file}")

    elif single:
        summary = run(args)
        append_txt_log(log_file, summary)
        print(f"log cumulativo -> {log_file}")

    else:
        raise SystemExit(
            "Specifica una modalita':\n"
            "  singola: --train F.csv --run-name TAG --seed N\n"
            "  multi  : --all-sets l1=train_l1.csv l2=train_l2.csv ... --seeds 13 42 97")


if __name__ == "__main__":
    main()
