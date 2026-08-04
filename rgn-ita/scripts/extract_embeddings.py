#!/usr/bin/env python3
"""
Extrac embeddings
"""
import argparse
import json
from pathlib import Path

import numpy as np

from intrinsic_common import load_glossary, all_forms


def resolve_checkpoint(run_dir: Path):
    summ = run_dir / "summary.json"
    if summ.exists():
        try:
            s = json.loads(summ.read_text(encoding="utf-8"))
            bc = s.get("best_checkpoint")
            if bc and bc != "final" and Path(bc).exists():
                return bc
        except Exception:
            pass
    if (run_dir / "config.json").exists() or (run_dir / "modules.json").exists():
        return str(run_dir)
    ckpt_dir = run_dir / "ckpt"
    if ckpt_dir.exists():
        cks = sorted(ckpt_dir.glob("checkpoint-*"),
                     key=lambda p: int(p.name.split("-")[-1]))
        if cks:
            return str(cks[-1])
    return None


def encode_all(model, forms, batch_size=64):
    embs = model.encode(forms, normalize_embeddings=True, batch_size=batch_size,
                        show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(embs, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Estrae embedding type-level per l'analisi intrinseca.")
    ap.add_argument("--runs-dir", default="out/runs")
    ap.add_argument("--glossary", required=True, help="glossario WIDE held-out (ita_word, rgn_var1..k)")
    ap.add_argument("--sets", default="l0,l1,l2,l3,l4,l5")
    ap.add_argument("--seeds", default="13,42,97,7,123")
    ap.add_argument("--out", default="embeddings.npz")
    ap.add_argument("--base-model", default="sentence-transformers/LaBSE",
                    help="modello base per il LaBSE zero-shot")
    ap.add_argument("--no-zeroshot", action="store_true",
                    help="non estrarre il baseline zero-shot")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    import gc, torch

    def _free_gpu():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sets = [s.strip() for s in args.sets.split(",")]
    seeds = [s.strip() for s in args.seeds.split(",")]

    lem2vars = load_glossary(args.glossary)
    forms = all_forms(lem2vars)
    print(f"[data] {len(lem2vars)} lemmi | {len(forms)} forme held-out da encodare")
    if not forms:
        raise SystemExit("Nessuna forma nel glossario: controlla il file / le colonne rgn_var*.")

    store = {}        
    d_model = None
    n_ok, n_missing = 0, 0
    missing_runs = []

    for st in sets:
        for sd in seeds:
            tag = f"{st}_seed{sd}"
            run_dir = Path(args.runs_dir) / tag
            ckpt = resolve_checkpoint(run_dir) if run_dir.exists() else None
            if not ckpt:
                n_missing += 1
                missing_runs.append(tag)
                print(f"  [skip] {tag}: checkpoint non trovato")
                continue
            model = SentenceTransformer(ckpt)
            mat = encode_all(model, forms, args.batch_size)
            if d_model is None:
                d_model = mat.shape[1]
            store[tag] = mat
            n_ok += 1
            print(f"  [ok]   {tag}: {mat.shape} da {ckpt}")
            del model
            _free_gpu()

    if not args.no_zeroshot:
        print(f"[zeroshot] carico {args.base_model}")
        zs = SentenceTransformer(args.base_model)
        store["zeroshot"] = encode_all(zs, forms, args.batch_size)
        if d_model is None:
            d_model = store["zeroshot"].shape[1]
        del zs
        _free_gpu()

    if not store:
        raise SystemExit("Nessun embedding estratto: nessun checkpoint valido trovato.")

    meta = {
        "base_model": args.base_model,
        "sets": sets, "seeds": seeds,
        "n_forms": len(forms), "dim": int(d_model),
        "n_runs_ok": n_ok, "n_runs_missing": n_missing,
        "missing_runs": missing_runs,
        "has_zeroshot": (not args.no_zeroshot),
    }

    np.savez_compressed(
        args.out,
        forms=np.array(forms, dtype=object),
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
        **store,
    )
    print(f"\n[out] {args.out}")
    print(f"      {n_ok} run estratti, {n_missing} mancanti"
          + (f" ({', '.join(missing_runs)})" if missing_runs else ""))
    if not args.no_zeroshot:
        print("      + zeroshot")


if __name__ == "__main__":
    main()
