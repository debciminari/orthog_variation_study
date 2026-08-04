#!/usr/bin/env python3
"""
Generate distractors for CLSD
"""

import argparse
import json
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

DEFAULTS = dict(
    n_distractors=3,
    n_candidates=8, 
    mmr_lambda=0.7,      
    dedup_thr=0.99,          
    llm_model="Qwen/Qwen3-4B-Instruct-2507",
    llm_batch_size=16,       
    llm_max_new_tokens=0,    
    encoder="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    source_field="ita",      
)

#-----------------------------------
SYSTEM_MSG = ("You are a native Italian speaker."
              "You always reply with valid JSON and nothing else. "
              "All generated sentences MUST be written in Italian.")

USER_TEMPLATE = """You are given an Italian sentence. Generate exactly {k} Italian sentences to be used as DISTRACTORS in a translation test.

Each distractor MUST:
- be SUPERFICIALLY SIMILAR to the original (same or near words, similar length);
- be SEMANTICALLY DIFFERENT: it must mean something DIFFERENT, so that it is NOT a
  valid translation of the same source (change the subject, object, action, negation,
  quantity, or a named entity), while staying fluent and grammatical in Italian.

Do NOT paraphrase while keeping the meaning. Do NOT just swap synonyms.
IMPORTANT: every output sentence MUST be in Italian, never in English.

Original sentence:
"{sentence}"

Reply with ONLY a JSON array of {k} strings. No other text, no markdown.
Format: ["...", "...", "..."]"""


def _extract_json_array(text: str):
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return [str(x) for x in v]
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list):
                return [str(x) for x in obj]
        except Exception:
            pass
    
    open_br = text.find("[")
    if open_br != -1:
        strings = re.findall(r'"((?:[^"\\]|\\.)*)"', text[open_br:])
        cleaned = [s.encode().decode("unicode_escape") if "\\" in s else s
                   for s in strings]
        cleaned = [s.strip() for s in cleaned if s.strip()]
        if cleaned:
            return cleaned
    return []


class LLMGenerator:
    

    def __init__(self, cfg: dict):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                   BitsAndBytesConfig)

        self.cfg = cfg
        self.torch = torch

        print(f"Carico LLM 4-bit NF4: {cfg['llm_model']}", file=sys.stderr)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tok = AutoTokenizer.from_pretrained(cfg["llm_model"])
    
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["llm_model"],
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def _build_prompt(self, sentence: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_TEMPLATE.format(
                k=self.cfg["n_candidates"], sentence=sentence)},
        ]
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def generate(self, sentences):
        torch = self.torch
        results = []
        bs = self.cfg["llm_batch_size"]
        total = len(sentences)

    
        mnt = self.cfg["llm_max_new_tokens"]
        if mnt <= 0:
            mnt = 64 + 40 * self.cfg["n_candidates"]
        print(f"  max_new_tokens = {mnt} (candidati richiesti: {self.cfg['n_candidates']})",
              file=sys.stderr)

        for start in range(0, total, bs):
            batch = sentences[start:start + bs]
            prompts = [self._build_prompt(s) for s in batch]
            enc = self.tok(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=mnt,
                    do_sample=True, temperature=0.9, top_p=0.95,
                    pad_token_id=self.tok.pad_token_id,
                )
    
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = self.tok.batch_decode(gen, skip_special_tokens=True)
            for t in texts:
                results.append(_extract_json_array(t))
            done = min(start + bs, total)
            print(f"  LLM: {done}/{total} righe", file=sys.stderr)
        return results


# ---------------------------
ANTONYM_SWAPS = {
    r"\bsempre\b": "mai", r"\bmai\b": "sempre",
    r"\bpiù\b": "meno", r"\bmeno\b": "più",
    r"\btutti\b": "nessuno", r"\bnessuno\b": "tutti",
    r"\bmolto\b": "poco", r"\bpoco\b": "molto",
    r"\bgrande\b": "piccolo", r"\bpiccolo\b": "grande",
    r"\bprima\b": "dopo", r"\bdopo\b": "prima",
    r"\bsopra\b": "sotto", r"\bsotto\b": "sopra",
    r"\bvecchia\b": "nuova", r"\bnuova\b": "vecchia",
    r"\bvecchio\b": "nuovo", r"\bnuovo\b": "vecchio",
}


NEGATION_TRIGGERS = [
    (r"\b(si)\s+(dice|fa|vede|sente|può|deve)\b", r"non \1 \2"),
    (r"(?<!non )\b(è)\b", r"non \1"),
    (r"(?<!non )\b(ha)\b", r"non \1"),
    (r"(?<!non )\b(sono)\b", r"non \1"),
]


def perturb(sentence: str, max_out: int = 4):
    outs = []
    for pat, repl in ANTONYM_SWAPS.items():
        new = re.sub(pat, repl, sentence, count=1, flags=re.IGNORECASE)
        if new != sentence:
            outs.append(new)
    if "non " not in sentence.lower():
        for pat, repl in NEGATION_TRIGGERS:
            new = re.sub(pat, repl, sentence, count=1, flags=re.IGNORECASE)
            if new != sentence and "non non" not in new.lower():
                outs.append(new)
                break
    seen, uniq = set(), []
    for o in outs:
        k = o.lower().strip()
        if k not in seen:
            seen.add(k)
            uniq.append(o)
    return uniq[:max_out]


# -------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n\r\"'“”‘’.,;:!?()")
    return s


def clean_candidates(original: str, raw_cands):
    orig_key = _norm(original)
    seen = {orig_key}
    out = []
    for c in raw_cands:
        if not isinstance(c, str):
            continue
        c = c.strip()
        if len(c) < 2:
            continue
        k = _norm(c)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


#---------------------------
def mmr_select(orig_vec, cand_vecs, cand_texts, n, lam, dedup_thr):
 
    orig = orig_vec / (np.linalg.norm(orig_vec) + 1e-9)
    cands = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)
    rel = cands @ orig

 
    keep = [i for i in range(len(cand_texts)) if rel[i] < dedup_thr]
    if not keep:                   
        keep = list(range(len(cand_texts)))

    selected, pool = [], list(keep)
    while pool and len(selected) < n:
        best_i, best_score = None, -1e9
        for i in pool:
            div = max((float(cands[i] @ cands[j]) for j in selected), default=0.0)
            score = lam * float(rel[i]) - (1 - lam) * div
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        pool.remove(best_i)
    return [cand_texts[i] for i in selected], [float(rel[i]) for i in selected]


# Main
def main():
    ap = argparse.ArgumentParser(description="Genera distrattori CLSD (rgn,ita,title).")
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--n", type=int, default=DEFAULTS["n_distractors"])
    ap.add_argument("--candidates", type=int, default=DEFAULTS["n_candidates"])
    ap.add_argument("--lambda", dest="lam", type=float, default=DEFAULTS["mmr_lambda"])
    ap.add_argument("--dedup-thr", type=float, default=DEFAULTS["dedup_thr"],
                    help="soglia di sola identita': scarta candidati con cos>=soglia "
                         "rispetto all'originale (ri-enunciazioni, non distrattori)")
    ap.add_argument("--model", default=DEFAULTS["llm_model"], help="repo HF del modello LLM")
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["llm_batch_size"])
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULTS["llm_max_new_tokens"],
                    help="token max per la generazione LLM. 0 = auto (scala coi candidati)")
    ap.add_argument("--encoder", default=DEFAULTS["encoder"])
    ap.add_argument("--field", default=DEFAULTS["source_field"])
    ap.add_argument("--no-llm", action="store_true", help="solo perturbazioni (no GPU)")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update(dict(
        n_distractors=args.n, n_candidates=args.candidates, mmr_lambda=args.lam,
        dedup_thr=args.dedup_thr, llm_model=args.model,
        llm_batch_size=args.batch_size, llm_max_new_tokens=args.max_new_tokens,
        encoder=args.encoder, source_field=args.field,
    ))

    df = pd.read_csv(args.input_csv)
    if cfg["source_field"] not in df.columns:
        sys.exit(f"ERRORE: colonna '{cfg['source_field']}' assente. Colonne: {list(df.columns)}")
    sentences = df[cfg["source_field"]].fillna("").astype(str).tolist()
    print(f"Righe: {len(sentences)} | campo sorgente: '{cfg['source_field']}'", file=sys.stderr)

    
    if args.no_llm:
        llm_cands = [[] for _ in sentences]
    else:
        gen = LLMGenerator(cfg)
        llm_cands = gen.generate(sentences)
    
        import gc, torch
        del gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    all_cands = []
    for sent, lc in zip(sentences, llm_cands):
        merged = list(lc) + perturb(sent)
        all_cands.append(clean_candidates(sent, merged))

    
    from sentence_transformers import SentenceTransformer
    print(f"Carico encoder: {cfg['encoder']}", file=sys.stderr)
    model = SentenceTransformer(cfg["encoder"])

    flat_texts, offsets = [], []
    for sent, cands in zip(sentences, all_cands):
        start = len(flat_texts)
        flat_texts.append(sent)
        flat_texts.extend(cands)
        offsets.append((start, len(cands)))

    print(f"Encoding di {len(flat_texts)} testi...", file=sys.stderr)
    emb = model.encode(flat_texts, batch_size=64, convert_to_numpy=True,
                       show_progress_bar=True, normalize_embeddings=False)

    
    out_cols = {f"distractor_{i+1}": [] for i in range(cfg["n_distractors"])}
    sim_cols = {f"distractor_{i+1}_sim": [] for i in range(cfg["n_distractors"])}
    n_short = 0
    for (start, ncand), sent, cands in zip(offsets, sentences, all_cands):
        if ncand == 0:
            picks, sims = [], []
        else:
            picks, sims = mmr_select(
                emb[start], emb[start + 1: start + 1 + ncand], cands,
                n=cfg["n_distractors"], lam=cfg["mmr_lambda"],
                dedup_thr=cfg["dedup_thr"])
        if len(picks) < cfg["n_distractors"]:
            n_short += 1
        for i in range(cfg["n_distractors"]):
            out_cols[f"distractor_{i+1}"].append(picks[i] if i < len(picks) else "")
            sim_cols[f"distractor_{i+1}_sim"].append(round(sims[i], 4) if i < len(sims) else "")

    for k, v in out_cols.items():
        df[k] = v
    for k, v in sim_cols.items():
        df[k] = v

    df.to_csv(args.output_csv, index=False)
    print(f"\nFatto -> {args.output_csv}", file=sys.stderr)
    if n_short:
        print(f"ATTENZIONE: {n_short} righe con meno di {cfg['n_distractors']} distrattori. "
              f"Prova --candidates piu' alto o soglie sim piu' larghe.", file=sys.stderr)


if __name__ == "__main__":
    main()
