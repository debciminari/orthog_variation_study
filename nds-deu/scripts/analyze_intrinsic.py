#!/usr/bin/env python3
"""
Compute measures for intrinsic evaluate
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

from intrinsic_common import (load_glossary, all_forms, lemma_pairs,
                              linear_cka, cosine)


def load_emb(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    forms = list(z["forms"])
    meta = json.loads(str(z["meta"]))
    mats = {k: z[k] for k in z.files if k not in ("forms", "meta")}
    idx = {f: i for i, f in enumerate(forms)}
    return forms, idx, meta, mats


def emb_of(mats, tag, idx, form):
    return mats[tag][idx[form]]


# ---- 1. intra-lemma cohesion  -------------------------------------------------
def lemma_cohesion(mats, tag, idx, lem2vars):
    import itertools
    out = {}
    for lemma, vs in lem2vars.items():
        sims = [cosine(emb_of(mats, tag, idx, a), emb_of(mats, tag, idx, b))
                for a, b in itertools.combinations(vs, 2)]
        if sims:
            out[lemma] = float(np.mean(sims))
    return out


# ---- 2. cosine similarities - levensthein distance  ---------------------
def pairwise_by_bin(mats, tag, idx, pairs):
    by = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        c = cosine(emb_of(mats, tag, idx, p["form_a"]),
                   emb_of(mats, tag, idx, p["form_b"]))
        by[p["bin"]][p["lemma"]].append(c)
    out = {}
    for b in ("low", "med", "high"):
        lem_means = [np.mean(v) for v in by[b].values()]
        out[b] = (float(np.mean(lem_means)) if lem_means else float("nan"),
                  len(lem_means))
    return out


def overall_pair_cosine(mats, tag, idx, pairs):
    by = defaultdict(list)
    for p in pairs:
        by[p["lemma"]].append(
            cosine(emb_of(mats, tag, idx, p["form_a"]),
                   emb_of(mats, tag, idx, p["form_b"])))
    lem_means = [np.mean(v) for v in by.values()]
    return float(np.mean(lem_means)) if lem_means else float("nan")



# ---- inter-lemma cohesion -----------
def inter_lemma_cosine(mats, tag, idx, lem2vars, n_samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    lemmas = list(lem2vars)
    forms_by_lemma = {l: lem2vars[l] for l in lemmas}
    sims = []
    for _ in range(n_samples):
        la, lb = rng.choice(len(lemmas), size=2, replace=False)
        la, lb = lemmas[la], lemmas[lb]
        fa = forms_by_lemma[la][rng.integers(len(forms_by_lemma[la]))]
        fb = forms_by_lemma[lb][rng.integers(len(forms_by_lemma[lb]))]
        sims.append(cosine(emb_of(mats, tag, idx, fa), emb_of(mats, tag, idx, fb)))
    a = np.array(sims)
    return float(a.mean()), float(a.std())


def cohesion_mean(mats, tag, idx, lem2vars):
    return float(np.mean(list(lemma_cohesion(mats, tag, idx, lem2vars).values())))


def tokenization_report(forms, base_model, lem2vars):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model)

    def pieces(w):
        return tok.tokenize(w)

    n_multi = 0
    per_form = {}
    for f in forms:
        ps = pieces(f)
        per_form[f] = ps
        if len(ps) > 1:
            n_multi += 1

    overlaps = []
    for lemma, vs in lem2vars.items():
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                sa, sb = set(per_form[vs[i]]), set(per_form[vs[j]])
                if sa or sb:
                    overlaps.append(len(sa & sb) / len(sa | sb))
    return {
        "n_forms": len(forms),
        "n_multi_subword": n_multi,
        "pct_multi_subword": round(100 * n_multi / max(1, len(forms)), 1),
        "mean_subword_jaccard_within_lemma": round(float(np.mean(overlaps)), 3) if overlaps else None,
        "examples": {f: per_form[f] for f in forms[:8]},
    }


def agg_seeds(per_seed_values):
    a = np.array([v for v in per_seed_values if v == v]) 
    if len(a) == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--glossary", required=True,
                    help="glossario WIDE held-out (de_word, nds_var1..k)")
    ap.add_argument("--sets", default="l1,l2,l3,l4,l5")
    ap.add_argument("--seeds", default="13,42,97")
    ap.add_argument("--baseline", default="l1",
                    help="livello di riferimento per la CKA gradiente")
    ap.add_argument("--out-dir", default="analysis")
    ap.add_argument("--tokenizer-check", action="store_true",
                    help="controllo 2: analizza la tokenizzazione subword delle forme")
    ap.add_argument("--base-model", default="sentence-transformers/LaBSE",
                    help="modello base per il tokenizer (controllo 2)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sets = [s.strip() for s in args.sets.split(",")]
    seeds = [s.strip() for s in args.seeds.split(",")]

    lem2vars = load_glossary(args.glossary)
    forms = all_forms(lem2vars)
    pairs = lemma_pairs(lem2vars)
    _, idx, meta, mats = load_emb(args.emb)

    def tag(st, sd):
        return f"{st}_seed{sd}"

    available = lambda t: t in mats

    results = {"per_level": {}, "bins_per_level": {}, "config": {
        "sets": sets, "seeds": seeds, "baseline": args.baseline,
        "n_lemmi": len(lem2vars), "n_forms": len(forms), "n_pairs": len(pairs)}}

    print(f"[data] {len(lem2vars)} lemmi | {len(forms)} forme | {len(pairs)} coppie")
    print(f"\n{'livello':>8} {'coesione':>18} {'coseno_coppia':>18}")
    for st in sets:
        coh_seed, cos_seed, bins_seed = [], [], []
        for sd in seeds:
            t = tag(st, sd)
            if not available(t):
                continue
            coh = lemma_cohesion(mats, t, idx, lem2vars)
            coh_seed.append(float(np.mean(list(coh.values()))))
            cos_seed.append(overall_pair_cosine(mats, t, idx, pairs))
            bins_seed.append(pairwise_by_bin(mats, t, idx, pairs))

        coh_m, coh_s = agg_seeds(coh_seed)
        cos_m, cos_s = agg_seeds(cos_seed)
        results["per_level"][st] = {"cohesion_mean": coh_m, "cohesion_std": coh_s,
                                    "pair_cosine_mean": cos_m, "pair_cosine_std": cos_s,
                                    }

        binagg = {}
        for b in ("low", "med", "high"):
            vals = [bs[b][0] for bs in bins_seed if b in bs]
            ns = [bs[b][1] for bs in bins_seed if b in bs]
            m, s = agg_seeds(vals)
            binagg[b] = {"cosine_mean": m, "cosine_std": s,
                         "n_lemmi": int(np.median(ns)) if ns else 0}
        results["bins_per_level"][st] = binagg
        print(f"{st:>8} {coh_m:>10.4f}±{coh_s:<6.4f} {cos_m:>10.4f}±{cos_s:<6.4f}")

    # ---- 3 CKA ----------------------
    print(f"\n[CKA] rispetto a {args.baseline} (media sui seed) + within-level baseline")
    cka = {"vs_baseline": {}, "within_level": {}, "matrix": {}}

    def mat_for(t):
        return mats[t] 

    for st in sets:
        ts = [tag(st, sd) for sd in seeds if available(tag(st, sd))]
        vals = []
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                vals.append(linear_cka(mat_for(ts[i]), mat_for(ts[j])))
        m, s = agg_seeds(vals)
        cka["within_level"][st] = {"mean": m, "std": s, "n_pairs": len(vals)}

    base_ts = [tag(args.baseline, sd) for sd in seeds if available(tag(args.baseline, sd))]
    for st in sets:
        ts = [tag(st, sd) for sd in seeds if available(tag(st, sd))]
        vals = []
        for a in ts:
            for b in base_ts:
                if a == b:
                    continue
                vals.append(linear_cka(mat_for(a), mat_for(b)))
        m, s = agg_seeds(vals)
        cka["vs_baseline"][st] = {"mean": m, "std": s, "n_pairs": len(vals)}
        wl = cka["within_level"][st]["mean"]
        print(f"   {st} vs {args.baseline}: CKA={m:.4f}±{s:.4f}  "
              f"(within-{st} floor={wl:.4f})")

    for a in sets:
        cka["matrix"][a] = {}
        a_ts = [tag(a, sd) for sd in seeds if available(tag(a, sd))]
        for b in sets:
            b_ts = [tag(b, sd) for sd in seeds if available(tag(b, sd))]
            vals = [linear_cka(mat_for(x), mat_for(y))
                    for x in a_ts for y in b_ts if x != y]
            m, _ = agg_seeds(vals)
            cka["matrix"][a][b] = round(m, 4)

    results["cka"] = cka

    diag = {}
    print(f"\n{'='*60}\nCONTROLLI DIAGNOSTICI\n{'='*60}")

    if "zeroshot" in mats:
        zs_coh = cohesion_mean(mats, "zeroshot", idx, lem2vars)
        zs_inter = inter_lemma_cosine(mats, "zeroshot", idx, lem2vars)
        diag["zeroshot"] = {"cohesion": zs_coh,
                            "inter_lemma_cosine_mean": zs_inter[0],
                            "inter_lemma_cosine_std": zs_inter[1]}
        ft_cohs = [cohesion_mean(mats, tag(st, sd), idx, lem2vars)
                   for st in sets for sd in seeds if available(tag(st, sd))]
        ft_coh = float(np.mean(ft_cohs)) if ft_cohs else float("nan")
        print(f"\n[1] Zero-shot vs fine-tunato (coesione intra-lemma):")
        print(f"    LaBSE zero-shot          : {zs_coh:.4f}")
        print(f"    fine-tunato (media L1-L5): {ft_coh:.4f}")
        delta = ft_coh - zs_coh
        muove = "il FT muove le rappresentazioni" if abs(delta) > 0.02 \
            else "il FT NON muove le forme held-out (dominano i pesi pre-addestrati)"
        print(f"    delta = {delta:+.4f}  ->  {muove}")
        diag["finetuned_mean_cohesion"] = ft_coh
        diag["ft_minus_zeroshot"] = delta
    else:
        print("\n[1] zero-shot assente (rilancia extract_embeddings senza --no-zeroshot)")

    print(f"\n[3] Coesione intra-lemma vs pavimento inter-lemma (anisotropia):")
    inter_by_level = {}
    for st in sets:
        t = next((tag(st, sd) for sd in seeds if available(tag(st, sd))), None)
        if not t:
            continue
        intra = cohesion_mean(mats, t, idx, lem2vars)
        inter_m, inter_s = inter_lemma_cosine(mats, t, idx, lem2vars)
        inter_by_level[st] = {"intra": intra, "inter_mean": inter_m,
                              "inter_std": inter_s, "gap": intra - inter_m}
        verdict = "(discrimina)" if intra - inter_m > 0.05 \
            else "(SATURATA: metrica non discrimina)"
        print(f"    {st}: intra={intra:.4f}  inter={inter_m:.4f}  "
              f"gap={intra-inter_m:+.4f}  {verdict}")
    diag["inter_lemma_by_level"] = inter_by_level

    if args.tokenizer_check:
        print(f"\n[2] Tokenizzazione delle forme held-out:")
        tokr = tokenization_report(forms, args.base_model, lem2vars)
        jac = tokr['mean_subword_jaccard_within_lemma']
        print(f"    forme spezzate in >1 subword: {tokr['n_multi_subword']}/{tokr['n_forms']} "
              f"({tokr['pct_multi_subword']}%)")
        print(f"    Jaccard subword medio tra varianti stesso lemma: {jac}")
        if jac is not None and jac >= 0.5:
            print(f"    -> Jaccard ALTO: varianti quasi identiche a livello subword; "
                  f"la coesione potrebbe essere un artefatto di tokenizzazione, non "
                  f"apprendimento.")
        else:
            print(f"    -> Jaccard BASSO: le varianti NON condividono i subword, hanno "
                  f"tokenizzazioni diverse; la coesione osservata e' quindi genuina "
                  f"(il modello avvicina forme che il tokenizer tratta diversamente), "
                  f"non un artefatto.")
        diag["tokenization"] = tokr
    else:
        print(f"\n[2] tokenizzazione: salta (--tokenizer-check per attivare)")

    results["diagnostics"] = diag

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_dir / "per_level.csv", "w", encoding="utf-8") as f:
        f.write("level,cohesion_mean,cohesion_std,pair_cosine_mean,pair_cosine_std,"
                "cka_vs_baseline,cka_within\n")
        for st in sets:
            pl = results["per_level"].get(st, {})
            f.write(f"{st},{pl.get('cohesion_mean','')},{pl.get('cohesion_std','')},"
                    f"{pl.get('pair_cosine_mean','')},{pl.get('pair_cosine_std','')},"
                    f"{cka['vs_baseline'].get(st,{}).get('mean','')},"
                    f"{cka['within_level'].get(st,{}).get('mean','')}\n")

    with open(out_dir / "bins_per_level.csv", "w", encoding="utf-8") as f:
        f.write("level,bin,cosine_mean,cosine_std,n_lemmi\n")
        for st in sets:
            for b in ("low", "med", "high"):
                bb = results["bins_per_level"].get(st, {}).get(b, {})
                f.write(f"{st},{b},{bb.get('cosine_mean','')},"
                        f"{bb.get('cosine_std','')},{bb.get('n_lemmi','')}\n")

    print(f"\n[out] {out_dir}/results.json, per_level.csv, bins_per_level.csv")


if __name__ == "__main__":
    main()
