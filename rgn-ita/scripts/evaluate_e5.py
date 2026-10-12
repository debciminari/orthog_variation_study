"""
Aggregation of CLSD runs for multilingual-e5-base.

Identical to the LaBSE aggregation script (metrics, paired bootstrap,
entropy vs P@1 trend test, error analysis). Differences, marked [E5]:
  1. zero-shot baseline = multilingual-e5-base (not LaBSE), with the
     "query: " prefix as in fine-tuning; model and prefix are imported from
     finetune_e5.py, so they are guaranteed to be identical
  2. the baseline is named "e5_zeroshot"
  3. run tags (e5_l0_seed13 ...) are read as l0..l5, so that
     the default entropy map works and the tables can be
     compared row by row with the LaBSE ones
  4. defaults --runs-dir out/runs_e5 and --out-dir out/e5, so that
     the LaBSE outputs in out/ are not overwritten
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import clsd_common as C
from finetune_e5 import E5_MODEL, e5 as e5_prefix   # [E5] same config as fine-tuning

ZS = "e5_zeroshot"   # [E5] baseline name
RUN_PREFIX = "e5_"   # [E5] tag prefix of the e5 runs


def load_cands_csv(path, src_col="rgn", tgt_col="ita"):

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        dcols = sorted(
            [c for c in cols if c.startswith("distractor_") and not c.endswith("_sim")],
            key=lambda c: int(c.split("_")[1]),
        )
        items = []
        for i, r in enumerate(reader):
            src = (r.get(src_col) or "").strip()
            true = (r.get(tgt_col) or "").strip()
            if not src or not true:
                continue
            distr = [(r.get(c) or "").strip() for c in dcols]
            distr = [d for d in distr if d and d != true]
            if not distr:
                continue
            items.append({"id": i, "src": src, "true": true, "distractors": distr})
    return items


def metrics_from_preds(recs):
    ranks = np.array([r["rank_true"] for r in recs], dtype=float)
    return {
        "n": len(recs),
        "p_at_1": float((ranks == 0).mean()),
        "mrr": float((1.0 / (ranks + 1.0)).mean()),
        "mean_rank": float(ranks.mean() + 1),
    }


def mean_std(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def paired_bootstrap(hits_a, hits_b, n_boot=10000, seed=42):

    rng = np.random.default_rng(seed)
    a, b = np.asarray(hits_a, float), np.asarray(hits_b, float)
    obs = a.mean() - b.mean()
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return round(float(obs), 4), round(float(min(p, 1.0)), 4)


def baseline_preds_e5(test_cands_csv):
    """[E5] Zero-shot baseline of multilingual-e5-base, with the e5 prefix
    applied to src and candidates, exactly as in clsd_p_at_1 of
    finetune_e5.py."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(E5_MODEL)
    recs = []
    for it in load_cands_csv(test_cands_csv):
        cands = [it["true"]] + it["distractors"]
        e_src = model.encode(e5_prefix(it["src"]), normalize_embeddings=True)
        e_c = model.encode([e5_prefix(c) for c in cands], normalize_embeddings=True)
        sims = e_c @ e_src
        rank = int((sims > sims[0]).sum())
        recs.append({
            "id": it.get("id"), "src": it["src"], "true": it["true"],
            "distractors": it["distractors"], "sim_true": float(sims[0]),
            "sim_distr": [float(x) for x in sims[1:]], "rank_true": rank,
            "hit": int(rank == 0), "flagged": it.get("flagged", False),
        })
    return recs


def _corr_stats(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    out = {"n": n}
    try:
        from scipy import stats as ss
        pr = ss.pearsonr(x, y)
        sr = ss.spearmanr(x, y)
        lr = ss.linregress(x, y)
        out["pearson_r"] = round(float(pr[0]), 4)
        out["pearson_p"] = round(float(pr[1]), 4)
        out["spearman_rho"] = round(float(sr.correlation), 4)
        out["spearman_p"] = round(float(sr.pvalue), 4)
        out["slope"] = round(float(lr.slope), 4)
        out["intercept"] = round(float(lr.intercept), 4)
        tcrit = float(ss.t.ppf(0.975, n - 2))
        out["slope_ci95"] = [round(lr.slope - tcrit * lr.stderr, 4),
                             round(lr.slope + tcrit * lr.stderr, 4)]
        out["backend"] = "scipy"
    except Exception:
        raise NotImplementedError("numpy fallback for _corr_stats not yet implemented")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Aggrega run CLSD di multilingual-e5: metriche, significativita', error analysis.")
    ap.add_argument("--runs-dir", default="out/runs_e5")          # [E5]
    ap.add_argument("--test-cands", required=True,
                    help="test candidates frozen in CSV wide (per il baseline)")
    ap.add_argument("--out-dir", default="out/e5")                # [E5]
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--entropy-map", default=None,
                    help="mappa set->entropia H per il test di trend, es. "
                         "'l1=0.0,l2=0.3269,l3=0.5676,l0=0.7096,l4=0.7414,l5=0.7687'. "
                         "Se assente, usa i valori di default dalla tabella del paper.")
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    DEFAULT_ENTROPY = {
        "l1": 0.0000, "l2": 0.3269, "l3": 0.5676,
        "l0": 0.7096, "l4": 0.7414, "l5": 0.7687,
    }
    if args.entropy_map:
        entropy_H = {}
        for tok in args.entropy_map.split(","):
            k, v = tok.split("=")
            entropy_H[k.strip()] = float(v)
    else:
        entropy_H = DEFAULT_ENTROPY

    run_dirs = [p for p in Path(args.runs_dir).glob("*_seed*")
                if (p / "test_predictions.jsonl").exists()]
    by_set = defaultdict(list)
    for rd in run_dirs:
        set_tag = rd.name.rsplit("_seed", 1)[0]
        if set_tag.startswith(RUN_PREFIX):                        # [E5] e5_l0 -> l0
            set_tag = set_tag[len(RUN_PREFIX):]
        by_set[set_tag].append(rd)

    table = []
    seed_hit_by_set = {}
    p1_per_seed_by_set = {}
    for set_tag, dirs in sorted(by_set.items()):
        per_seed = {"p_at_1": [], "mrr": [], "mean_rank": []}
        hit_vectors = []
        for rd in sorted(dirs):
            recs = sorted(C.read_jsonl(rd / "test_predictions.jsonl"), key=lambda r: r["id"])
            m = metrics_from_preds(recs)
            for k in per_seed:
                per_seed[k].append(m[k])
            hit_vectors.append(np.array([r["hit"] for r in recs], float))
        p1_per_seed_by_set[set_tag] = list(per_seed["p_at_1"])
        row = {"run": set_tag, "n_seeds": len(dirs), "n": metrics_from_preds(
                    C.read_jsonl(sorted(dirs)[0] / "test_predictions.jsonl"))["n"]}
        for k in per_seed:
            mu, sd = mean_std(per_seed[k])
            row[k + "_mean"] = round(mu, 4)
            row[k + "_std"] = round(sd, 4)
        train_secs = []
        for rd in sorted(dirs):
            sp = rd / "summary.json"
            if sp.exists():
                s = json.loads(sp.read_text(encoding="utf-8"))
                if "timing" in s:
                    train_secs.append(s["timing"]["train_seconds"])
        if train_secs:
            tmu, tsd = mean_std(train_secs)
            row["train_seconds_mean"] = round(tmu, 1)
            row["train_seconds_std"] = round(tsd, 1)
        table.append(row)
        seed_hit_by_set[set_tag] = np.mean(hit_vectors, axis=0)

    if not args.no_baseline:
        brecs = sorted(baseline_preds_e5(args.test_cands), key=lambda r: r["id"])
        C.write_jsonl(Path(args.out_dir) / "zeroshot_predictions.jsonl", brecs)  # [E5]
        bm = metrics_from_preds(brecs)
        table.append({"run": ZS, "n_seeds": 1, "n": bm["n"],
                      "p_at_1_mean": round(bm["p_at_1"], 4), "p_at_1_std": 0.0,
                      "mrr_mean": round(bm["mrr"], 4), "mrr_std": 0.0,
                      "mean_rank_mean": round(bm["mean_rank"], 4), "mean_rank_std": 0.0})
        seed_hit_by_set[ZS] = np.array([r["hit"] for r in brecs], float)

    table.sort(key=lambda x: -x["p_at_1_mean"])
    n_seed_max = max((len(d) for d in by_set.values()), default=0)
    print(f"\n=== [multilingual-e5-base] Translation Ranking (CLSD) — metriche Test "
          f"(media +/- stdev su {n_seed_max} seed) ===")
    print(f"{'set':<16}{'n':>5}{'P@1':>18}{'MRR':>16}{'meanRank':>16}{'train_s':>14}")
    for r in table:
        p = f"{r['p_at_1_mean']:.3f}±{r['p_at_1_std']:.3f}"
        mr = f"{r['mrr_mean']:.3f}±{r['mrr_std']:.3f}"
        rk = f"{r['mean_rank_mean']:.2f}±{r['mean_rank_std']:.2f}"
        ts = (f"{r['train_seconds_mean']:.0f}±{r['train_seconds_std']:.0f}"
              if "train_seconds_mean" in r else "-")
        print(f"{r['run']:<16}{r['n']:>5}{p:>18}{mr:>16}{rk:>16}{ts:>14}")
    C.dump_json(Path(args.out_dir) / "metrics_table.json", table)

    first = sorted(by_set[sorted(by_set)[0]])[0] if by_set else None
    if first and (first / "summary.json").exists():
        s0 = json.loads((first / "summary.json").read_text(encoding="utf-8"))
        if "environment" in s0:
            C.dump_json(Path(args.out_dir) / "environment.json", s0["environment"])

    ref = ZS if ZS in seed_hit_by_set else \
          max(seed_hit_by_set, key=lambda k: seed_hit_by_set[k].mean())
    sig = []
    for name, hits in seed_hit_by_set.items():
        if name == ref:
            continue
        obs, p = paired_bootstrap(hits, seed_hit_by_set[ref])
        sig.append({"set": name, "vs": ref, "delta_p1": obs, "p_value": p})
    print(f"\n=== Bootstrap appaiato vs {ref} (10k resample, hit mediati sui seed) ===")
    for s in sorted(sig, key=lambda x: -x["delta_p1"]):
        star = " *" if s["p_value"] < 0.05 else ""
        print(f"{s['set']:<16} deltaP@1={s['delta_p1']:+.4f}  p={s['p_value']:.4f}{star}")
    C.dump_json(Path(args.out_dir) / "significance.json", sig)

    set_names = sorted(
        [k for k in seed_hit_by_set if k != ZS],
        key=lambda k: -seed_hit_by_set[k].mean(),
    )
    pairwise = []
    for i in range(len(set_names)):
        for j in range(i + 1, len(set_names)):
            a, b = set_names[i], set_names[j]
            obs, p = paired_bootstrap(seed_hit_by_set[a], seed_hit_by_set[b])
            pairwise.append({"a": a, "b": b, "delta_p1": obs, "p_value": p})
    if pairwise:
        print("\n=== Bootstrap appaiato SET vs SET (10k resample, hit mediati sui seed) ===")
        n_sig = sum(1 for pw in pairwise if pw["p_value"] < 0.05)
        print(f"coppie testate: {len(pairwise)} | significative (p<0.05): {n_sig}")
        for pw in sorted(pairwise, key=lambda x: x["p_value"]):
            star = " *" if pw["p_value"] < 0.05 else ""
            print(f"  {pw['a']:<6} vs {pw['b']:<6}  "
                  f"deltaP@1={pw['delta_p1']:+.4f}  p={pw['p_value']:.4f}{star}")
        print("\n  matrice p-value (righe vs colonne):")
        header = "        " + "".join(f"{s:>8}" for s in set_names)
        print(header)
        pmat = {(pw["a"], pw["b"]): pw["p_value"] for pw in pairwise}
        for ri, a in enumerate(set_names):
            cells = []
            for ci, b in enumerate(set_names):
                if a == b:
                    cells.append(f"{'—':>8}")
                else:
                    key = (a, b) if (a, b) in pmat else (b, a)
                    cells.append(f"{pmat[key]:>8.3f}")
            print(f"  {a:<6}" + "".join(cells))
        if n_sig == 0:
            print("\n  => nessuna coppia di set differisce significativamente: "
                  "i divari di P@1 tra set sono compatibili col rumore.")
    C.dump_json(Path(args.out_dir) / "significance_pairwise.json", pairwise)

    have_H = {s: entropy_H[s] for s in p1_per_seed_by_set if s in entropy_H}
    missing = [s for s in p1_per_seed_by_set if s not in entropy_H]
    if len(have_H) >= 3:
        xs, ys = [], []
        for s, H in have_H.items():
            for p1 in p1_per_seed_by_set[s]:
                xs.append(H); ys.append(p1)
        trend_perseed = _corr_stats(xs, ys)
        xm = [have_H[s] for s in have_H]
        ym = [float(np.mean(p1_per_seed_by_set[s])) for s in have_H]
        trend_mean = _corr_stats(xm, ym)

        print("\n=== Test di trend: entropia ortografica H vs P@1 ===")
        print("  ordine per H:", ", ".join(
            f"{s}(H={have_H[s]:.3f})" for s in sorted(have_H, key=have_H.get)))
        if missing:
            print(f"  NB set senza H, esclusi dal trend: {missing}")

        def _fmt(t, label):
            line = f"  [{label}] n={t['n']}  Pearson r={t['pearson_r']}"
            if "pearson_p" in t:
                line += f" (p={t['pearson_p']})"
            line += f"  Spearman rho={t['spearman_rho']}"
            if "spearman_p" in t:
                line += f" (p={t['spearman_p']})"
            print(line)
            sl = f"       slope={t['slope']} P@1 per bit/class"
            if "slope_ci95" in t:
                sl += f"  CI95={t['slope_ci95']}"
            print(sl)

        _fmt(trend_perseed, f"per-seed {len(xs)} pt")
        _fmt(trend_mean, f"medie {len(xm)} pt")
        if trend_perseed["backend"].startswith("numpy"):
            print(f"  ({trend_perseed['backend']})")

        sp = trend_perseed.get("spearman_p")
        if sp is not None:
            if sp < 0.05 and trend_perseed["spearman_rho"] > 0:
                verdict = ("trend positivo significativo: P@1 cresce con l'entropia "
                           "ortografica del training.")
            elif sp < 0.05 and trend_perseed["spearman_rho"] < 0:
                verdict = "trend NEGATIVO significativo: piu' entropia -> P@1 piu' basso."
            else:
                verdict = ("nessun trend monotono significativo tra entropia e P@1 "
                           f"(Spearman p={sp}): l'effetto e' compatibile col rumore.")
            print("  =>", verdict)

        C.dump_json(Path(args.out_dir) / "trend_entropy.json",
                    {"entropy_H": have_H,
                     "p1_per_seed": {s: p1_per_seed_by_set[s] for s in have_H},
                     "per_seed": trend_perseed, "mean": trend_mean})
    else:
        print("\n[trend] entropia nota per <3 set: test di trend saltato. "
              "Passa --entropy-map se i tag non sono l0..l5.")

    ea_path = Path(args.out_dir) / "error_analysis.csv"
    with open(ea_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["set", "id", "src", "true", "seeds_failed", "winning_distractor",
                    "win_sim", "true_sim", "sim_gap", "winner_jaccard_to_true", "flagged"])
        for set_tag, dirs in sorted(by_set.items()):
            maj = math.ceil(len(dirs) / 2)
            per_id = defaultdict(list)
            for rd in sorted(dirs):
                for r in C.read_jsonl(rd / "test_predictions.jsonl"):
                    per_id[r["id"]].append(r)
            for rid, recs in per_id.items():
                fails = [r for r in recs if r["rank_true"] != 0]
                if len(fails) < maj:
                    continue
                rec = fails[0]
                sd = rec["sim_distr"]
                wi = int(np.argmax(sd))
                winner = rec["distractors"][wi] if wi < len(rec["distractors"]) else ""
                w.writerow([set_tag, rid, rec["src"], rec["true"], len(fails), winner,
                            round(sd[wi], 4), round(rec["sim_true"], 4),
                            round(rec["sim_true"] - sd[wi], 4),
                            C.jaccard_words(rec["true"], winner) if winner else "",
                            rec.get("flagged", False)])
    print(f"\n[scritto] {ea_path}  (item falliti dalla maggioranza dei seed, per set)")


if __name__ == "__main__":
    main()
