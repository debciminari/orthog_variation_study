#!/usr/bin/env python3
import csv, random, math
from pathlib import Path
from collections import defaultdict, Counter
from tokenization import tokenize, normalize_text, WORD_RE
from heldout import load_heldout_list, load_variants_with_freq

SEED = 42
OUT = Path("./out"); OUT.mkdir(parents=True, exist_ok=True)
TRAIN = Path("./train.csv"); MIN_LEN = 3


def variants_to_groups(variants_path):
    gl = load_variants_with_freq(variants_path, sim_threshold=None)
    return {de: {form: freq for form, freq in cells} for de, cells in gl.items()}


def build_maps(groups, heldout_lemmas=None):
    heldout_lemmas = heldout_lemmas or set()
    class_members = {}                 
    best = {}
    n_skipped = 0
    for de, ff in groups.items():
        if de in heldout_lemmas:       
            n_skipped += 1
            continue
        forms = sorted(ff, key=lambda w: (-ff[w], w)); canon = forms[0]
        if len(canon) < MIN_LEN:
            continue
        good = [v for v in forms[1:] if len(v) >= MIN_LEN]
        if not good:
            continue
        cl = canon.lower()
        members = [cl] + [v.lower() for v in good]
        class_members[cl] = members
        for orig in [canon] + good:
            m = orig.lower()
            freq = ff[orig]
            cur = best.get(m)
            if cur is None or (freq, ) > (cur[0], ) or (freq == cur[0] and cl < cur[1]):
                best[m] = (freq, cl)
    to_canon = {m: cl for m, (_, cl) in best.items()}
    cleaned = {}
    for cl, members in class_members.items():
        keep = [m for m in members if to_canon.get(m) == cl]
        if cl in keep and len(keep) >= 2:
            cleaned[cl] = keep
    valid_forms = {m for members in cleaned.values() for m in members}
    to_canon = {m: cl for m, cl in to_canon.items() if m in valid_forms}
    if heldout_lemmas:
        print(f"held-out: {n_skipped} lemmi esclusi dalla diversificazione")
    return to_canon, cleaned

def _case(orig,repl):
    if orig.isupper(): return repl.upper()
    if orig[:1].isupper(): return repl[:1].upper()+repl[1:]
    return repl

def _key(token):
    return normalize_text(token)

# ---------------------------------------------------------------------------
# Verify entropy
# ---------------------------------------------------------------------------
def build_form2class(groups, heldout_lemmas=None):
    to_canon, _ = build_maps(groups, heldout_lemmas=heldout_lemmas)
    return dict(to_canon)

def shannon(counter):
    tot=sum(counter.values())
    if tot==0: return 0.0
    return -sum((n/tot)*math.log2(n/tot) for n in counter.values())

def observe_classes(ds, form2class):
    obs=defaultdict(Counter)
    for row in ds:
        for w in tokenize(row["nds"]):
            if w in form2class: obs[form2class[w]][w]+=1
    return obs

def entropy_metrics(ds, form2class, grid=None):
    obs=observe_classes(ds, form2class)
    if grid is None:
        classes=[c for c in obs if sum(obs[c].values())>0]
    else:
        classes=sorted(grid)
    if not classes: return 0.0, 0.0, 0
    H=sum(shannon(obs[c]) for c in classes)/len(classes)
    S=sum(len(obs[c]) for c in classes)/len(classes)
    return H, S, len(classes)

def run_entropy_report(train_path, out_dir, form2class):
    orig=list(csv.DictReader(open(train_path,encoding="utf-8")))
    orig_obs=observe_classes(orig, form2class)
    grid={c for c in orig_obs if sum(orig_obs[c].values())>0}

    print(f"\n{'='*52}\nVERIFICA ENTROPIA (bit/classe)\n{'='*52}")
    print(f"griglia fissa: {len(grid)} classi attive in ORIG\n")
    print(f"{'set':<6}{'H (bit/classe)':>16}{'#forme (supporto)':>20}{'#classi':>10}")

    report_rows=[]  
    H,S,nc=entropy_metrics(orig, form2class, grid=grid)
    print(f"{'ORIG':<6}{H:>16.4f}{S:>20.3f}{nc:>10}")
    report_rows.append(("ORIG", H, S, nc))
    for lv in range(1,6):
        d=list(csv.DictReader(open(out_dir/f"train_L{lv}.csv",encoding="utf-8")))
        H,S,nc=entropy_metrics(d, form2class, grid=grid)
        print(f"{'L'+str(lv):<6}{H:>16.4f}{S:>20.3f}{nc:>10}")
        report_rows.append((f"L{lv}", H, S, nc))

    csv_path=out_dir/"entropy_report.csv"
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["set","H_bit_per_class","support_mean_forms","n_classes_grid"])
        for name,H,S,nc in report_rows:
            w.writerow([name, f"{H:.6f}", f"{S:.6f}", nc])
    print(f"\nreport entropia -> {csv_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Genera i 5 training set da variants.csv e verifica l'entropia")
    ap.add_argument("--variants", required=True,
                    help="glossario variants.csv (con freq) estratto da build_dataset.py")
    ap.add_argument("--train", default=str(TRAIN), help="train.csv da trasformare")
    ap.add_argument("--heldout", default=None,
                    help="heldout_lemmas.txt: lemmi esclusi dalla diversificazione (A1)")
    ap.add_argument("--no-entropy", action="store_true",
                    help="salta la verifica di entropia finale")
    args = ap.parse_args()

    heldout_lemmas = load_heldout_list(args.heldout) if args.heldout else set()
    out_dir = OUT; out_dir.mkdir(parents=True, exist_ok=True)

    groups = variants_to_groups(args.variants)
    to_canon, class_members = build_maps(groups, heldout_lemmas)
    print(f"lemmi >1 variante: {len(groups)} | forme mappate: {len(to_canon)} | classi: {len(class_members)}\n")
    rows=list(csv.DictReader(open(args.train,encoding="utf-8"))); n=len(rows)

    def norm(m):
        t=m.group(0); low=_key(t)
        return _case(t, to_canon[low]) if low in to_canon else t

    def make_diversifier():
        counter=defaultdict(int)
        def div(m):
            t=m.group(0); low=_key(t)
            if low not in to_canon: return t
            cl=to_canon[low]; members=class_members[cl]
            pool=[f for f in members if f!=low] or members
            pick=pool[counter[cl] % len(pool)]
            counter[cl]+=1
            return _case(t, pick)
        return div

    frac={1:0.0,2:.25,3:.5,4:.75,5:1.0}
    rnd=random.Random(SEED); order=list(range(n)); rnd.shuffle(order)
    for lv in range(1,6):
        k=round(frac[lv]*n); div_set=set(order[:k])
        diversify=make_diversifier(); out=[]
        for i,row in enumerate(rows):
            fn = diversify if i in div_set else norm
            out.append({"nds":WORD_RE.sub(fn,row["nds"]),"de":row["de"],"title":row.get("title","")})
        with open(out_dir/f"train_L{lv}.csv","w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["nds","de","title"]); w.writeheader(); w.writerows(out)
        ch=sum(1 for a,b in zip(rows,out) if a["nds"]!=b["nds"])
        print(f"L{lv}: {len(out)} righe | cambiate {ch:4d} ({100*ch/n:.1f}%) | frac_div={frac[lv]}")

    if not args.no_entropy:
        form2class=build_form2class(groups)
        run_entropy_report(args.train, out_dir, form2class)

if __name__=="__main__": main()
