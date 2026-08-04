"""
Held-out glossaries 
"""
import csv
from collections import defaultdict

from distance import normalised_similarity, SIM_THRESHOLD
from tokenization import tokenize

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


def extract_variants(src_sentences, trg_sentences, out_path,
                     sim_threshold=SIM_THRESHOLD):
    from alignment import word_alignment

    links = word_alignment(src_sentences, trg_sentences)

    groups = defaultdict(lambda: defaultdict(int))   
    for rt, it, ls in zip(src_sentences, trg_sentences, links):
        for ri, ii in ls:
            if ri < len(rt) and ii < len(it):
                groups[it[ii]][rt[ri]] += 1
    groups = {k: dict(v) for k, v in groups.items() if len(v) > 1}

    rows = []
    for ita, ff in groups.items():
        forms = sorted(ff, key=lambda w: (-ff[w], w))  
        canon = forms[0]
        kept = [canon] + [v for v in forms[1:]
                          if normalised_similarity(v, canon) >= sim_threshold]
        if len(kept) >= 2:
            rows.append((ita, [(v, ff[v]) for v in kept]))

    ncols = max((len(cells) for _, cells in rows), default=0)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ita_word"] + [f"rgn_var{i+1}" for i in range(ncols)])
        for ita, cells in rows:
            w.writerow([ita] + [f"{form}:{freq}" for form, freq in cells])
    return out_path, len(rows)


def _norm_apo(s):
    return str(s).replace("\u2019", "'").replace("\u2018", "'")


def _parse_cell(cell):
    """Accetta sia 'forma' sia 'forma:freq'. Ritorna (forma_lower, freq_or_None)."""
    cell = _norm_apo(cell).strip()
    if not cell:
        return None, None
    if ":" in cell:
        form, _, fr = cell.rpartition(":")
        try:
            return form.strip().lower(), int(fr)
        except ValueError:
            return cell.lower(), None   
    return cell.lower(), None


def load_filtered_variants(variants_path, sim_threshold=SIM_THRESHOLD):
    lemma2vars, var2lemmas = {}, defaultdict(set)
    with open(variants_path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            if not row or not row[0].strip():
                continue
            lemma = row[0].strip()
            forms = []
            for c in row[1:]:
                form, _ = _parse_cell(c)
                if form:
                    forms.append(form)
            if len(forms) < 2:
                continue
            canon = forms[0]
            kept = [canon] + [v for v in forms[1:]
                              if normalised_similarity(v, canon) >= sim_threshold]
            kept = list(dict.fromkeys(kept))
            if len(kept) >= 2:
                lemma2vars[lemma] = set(kept)
                for v in kept:
                    var2lemmas[v].add(lemma)
    return lemma2vars, var2lemmas


def load_variants_with_freq(variants_path, sim_threshold=SIM_THRESHOLD):
    out = {}
    with open(variants_path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            if not row or not row[0].strip():
                continue
            lemma = row[0].strip()
            cells = []
            for c in row[1:]:
                form, fr = _parse_cell(c)
                if form:
                    cells.append((form, fr if fr is not None else 0))
            if len(cells) < 2:
                continue
            canon = cells[0][0]
            if sim_threshold is None:
                kept = cells
            else:
                kept = [cells[0]] + [(v, fr) for v, fr in cells[1:]
                                     if normalised_similarity(v, canon) >= sim_threshold]
            seen, dd = set(), []
            for v, fr in kept:
                if v not in seen:
                    seen.add(v); dd.append((v, fr))
            if len(dd) >= 2:
                out[lemma] = dd
    return out


def _cost_on_rows(rgn_series, var2lemmas):
    hits = defaultdict(int)
    for sent in rgn_series:
        touched = set()
        for tok in set(tokenize(sent)):
            if tok in var2lemmas:
                touched.update(var2lemmas[tok])
        for lem in touched:
            hits[lem] += 1
    return hits


def _variants_in_test(test_rgn, var2lemmas):
    test_tokens = set()
    for sent in test_rgn:
        test_tokens |= set(tokenize(sent))
    per_lemma = defaultdict(set)
    for tok in test_tokens:
        for lem in var2lemmas.get(tok, ()):
            per_lemma[lem].add(tok)
    return {lem: len(vs) for lem, vs in per_lemma.items()}


def select_heldout(lemma2vars, var2lemmas, train_rgn, val_rgn, test_rgn, k=100,
                   spacy_model="it_core_news_sm"):
    import spacy
    nlp = spacy.load(spacy_model)
    lemmas = list(lemma2vars)
    pos = {}
    for doc in nlp.pipe(lemmas, batch_size=256):
        pos[doc.text] = doc[0].pos_ if len(doc) else "X"

    def unambiguous(l):
        return all(len(var2lemmas[v]) == 1 for v in lemma2vars[l])

    tr = _cost_on_rows(train_rgn, var2lemmas)
    va = _cost_on_rows(val_rgn, var2lemmas)
    in_test = _variants_in_test(test_rgn, var2lemmas)

    report = []
    for lem, vs in lemma2vars.items():
        p = pos.get(lem, "X")
        content = p in CONTENT_POS
        unamb = unambiguous(lem)
        n_in_test = in_test.get(lem, 0)
        evaluable = n_in_test > 0
        cost = tr.get(lem, 0) + va.get(lem, 0)
        report.append({"lemma": lem, "pos": p, "n_var": len(vs),
                       "content": int(content), "unambiguous": int(unamb),
                       "n_in_test": n_in_test, "evaluable": int(evaluable),
                       "cost_total": cost,
                       "valid": int(content and unamb and evaluable)})

    valid = [r for r in report if r["valid"]]
    valid.sort(key=lambda r: (-r["n_in_test"], r["cost_total"], -r["n_var"]))
    chosen = {r["lemma"] for r in valid[:k]}
    for r in report:
        r["chosen"] = int(r["lemma"] in chosen)
    return chosen, report


def heldout_variant_set(chosen, lemma2vars):
    out = set()
    for lem in chosen:
        out |= lemma2vars.get(lem, set())
    return out


def sentence_touches_heldout(rgn_sentence, heldout_vars):
    return bool(set(tokenize(rgn_sentence)) & heldout_vars)


def load_heldout_list(path):
    with open(path, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip()}


def save_heldout_list(chosen, path):
    with open(path, "w", encoding="utf-8") as f:
        for lem in sorted(chosen):
            f.write(lem + "\n")
