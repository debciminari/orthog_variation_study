#!/usr/bin/env python3
import argparse
import json
import re
import random
import statistics as st
import csv
from collections import Counter
from pathlib import Path
import pandas as pd

from distance import SIM_THRESHOLD
from tokenization import tokenize
from tokenization import tokenize as _tok
from heldout import (load_filtered_variants, select_heldout, heldout_variant_set,
                     sentence_touches_heldout, save_heldout_list,
                     extract_variants)

URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)


def clean_cell(text: str) -> str:
    text = URL_RE.sub('', text)
    return re.sub(r'[ \t]+', ' ', text).strip()


def load_tsv(path: Path, nds_col: int, de_col: int, has_header: bool):
    rows, n_skipped = [], 0
    need = max(nds_col, de_col)
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f, delimiter='\t')
        if has_header:
            next(rd, None)
        for r in rd:
            if len(r) <= need:
                n_skipped += 1
                continue
            nds = clean_cell(r[nds_col])
            de = clean_cell(r[de_col])
            if nds and de:
                rows.append({'nds': nds, 'de': de})
            else:
                n_skipped += 1
    return rows, n_skipped


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate(df):
    def norm(s):
        return re.sub(r'\s+', ' ', str(s)).strip().casefold()
    key = df['nds'].map(norm) + '\x1f' + df['de'].map(norm)
    is_dup = key.duplicated(keep='first')
    removed = df[is_dup].copy()
    deduped = df[~is_dup].copy().reset_index(drop=True)
    return deduped, removed


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def _cluster_key(nds_sentence: str) -> str:
    toks = sorted(set(tokenize(nds_sentence)))
    return ' '.join(toks) if toks else nds_sentence.strip().casefold()


def subsample(df, fraction, split_mode, seed=42):
    if fraction >= 1.0:
        return df
    if not (0.0 < fraction < 1.0):
        raise SystemExit(f"--fraction must be in (0, 1], got {fraction}")
    rng = random.Random(seed)
    df = df.reset_index(drop=True)
    target = int(round(fraction * len(df)))
    if split_mode == 'cluster':
        df = df.copy()
        df['_ckey'] = df['nds'].map(_cluster_key)
        clusters = df.groupby('_ckey').indices
        keys = list(clusters)
        rng.shuffle(keys)
        keep_idx, acc = [], 0
        for k in keys:
            if acc >= target:
                break
            members = list(clusters[k])
            keep_idx += members
            acc += len(members)
        kept = df.iloc[keep_idx][['nds', 'de']].reset_index(drop=True)
    else:
        idx = list(df.index)
        rng.shuffle(idx)
        kept = df.iloc[idx[:target]][['nds', 'de']].reset_index(drop=True)
    return kept


def cluster_split(df, seed=42, test_frac=0.15, val_frac=0.15):
    rng = random.Random(seed)
    df = df.reset_index(drop=True)
    df['_ckey'] = df['nds'].map(_cluster_key)
    clusters = df.groupby('_ckey').indices           
    keys = list(clusters)
    rng.shuffle(keys)
    total = len(df)

    test_idx, val_idx, train_idx = [], [], []
    acc_test = acc_val = 0
    for k in keys:
        members = list(clusters[k])
        if acc_test < test_frac * total:
            test_idx += members; acc_test += len(members)
        elif acc_val < val_frac * total:
            val_idx += members; acc_val += len(members)
        else:
            train_idx += members
    cols = ['nds', 'de']
    train_df = df.iloc[train_idx][cols].reset_index(drop=True)
    val_df = df.iloc[val_idx][cols].reset_index(drop=True)
    test_df = df.iloc[test_idx][cols].reset_index(drop=True)
    return train_df, val_df, test_df


def random_split(df, seed=42, test_frac=0.15, val_frac=0.15):
    rng = random.Random(seed)
    df = df.reset_index(drop=True)
    idx = list(df.index)
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    cols = ['nds', 'de']
    return (df.iloc[train_idx][cols].reset_index(drop=True),
            df.iloc[val_idx][cols].reset_index(drop=True),
            df.iloc[test_idx][cols].reset_index(drop=True))


def subsample_fraction(df, fraction, seed=42, split='cluster'):
    if fraction >= 1.0:
        return df
    if fraction <= 0.0:
        raise SystemExit('--fraction must be in (0, 1].')
    rng = random.Random(seed)
    df = df.reset_index(drop=True)

    if split == 'cluster':
        df = df.copy()
        df['_ckey'] = df['nds'].map(_cluster_key)
        clusters = df.groupby('_ckey').indices
        keys = list(clusters)
        rng.shuffle(keys)
        target = fraction * len(df)
        keep_pos, acc = [], 0
        for k in keys:
            if acc >= target:
                break
            members = list(clusters[k])
            keep_pos += members
            acc += len(members)
        out = df.iloc[sorted(keep_pos)][['nds', 'de']].reset_index(drop=True)
    else:
        idx = list(df.index)
        rng.shuffle(idx)
        k = round(fraction * len(idx))
        out = df.iloc[sorted(idx[:k])][['nds', 'de']].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
def apply_heldout(train_df, val_df, heldout_vars):
    def mask_touch(df):
        keep_idx, drop_idx = [], []
        for i, sent in zip(df.index, df['nds']):
            if sentence_touches_heldout(sent, heldout_vars):
                drop_idx.append(i)
            else:
                keep_idx.append(i)
        return df.loc[keep_idx].copy(), df.loc[drop_idx].copy()
    tr_keep, tr_drop = mask_touch(train_df)
    va_keep, va_drop = mask_touch(val_df)
    return tr_keep, va_keep, tr_drop, va_drop


def write_heldout_glossary(out_dir, chosen_lemmas, lemma2vars, test_df):
    test_tokens = set()
    for sent in test_df['nds']:
        test_tokens |= set(_tok(sent))

    rows = []
    max_vars = 0
    for lem in sorted(chosen_lemmas):
        vars_nds = sorted(lemma2vars[lem])
        n_in_test = sum(1 for v in vars_nds if v in test_tokens)
        rows.append((lem, vars_nds, n_in_test))
        max_vars = max(max_vars, len(vars_nds))

    path = out_dir / 'heldout_glossary.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['de_word'] + [f'nds_var{i + 1}' for i in range(max_vars)]
                   + ['n_variants', 'n_in_test'])
        for lem, vars_nds, n_in_test in rows:
            padded = vars_nds + [''] * (max_vars - len(vars_nds))
            w.writerow([lem] + padded + [len(vars_nds), n_in_test])
    return path


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _dist(values):
    if not values:
        return {}
    values = sorted(values)
    return {
        'n': len(values),
        'sum': sum(values),
        'mean': round(st.mean(values), 2),
        'median': st.median(values),
        'std': round(st.pstdev(values), 2) if len(values) > 1 else 0.0,
        'min': values[0],
        'max': values[-1],
        'p05': values[int(0.05 * (len(values) - 1))],
        'p95': values[int(0.95 * (len(values) - 1))],
    }


def lang_stats(series):
    tok_per_line, char_per_line = [], []
    vocab = Counter()
    total_chars = 0
    for text in series:
        toks = tokenize(text)
        tok_per_line.append(len(toks))
        char_per_line.append(len(text))
        total_chars += len(text)
        vocab.update(toks)
    n_tokens = sum(vocab.values())
    n_types = len(vocab)
    return {
        'lines': len(series),
        'tokens': n_tokens,
        'types': n_types,
        'ttr': round(n_types / n_tokens, 4) if n_tokens else 0.0,
        'chars': total_chars,
        'tokens_per_line': _dist(tok_per_line),
        'chars_per_line': _dist(char_per_line),
        'hapax': sum(1 for _, c in vocab.items() if c == 1),
    }, vocab


def compute_stats(df, train_df, val_df, test_df, n_dupes=0, split_predrop=None,
                  split_mode='cluster'):
    stats = {}
    stats['overview'] = {
        'split_mode': split_mode,
        'aligned_lines': len(df),
        'duplicate_pairs_removed': int(n_dupes),
    }
    nds_stats, nds_vocab = lang_stats(df['nds'])
    de_stats, de_vocab = lang_stats(df['de'])
    stats['language'] = {'nds': nds_stats, 'de': de_stats}

    if split_predrop is not None:
        tr_nds_stats, _ = lang_stats(train_df['nds'])
        tr_de_stats, _ = lang_stats(train_df['de'])
        va_nds_stats, _ = lang_stats(val_df['nds'])
        va_de_stats, _ = lang_stats(val_df['de'])
        stats['language_post_a1'] = {
            'train': {'nds': tr_nds_stats, 'de': tr_de_stats},
            'val':   {'nds': va_nds_stats, 'de': va_de_stats},
        }

    ratios = []
    for r, i in zip(df['nds'], df['de']):
        it = len(tokenize(i))
        if it:
            ratios.append(len(tokenize(r)) / it)
    stats['length_ratio_nds_over_de'] = _dist([round(x, 3) for x in ratios])

    drawn = split_predrop if split_predrop is not None else (train_df, val_df, test_df)
    drawn_map = dict(zip(('train', 'val', 'test'), drawn))
    post_map = {'train': train_df, 'val': val_df, 'test': test_df}
    n_drawn = sum(len(p) for p in drawn)
    split = {}
    for name in ('train', 'val', 'test'):
        drawn_part = drawn_map[name]
        part = post_map[name]
        split[name] = {
            'lines_drawn': len(drawn_part),
            'lines_pct_drawn': round(100 * len(drawn_part) / n_drawn, 1) if n_drawn else 0.0,
            'lines': len(part),
            'nds_tokens': sum(len(tokenize(t)) for t in part['nds']),
            'de_tokens': sum(len(tokenize(t)) for t in part['de']),
        }
    stats['split'] = split

    def vocab_of(part, col):
        v = set()
        for t in part[col]:
            v.update(tokenize(t))
        return v

    def oov_block(eval_df, train_vocab):
        types = vocab_of(eval_df, 'nds')
        n_type_oov = len(types - train_vocab)
        tok_total = tok_oov = 0
        for t in eval_df['nds']:
            for w in tokenize(t):
                tok_total += 1
                if w not in train_vocab:
                    tok_oov += 1
        return {
            'types': len(types),
            'type_oov': n_type_oov,
            'type_oov_rate': round(n_type_oov / len(types), 4) if types else 0.0,
            'tokens': tok_total,
            'token_oov': tok_oov,
            'token_oov_rate': round(tok_oov / tok_total, 4) if tok_total else 0.0,
        }

    tr_vocab_model = vocab_of(train_df, 'nds')
    drawn_train = drawn_map['train']
    tr_vocab_corpus = vocab_of(drawn_train, 'nds')
    stats['oov'] = {
        'note': ('type/token OOV of nds vs train vocabulary. *_model uses the '
                 'post-A1 train vocab (what the model is trained on); *_corpus '
                 'uses the pre-A1 train vocab (a property of the split itself, '
                 'not inflated by held-out removal). They coincide when A1 is off.'),
        'test_model': oov_block(test_df, tr_vocab_model),
        'test_corpus': oov_block(test_df, tr_vocab_corpus),
        'val_model': oov_block(post_map['val'], tr_vocab_model),
        'val_corpus': oov_block(drawn_map['val'], tr_vocab_corpus),
    }
    return stats


def render_stats_txt(stats):
    L = []
    def line(s=''): L.append(s)
    o = stats['overview']
    line('=' * 60)
    line('CORPUS STATISTICS')
    line('=' * 60)
    line(f"Split mode    : {o['split_mode']}")
    line(f"Aligned lines : {o['aligned_lines']}")
    line(f"Dupes removed : {o['duplicate_pairs_removed']}")
    line('')
    line('-' * 60)
    line('PER LANGUAGE')
    line('-' * 60)
    line(f"{'':14s}{'NDS':>14s}{'DE':>14s}")
    r, i = stats['language']['nds'], stats['language']['de']
    for label, key in (('tokens', 'tokens'), ('types', 'types'),
                       ('TTR', 'ttr'), ('hapax', 'hapax'), ('chars', 'chars')):
        line(f"{label:14s}{str(r[key]):>14s}{str(i[key]):>14s}")
    line(f"{'tok/line mean':14s}"
         f"{str(r['tokens_per_line']['mean']):>14s}"
         f"{str(i['tokens_per_line']['mean']):>14s}")
    if 'language_post_a1' in stats:
        for split_name in ('train', 'val'):
            line('')
            line('-' * 60)
            line(f'PER LANGUAGE ({split_name}, post-A1 — what the model sees)')
            line('-' * 60)
            line(f"{'':14s}{'NDS':>14s}{'DE':>14s}")
            rt = stats['language_post_a1'][split_name]['nds']
            it = stats['language_post_a1'][split_name]['de']
            line(f"{'lines':14s}{str(rt['lines']):>14s}{str(it['lines']):>14s}")
            for label, key in (('tokens', 'tokens'), ('types', 'types'),
                               ('TTR', 'ttr'), ('hapax', 'hapax'),
                               ('chars', 'chars')):
                line(f"{label:14s}{str(rt[key]):>14s}{str(it[key]):>14s}")
            line(f"{'tok/line mean':14s}"
                 f"{str(rt['tokens_per_line']['mean']):>14s}"
                 f"{str(it['tokens_per_line']['mean']):>14s}")
    lr = stats['length_ratio_nds_over_de']
    line(f"\nLength ratio nds/de per line: mean {lr.get('mean')}, "
         f"median {lr.get('median')}, p05 {lr.get('p05')}, p95 {lr.get('p95')}")
    line('')
    line('-' * 60)
    line('SPLIT (70/15/15 target)')
    line('-' * 60)
    line(f"{'set':6s}{'drawn':>8s}{'%drawn':>8s}{'kept':>8s}"
         f"{'nds_tok':>9s}{'de_tok':>9s}")
    for name in ('train', 'val', 'test'):
        s = stats['split'][name]
        line(f"{name:6s}{s['lines_drawn']:>8d}{s['lines_pct_drawn']:>8.1f}"
             f"{s['lines']:>8d}{s['nds_tokens']:>9d}{s['de_tokens']:>9d}")
    if 'heldout' in stats:
        h = stats['heldout']
        line(f"A1 held-out removed: train -{h['train_dropped']} "
             f"({h['train_dropped_pct']}%), val -{h['val_dropped']} "
             f"({h['val_dropped_pct']}%); test untouched")
    line('')
    ov = stats['oov']
    line('-' * 60)
    line('NDS OOV vs train vocabulary')
    line('-' * 60)
    line(f"{'':16s}{'type-OOV':>10s}{'token-OOV':>11s}")
    def oov_row(label, blk):
        line(f"{label:16s}{blk['type_oov_rate']:>9.1%}{blk['token_oov_rate']:>11.1%}")
    oov_row('test (corpus)', ov['test_corpus'])
    oov_row('test (model)', ov['test_model'])
    oov_row('val  (corpus)', ov['val_corpus'])
    oov_row('val  (model)', ov['val_model'])
    line("  corpus = vs pre-A1 train vocab (split property); "
         "model = vs post-A1 train vocab (what the model sees)")
    if 'heldout' in stats and 'heldout_eval_lines' in stats['heldout']:
        line('')
        line(f"Held-out eval set (withheld train+val lines, reused as probe): "
             f"{stats['heldout']['heldout_eval_lines']} lines -> heldout_eval.csv")
    return '\n'.join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input_tsv', help='TSV of nds/de sentence pairs')
    ap.add_argument('-o', '--output_dir', default='.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--nds-col', type=int, default=1,
                    help='0-based index of the nds sentence column (default 1)')
    ap.add_argument('--de-col', type=int, default=3,
                    help='0-based index of the de sentence column (default 3)')
    ap.add_argument('--has-header', action='store_true',
                    help='skip the first TSV row as a header')
    ap.add_argument('--split', choices=['cluster', 'random'], default='cluster',
                    help='cluster (default, near-duplicate-safe) or random')
    ap.add_argument('--fraction', type=float, default=1.0,
                    help='keep only this fraction (0<f<=1) of the corpus BEFORE '
                         'splitting, preserving the 70/15/15 proportions, to make '
                         'fine-tuning faster. Sampled by whole clusters when '
                         '--split cluster, by rows when --split random. Default 1.0 '
                         '(use everything).')
    ap.add_argument('--fraction-seed', type=int, default=0,
                    help='seed for --fraction subsampling (independent of --seed, '
                         'so the subsample is stable across split re-runs)')
    ap.add_argument('--cols', default='nds,de',
                    help='columns to keep in the split csvs')
    ap.add_argument('--variants', default=None,
                    help='glossario de_word -> varianti nds (attiva held-out A1). '
                         'Alternativa a --extract-variants. Se assenti entrambi, split invariato.')
    ap.add_argument('--extract-variants', action='store_true',
                    help='estrae variants.csv da TRAIN+VAL qui (eflomal una volta '
                         'sola, test mai visto); fonte unica per build_lexical.py')
    ap.add_argument('--heldout-k', type=int, default=100,
                    help='quanti lemmi tenere held-out da train+val (A1)')
    ap.add_argument('--sim-threshold', type=float, default=SIM_THRESHOLD,
                    help='filtro falsi cugini (default: distance.SIM_THRESHOLD, '
                         'unica fonte condivisa con build_lexical.py e heldout.py)')
    ap.add_argument('--spacy-model', default='de_core_news_sm',
                    help='modello spaCy TEDESCO per il POS su de_word')
    args = ap.parse_args()

    in_tsv = Path(args.input_tsv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, n_skipped = load_tsv(in_tsv, args.nds_col, args.de_col, args.has_header)
    if not rows:
        raise SystemExit('No data extracted. Check --nds-col/--de-col and the TSV.')
    print(f'Loaded {len(rows)} pairs (skipped {n_skipped} malformed/empty)')

    df = pd.DataFrame(rows, columns=['nds', 'de'])

    n_before = len(df)
    df, removed = deduplicate(df)
    n_dupes = len(removed)
    if n_dupes:
        removed.to_csv(out_dir / 'duplicates_removed.csv', index=False)
    print(f'Deduplication: removed {n_dupes} duplicate pair(s) '
          f'({n_before} -> {len(df)} rows)')

    if args.fraction < 1.0:
        n_full = len(df)
        df = subsample_fraction(df, args.fraction, seed=args.fraction_seed,
                                split=args.split)
        print(f'Subsample (--fraction {args.fraction}, by '
              f'{"cluster" if args.split == "cluster" else "row"}): '
              f'{n_full} -> {len(df)} rows ({100*len(df)/n_full:.1f}%)')

    df.to_csv(out_dir / 'full.csv', index=False)

    if args.split == 'cluster':
        train_df, val_df, test_df = cluster_split(df, seed=args.seed)
    else:
        train_df, val_df, test_df = random_split(df, seed=args.seed)
    print(f'Split ({args.split}): train {len(train_df)} | '
          f'val {len(val_df)} | test {len(test_df)}')

    variants_path = args.variants
    if args.extract_variants:
        variants_path = str(out_dir / 'variants.csv')
        src, trg = [], []
        for nds, de in zip(pd.concat([train_df['nds'], val_df['nds']]),
                           pd.concat([train_df['de'], val_df['de']])):
            r, i = _tok(nds), _tok(de)
            if r and i:
                src.append(r); trg.append(i)
        _, n_lemmas = extract_variants(
            src, trg, variants_path, sim_threshold=args.sim_threshold)
        print(f'Glossario estratto da train+val: {n_lemmas} lemmi '
              f'-> {variants_path}')

    heldout_info = None
    split_predrop = None
    if variants_path:
        lemma2vars, var2lemmas = load_filtered_variants(
            variants_path, sim_threshold=args.sim_threshold)
        chosen, ho_report = select_heldout(
            lemma2vars, var2lemmas, train_df['nds'], val_df['nds'],
            test_df['nds'], k=args.heldout_k, spacy_model=args.spacy_model)
        heldout_vars = heldout_variant_set(chosen, lemma2vars)
        n_tr0, n_va0 = len(train_df), len(val_df)
        split_predrop = (train_df.copy(), val_df.copy(), test_df.copy())
        train_df, val_df, tr_drop, va_drop = apply_heldout(
            train_df, val_df, heldout_vars)

        save_heldout_list(chosen, out_dir / 'heldout_lemmas.txt')
        gloss_path = write_heldout_glossary(out_dir, chosen, lemma2vars, test_df)
        ho_df = pd.DataFrame(ho_report)
        if {'valid', 'cost_total'}.issubset(ho_df.columns):
            ho_df = ho_df.sort_values(['valid', 'cost_total'],
                                      ascending=[False, True])
        ho_df.to_csv(out_dir / 'heldout_report.csv', index=False)
        tr_drop.to_csv(out_dir / 'dropped_train.csv', index=False)
        va_drop.to_csv(out_dir / 'dropped_val.csv', index=False)

        heldout_eval = pd.concat([tr_drop, va_drop], ignore_index=True)
        heldout_eval.insert(0, 'origin',
                            ['train'] * len(tr_drop) + ['val'] * len(va_drop))
        heldout_eval.to_csv(out_dir / 'heldout_eval.csv', index=False)

        test_tokens = set()
        for sent in test_df['nds']:
            test_tokens |= set(_tok(sent))
        n_in_test = sum(1 for v in heldout_vars if v in test_tokens)
        heldout_info = {
            'heldout_lemmas': len(chosen),
            'heldout_variants': len(heldout_vars),
            'heldout_variants_in_test': n_in_test,
            'train_before': n_tr0, 'train_after': len(train_df),
            'train_dropped': len(tr_drop),
            'train_dropped_pct': round(100 * len(tr_drop) / max(1, n_tr0), 2),
            'val_before': n_va0, 'val_after': len(val_df),
            'val_dropped': len(va_drop),
            'val_dropped_pct': round(100 * len(va_drop) / max(1, n_va0), 2),
            'heldout_eval_lines': len(heldout_eval),
        }
        print(f"Held-out (A1): {len(chosen)} lemmi, {len(heldout_vars)} varianti | "
              f"train -{len(tr_drop)} ({heldout_info['train_dropped_pct']}%), "
              f"val -{len(va_drop)} ({heldout_info['val_dropped_pct']}%) | "
              f"{n_in_test} varianti presenti nel test")
        print(f"  glossario held-out -> {gloss_path.name}")

    keep = [c.strip() for c in args.cols.split(',')]
    for name, part in (('train', train_df), ('val', val_df), ('test', test_df)):
        part[keep].to_csv(out_dir / f'{name}.csv', index=False)

    stats = compute_stats(df, train_df, val_df, test_df, n_dupes=n_dupes,
                          split_predrop=split_predrop, split_mode=args.split)
    if heldout_info is not None:
        stats['heldout'] = heldout_info
    (out_dir / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    report = render_stats_txt(stats)
    (out_dir / 'stats.txt').write_text(report)
    print('\n' + report)


if __name__ == '__main__':
    main()
