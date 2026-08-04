#!/usr/bin/env python3
"""
Creation of parallel corpus
"""
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
# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------
URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
TAG_RE = re.compile(r'<[^>]+>')                    
TITLE_TAG_RE = re.compile(r'<title>(.*?)</title>', re.IGNORECASE | re.DOTALL)
SEP_RE = re.compile(r'^\s*#+\s*$')                   
NONE_RE = re.compile(r'^\s*none\s*$', re.IGNORECASE)  
RGN_SUFFIXES = ('_rgn', '_rmg', '_rom')
ITA_SUFFIXES = ('_ita', '_it')
VARIANT_RE = re.compile(r'^(.*)_(\d+)$')  


from tokenization import tokenize
# ---------------------------------------------------------------------------
# Line-level cleaning
# ---------------------------------------------------------------------------
def clean_line(line: str) -> str:
    if SEP_RE.match(line):        
        return ''
    if NONE_RE.match(line):       
        return ''
    line = URL_RE.sub('', line)
    line = TAG_RE.sub('', line)
    return re.sub(r'[ \t]+', ' ', line).strip()
def is_url_only(line: str) -> bool:
    return bool(line.strip()) and not URL_RE.sub('', line).strip()
# ---------------------------------------------------------------------------
# Poem segmentation
# ---------------------------------------------------------------------------
def segment_poems(lines):
    segments, current, blank_run = [], [], 0
    for i, raw in enumerate(lines):
        if SEP_RE.match(raw):
            if current:
                segments.append(current)
                current = []
            blank_run = 0
        elif raw.strip() == '':
            blank_run += 1
            if blank_run >= 2 and current:
                segments.append(current)
                current = []
  
            if blank_run == 1:
                current.append((i, raw))
        else:
            blank_run = 0
            current.append((i, raw))
    if current:
        segments.append(current)
  
    poems = []
    for seg in segments:
  
        while seg and seg[0][1].strip() == '':
            seg = seg[1:]
        while seg and seg[-1][1].strip() == '':
            seg = seg[:-1]
        if not seg:
            continue
        title_idx = None
        title = ''
        body_idx = [idx for idx, _ in seg]
        first_idx, first_raw = seg[0]
  
        has_blank_after_first = len(seg) > 1 and seg[1][1].strip() == ''
        if has_blank_after_first:
            title_idx = first_idx
            title = clean_line(first_raw)         
  
            body_idx = [idx for idx, raw in seg[1:] if raw.strip() != '']
        else:
          
            body_idx = [idx for idx, raw in seg if raw.strip() != '']
        poems.append({'title': title, 'title_idx': title_idx,
                      'body_idx': body_idx})
    return poems
# ---------------------------------------------------------------------------
# Per-file / per-pair processing
# ---------------------------------------------------------------------------
def base_and_lang(path: Path):
    stem = path.stem
    variant = None
    m = VARIANT_RE.match(stem)
    if m:
        stem, variant = m.group(1), m.group(2)
    for suf in RGN_SUFFIXES:
        if stem.lower().endswith(suf):
            return stem[:-len(suf)], 'rgn', variant
    for suf in ITA_SUFFIXES:
        if stem.lower().endswith(suf):
            return stem[:-len(suf)], 'ita', variant
    return None, None, None
def read_lines(path: Path):
    return path.read_text(encoding='utf-8', errors='replace').splitlines()
def strip_leading_urls(lines):
    out = list(lines)
    while out and is_url_only(out[0]):
        out.pop(0)
    return out
def process_pair(rgn_path: Path, ita_path: Path, author: str, align_flags: list,
                  poem_prefix: 'str | None' = None):

    if poem_prefix is None:
        poem_prefix = author
    rgn_raw_all = read_lines(rgn_path)
    ita_raw_all = read_lines(ita_path)

    rgn_body = strip_leading_urls(rgn_raw_all)
    ita_body = strip_leading_urls(ita_raw_all)
    rgn_url = len(rgn_raw_all) - len(rgn_body)
    ita_url = len(ita_raw_all) - len(ita_body)
    pad = rgn_url - ita_url
    if pad > 0:                      
        ita_lines = [''] * pad + ita_body
        rgn_lines = rgn_body
    elif pad < 0:                    
        rgn_lines = [''] * (-pad) + rgn_body
        ita_lines = ita_body
    else:
        rgn_lines, ita_lines = rgn_body, ita_body
    # ---- file-level alignment check --------------------------------------
    if len(rgn_lines) != len(ita_lines):
        align_flags.append({
            'author': author, 'poem_id': f'{poem_prefix}__FILE',
            'level': 'file', 'rgn_lines': len(rgn_lines),
            'ita_lines': len(ita_lines),
            'diff': len(rgn_lines) - len(ita_lines),
            'note': 'file line counts differ; per-line pairing may drift',
        })
    poems = segment_poems(rgn_lines)
    rows = []
    for poem_local_id, poem in enumerate(poems):
        body_idx = poem['body_idx']
        poem_id = f'{poem_prefix}__{poem_local_id}'
       
        title = ''
        t_idx = poem['title_idx']
        if t_idx is not None:
            ita_title = clean_line(ita_lines[t_idx]) if t_idx < len(ita_lines) else ''
            title = ita_title or poem['title']
        rgn_nonempty = ita_nonempty = 0
        rgn_tok = ita_tok = 0
        n_one_sided = 0
        for idx in body_idx:
            rgn_clean = clean_line(rgn_lines[idx])
            ita_raw = ita_lines[idx] if idx < len(ita_lines) else ''
            ita_clean = clean_line(ita_raw)
            if not rgn_clean and not ita_clean:
                continue
            rgn_nonempty += bool(rgn_clean)
            ita_nonempty += bool(ita_clean)
       
            if not rgn_clean or not ita_clean:
                n_one_sided += 1
                continue
            rgn_tok += len(tokenize(rgn_clean))
            ita_tok += len(tokenize(ita_clean))
            rows.append({
                'rgn': rgn_clean, 'ita': ita_clean, 'title': title,
                'author': author, 'poem_id': poem_id,
            })
       
        if rgn_nonempty == 0 and ita_nonempty == 0:
            continue
        # ---- poem-level alignment checks ---------------------------------
        if rgn_nonempty != ita_nonempty:
            align_flags.append({
                'author': author, 'poem_id': poem_id, 'level': 'poem',
                'rgn_lines': rgn_nonempty, 'ita_lines': ita_nonempty,
                'diff': rgn_nonempty - ita_nonempty,
                'note': f'one side has empty/missing lines; '
                        f'{n_one_sided} one-sided row(s) dropped',
            })
        if ita_tok and not (0.5 <= rgn_tok / ita_tok <= 2.0):
            align_flags.append({
                'author': author, 'poem_id': poem_id, 'level': 'ratio',
                'rgn_lines': rgn_nonempty, 'ita_lines': ita_nonempty,
                'diff': rgn_tok - ita_tok,
                'note': f'token ratio rgn/ita={rgn_tok/ita_tok:.2f} out of [0.5,2.0]',
            })
    return rows
# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate(df):
    def norm(s):
        return re.sub(r'\s+', ' ', str(s)).strip().casefold()
    key = df['rgn'].map(norm) + '\x1f' + df['ita'].map(norm)
    is_dup = key.duplicated(keep='first')
    removed = df[is_dup].copy()
    deduped = df[~is_dup].copy().reset_index(drop=True)
    return deduped, removed
# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def author_level_split(df, seed=42, test_frac=0.15, val_frac=0.15):
    rng = random.Random(seed)
    authors = sorted(df['author'].unique())
    rng.shuffle(authors)
    total = len(df)
    counts = df.groupby('author').size().to_dict()
    test_authors, acc = set(), 0
    for a in authors:
        if acc >= test_frac * total:
            break
        test_authors.add(a)
        acc += counts[a]
    test_df = df[df['author'].isin(test_authors)].copy()
    rest = df[~df['author'].isin(test_authors)].copy()
    poems = sorted(rest['poem_id'].unique())
    rng.shuffle(poems)
    poem_counts = rest.groupby('poem_id').size().to_dict()
    val_poems, acc = set(), 0
    for p in poems:
        if acc >= val_frac * total:
            break
        val_poems.add(p)
        acc += poem_counts[p]
    val_df = rest[rest['poem_id'].isin(val_poems)].copy()
    train_df = rest[~rest['poem_id'].isin(val_poems)].copy()
    return train_df, val_df, test_df
# ---------------------------------------------------------------------------
# Orthographic held-out 
# ---------------------------------------------------------------------------
from tokenization import tokenize as _tok
from heldout import (load_filtered_variants, select_heldout, heldout_variant_set,
                     sentence_touches_heldout, save_heldout_list,
                     extract_variants)
def apply_heldout(train_df, val_df, heldout_vars):
    def mask_touch(df):
        keep_idx, drop_idx = [], []
        for i, sent in zip(df.index, df['rgn']):
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
    for sent in test_df['rgn']:
        test_tokens |= set(_tok(sent))

   
    rows = []
    max_vars = 0
    for lem in sorted(chosen_lemmas):
        vars_rgn = sorted(lemma2vars[lem])
        n_in_test = sum(1 for v in vars_rgn if v in test_tokens)
        rows.append((lem, vars_rgn, n_in_test))
        max_vars = max(max_vars, len(vars_rgn))

    path = out_dir / 'heldout_glossary.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ita_word'] + [f'rgn_var{i + 1}' for i in range(max_vars)]
                   + ['n_variants', 'n_in_test'])
        for lem, vars_rgn, n_in_test in rows:
            padded = vars_rgn + [''] * (max_vars - len(vars_rgn))
            w.writerow([lem] + padded + [len(vars_rgn), n_in_test])
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
def compute_stats(df, train_df, val_df, test_df, align_flags, n_dupes=0,
                  split_predrop=None):
   
    stats = {}
    stats['overview'] = {
        'aligned_lines': len(df),
        'poems': df['poem_id'].nunique(),
        'authors': df['author'].nunique(),
        'titled_poems': int(df[df['title'] != '']['poem_id'].nunique()),
        'untitled_poems': int(df[df['title'] == '']['poem_id'].nunique()),
        'duplicate_pairs_removed': int(n_dupes),
        'lines_per_poem': _dist(df.groupby('poem_id').size().tolist()),
        'poems_per_author': _dist(df.groupby('author')['poem_id']
                                  .nunique().tolist()),
        'lines_per_author': _dist(df.groupby('author').size().tolist()),
    }
    rgn_stats, rgn_vocab = lang_stats(df['rgn'])
    ita_stats, ita_vocab = lang_stats(df['ita'])
    stats['language'] = {'rgn': rgn_stats, 'ita': ita_stats}
  
    if split_predrop is not None:
        tr_rgn_stats, _ = lang_stats(train_df['rgn'])
        tr_ita_stats, _ = lang_stats(train_df['ita'])
        va_rgn_stats, _ = lang_stats(val_df['rgn'])
        va_ita_stats, _ = lang_stats(val_df['ita'])
        stats['language_post_a1'] = {
            'train': {'rgn': tr_rgn_stats, 'ita': tr_ita_stats},
            'val':   {'rgn': va_rgn_stats, 'ita': va_ita_stats},
        }
    ratios = []
    for r, i in zip(df['rgn'], df['ita']):
        it = len(tokenize(i))
        if it:
            ratios.append(len(tokenize(r)) / it)
    stats['length_ratio_rgn_over_ita'] = _dist([round(x, 3) for x in ratios])
  
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
            'poems': int(part['poem_id'].nunique()),
            'authors': int(part['author'].nunique()),
            'rgn_tokens': sum(len(tokenize(t)) for t in part['rgn']),
            'ita_tokens': sum(len(tokenize(t)) for t in part['ita']),
        }
    stats['split'] = split
    tr_auth, va_auth = set(train_df['author']), set(val_df['author'])
    te_auth = set(test_df['author'])
    tr_poem, te_poem = set(train_df['poem_id']), set(test_df['poem_id'])
    va_poem = set(val_df['poem_id'])
    stats['integrity'] = {
        'test_authors_disjoint_from_trainval':
            not (te_auth & (tr_auth | va_auth)),
        'test_poems_disjoint_from_trainval':
            not (te_poem & (tr_poem | va_poem)),
        'trainval_share_authors': bool(tr_auth & va_auth),
        'test_authors': sorted(te_auth),
    }
    def vocab_of(part, col):
        v = set()
        for t in part[col]:
            v.update(tokenize(t))
        return v

    def oov_block(eval_df, train_vocab):
        """Type- and token-level OOV of eval_df['rgn'] against a train vocab."""
        types = vocab_of(eval_df, 'rgn')
        n_type_oov = len(types - train_vocab)
        tok_total = tok_oov = 0
        for t in eval_df['rgn']:
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

   
    tr_vocab_model = vocab_of(train_df, 'rgn')
   
    drawn_train = drawn_map['train']
    tr_vocab_corpus = vocab_of(drawn_train, 'rgn')
    stats['oov'] = {
        'note': ('type/token OOV of rgn vs train vocabulary. *_model uses the '
                 'post-A1 train vocab (what the model is trained on); *_corpus '
                 'uses the pre-A1 train vocab (a property of the split itself, '
                 'not inflated by held-out removal). They coincide when A1 is off.'),
        'test_model': oov_block(test_df, tr_vocab_model),
        'test_corpus': oov_block(test_df, tr_vocab_corpus),
        'val_model': oov_block(post_map['val'], tr_vocab_model),
        'val_corpus': oov_block(drawn_map['val'], tr_vocab_corpus),
    }
    by_level = Counter(f['level'] for f in align_flags)
    stats['alignment'] = {
        'flagged_total': len(align_flags),
        'by_level': dict(by_level),
        'poems_flagged': len({f['poem_id'] for f in align_flags
                              if f['level'] != 'file'}),
    }
    return stats
def render_stats_txt(stats):
    L = []
    def line(s=''): L.append(s)
    o = stats['overview']
    line('=' * 60)
    line('CORPUS STATISTICS')
    line('=' * 60)
    line(f"Aligned lines : {o['aligned_lines']}")
    line(f"Poems         : {o['poems']}  "
         f"(titled {o['titled_poems']}, untitled {o['untitled_poems']})")
    line(f"Authors       : {o['authors']}")
    line(f"Dupes removed : {o['duplicate_pairs_removed']}")
    line(f"Lines/poem    : mean {o['lines_per_poem'].get('mean')}, "
         f"median {o['lines_per_poem'].get('median')}, "
         f"range {o['lines_per_poem'].get('min')}-{o['lines_per_poem'].get('max')}")
    line(f"Poems/author  : mean {o['poems_per_author'].get('mean')}, "
         f"range {o['poems_per_author'].get('min')}-{o['poems_per_author'].get('max')}")
    line('')
    line('-' * 60)
    line('PER LANGUAGE')
    line('-' * 60)
    line(f"{'':14s}{'RGN':>14s}{'ITA':>14s}")
    r, i = stats['language']['rgn'], stats['language']['ita']
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
            line(f"{'':14s}{'RGN':>14s}{'ITA':>14s}")
            rt = stats['language_post_a1'][split_name]['rgn']
            it = stats['language_post_a1'][split_name]['ita']
            line(f"{'lines':14s}{str(rt['lines']):>14s}{str(it['lines']):>14s}")
            for label, key in (('tokens', 'tokens'), ('types', 'types'),
                               ('TTR', 'ttr'), ('hapax', 'hapax'),
                               ('chars', 'chars')):
                line(f"{label:14s}{str(rt[key]):>14s}{str(it[key]):>14s}")
            line(f"{'tok/line mean':14s}"
                 f"{str(rt['tokens_per_line']['mean']):>14s}"
                 f"{str(it['tokens_per_line']['mean']):>14s}")
    lr = stats['length_ratio_rgn_over_ita']
    line(f"\nLength ratio rgn/ita per line: mean {lr.get('mean')}, "
         f"median {lr.get('median')}, p05 {lr.get('p05')}, p95 {lr.get('p95')}")
    line('')
    line('-' * 60)
    line('SPLIT (70/15/15 target, author-disjoint test)')
    line('-' * 60)
   
    line(f"{'set':6s}{'drawn':>8s}{'%drawn':>8s}{'kept':>8s}{'poems':>7s}"
         f"{'auth':>6s}{'rgn_tok':>9s}{'ita_tok':>9s}")
    for name in ('train', 'val', 'test'):
        s = stats['split'][name]
        line(f"{name:6s}{s['lines_drawn']:>8d}{s['lines_pct_drawn']:>8.1f}"
             f"{s['lines']:>8d}{s['poems']:>7d}{s['authors']:>6d}"
             f"{s['rgn_tokens']:>9d}{s['ita_tokens']:>9d}")
    if 'heldout' in stats:
        h = stats['heldout']
        line(f"A1 held-out removed: train -{h['train_dropped']} "
             f"({h['train_dropped_pct']}%), val -{h['val_dropped']} "
             f"({h['val_dropped_pct']}%); test untouched")
    line('')
    ig = stats['integrity']
    line('-' * 60)
    line('INTEGRITY')
    line('-' * 60)
    line(f"test authors disjoint from train/val : "
         f"{ig['test_authors_disjoint_from_trainval']}")
    line(f"test poems   disjoint from train/val : "
         f"{ig['test_poems_disjoint_from_trainval']}")
    line(f"train & val share authors            : {ig['trainval_share_authors']}")
    line(f"held-out test authors                : {', '.join(ig['test_authors'])}")
    line('')
    ov = stats['oov']
    line('-' * 60)
    line('RGN OOV vs train vocabulary')
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
    line('')
    al = stats['alignment']
    line('-' * 60)
    line('ALIGNMENT')
    line('-' * 60)
    line(f"flagged items : {al['flagged_total']} "
         f"({al['poems_flagged']} distinct poems)")
    line(f"by level      : {al['by_level']}")
    line('  see alignment_report.csv for details')
    return '\n'.join(L)
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input_dir', help='folder with the parallel .txt files')
    ap.add_argument('-o', '--output_dir', default='.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cols', default='rgn,ita,title',
                    help='columns to keep in the split csvs')
    ap.add_argument('--variants', default=None,
                    help='glossario ita_word -> varianti rgn (attiva held-out A1). '
                         'Alternativa a --extract-variants. Se assente entrambi, split invariato.')
    ap.add_argument('--extract-variants', action='store_true',
                    help='estrae variants.csv da TRAIN+VAL qui (eflomal una volta '
                         'sola, test mai visto); fonte unica per build_lexical.py')
    ap.add_argument('--heldout-k', type=int, default=100,
                    help='quanti lemmi tenere held-out da train+val (A1)')
    ap.add_argument('--sim-threshold', type=float, default=SIM_THRESHOLD,
                    help='filtro falsi cugini (default: distance.SIM_THRESHOLD, '
                         'unica fonte condivisa con build_lexical.py e heldout.py)')
    ap.add_argument('--spacy-model', default='it_core_news_sm',
                    help='modello spaCy per il POS su ita_word')
    args = ap.parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = {}
    for p in sorted(in_dir.glob('*.txt')):
        base, lang, variant = base_and_lang(p)
        if base is None:
            print(f'  skip (no lang suffix): {p.name}')
            continue
   
        key = f'{base}__{variant}' if variant else base
        entry = pairs.setdefault(key, {'author': base})
        entry[lang] = p
    all_rows, align_flags = [], []
    for key, d in sorted(pairs.items()):
        author = d['author']
        if 'rgn' not in d or 'ita' not in d:
            print(f'  skip (missing pair): {key} -> '
                  f'{[k for k in d if k != "author"]}')
            continue
        rows = process_pair(d['rgn'], d['ita'], author, align_flags,
                            poem_prefix=key)
        all_rows.extend(rows)
        print(f'  {key} (author={author}): {len(rows)} aligned lines')
    if not all_rows:
        raise SystemExit('No data extracted. Check filenames/suffixes.')
    df = pd.DataFrame(all_rows, columns=['rgn', 'ita', 'title', 'author', 'poem_id'])
   
    n_before = len(df)
    df, removed = deduplicate(df)
    n_dupes = len(removed)
    if n_dupes:
        removed.to_csv(out_dir / 'duplicates_removed.csv', index=False)
    print(f'\nDeduplication: removed {n_dupes} duplicate pair(s) '
          f'({n_before} -> {len(df)} rows)')
    full_path = out_dir / 'full.csv'
    df.to_csv(full_path, index=False)
    train_df, val_df, test_df = author_level_split(df, seed=args.seed)
   
    variants_path = args.variants
    if args.extract_variants:
        variants_path = str(out_dir / 'variants.csv')
        src, trg = [], []
        for rgn, ita in zip(pd.concat([train_df['rgn'], val_df['rgn']]),
                            pd.concat([train_df['ita'], val_df['ita']])):
            r, i = _tok(rgn), _tok(ita)
            if r and i:
                src.append(r); trg.append(i)
        _, n_lemmas = extract_variants(
            src, trg, variants_path, sim_threshold=args.sim_threshold)
        print(f'\nGlossario estratto da train+val: {n_lemmas} lemmi '
              f'-> {variants_path}')
    # ---- Orthographic held-out ------------------------------
   
    heldout_info = None
    split_predrop = None
    if variants_path:
        lemma2vars, var2lemmas = load_filtered_variants(
            variants_path, sim_threshold=args.sim_threshold)
        chosen, ho_report = select_heldout(
            lemma2vars, var2lemmas, train_df['rgn'], val_df['rgn'],
            test_df['rgn'], k=args.heldout_k, spacy_model=args.spacy_model)
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
        for sent in test_df['rgn']:
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
        print(f"\nHeld-out (A1): {len(chosen)} lemmi, {len(heldout_vars)} varianti | "
              f"train -{len(tr_drop)} ({heldout_info['train_dropped_pct']}%), "
              f"val -{len(va_drop)} ({heldout_info['val_dropped_pct']}%) | "
              f"{n_in_test} varianti presenti nel test")
        print(f"  glossario held-out -> {gloss_path.name}")
    keep = [c.strip() for c in args.cols.split(',')]
    for name, part in (('train', train_df), ('val', val_df), ('test', test_df)):
        part[keep].to_csv(out_dir / f'{name}.csv', index=False)
    if align_flags:
        pd.DataFrame(align_flags).to_csv(out_dir / 'alignment_report.csv', index=False)
    else:
        pd.DataFrame(columns=['author', 'poem_id', 'level', 'rgn_lines',
                              'ita_lines', 'diff', 'note']
                     ).to_csv(out_dir / 'alignment_report.csv', index=False)
    stats = compute_stats(df, train_df, val_df, test_df, align_flags,
                          n_dupes=n_dupes, split_predrop=split_predrop)
    if heldout_info is not None:
        stats['heldout'] = heldout_info
    (out_dir / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    report = render_stats_txt(stats)
    (out_dir / 'stats.txt').write_text(report)
    print('\n' + report)
    if not stats['integrity']['test_authors_disjoint_from_trainval']:
        raise SystemExit('ERROR: author leakage into test set!')
if __name__ == '__main__':
    main()
