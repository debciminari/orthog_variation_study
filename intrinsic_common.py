#!/usr/bin/env python3
import csv
import itertools
from collections import defaultdict

import numpy as np
from rapidfuzz.distance import Levenshtein


# ---------------------------------------------------------------------------
# Glossario
# ---------------------------------------------------------------------------
def norm_apostrophes(s):
    return str(s).replace("\u2019", "'").replace("\u2018", "'")


def load_glossary(path):
    lem2vars = {}
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        var_cols = [c for c in rd.fieldnames if c.lower().startswith("rgn_var")]
        for r in rd:
            lemma = r["ita_word"].strip()
            variants = []
            seen = set()
            for c in var_cols:
                v = norm_apostrophes(r.get(c, "")).strip()
                if v and v.lower() not in seen:
                    seen.add(v.lower())
                    variants.append(v)
            if len(variants) >= 2:
                lem2vars[lemma] = variants
    return lem2vars


def all_forms(lem2vars):
    forms = []
    seen = set()
    for vs in lem2vars.values():
        for v in vs:
            if v not in seen:
                seen.add(v)
                forms.append(v)
    return forms


# ---------------------------------------------------------------------------
def lemma_pairs(lem2vars):
    out = []
    for lemma, vs in lem2vars.items():
        for a, b in itertools.combinations(vs, 2):
            d = Levenshtein.distance(a.lower(), b.lower())
            out.append({"lemma": lemma, "form_a": a, "form_b": b,
                        "lev": d, "bin": lev_bin(d)})
    return out


def lev_bin(d):
    if d <= 1:
        return "low"
    if d <= 3:
        return "med"
    return "high"


# ---------------------------------------------------------------------------
def encode_forms(model, forms, batch_size=64):
    embs = model.encode(forms, normalize_embeddings=True,
                        batch_size=batch_size, show_progress_bar=False)
    return {f: np.asarray(e, dtype=np.float64) for f, e in zip(forms, embs)}


def cosine(a, b):
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
def _center(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    Kx = X @ X.T
    Ky = Y @ Y.T
    hsic = np.sum(_center(Kx) * _center(Ky))
    nx = np.sqrt(np.sum(_center(Kx) * _center(Kx)))
    ny = np.sqrt(np.sum(_center(Ky) * _center(Ky)))
    denom = nx * ny
    return float(hsic / denom) if denom > 0 else 0.0


def forms_matrix(emb_dict, forms):
    return np.vstack([emb_dict[f] for f in forms])


# ---------------------------------------------------------------------------
def fit_whitening(X):
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + 1e-8))
    return mu, W


def apply_whitening(X, mu, W, renorm=True):
    Y = (X - mu) @ W
    if renorm:
        n = np.linalg.norm(Y, axis=1, keepdims=True)
        n[n == 0] = 1.0
        Y = Y / n
    return Y
