#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np

def norm(s) -> str:
    return unicodedata.normalize("NFC", str(s)).strip()


def jaccard_words(a: str, b: str) -> float:
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / max(1, len(wa | wb))


def levenshtein_ratio(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    try:
        import Levenshtein  
        d = Levenshtein.distance(a, b)
        return 1.0 - d / max(len(a), len(b))
    except ImportError:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()


def word_edit_distance(a: str, b: str) -> int:
    ta = re.findall(r"\w+", a.lower())
    tb = re.findall(r"\w+", b.lower())
    n, m = len(ta), len(tb)
    if n == 0 and m == 0:
        return 0
    if n == 0 or m == 0:
        return max(n, m)

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ta[i - 1] != tb[j - 1]))
            prev = cur
    return dp[m]


def check_gpu(require: bool = True) -> bool:
    import torch

    if not torch.cuda.is_available():
        print(
            "[GPU CHECK] torch.cuda.is_available() = False -> nessuna GPU usabile.\n"
            "  Probabile mismatch torch/driver CUDA. Controlla:\n"
            "    nvidia-smi\n"
            "    cat /proc/driver/nvidia/version   (versione driver)\n"
            '    python -c "import torch; print(torch.version.cuda)"  (CUDA di torch)\n'
            "  Installa un torch compilato per una CUDA <= max del driver (es. cu121).",
            file=sys.stdout,
            flush=True,
        )
        if require:
            sys.exit(1)
        return False
    try:
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free, _ = torch.cuda.mem_get_info()
        free_gb = free / 1e9
        _ = torch.zeros(8, device="cuda")
        torch.cuda.synchronize()
        print(
            f"[GPU CHECK] OK: {name} | {total:.1f}GB tot, {free_gb:.1f}GB liberi | "
            f"torch {torch.__version__} (CUDA {torch.version.cuda})",
            file=sys.stdout,
            flush=True,
        )
        if free_gb < 6.0:
            print(
                f"[GPU CHECK] nota: {free_gb:.1f}GB liberi (serve solo a LaBSE; "
                f"la generazione gira a parte). Se e' poco, un altro processo occupa memoria.",
                file=sys.stdout,
                flush=True,
            )
        return True
    except Exception as e: 
        print(
            f"[GPU CHECK] CUDA presente ma inutilizzabile: {e}\n"
            "  Il runtime non riesce a usare la GPU (mismatch driver/runtime).",
            file=sys.stdout,
            flush=True,
        )
        if require:
            sys.exit(1)
        return False


def _hf_cache_dir() -> str:
    import os

    root = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
    return root if root.rstrip("/").endswith("hub") else os.path.join(root, "hub")


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


class LabseEncoder:
    def __init__(self, model_name: str = "sentence-transformers/LaBSE"):
        import os

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, cache_folder=_hf_cache_dir())

    def encode(self, texts, batch_size: int = 64):
        return self.model.encode(
            texts, normalize_embeddings=True, batch_size=batch_size
        )

    def cos_to(self, ref_text: str, cands: list[str]) -> list[float]:
        if not cands:
            return []
        e_ref = self.encode(ref_text)
        e_c = self.encode(cands)
        return [cos(e_ref, e) for e in e_c]


def read_pairs(infile: str, src_col: str, tgt_col: str) -> list[dict]:
    if infile.endswith(".jsonl"):
        return [
            json.loads(l)
            for l in Path(infile).read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
    with open(infile, newline="", encoding="utf-8") as f:
        return [
            {"id": i, "src": norm(r[src_col]), "true": norm(r[tgt_col])}
            for i, r in enumerate(csv.DictReader(f))
        ]


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(l)
        for l in Path(path).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dump_json(path: str | Path, obj) -> None:
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def quantile(arr, p: float):
    a = np.asarray(arr, dtype=float)
    return round(float(np.quantile(a, p)), 4) if a.size else None


class Timer:

    def __init__(self, label: str | None = None, stream=None):
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.seconds = 0.0

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self._t0
        if self.label:
            print(f"[time] {self.label}: {self.seconds:.2f}s",
                  file=self.stream, flush=True)
        return False
