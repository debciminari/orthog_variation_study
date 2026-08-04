"""
Word alignment
"""
import io

from eflomal import Aligner


def word_alignment(src_sentences, trg_sentences):
    src_txt = "\n".join(" ".join(t) for t in src_sentences) + "\n"
    trg_txt = "\n".join(" ".join(t) for t in trg_sentences) + "\n"

    Aligner().align(
        io.StringIO(src_txt), io.StringIO(trg_txt),
        links_filename_fwd="/tmp/nds_fwd.align",
        links_filename_rev="/tmp/nds_rev.align",
        quiet=True,
    )
    fwd = open("/tmp/nds_fwd.align").read().splitlines()
    rev = open("/tmp/nds_rev.align").read().splitlines()

    def parse(line):
        out = set()
        for tok in line.split():
            a, b = tok.split("-")
            out.add((int(a), int(b)))
        return out

    return [parse(f) & parse(r) for f, r in zip(fwd, rev)]
