import re
import unicodedata

WORD_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", flags=re.UNICODE)


def _nfc(s):
    return unicodedata.normalize("NFC", str(s))


def normalize_text(sentence):
    s = _nfc(sentence).lower().strip()
    return s.replace("\u2019", "'").replace("\u2018", "'")


def tokenize(sentence):
    return [t for t in WORD_RE.findall(normalize_text(sentence)) if t]
