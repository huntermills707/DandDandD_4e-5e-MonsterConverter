"""Pairwise metadata features for 4e <-> 5e matching (no numeric stats).

META_FEATURES are non-numeric metadata agreement, TF-IDF similarity of the
masked block text, and sentence-embedding similarities. Names are removed
from everything the model sees: its training labels come from names, so
name features would leak (name evidence is applied separately in
pipeline/matcher.py).
"""

import hashlib
import re
from pathlib import Path

import numpy as np

from pipeline.metadata import SIZES, clean_text, profile_text

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "processed" / "embeddings"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

META_FEATURES = [
    "size_eq", "size_dist", "types_jacc", "moral_eq", "lang_jacc",
    "lang_both_none", "senses_jacc", "moves_jacc", "skills_jacc",
    "attacks_jacc", "damage_jacc", "emb_cos", "text_tfidf_cos", "text_emb_cos",
]

AGE_SYNONYMS = {"elder": "ancient"}


def norm_name(name):
    s = re.sub(r"[^\w\s]", " ", name.lower())
    s = re.sub(r"\d+", "", s)
    return " ".join(AGE_SYNONYMS.get(t, t) for t in s.split())


def name_tokens(name):
    return {t.rstrip("s") for t in norm_name(name).split() if len(t) > 2}


def strip_own_name(profile):
    """Copy of the profile with words of its own names removed from attack
    words (4e powers often embed the monster name: 'Hobgoblin Resilience')."""
    own = set()
    for n in [profile["name"], *profile["alt_names"], profile["family"] or ""]:
        own |= name_tokens(n)
    p = dict(profile)
    p["attacks"] = {w for w in profile["attacks"] if w.rstrip("s") not in own}
    full_words = set()
    for n in [profile["name"], *profile["alt_names"], profile["family"] or ""]:
        full_words |= set(norm_name(n).split())
    p["text"] = clean_text(profile["text"], sorted(full_words | own, key=len, reverse=True))
    return p


def jacc(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def embed(texts, model_name=DEFAULT_MODEL):
    """Unit-normalized sentence embeddings, cached on disk by content hash."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((model_name + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    vecs = model.encode(texts, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False)
    np.save(path, vecs)
    return vecs


def tfidf_cos(texts4, texts5):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5,
                          ngram_range=(1, 2), stop_words="english")
    M = vec.fit_transform(list(texts4) + list(texts5))
    return (M[: len(texts4)] @ M[len(texts4):].T).toarray()


def meta_matrix(p4s, p5s, model_name=DEFAULT_MODEL):
    """Feature tensor [n4, n5, len(META_FEATURES)] over all pairs."""
    p4s = [strip_own_name(p) for p in p4s]
    p5s = [strip_own_name(p) for p in p5s]
    e4 = embed([profile_text(p) for p in p4s], model_name)
    e5 = embed([profile_text(p) for p in p5s], model_name)
    emb_cos = e4 @ e5.T
    t4, t5 = [p["text"] for p in p4s], [p["text"] for p in p5s]
    text_tfidf = tfidf_cos(t4, t5)
    text_emb = embed(t4, model_name) @ embed(t5, model_name).T

    size_idx = {s: i for i, s in enumerate(SIZES)}
    X = np.zeros((len(p4s), len(p5s), len(META_FEATURES)))
    for i, a in enumerate(p4s):
        for j, b in enumerate(p5s):
            sa, sb = size_idx.get(a["size"]), size_idx.get(b["size"])
            if "any" in (a["moral"], b["moral"]):
                moral = 0.5
            else:
                moral = float(a["moral"] == b["moral"])
            X[i, j] = [
                float(sa is not None and sa == sb),
                abs(sa - sb) if sa is not None and sb is not None else 2.0,
                jacc(a["types"], b["types"]),
                moral,
                jacc(a["languages"], b["languages"]),
                float(not a["languages"] and not b["languages"]),
                jacc(a["senses"], b["senses"]),
                jacc(a["moves"], b["moves"]),
                jacc(a["skills"], b["skills"]),
                jacc(a["attacks"], b["attacks"]),
                jacc(a["damage"], b["damage"]),
                emb_cos[i, j],
                text_tfidf[i, j],
                text_emb[i, j],
            ]
    return X

