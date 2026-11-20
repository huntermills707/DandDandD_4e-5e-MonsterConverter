"""Name similarity measures for accepting metadata-proposed matches.

The main measure is an IDF-weighted, fuzzy ROUGE-1 recall: the share of the
5e name's words that also appear in the 4e name, where rare words count more
than common ones and near-identical words match ("efreet" ~ "efreeti").

Recall rather than BLEU-style precision fits 4e naming: 4e variants add
words to a base name ("War Troll" -> "Troll"), which recall scores 1.0 but
BLEU's brevity penalty punishes. IDF weighting is what separates siblings:
"Ancient Red Dragon" vs "Ancient Gold Dragon" share the common words
"ancient" and "dragon" but differ in the rare, decisive "red"/"gold".
"""

import collections
import difflib
import math

import numpy as np

from pipeline.match_features import norm_name

# 0.85 keeps 'efreet' ~ 'efreeti' (0.92) but not 'ghost' ~ 'ghast' (0.80)
FUZZY_WORD_RATIO = 0.85


def words(name):
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in norm_name(name).split()]


def word_match(a, b):
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= FUZZY_WORD_RATIO


class NameSimilarity:
    """Similarity measures with IDF fitted on a corpus of names."""

    def __init__(self, names):
        df = collections.Counter(w for n in names for w in set(words(n)))
        self.n = len(names)
        self.df = df

    def idf(self, w):
        return math.log((self.n + 1) / (self.df.get(w, 0) + 1)) + 1

    def rouge1_recall(self, name4, name5):
        """IDF-weighted share of the 5e name's words found (fuzzily) in the 4e name."""
        w4, w5 = words(name4), words(name5)
        if not w5:
            return 0.0
        total = sum(self.idf(w) for w in w5)
        hit = sum(self.idf(w) for w in w5 if any(word_match(w, x) for x in w4))
        return hit / total

    @staticmethod
    def bleu1(name4, name5):
        """Unigram BLEU with brevity penalty, 5e name as the candidate."""
        c, r = words(name5), words(name4)
        if not c:
            return 0.0
        rc = collections.Counter(r)
        m = sum(min(v, rc[k]) for k, v in collections.Counter(c).items())
        bp = 1.0 if len(c) > len(r) else math.exp(1 - len(r) / len(c))
        return bp * m / len(c)

    @staticmethod
    def jaccard(name4, name5):
        a, b = set(words(name4)), set(words(name5))
        return len(a & b) / len(a | b) if a | b else 0.0

    @staticmethod
    def chrf(name4, name5, n=4, beta=2.0):
        """Character n-gram F-score, recall-weighted toward the 5e name."""
        a = norm_name(name4).replace(" ", "")
        b = norm_name(name5).replace(" ", "")
        ps, rs = [], []
        for k in range(1, n + 1):
            A = collections.Counter(a[i:i + k] for i in range(len(a) - k + 1))
            B = collections.Counter(b[i:i + k] for i in range(len(b) - k + 1))
            if not A or not B:
                continue
            m = sum((A & B).values())
            ps.append(m / sum(A.values()))
            rs.append(m / sum(B.values()))
        if not ps:
            return 0.0
        p, r = float(np.mean(ps)), float(np.mean(rs))
        return 0.0 if p + r == 0 else (1 + beta**2) * p * r / (beta**2 * p + r)

    def measures(self):
        return {
            "rouge1_recall_idf": self.rouge1_recall,
            "bleu1": self.bleu1,
            "chrf": self.chrf,
            "jaccard": self.jaccard,
        }
