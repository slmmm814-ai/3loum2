# -*- coding: utf-8 -*-
"""
نفس منهجية 07_build_bm25_and_stems.py لكن مُعاد هيكلتها لتعمل ضمن حاوية بذاكرة
محدودة (~3.9GB) مع مدونة أكبر بخمس مرات (358,934 فقرة مقابل 73,199).

التعديلات لتفادي نفاد الذاكرة (Killed سابقاً):
  1) مرحلتان منفصلتان (سطحي ثم جذري) بدل بناء كل شيء دفعة واحدة، مع gc.collect()
     بينهما لتحرير الذاكرة فعلياً قبل بدء المرحلة الثانية.
  2) sys.intern() على كل رمز — الكلمات العربية تتكرر آلاف المرات، والدمج المرجعي
     (interning) يقلل استهلاك الذاكرة فعلياً بدل تخزين نسخة نصية جديدة لكل تكرار.
  3) تحديد سقف 40 كلمة لكل جذر في root_to_words بدل تخزين كل الصيغ بلا حدود.
"""
import sqlite3, re, pickle, time, gc, sys
from collections import defaultdict
from nltk.stem.isri import ISRIStemmer
from rank_bm25 import BM25Okapi

DB = "فهرس_الادب_part1_sqlite3"
AR_TOKEN_RE = re.compile(r"[\u0621-\u064A]+")
DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")
stemmer = ISRIStemmer()
MAX_WORDS_PER_ROOT = 40


def normalize_ar(text):
    text = DIACRITICS_RE.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def tokenize(text):
    return [sys.intern(normalize_ar(t)) for t in AR_TOKEN_RE.findall(text) if len(t) > 1]


def load_passages():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT passage_id, text FROM passages ORDER BY passage_id")
    return cur.fetchall()


def phase_surface(rows):
    t0 = time.time()
    passage_ids = []
    tokenized_corpus = []
    vocab_surface = set()
    for pid, text in rows:
        toks = tokenize(text or "")
        vocab_surface.update(toks)
        passage_ids.append(pid)
        tokenized_corpus.append(toks)
    print(f"[surface] tokenized in {time.time()-t0:.1f}s, vocab={len(vocab_surface)}", file=sys.stderr)

    t0 = time.time()
    bm25_surface = BM25Okapi(tokenized_corpus)
    print(f"[surface] BM25 built in {time.time()-t0:.1f}s", file=sys.stderr)

    with open("bm25_surface_adab.pkl", "wb") as f:
        pickle.dump(bm25_surface, f)
    with open("bm25_passage_ids_adab.pkl", "wb") as f:
        pickle.dump(passage_ids, f)
    with open("vocab_surface_adab.pkl", "wb") as f:
        pickle.dump(sorted(vocab_surface), f)

    del bm25_surface, tokenized_corpus, vocab_surface, passage_ids
    gc.collect()
    print("[surface] saved + freed memory", file=sys.stderr)


def phase_stem(rows):
    t0 = time.time()
    stemmed_corpus = []
    root_to_words = defaultdict(set)
    stem_cache = {}
    for pid, text in rows:
        toks = tokenize(text or "")
        stems = []
        for tk in toks:
            r = stem_cache.get(tk)
            if r is None:
                try:
                    r = sys.intern(stemmer.stem(tk))
                except Exception:
                    r = tk
                stem_cache[tk] = r
            if len(root_to_words[r]) < MAX_WORDS_PER_ROOT:
                root_to_words[r].add(tk)
            stems.append(r)
        stemmed_corpus.append(stems)
    print(f"[stem] tokenized+stemmed in {time.time()-t0:.1f}s, roots={len(root_to_words)}", file=sys.stderr)

    t0 = time.time()
    bm25_stem = BM25Okapi(stemmed_corpus)
    print(f"[stem] BM25 built in {time.time()-t0:.1f}s", file=sys.stderr)

    with open("bm25_stem_adab.pkl", "wb") as f:
        pickle.dump(bm25_stem, f)
    with open("root_to_words_adab.pkl", "wb") as f:
        pickle.dump(dict(root_to_words), f)

    print("[stem] saved", file=sys.stderr)


def main():
    t0 = time.time()
    rows = load_passages()
    print(f"loaded {len(rows)} passages in {time.time()-t0:.1f}s", file=sys.stderr)

    phase_surface(rows)
    gc.collect()
    phase_stem(rows)
    print("DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
