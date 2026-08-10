# -*- coding: utf-8 -*-
"""
Phase 7: بناء طبقات تحسين حقيقية فوق ما هو موجود:
  1) BM25 حقيقي (rank_bm25) على الرموز بعد تجذيع ISRI (يوحّد "استعارة/مستعار/الاستعارات")
  2) قاموس رموز فريدة لكل الفقرات لدعم fuzzy حقيقي بمسافة تحرير (rapidfuzz) بدل التشكيل فقط
  3) خرائط جذر -> كلمات لدعم التوسّع الصرفي في البحث
يحفظ النتائج كـ pickle لإعادة الاستخدام السريع (بدون إعادة البناء في كل بحث).
"""
import sqlite3, re, pickle, time
from collections import defaultdict
from nltk.stem.isri import ISRIStemmer
from rank_bm25 import BM25Okapi

DB = "فهرس_البلاغة_full_sqlite3_enhanced_v2_db"
AR_TOKEN_RE = re.compile(r"[\u0621-\u064A]+")
DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")

stemmer = ISRIStemmer()


def normalize_ar(text):
    text = DIACRITICS_RE.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def tokenize(text):
    return [normalize_ar(t) for t in AR_TOKEN_RE.findall(text) if len(t) > 1]


def main():
    t0 = time.time()
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT passage_id, text FROM passages ORDER BY passage_id")
    rows = cur.fetchall()
    print(f"loaded {len(rows)} passages in {time.time()-t0:.1f}s")

    passage_ids = []
    tokenized_corpus = []      # للبحث بالكلمات كما هي (بعد التطبيع)
    stemmed_corpus = []        # للبحث بالجذر (يوحّد التصريف)
    vocab_surface = set()      # كل الكلمات الفعلية لدعم fuzzy
    root_to_words = defaultdict(set)

    t0 = time.time()
    for pid, text in rows:
        toks = tokenize(text or "")
        stems = []
        for tk in toks:
            vocab_surface.add(tk)
            try:
                r = stemmer.stem(tk)
            except Exception:
                r = tk
            stems.append(r)
            root_to_words[r].add(tk)
        passage_ids.append(pid)
        tokenized_corpus.append(toks)
        stemmed_corpus.append(stems)
    print(f"tokenized+stemmed in {time.time()-t0:.1f}s, vocab_surface={len(vocab_surface)}, roots={len(root_to_words)}")

    t0 = time.time()
    bm25_surface = BM25Okapi(tokenized_corpus)
    print(f"BM25 (surface words) built in {time.time()-t0:.1f}s")

    t0 = time.time()
    bm25_stem = BM25Okapi(stemmed_corpus)
    print(f"BM25 (stemmed/root) built in {time.time()-t0:.1f}s")

    with open("bm25_surface.pkl", "wb") as f:
        pickle.dump(bm25_surface, f)
    with open("bm25_stem.pkl", "wb") as f:
        pickle.dump(bm25_stem, f)
    with open("bm25_passage_ids.pkl", "wb") as f:
        pickle.dump(passage_ids, f)
    with open("vocab_surface.pkl", "wb") as f:
        pickle.dump(sorted(vocab_surface), f)
    with open("root_to_words.pkl", "wb") as f:
        pickle.dump(dict(root_to_words), f)

    print("saved: bm25_surface.pkl, bm25_stem.pkl, bm25_passage_ids.pkl, vocab_surface.pkl, root_to_words.pkl")


if __name__ == "__main__":
    main()
