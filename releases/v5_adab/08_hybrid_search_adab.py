# -*- coding: utf-8 -*-
"""
محرك بحث لقاعدة "فهرس الأدب" (part1) — نفس منهجية v2 المُثبَتة على قاعدة
البلاغة (BM25 حقيقي على الجذر ISRI + fuzzy بمسافة تحرير حقيقية + RRF)، لكن
بدون طبقة FastText/LSA الدلالية: لم تُبنَ لهذه القاعدة لأن حجم المدونة هنا
5 أضعاف (358,934 فقرة مقابل 73,199)، وتدريب FastText من الصفر على هذا الحجم
ضمن حاوية بذاكرة 3.9GB يحتاج بناءً منفصلاً وأطول. يمكن إضافتها لاحقاً بنفس
أسلوب v3 الأصلي إن رغبت.
"""
import sqlite3, re, pickle, sys, time
import numpy as np
from collections import defaultdict
from nltk.stem.isri import ISRIStemmer
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

DB = "فهرس_الادب_part1_sqlite3"
AR_TOKEN_RE = re.compile(r"[\u0621-\u064A]+")
DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")
stemmer = ISRIStemmer()

_loaded = {}


def normalize_ar(text):
    text = DIACRITICS_RE.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def tokenize(text):
    return [normalize_ar(t) for t in AR_TOKEN_RE.findall(text) if len(t) > 1]


def _lazy_load():
    if _loaded:
        return _loaded
    t0 = time.time()
    with open("bm25_surface_adab.pkl", "rb") as f:
        _loaded["bm25_surface"] = pickle.load(f)
    with open("bm25_stem_adab.pkl", "rb") as f:
        _loaded["bm25_stem"] = pickle.load(f)
    with open("bm25_passage_ids_adab.pkl", "rb") as f:
        _loaded["bm25_passage_ids"] = pickle.load(f)
    with open("vocab_surface_adab.pkl", "rb") as f:
        _loaded["vocab_surface"] = set(pickle.load(f))
    with open("root_to_words_adab.pkl", "rb") as f:
        _loaded["root_to_words"] = pickle.load(f)

    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT passage_id, book_name, pg, printed_page, passage_type, text FROM passages")
    _loaded["meta"] = {r[0]: r[1:] for r in cur.fetchall()}
    print(f"[adab] lazy load done in {time.time()-t0:.1f}s", file=sys.stderr)
    return _loaded


def expand_query_fuzzy(tokens, vocab, limit_per_token=3, score_cutoff=82):
    expanded = list(tokens)
    corrections = {}
    for tok in tokens:
        if tok in vocab:
            continue
        candidates = [v for v in vocab if abs(len(v) - len(tok)) <= 2 and len(v) >= 3]
        matches = rf_process.extract(tok, candidates, scorer=rf_fuzz.ratio,
                                      limit=limit_per_token, score_cutoff=score_cutoff)
        if matches:
            corrections[tok] = [m[0] for m in matches]
            expanded.extend(m[0] for m in matches)
    return expanded, corrections


def expand_query_roots(tokens, root_to_words, max_per_root=5):
    expanded = list(tokens)
    for tok in tokens:
        r = stemmer.stem(tok)
        words = root_to_words.get(r)
        if words:
            expanded.extend(list(words)[:max_per_root])
    return expanded


def rrf_fuse(rank_lists, k=60):
    scores = defaultdict(float)
    for ranked in rank_lists:
        for i, pid in enumerate(ranked):
            scores[pid] += 1.0 / (k + i + 1)
    return scores


def hybrid_search_adab(query, top_k=10, debug=False):
    L = _lazy_load()
    raw_tokens = tokenize(query)
    if not raw_tokens:
        return [], {}

    vocab = L["vocab_surface"]
    fuzzy_expanded, corrections = expand_query_fuzzy(raw_tokens, vocab)
    root_expanded = expand_query_roots(raw_tokens, L["root_to_words"])
    all_surface_query = list(set(raw_tokens + fuzzy_expanded))
    stem_query = [stemmer.stem(t) for t in set(raw_tokens + fuzzy_expanded)]

    pids = L["bm25_passage_ids"]

    scores_surface = L["bm25_surface"].get_scores(all_surface_query)
    order_surface = np.argsort(-scores_surface)[:200]
    rank_surface = [pids[i] for i in order_surface if scores_surface[i] > 0]

    scores_stem = L["bm25_stem"].get_scores(stem_query)
    order_stem = np.argsort(-scores_stem)[:200]
    rank_stem = [pids[i] for i in order_stem if scores_stem[i] > 0]

    scores_morph = L["bm25_surface"].get_scores(list(set(root_expanded)))
    order_morph = np.argsort(-scores_morph)[:200]
    rank_morph = [pids[i] for i in order_morph if scores_morph[i] > 0]

    fused = rrf_fuse([rank_surface, rank_stem, rank_morph])
    top = sorted(fused.items(), key=lambda x: -x[1])[:top_k]

    results = []
    meta = L["meta"]
    for pid, score in top:
        m = meta.get(pid)
        if not m:
            continue
        book_name, pg, printed_page, ptype, text = m
        results.append({
            "passage_id": pid,
            "rrf_score": round(score, 5),
            "book_name": book_name,
            "pg": pg,
            "printed_page": printed_page,
            "type": ptype,
            "text": text[:180],
        })

    if debug:
        print("query tokens:", raw_tokens, file=sys.stderr)
        print("fuzzy corrections:", corrections, file=sys.stderr)

    return results, corrections


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "الأدب والشعر"
    results, corrections = hybrid_search_adab(q, top_k=8, debug=True)
    if corrections:
        print(f"[تصحيح تلقائي] {corrections}")
    for r in results:
        print(f"[{r['rrf_score']}] {r['book_name']} صفحة {r['pg']} :: {r['text']}")
