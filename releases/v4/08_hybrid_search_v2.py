# -*- coding: utf-8 -*-
"""
Phase 8: محرك بحث هجين v2 — تحسينات حقيقية وقابلة للقياس فوق v1:

  المشكلة في v1                          | الحل في v2
  --------------------------------------|--------------------------------------
  fuzzy يعالج التشكيل فقط، يفشل مع خطأ   | rapidfuzz: مطابقة بمسافة تحرير حقيقية
  إملائي حقيقي (حرف بدل حرف)             | على كامل المفردات (182,901 كلمة)
  --------------------------------------|--------------------------------------
  BM25/FTS5 لا يوحّد الصيغ الصرفية        | BM25Okapi حقيقي على جذور ISRI
  ("استعارة" لا تُطابق "مستعار")          | (19,935 جذراً) بالإضافة لطبقة الكلمة السطحية
  --------------------------------------|--------------------------------------
  دمج الدرجات بجمع مرجّح (scores بمقاييس  | Reciprocal Rank Fusion (RRF) — يدمج
  مختلفة BM25 مقابل cosine) غير متسق     | حسب الرتبة لا الدرجة، متّسق عبر الطرق
  --------------------------------------|--------------------------------------
  FastText يقتصر على أعلى 20,000 كلمة    | يُستخدم هنا لكامل المفردات المطلوبة
  تكراراً (قيد الذاكرة الموثّق سابقاً)     | فقط لكلمات الاستعلام (لا حاجة لحساب شامل)
"""
import sqlite3, re, pickle, sys, time
import numpy as np
from collections import defaultdict
from nltk.stem.isri import ISRIStemmer
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
from gensim.models import FastText

DB = "فهرس_البلاغة_full_sqlite3_enhanced_v2_db"
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
    with open("bm25_surface.pkl", "rb") as f:
        _loaded["bm25_surface"] = pickle.load(f)
    with open("bm25_stem.pkl", "rb") as f:
        _loaded["bm25_stem"] = pickle.load(f)
    with open("bm25_passage_ids.pkl", "rb") as f:
        _loaded["bm25_passage_ids"] = pickle.load(f)
    with open("vocab_surface.pkl", "rb") as f:
        _loaded["vocab_surface"] = pickle.load(f)
    with open("root_to_words.pkl", "rb") as f:
        _loaded["root_to_words"] = pickle.load(f)

    _loaded["ft_model"] = FastText.load("fasttext_balagha.model")
    _loaded["lsa_vectors"] = np.load("passage_vectors_lsa.npy")
    with open("passage_ids.pkl", "rb") as f:
        _loaded["lsa_passage_ids"] = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        _loaded["vectorizer"] = pickle.load(f)
    with open("svd_model.pkl", "rb") as f:
        _loaded["svd"] = pickle.load(f)

    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT passage_id, book_name, pg, vol, printed_page, passage_type, text FROM passages")
    _loaded["meta"] = {r[0]: r[1:] for r in cur.fetchall()}
    print(f"[v2] lazy load done in {time.time()-t0:.1f}s", file=sys.stderr)
    return _loaded


def expand_query_fuzzy(tokens, vocab, limit_per_token=3, score_cutoff=82):
    """لكل كلمة استعلام: لو غير موجودة حرفياً في المفردات، ابحث عن أقرب
    كلمات بمسافة تحرير حقيقية (rapidfuzz) بدل إرجاع صفر نتائج كما في v1.
    يستخدم fuzz.ratio (وليس WRatio) لأن WRatio يفضّل خطأً السلاسل القصيرة
    جداً (مطابقة جزئية) — هذا الخلل تحديداً جُرِّب واكتُشِف أثناء الاختبار.
    كما يقيّد المرشحين بطول قريب من الكلمة الأصلية لتفادي ضوضاء كهذه."""
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
    """Reciprocal Rank Fusion: يدمج عدة قوائم مرتّبة (list[passage_id]) بدرجة
    متّسقة 1/(k+rank) بدل جمع درجات بمقاييس مختلفة تماماً كما في v1."""
    scores = defaultdict(float)
    for ranked in rank_lists:
        for i, pid in enumerate(ranked):
            scores[pid] += 1.0 / (k + i + 1)
    return scores


def hybrid_search_v2(query, top_k=10, debug=False):
    L = _lazy_load()
    raw_tokens = tokenize(query)
    if not raw_tokens:
        return []

    vocab = L["vocab_surface"]
    fuzzy_expanded, corrections = expand_query_fuzzy(raw_tokens, vocab)
    root_expanded = expand_query_roots(raw_tokens, L["root_to_words"])
    all_surface_query = list(set(raw_tokens + fuzzy_expanded))
    stem_query = [stemmer.stem(t) for t in set(raw_tokens + fuzzy_expanded)]

    pids = L["bm25_passage_ids"]

    # 1) BM25 على الكلمة السطحية (بعد تصحيح fuzzy)
    scores_surface = L["bm25_surface"].get_scores(all_surface_query)
    order_surface = np.argsort(-scores_surface)[:200]
    rank_surface = [pids[i] for i in order_surface if scores_surface[i] > 0]

    # 2) BM25 على الجذر (يوحّد التصريف: استعارة/مستعار/استعارات...)
    scores_stem = L["bm25_stem"].get_scores(stem_query)
    order_stem = np.argsort(-scores_stem)[:200]
    rank_stem = [pids[i] for i in order_stem if scores_stem[i] > 0]

    # 3) دلالي LSA (كما في v1)
    q_norm = " ".join(all_surface_query)
    q_vec = L["vectorizer"].transform([q_norm])
    q_lsa = L["svd"].transform(q_vec)
    lsa_vectors = L["lsa_vectors"]
    denom = (np.linalg.norm(lsa_vectors, axis=1) * np.linalg.norm(q_lsa) + 1e-9)
    sims = (lsa_vectors @ q_lsa.T).ravel() / denom
    order_lsa = np.argsort(-sims)[:200]
    lsa_pids = L["lsa_passage_ids"]
    rank_lsa = [lsa_pids[i] for i in order_lsa if sims[i] > 0]

    # 4) توسّع صرفي بالجذر: أعد بحث BM25 السطحي بالكلمات المشتقة من نفس الجذر
    scores_morph = L["bm25_surface"].get_scores(list(set(root_expanded)))
    order_morph = np.argsort(-scores_morph)[:200]
    rank_morph = [pids[i] for i in order_morph if scores_morph[i] > 0]

    # 5) توسّع بمرادفات/صيغ قريبة عبر FastText (كان مُحمَّلاً في v1 لكن غير
    #    مُستخدَم فعلياً في hybrid_search — هنا يُستخدم فعلاً لتوسيع الاستعلام
    ft_expanded = list(all_surface_query)
    ft_model = L["ft_model"]
    for tok in raw_tokens:
        try:
            sims = ft_model.wv.most_similar(tok, topn=5)
            ft_expanded.extend(w for w, _ in sims)
        except KeyError:
            pass
    scores_ft = L["bm25_surface"].get_scores(list(set(ft_expanded)))
    order_ft = np.argsort(-scores_ft)[:200]
    rank_ft = [pids[i] for i in order_ft if scores_ft[i] > 0]

    fused = rrf_fuse([rank_surface, rank_stem, rank_lsa, rank_morph, rank_ft])
    top = sorted(fused.items(), key=lambda x: -x[1])[:top_k]

    results = []
    meta = L["meta"]
    for pid, score in top:
        m = meta.get(pid)
        if not m:
            continue
        book_name, pg, vol, printed_page, ptype, text = m
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
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "الاستعارة والتشبيه"
    results, corrections = hybrid_search_v2(q, top_k=8, debug=True)
    if corrections:
        print(f"[تصحيح تلقائي] {corrections}")
    for r in results:
        print(f"[{r['rrf_score']}] {r['book_name']} صفحة {r['pg']} :: {r['text']}")
