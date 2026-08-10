# -*- coding: utf-8 -*-
"""
واجهة API محدّثة تستخدم hybrid_search_v2 (BM25 حقيقي بالجذر + fuzzy بمسافة
تحرير + FastText موصول فعلياً + LSA + دمج RRF) بدل hybrid_search v1.
تشغيل: python3 09_search_api_v2.py
ثم:   curl "http://localhost:8009/search?q=الاستعارة&top_k=10"
"""
from flask import Flask, request, jsonify
import importlib.util, os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("hybrid_v2", os.path.join(_here, "08_hybrid_search_v2.py"))
hv2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hv2)

app = Flask(__name__)


@app.route("/search", methods=["GET"])
def search():
    q = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", 10))
    if not q:
        return jsonify({"error": "أضف معامل q في الرابط، مثل ?q=الاستعارة"}), 400
    results, corrections = hv2.hybrid_search_v2(q, top_k=top_k)
    return jsonify({
        "query": q,
        "auto_corrections": corrections,
        "count": len(results),
        "results": results,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8009)
