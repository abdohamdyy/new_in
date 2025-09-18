# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
from flask import Flask, request, jsonify
from drug_search import DrugSearch, DEFAULT_TOP_K, DEFAULT_MIN_SCORE10

app = Flask(__name__)

# إعدادات قابلة للتعديل من env (اختياري)
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", DEFAULT_TOP_K))
SEARCH_MIN_SCORE10 = float(os.getenv("SEARCH_MIN_SCORE10", DEFAULT_MIN_SCORE10))
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", 50))

# مهيّئ واحد للـ DrugSearch (أفضل للأداء)
searcher = DrugSearch(top_k=SEARCH_TOP_K, min_score10=SEARCH_MIN_SCORE10)

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/search")
def search_get():
    """
    GET /search?q=اسم_الدواء&limit=50&minscore=5&topk=400
    يرجّع قائمة JSON بكل المعلومات (meta كاملة)
    """
    q = request.args.get("q")
    if not q or not q.strip():
        return jsonify({"error": "missing 'q' parameter"}), 400

    limit = request.args.get("limit", type=int) or SEARCH_LIMIT
    minscore = request.args.get("minscore", type=float) or SEARCH_MIN_SCORE10
    topk = request.args.get("topk", type=int) or SEARCH_TOP_K

    try:
        results = searcher.search_one(q.strip(), top_k=topk, min_score10=minscore, limit=limit)
        return app.response_class(
            response=json.dumps(results, ensure_ascii=False),
            status=200,
            mimetype="application/json; charset=utf-8",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/search")
def search_post():
    """
    POST /search
    Body (JSON): {"query": "اسم الدواء", "limit": 50, "minscore": 5, "topk": 400}
    بيركّز على query واحدة فقط.
    """
    if not request.is_json:
        return jsonify({"error": "expected application/json body"}), 400
    payload = request.get_json(silent=True) or {}
    q = (payload.get("query") or "").strip()
    if not q:
        return jsonify({"error": "missing 'query' in JSON body"}), 400

    limit = int(payload.get("limit") or SEARCH_LIMIT)
    minscore = float(payload.get("minscore") or SEARCH_MIN_SCORE10)
    topk = int(payload.get("topk") or SEARCH_TOP_K)

    try:
        results = searcher.search_one(q, top_k=topk, min_score10=minscore, limit=limit)
        return app.response_class(
            response=json.dumps(results, ensure_ascii=False),
            status=200,
            mimetype="application/json; charset=utf-8",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # شغّل السيرفر المحلي
    # لو داخل Docker/Gunicorn سيب بلوك التشغيل لمدير التشغيل
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5543")), debug=False)
