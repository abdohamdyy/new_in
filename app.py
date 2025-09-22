# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import logging
from flask import Flask, request, jsonify

from drug_search import DrugSearch, DEFAULT_TOP_K, DEFAULT_MIN_SCORE10
from equivalents_search import EquivalentsFinder  # ملف المثائل

# ================= إعداد اللوجينغ =================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger("duaya_api")

app = Flask(__name__)

# ================= إعدادات بحث الاسم التجاري =================
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", DEFAULT_TOP_K))
SEARCH_MIN_SCORE10 = float(os.getenv("SEARCH_MIN_SCORE10", DEFAULT_MIN_SCORE10))
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", 50))

searcher = DrugSearch(top_k=SEARCH_TOP_K, min_score10=SEARCH_MIN_SCORE10)

# ================= إعدادات بحث المثائل =================
EQ_TOP_K = int(os.getenv("EQ_TOP_K", 800))
EQ_MIN_BASE10 = float(os.getenv("EQ_MIN_BASE10", 0.0))
EQ_LIMIT = int(os.getenv("EQ_LIMIT", 50))

eq_finder = EquivalentsFinder(top_k=EQ_TOP_K, min_base10=EQ_MIN_BASE10)

# ================= أدوات مساعدة =================
def parse_bool(val, default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "on", "اه", "ايوه")

@app.get("/health")
def health():
    return jsonify({"ok": True})

# ================= بحث الاسم التجاري =================
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
        logger.exception("search_get failed")
        return jsonify({"error": str(e)}), 500

@app.post("/search")
def search_post():
    """
    POST /search
    Body: {"query":"اسم الدواء","limit":50,"minscore":5,"topk":400}
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
        logger.exception("search_post failed")
        return jsonify({"error": str(e)}), 500

# ==================== Endpoints المثائل ====================
@app.get("/equivalents")
def equivalents_get():
    """
    GET /equivalents?active=باراسيتامول&mg=500&form=أقراص&tol=0&allow_per_ml=true&strict_form=true&limit=50&topk=800&minbase=0&debug=1
    - active: إجباري (المادة الفعّالة/الاسم العلمي)
    - mg: إجباري (تركيز بالمليجرام)
    - form: اختياري (أقراص/كبسول/شراب/أمبول/...)
    - debug: اختياري (لو true يطلع لوجز تفصيلية في docker logs)
    """
    active = request.args.get("active", type=str)
    mg = request.args.get("mg", type=str)
    if not active or not active.strip():
        return jsonify({"error": "missing 'active' parameter"}), 400
    try:
        mg_val = float(str(mg).strip())
    except Exception:
        return jsonify({"error": "invalid 'mg' parameter, must be a number (mg)"}), 400

    form = request.args.get("form", type=str)
    tol = request.args.get("tol", type=float) or 0.0
    allow_per_ml = parse_bool(request.args.get("allow_per_ml"), True)
    strict_form = parse_bool(request.args.get("strict_form"), True)
    limit = request.args.get("limit", type=int) or EQ_LIMIT
    topk = request.args.get("topk", type=int) or EQ_TOP_K
    minbase = request.args.get("minbase", type=float) or EQ_MIN_BASE10
    debug_flag = parse_bool(request.args.get("debug"), False)

    try:
        if debug_flag:
            logger.setLevel(logging.DEBUG)
        logger.info(f"/equivalents GET active='{active}' mg={mg_val} form='{form}' tol={tol} strict={strict_form} per_ml={allow_per_ml} topk={topk} minbase={minbase}")

        local_finder = eq_finder
        if topk != EQ_TOP_K or minbase != EQ_MIN_BASE10:
            local_finder = EquivalentsFinder(top_k=topk, min_base10=minbase)

        eq_results = local_finder.find_equivalents(
            active_query=active.strip(),
            target_mg=mg_val,
            tolerance_mg=tol,
            allow_per_ml=allow_per_ml,
            target_form=form,
            strict_form=strict_form,
            limit=limit,
            debug=debug_flag,
        )
        exclude_ids = {str(r.get('id')) for r in eq_results}
        alt_results = local_finder.find_alternatives(
            active_query=active.strip(),
            target_form=form,
            strict_form=strict_form,
            limit=limit,
            exclude_ids=exclude_ids,
            debug=debug_flag,
        )
        logger.info(f"/equivalents -> equivalents={len(eq_results)} alternatives={len(alt_results)}")
        return app.response_class(
            response=json.dumps({
                "equivalents": eq_results,
                "alternatives": alt_results,
            }, ensure_ascii=False),
            status=200,
            mimetype="application/json; charset=utf-8",
        )
    except Exception as e:
        logger.exception("equivalents_get failed")
        return jsonify({"error": str(e)}), 500

@app.post("/equivalents")
def equivalents_post():
    """
    POST /equivalents
    Body مثال:
      {
        "active": "باراسيتامول",
        "mg": 500,
        "form": "أقراص",
        "tol": 0,
        "allow_per_ml": true,
        "strict_form": true,
        "limit": 50,
        "topk": 800,
        "minbase": 0.0,
        "debug": true
      }
    """
    if not request.is_json:
        return jsonify({"error": "expected application/json body"}), 400
    payload = request.get_json(silent=True) or {}

    active = (payload.get("active") or "").strip()
    if not active:
        return jsonify({"error": "missing 'active' in JSON body"}), 400

    try:
        mg_val = float(payload.get("mg"))
    except Exception:
        return jsonify({"error": "invalid 'mg' in JSON body, must be a number (mg)"}), 400

    form = payload.get("form")
    tol = float(payload.get("tol") or 0.0)
    allow_per_ml = parse_bool(payload.get("allow_per_ml"), True)
    strict_form = parse_bool(payload.get("strict_form"), True)
    limit = int(payload.get("limit") or EQ_LIMIT)
    topk = int(payload.get("topk") or EQ_TOP_K)
    minbase = float(payload.get("minbase") or EQ_MIN_BASE10)
    debug_flag = parse_bool(payload.get("debug"), False)

    try:
        if debug_flag:
            logger.setLevel(logging.DEBUG)
        logger.info(f"/equivalents POST active='{active}' mg={mg_val} form='{form}' tol={tol} strict={strict_form} per_ml={allow_per_ml} topk={topk} minbase={minbase}")

        local_finder = eq_finder
        if topk != EQ_TOP_K or minbase != EQ_MIN_BASE10:
            local_finder = EquivalentsFinder(top_k=topk, min_base10=minbase)

        eq_results = local_finder.find_equivalents(
            active_query=active,
            target_mg=mg_val,
            tolerance_mg=tol,
            allow_per_ml=allow_per_ml,
            target_form=form,
            strict_form=strict_form,
            limit=limit,
            debug=debug_flag,
        )
        exclude_ids = {str(r.get('id')) for r in eq_results}
        alt_results = local_finder.find_alternatives(
            active_query=active,
            target_form=form,
            strict_form=strict_form,
            limit=limit,
            exclude_ids=exclude_ids,
            debug=debug_flag,
        )
        logger.info(f"/equivalents -> equivalents={len(eq_results)} alternatives={len(alt_results)}")
        return app.response_class(
            response=json.dumps({
                "equivalents": eq_results,
                "alternatives": alt_results,
            }, ensure_ascii=False),
            status=200,
            mimetype="application/json; charset=utf-8",
        )
    except Exception as e:
        logger.exception("equivalents_post failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # في التطوير بنستخدم Flask مباشرة؛ داخل Docker يفضل gunicorn
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5543")), debug=False)
