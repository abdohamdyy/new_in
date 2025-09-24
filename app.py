# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import logging
import hmac
import hashlib
import time
from functools import wraps
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify, g

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

# ================= أمان HMAC حسب كل عميل =================
# يمكن تعريف الأسرار عبر متغير بيئة JSON: HMAC_CLIENT_SECRETS='{"duaya_index":"secret1","salamtk":"secret2"}'
# أو عبر متغيرات بيئة منفصلة: HMAC_SECRET_DUAYA_INDEX=... , HMAC_SECRET_SALAMTK=...
HMAC_TTL_SECONDS = int(os.getenv("HMAC_TTL_SECONDS", "300"))  # مهلة التوقيع (ثواني)

def _load_client_secrets() -> Dict[str, str]:
    secrets: Dict[str, str] = {}
    raw = os.getenv("HMAC_CLIENT_SECRETS", "").strip()
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(k, str) and isinstance(v, str) and v:
                        secrets[k.strip().lower()] = v.strip()
        except Exception:
            logger.warning("Failed to parse HMAC_CLIENT_SECRETS; falling back to HMAC_SECRET_* vars")
    # دعم HMAC_SECRET_*
    prefix = "HMAC_SECRET_"
    for env_name, env_value in os.environ.items():
        if env_name.startswith(prefix) and env_value:
            client = env_name[len(prefix):].strip().lower()
            if client and client not in secrets:
                secrets[client] = env_value.strip()
    return secrets

CLIENT_SECRETS: Dict[str, str] = _load_client_secrets()

def _get_client_secret(client_id: str) -> Optional[str]:
    if not client_id:
        return None
    return CLIENT_SECRETS.get(str(client_id).strip().lower())

def _build_payload_to_sign(req) -> str:
    # نبني تمثيلاً ثابتاً للباراميترز/البودي لضمان تطابق الحساب بين العميل والسيرفر
    if req.method == "GET":
        items: Dict[str, Any] = {}
        # request.args قد تحتوي على مفاتيح متعددة القيم؛ نحولها إلى list عند الحاجة
        auth_keys = {"client", "timestamp", "ts", "hashkey", "signature"}
        for key in sorted(req.args.keys()):
            if key.strip().lower() in auth_keys:
                continue
            values = req.args.getlist(key)
            if len(values) == 1:
                items[key] = values[0]
            else:
                items[key] = values
        try:
            return json.dumps(items, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except Exception:
            return ""
    else:
        if req.is_json:
            payload = req.get_json(silent=True)
            if isinstance(payload, dict):
                # احذف مفاتيح الأوث
                for k in ["client", "client_id", "timestamp", "ts", "hashKey", "signature"]:
                    if k in payload:
                        payload.pop(k)
            try:
                return json.dumps(payload if payload is not None else {}, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            except Exception:
                return ""
        data = req.get_data(cache=False) or b""
        try:
            return data.decode("utf-8")
        except Exception:
            return ""

def _compute_signature(secret: str, method: str, path: str, timestamp: str, payload_str: str) -> str:
    base = f"method={method.upper()}&path={path}&ts={timestamp}&payload={payload_str}"
    return hmac.new(secret.encode("utf-8"), base.encode("utf-8"), hashlib.sha256).hexdigest()

def require_hmac(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # ندعم إما الهيدر القياسي أو نمط الحقول داخل الاستعلام/البودي (client,timestamp,hashKey)
        client_id = request.headers.get("X-Client-Id") or request.args.get("client")
        ts = request.headers.get("X-Timestamp") or request.args.get("timestamp")
        sig = request.headers.get("X-Signature") or request.args.get("hashKey")

        if (not client_id or not ts or not sig) and request.is_json:
            body = request.get_json(silent=True) or {}
            client_id = client_id or body.get("client") or body.get("client_id")
            ts = ts or str(body.get("timestamp") or body.get("ts") or "")
            sig = sig or body.get("hashKey") or body.get("signature")

        if not client_id or not ts or not sig:
            return jsonify({"error": "missing HMAC auth (client/timestamp/signature)"}), 401

        try:
            ts_int = int(str(ts))
        except Exception:
            return jsonify({"error": "invalid timestamp"}), 401

        now = int(time.time())
        if abs(now - ts_int) > HMAC_TTL_SECONDS:
            return jsonify({"error": "timestamp expired"}), 401

        secret = _get_client_secret(str(client_id))
        if not secret:
            return jsonify({"error": "unknown client"}), 401

        payload_str = _build_payload_to_sign(request)
        expected = _compute_signature(secret, request.method, request.path, str(ts_int), payload_str)
        if not hmac.compare_digest(expected, str(sig)):
            return jsonify({"error": "invalid signature"}), 401

        # نمرر هوية العميل عبر g
        try:
            g.client_id = str(client_id)
        except Exception:
            pass
        return func(*args, **kwargs)
    return wrapper

@app.get("/health")
def health():
    return jsonify({"ok": True})

# ================= بحث الاسم التجاري =================
@app.get("/search")
@require_hmac
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
@require_hmac
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
@require_hmac
def equivalents_get():
    """
    GET /equivalents?active=باراسيتامول&mg=500&form=أقراص&tol=0&allow_per_ml=true&strict_form=true&limit=50&topk=800&minbase=0&debug=1
    - active: إجباري (المادة الفعّالة/الاسم العلمي)
    - mg: إجباري (تركيز بالمليجرام)
    - form: اختياري (أقراص/كبسول/شراب/أمبول/...)
    - debug: اختياري (لو true يطلع لوجز تفصيلية في docker logs)
    """
    # دعم إدخال متعدد المواد عبر actives (JSON list). لو غير متاح نرجع للنمط القديم active+mg
    actives_raw = request.args.get("actives", type=str)
    active = request.args.get("active", type=str)
    mg = request.args.get("mg", type=str)
    actives_list = None
    if actives_raw:
        try:
            actives_list = json.loads(actives_raw)
            if not isinstance(actives_list, list) or not actives_list:
                actives_list = None
        except Exception:
            actives_list = None
    mg_val = None
    if not actives_list:
        if not active or not active.strip():
            return jsonify({"error": "missing 'active' parameter or provide 'actives' list"}), 400
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
    debug_flag = parse_bool(request.args.get("debug"), True)

    try:
        if debug_flag:
            logger.setLevel(logging.DEBUG)
        if actives_list:
            logger.info(f"/equivalents GET actives(list) form='{form}' tol={tol} strict={strict_form} per_ml={allow_per_ml} topk={topk} minbase={minbase}")
        else:
            logger.info(f"/equivalents GET active='{active}' mg={mg_val} form='{form}' tol={tol} strict={strict_form} per_ml={allow_per_ml} topk={topk} minbase={minbase}")

        local_finder = eq_finder
        if topk != EQ_TOP_K or minbase != EQ_MIN_BASE10:
            local_finder = EquivalentsFinder(top_k=topk, min_base10=minbase)

        if actives_list:
            eq_results = local_finder.find_equivalents_multi(
                actives=actives_list,
                target_form=form,
                strict_form=strict_form,
                limit=limit,
                debug=debug_flag,
            )
            exclude_ids = {str(r.get('id')) for r in eq_results}
            alt_results = local_finder.find_alternatives_multi(
                actives=actives_list,
                target_form=form,
                strict_form=strict_form,
                limit=limit,
                exclude_ids=exclude_ids,
                debug=debug_flag,
            )
        else:
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
@require_hmac
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

    # دعم إدخال متعدد المواد عبر actives (list of dict/name+mg). لو غير متاح نرجع للنمط القديم
    actives_list = payload.get("actives")
    active = (payload.get("active") or "").strip()
    mg_val = None
    if not actives_list:
        if not active:
            return jsonify({"error": "missing 'active' in JSON body or provide 'actives' list"}), 400
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
    debug_flag = parse_bool(payload.get("debug"), True)

    try:
        if debug_flag:
            logger.setLevel(logging.DEBUG)
        if actives_list:
            logger.info(f"/equivalents POST actives(list) form='{form}' tol={tol} strict={strict_form} per_ml={allow_per_ml} topk={topk} minbase={minbase}")
        else:
            logger.info(f"/equivalents POST active='{active}' mg={mg_val} form='{form}' tol={tol} strict={strict_form} per_ml={allow_per_ml} topk={topk} minbase={minbase}")

        local_finder = eq_finder
        if topk != EQ_TOP_K or minbase != EQ_MIN_BASE10:
            local_finder = EquivalentsFinder(top_k=topk, min_base10=minbase)

        if actives_list:
            eq_results = local_finder.find_equivalents_multi(
                actives=actives_list,
                target_form=form,
                strict_form=strict_form,
                limit=limit,
                debug=debug_flag,
            )
            exclude_ids = {str(r.get('id')) for r in eq_results}
            alt_results = local_finder.find_alternatives_multi(
                actives=actives_list,
                target_form=form,
                strict_form=strict_form,
                limit=limit,
                exclude_ids=exclude_ids,
                debug=debug_flag,
            )
        else:
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
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5543")), debug=True)
