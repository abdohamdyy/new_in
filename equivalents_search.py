# -*- coding: utf-8 -*-
"""
equivalents_search.py  (rev: robust-strength+active+logs)
- إيجاد مثائل/بدائل لدواء بناءً على:
  * نفس المادة الفعّالة (أو احتوائها ضمن تركيبة)
  * نفس التركيز (± tolerance)
  * نفس الشكل الدوائي لو strict_form=True
- يعتمد على مرشّح أولي بالبحث المتجهي (ChromaDB) + فلترة دقيقة على الميتاداتا
- مزوّد بـ DEBUG logging + DROP-STATS لمعرفة أسباب الرفض.
"""

from __future__ import annotations
import os
import re
import json
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

# -------------------- Logger --------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("equivalents")
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[equivalents] %(levelname)s: %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# ====================== إعدادات افتراضية ======================
DEFAULT_EQ_TOP_K = 800          # عدد المرشحين من المتجهات
DEFAULT_EQ_MIN_BASE10 = 0.0     # أقل درجة أساسية للقبول (0..10)

# نفس دالة تحويل المسافة لسكور (0..10)
def distance_to_score10(d: Optional[float]) -> float:
    if d is None: return 0.0
    s = (1.0 - (d / 2.0)) * 10.0
    return max(0.0, min(10.0, float(s)))

# ====================== تطبيع عربي ======================
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06ED]")

def normalize_ar(text: str) -> str:
    if not isinstance(text, str): return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = (text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
                .replace("ى","ي").replace("ة","ه")
                .replace("ؤ","و").replace("ئ","ي"))
    return re.sub(r"\s+", " ", text).strip().lower()

# ====================== تطبيع الشكل الدوائي ======================
def normalize_form(value: str) -> str:
    if not value: return ""
    v = value.strip().lower()
    if "قرص" in v or "اقراص" in v:        return "أقراص"
    if "كبسول" in v or "كبسوله" in v:     return "كبسول"
    if "شراب" in v or "syrup" in v:        return "شراب"
    if "قطره" in v or "نقط" in v:         return "قطرة"
    if "مرهم" in v:                        return "مرهم"
    if "جل" in v:                          return "جل"
    if "لبوس" in v or "تحميله" in v:      return "لبوس"
    if "امبول" in v or "فيال" in v or "حقن" in v: return "أمبول"
    return value.strip()

# ====================== مفاتيح ميتاداتا شائعة ======================
ACTIVE_KEYS = (
    "scientific_name", "الاسم العلمي",
    "active_ingredients", "المواد الفعالة",
)

FORM_KEYS = (
    "pharma_form", "الشكل الدوائي", "form", "dosage_form", "pharmaceutical_form",
)

# هنا بنقرأ التركيز من أي حقل ممكن يظهر فيه أرقام التركيز
STRENGTH_KEYS = (
    "concentrations", "التراكيز",
    "strength", "dose_strength", "dosage_strength", "التركيز", "تركيز",
    "name", "اسم الدواء الأصلي", "commercial_name", "الاسم التجاري",
)

def get_first_str(meta: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    for k in keys:
        v = (meta or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def get_form(meta: Dict[str, Any]) -> str:
    return get_first_str(meta, FORM_KEYS)

def get_brand(meta: Dict[str, Any]) -> str:
    return get_first_str(meta, ("commercial_name","الاسم التجاري","الاسم_التجاري","name","اسم الدواء الأصلي"))

def get_scientific_all(meta: Dict[str, Any]) -> str:
    parts = []
    for k in ACTIVE_KEYS:
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    return " ، ".join(parts)

# ====================== Parser/Extractor للتراكيز ======================
# نمط يلقط:
#  - "باراسيتامول 500 ملغ"
#  - "كافيين 65 ملغ"
#  - "باراسيتامول 160 ملغ / 5 مل"  => per_ml = 32
#  - يدعم وحدات mg/mcg/g و"مجم/ملغ/جم"
_CONC_RE = re.compile(
    r"""
    (?P<drug>[^,\u061B\u060C؛\n\r]+?)      # اسم المادة حتى فاصل
    \s*
    (?P<amt>\d+(?:\.\d+)?)\s*
    (?P<unit>ملغ|مجم|mg|mcg|ميكروغرام|جم|g)?   # الوحدة (اختيارية)
    (?:\s*/\s*(?P<per>\d+(?:\.\d+)?)\s*(?:مل|ml))?  # لكل مل (اختيارية)
    """, re.VERBOSE | re.IGNORECASE
)

def _to_mg(amount: float, unit: Optional[str]) -> float:
    if not unit: return amount
    u = unit.lower()
    if u in ("ملغ", "مجم", "mg"): return amount
    if u in ("mcg", "ميكروغرام"): return amount / 1000.0
    if u in ("جم", "g"): return amount * 1000.0
    return amount

def extract_strengths(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """يستخرج (drug, mg, per_ml) من أي حقل متاح في STRENGTH_KEYS."""
    blobs: List[str] = []
    for k in STRENGTH_KEYS:
        v = meta.get(k)
        if isinstance(v, str) and v.strip() and v.strip() != "غير متوفر":
            blobs.append(v)
    text = " ، ".join(blobs)
    out: List[Dict[str, Any]] = []
    for m in _CONC_RE.finditer(text):
        drug = (m.group("drug") or "").strip(" -\u061F\u061B\u060C،")
        if not drug:
            continue
        amt = float(m.group("amt"))
        unit = m.group("unit") or "mg"
        per = m.group("per")
        mg = _to_mg(amt, unit)
        per_ml = None
        if per:
            vol = float(per)
            if vol > 0:
                per_ml = mg / vol  # mg per 1 ml
        out.append({"drug": drug, "mg": mg, "per_ml": per_ml})
    return out

# ====================== المادة الفعّالة ======================
_SPLIT_ACTIVE = re.compile(r"[,\+\-/\(\)\[\]{}؛;،]|(?:\s+و\s+)|(?:\s+and\s+)", re.IGNORECASE)

def extract_active_tokens(meta: Dict[str, Any]) -> List[str]:
    text = get_scientific_all(meta)
    raw = [t.strip() for t in _SPLIT_ACTIVE.split(text) if t and t.strip()]
    toks = []
    for t in raw:
        tn = normalize_ar(t)
        if len(tn) >= 3:
            toks.append(tn)
    return toks

def contains_active(meta: Dict[str, Any], active_query_norm: str) -> Tuple[bool, str]:
    # اسم المادة من scientific/active_ingredients
    tokens = extract_active_tokens(meta)
    for t in tokens:
        if active_query_norm in t:
            return True, t
    # fallback: أحيانًا المادة مكتوبة ضمن الاسم التجاري (نادر)
    name_blob = " ".join([
        get_first_str(meta, ("name","اسم الدواء الأصلي")),
        get_brand(meta)
    ])
    if name_blob and active_query_norm in normalize_ar(name_blob):
        return True, name_blob.strip()
    return False, ""

# ====================== فحص البديل ======================
def is_equivalent_item(
    meta: Dict[str, Any],
    active_query: str,
    target_mg: float,
    tolerance_mg: float = 0.0,
    allow_per_ml: bool = True,
    target_form: Optional[str] = None,
    strict_form: bool = True,
    debug: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    بديل صالح لو:
      - يحتوي نفس المادة (scientific/active_ingredients/name)
      - تركيز المادة داخل [target_mg ± tolerance_mg]
      - نفس الشكل لو strict_form=True
    """
    brand = get_brand(meta)

    # الشكل
    form_raw = get_form(meta)
    form_norm = normalize_form(str(form_raw))
    if strict_form and target_form:
        if normalize_form(target_form) != form_norm:
            if debug:
                logger.debug(f"SKIP form mismatch: wanted={normalize_form(target_form)} found={form_norm} brand={brand}")
            return None

    # المادة
    aq = normalize_ar(active_query)
    has_active, active_hit = contains_active(meta, aq)
    if not has_active:
        if debug:
            logger.debug(f"SKIP no active hit: active='{active_query}' brand={brand}")
        return None

    # التركيز
    strengths = extract_strengths(meta)
    if not strengths:
        if debug:
            cf = get_first_str(meta, ("concentrations","التراكيز")) or "غير متوفر"
            logger.debug(f"SKIP no concentrations parsed: brand={brand} conc_field='{cf[:80]}'")
        return None

    hit = None
    for c in strengths:
        # نطلب السطر الخاص بنفس الـ active (احتواء بالـ normalize)
        if aq in normalize_ar(c["drug"]):
            hit = c
            break
    if not hit:
        # لو ما اتكتبش اسم المادة جنب الرقم (يحصل في بعض الداتا) هنجرّب أي strength كـ fallback
        hit = strengths[0]

    # مقارنة الجرعة
    if hit["per_ml"] is not None:
        # شراب/محلول (mg/ml)
        if not allow_per_ml:
            if debug:
                logger.debug(f"SKIP per-ml not allowed: brand={brand}")
            return None
        if strict_form:
            # لو الشكل صارم غالبًا الشراب مش بديل للأقراص
            if debug:
                logger.debug(f"SKIP per-ml strict reject: brand={brand}")
            return None
        # لو strict=False هنرخّي في المطابقة التركيز
        mg_match = True
    else:
        mg = float(hit["mg"])
        mg_match = abs(mg - float(target_mg)) <= float(tolerance_mg)
        if not mg_match and debug:
            logger.debug(f"SKIP mg mismatch: target={target_mg}±{tolerance_mg} got={mg} brand={brand}")

    if not mg_match:
        return None

    return {
        "match": {
            "active": active_hit or hit["drug"],
            "mg": float(hit["mg"]),
            "per_ml": hit["per_ml"],
            "form": form_norm,
            "strength_hit": f"{hit['mg']} mg" if hit["per_ml"] is None else f"{hit['mg']} mg per ml={hit['per_ml']:.2f}",
        }
    }

# ====================== الكلاس الرئيسي ======================
class EquivalentsFinder:
    """
    يجهز Chroma + Embeddings
    find_equivalents(...) يرجّع مثائل مرتبة حسب درجة التشابه الأساسية (vector) مع فلترة دقيقة.
    """

    def __init__(
        self,
        collection_name: str = Config.COLLECTION_NAME,
        db_dir: str = Config.DB_DIRECTORY,
        top_k: int = DEFAULT_EQ_TOP_K,
        min_base10: float = DEFAULT_EQ_MIN_BASE10,
    ) -> None:
        self.collection_name = collection_name
        self.db_dir = db_dir
        self.top_k = top_k
        self.min_base10 = min_base10

        # Init Chroma (Persistent)
        self.client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(
                persist_directory=db_dir,
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
            )
        )
        self.collection = self.client.get_collection(collection_name)

        # Embeddings (e5) — نفس إعدادات Config
        self.emb = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_SETTINGS['model_name'],
            model_kwargs=Config.EMBEDDING_SETTINGS['model_kwargs'],
            encode_kwargs=Config.EMBEDDING_SETTINGS['encode_kwargs'],
        )

    def _query_once(self, active_query: str, target_mg: float) -> Dict[str, Any]:
        # بادئة "query: " مهمة مع e5
        q_text = f"active ingredient: {active_query} strength: {target_mg} mg"
        q_emb = self.emb.embed_query(f"query: {q_text}")
        return self.collection.query(
            query_embeddings=[q_emb],
            n_results=self.top_k,
            include=["distances", "metadatas", "documents"],
        )

    def find_equivalents(
        self,
        active_query: str,
        target_mg: float,
        tolerance_mg: float = 0.0,
        allow_per_ml: bool = True,
        target_form: Optional[str] = None,
        strict_form: bool = True,
        limit: int = 50,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        if debug:
            logger.setLevel(logging.DEBUG)
            logger.debug(f"REQ active='{active_query}' mg={target_mg} tol={tolerance_mg} form='{target_form}' strict={strict_form} allow_per_ml={allow_per_ml} topk={self.top_k} minbase={self.min_base10}")

        res = self._query_once(active_query, target_mg)
        ids_list   = res.get("ids", [[]])[0]
        dists_list = res.get("distances", [[]])[0]
        metas_list = res.get("metadatas", [[]])[0]
        docs_list  = res.get("documents", [[]])[0]

        if debug:
            logger.debug(f"CANDIDATES via vector: {len(ids_list)}")

        rows: List[Dict[str, Any]] = []
        dropped = {"base10":0, "no_active":0, "no_conc":0, "mg_mismatch":0, "form":0, "perml_reject":0}

        for _id, dist, meta, doc in zip(ids_list, dists_list, metas_list, docs_list):
            base_score = distance_to_score10(dist)
            if base_score < self.min_base10:
                dropped["base10"] += 1
                continue

            brand = get_brand(meta)

            # فحص بديل
            ok = is_equivalent_item(
                meta=meta,
                active_query=active_query,
                target_mg=float(target_mg),
                tolerance_mg=float(tolerance_mg),
                allow_per_ml=allow_per_ml,
                target_form=target_form,
                strict_form=strict_form,
                debug=debug,
            )

            if not ok:
                # نعيد حساب counters حسب الرسائل المحتملة (تقريبية للـ stats)
                # NOTE: الرسائل التفصيلية مطبوعة بالفعل من is_equivalent_item.
                # هنا نحاول نخمّن السبب من الميتاداتا بشكل مبسط:
                aq = normalize_ar(active_query)
                has_active, _ = contains_active(meta, aq)
                if not has_active:
                    dropped["no_active"] += 1
                else:
                    strengths = extract_strengths(meta)
                    if not strengths:
                        dropped["no_conc"] += 1
                    else:
                        # عند وجود strengths لكن الرفض كان غالبًا mg أو form/perml
                        form_raw = get_form(meta)
                        if strict_form and target_form and normalize_form(target_form) != normalize_form(str(form_raw)):
                            dropped["form"] += 1
                        else:
                            # إمّا mg mismatch أو per-ml reject
                            any_perml = any(s["per_ml"] is not None for s in strengths)
                            if any_perml and (strict_form or not allow_per_ml):
                                dropped["perml_reject"] += 1
                            else:
                                dropped["mg_mismatch"] += 1
                continue

            sci   = get_first_str(meta, ("scientific_name","الاسم العلمي"))
            form  = get_form(meta)

            row = {
                "id": _id,
                "query": {
                    "active": active_query,
                    "mg": float(target_mg),
                    "form": target_form,
                    "tolerance_mg": float(tolerance_mg),
                    "strict_form": bool(strict_form),
                    "allow_per_ml": bool(allow_per_ml),
                },
                "base_score10": round(base_score, 3),
                "brand": brand or None,
                "name": meta.get("name") or meta.get("اسم الدواء الأصلي"),
                "commercial_name": brand or meta.get("commercial_name") or meta.get("الاسم التجاري"),
                "scientific_name": sci or meta.get("الاسم العلمي"),
                "manufacturer": meta.get("manufacturer") or meta.get("الشركة المصنعة"),
                "form": form,
                "match": ok["match"],      # فيه active/mg/per_ml/strength_hit/form
                "meta": meta,              # كل الميتاداتا
                "doc": doc,                # نص الوثيقة (لو موجود)
                "tag": "equivalent"
            }
            rows.append(row)

        rows.sort(key=lambda x: x["base_score10"], reverse=True)
        if limit:
            rows = rows[:limit]

        if debug:
            logger.debug(f"RETURN {len(rows)} hits")
            logger.debug(f"DROP-STATS: {json.dumps(dropped, ensure_ascii=False)}")

        return rows
