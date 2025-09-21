# -*- coding: utf-8 -*-
"""
equivalents_search.py
- إيجاد مثائل/بدائل لمنتج دوائي بناءً على:
  * نفس المادة الفعّالة (أو احتوائها ضمن تركيبة)
  * نفس التركيز (± tolerance)
  * نفس الشكل الدوائي لو strict_form=True
- يعتمد على مرشّح أولي بالبحث المتجهي (ChromaDB) + فلترة دقيقة على الميتاداتا
"""

from __future__ import annotations
import re
import unicodedata
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

# ====================== إعدادات افتراضية ======================
DEFAULT_EQ_TOP_K = 800          # عدد المرشحين من المتجهات
DEFAULT_EQ_MIN_BASE10 = 0.0     # أقل درجة أساسية للقبول (0..10)

# نفس دالة تحويل المسافة لسكور من drug_search.py (للموضوعية في الترتيب)
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

# ====================== Parser للتراكيز ======================
# أمثلة:
# "باراسيتامول 500 ملغ، كافيين 65 ملغ"
# "باراسيتامول 160 ملغ / 5 مل, كلورفينيرامين 1 ملغ / 5 مل"
_CONC_RE = re.compile(
    r"""
    (?P<drug>[^,\u061B\u060C؛]+?)          # اسم المادة حتى فاصل
    \s*
    (?P<amt>\d+(?:\.\d+)?)\s*
    (?P<unit>ملغ|مجم|mg|mcg|ميكروغرام|جم|g)?  # وحدة اختيارية
    (?:\s*/\s*(?P<per>\d+(?:\.\d+)?)\s*(?:مل|ml))? # لكل مل اختيارية (شراب)
    """, re.VERBOSE | re.IGNORECASE
)

def _to_mg(amount: float, unit: str) -> float:
    if not unit: return amount
    u = unit.lower()
    if u in ("ملغ", "مجم", "mg"): return amount
    if u in ("mcg", "ميكروغرام"): return amount / 1000.0
    if u in ("جم", "g"): return amount * 1000.0
    return amount

def parse_concentrations(text: str) -> List[Dict[str, Any]]:
    """
    يرجّع قائمة عناصر بالشكل:
      [{'drug':'باراسيتامول', 'mg':500.0, 'per_ml': None}, ...]
    أو للشراب:
      [{'drug':'باراسيتامول', 'mg':160.0, 'per_ml': 32.0}]  # 160 mg / 5 ml => 32 mg/ml
    """
    results: List[Dict[str, Any]] = []
    if not text or not isinstance(text, str):
        return results
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
                per_ml = mg / vol
        results.append({"drug": drug.strip(), "mg": mg, "per_ml": per_ml})
    return results

# ====================== أدوات ميتاداتا ======================
def get_brand(meta: Dict[str, Any]) -> str:
    for k in ("commercial_name","الاسم التجاري","الاسم_التجاري","name","اسم الدواء الأصلي"):
        v = (meta or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def get_scientific(meta: Dict[str, Any]) -> str:
    for k in ("scientific_name","الاسم العلمي"):
        v = (meta or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def get_form(meta: Dict[str, Any]) -> str:
    for k in ("pharma_form","الشكل الدوائي","form","dosage_form","pharmaceutical_form"):
        v = (meta or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def get_conc_text(meta: Dict[str, Any]) -> str:
    # أثناء البناء بيتسجل المفتاح الإنجليزي "concentrations" (مهم جدًا)
    # ولو لسه قبل البناء/مصدر خام، ابحث في "التراكيز"
    for k in ("concentrations","التراكيز"):
        v = (meta or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

# ====================== فحص البديل ======================
def is_equivalent_item(
    meta: Dict[str, Any],
    active_query: str,
    target_mg: float,
    tolerance_mg: float = 0.0,
    allow_per_ml: bool = True,
    target_form: Optional[str] = None,
    strict_form: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    يقرر إذا كان المنتج meta بديل صالح:
      - يحتوي نفس المادة (أو تتضمنها ضمن تركيبة)
      - تركيز المادة داخل [target_mg ± tolerance_mg]
      - نفس الشكل لو strict_form=True
      - يدعم محلول/شراب (mg/ml) لكن بيستبعده كبديل مباشر لأقراص عند strict_form=True
    """
    # الشكل الدوائي
    form = get_form(meta)
    form_norm = normalize_form(str(form))
    if strict_form and target_form:
        if normalize_form(target_form) != form_norm:
            return None

    # التراكيز
    conc_text = get_conc_text(meta)
    conc_list = parse_concentrations(conc_text)

    aq = normalize_ar(active_query)
    def _match_active(x: str) -> bool:
        # نطبع ونسمح بالاحتواء (عشان التركيبات)
        return aq in normalize_ar(x)

    hit = None
    for c in conc_list:
        if _match_active(c["drug"]):
            hit = c
            break
    if not hit:
        return None

    # المقارنة على التركيز
    if hit["per_ml"] is not None:
        # ده منتج شراب/محلول
        if not allow_per_ml:
            return None
        # لو الهدف أقراص والمرشح شراب ومع strict_form=True اتشال بالفعل فوق
        # هنا هنرجّح تجاهل الشراب كبديل مباشر للقرص (إلا إذا strict_form=False)
        if strict_form:
            return None
        # لو عايز توسّع: ممكن تقارن mg/5ml مقابل mg/قرص (حسب سياسة الجرعات عندك)
        # حالياً هنعديه بشرط الشكل غير صارم
        mg_match = True  # تخفيف القيود لو المستخدم سمح بالشكل المختلف
    else:
        mg_match = abs(float(hit["mg"]) - float(target_mg)) <= float(tolerance_mg)

    if not mg_match:
        return None

    return {
        "match": {
            "active": hit["drug"],
            "mg": float(hit["mg"]),
            "per_ml": hit["per_ml"],
            "form": form_norm,
            "strength_hit": f"{hit['mg']} mg" if hit["per_ml"] is None else f"{hit['mg']} mg per {hit['per_ml']:.2f} mg/ml",
        }
    }

# ====================== الكلاس الرئيسي ======================
class EquivalentsFinder:
    """
    - يجهز Chroma + Embeddings
    - find_equivalents(...) بيرجع مثائل مرتبة حسب درجة التشابه الأساسية (vector) أولاً،
      مع الإبقاء فقط على اللي عدّوا فحص المادة/التركيز/الشكل.
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

        # نفس موديل التضمين وإعداداته المستخدمة في البناء والبحث (e5)
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
    ) -> List[Dict[str, Any]]:
        """
        يرجّع مثائل لها:
          - نفس المادة/احتواء المادة
          - نفس التركيز ± tolerance
          - (اختياري) نفس الشكل بدقة لو strict_form=True
        """
        if not active_query or target_mg is None:
            return []

        res = self._query_once(active_query, target_mg)
        ids_list   = res.get("ids", [[]])[0]
        dists_list = res.get("distances", [[]])[0]
        metas_list = res.get("metadatas", [[]])[0]
        docs_list  = res.get("documents", [[]])[0]

        rows: List[Dict[str, Any]] = []
        for _id, dist, meta, doc in zip(ids_list, dists_list, metas_list, docs_list):
            base_score = distance_to_score10(dist)
            if base_score < self.min_base10:
                continue

            ok = is_equivalent_item(
                meta=meta,
                active_query=active_query,
                target_mg=float(target_mg),
                tolerance_mg=float(tolerance_mg),
                allow_per_ml=allow_per_ml,
                target_form=target_form,
                strict_form=strict_form,
            )
            if not ok:
                continue

            brand = get_brand(meta)
            sci   = get_scientific(meta)
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
                "doc": doc,                # نص الوثيقة داخل Chroma (لو موجود)
                "tag": "equivalent"
            }
            rows.append(row)

        # رتب حسب أعلى سكور أساسي (اللي أقرب في المتجهات)
        rows.sort(key=lambda x: x["base_score10"], reverse=True)
        if limit:
            rows = rows[:limit]
        return rows
