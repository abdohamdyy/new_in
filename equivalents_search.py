# -*- coding: utf-8 -*-
"""
equivalents_search.py

🔎 استراتيجية البحث (Querying Order):
1) نبحث بالمواد الفعّالة أولًا (active_ingredients) + scientific hint
2) ثم الاسم العلمي فقط
3) ثم الاسم التجاري/الاسم (brand/commercial)
(التركيز لا يُستخدم في الاسترجاع.. هنستخدمه لاحقًا في الفرز)

📊 استراتيجية الفرز (Ranking in response):
- أولاً: النتائج المرتبطة علميًا (active/scientific ≥ 0.70)
- داخل المرتبطين علميًا: النتائج التي تحتوي رقم تركيز يساوي target_mg "بالظبط"
- بعد ذلك: ترتيب عادي حسب final_score (vector + fuzzy: active > scientific > brand)
- ثم tie-break بالـ base_score10

يحافظ على نفس أسلوب drug_search.py:
- normalize_ar
- distance_to_score10
- نفس هيكل الخرج: id/query/base_score10/final_score/bonus/tag/.../meta/doc
"""

from __future__ import annotations
import re
import unicodedata
import difflib
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

# ======================= ثوابت وأوزان =======================
# أوزان البونصات: المواد الفعّالة أعلى من العلمي، والعلمي أعلى من التجاري
ACTIVE_FUZZY_MIN        = 0.70
ACTIVE_BASE_BOOST       = 1.4
ACTIVE_STRONG_BOOST     = 1.6   # r>=0.90

SCIENTIFIC_FUZZY_MIN    = 0.70
SCIENTIFIC_BASE_BOOST   = 1.0
SCIENTIFIC_STRONG_BOOST = 1.2   # r>=0.90

BRAND_FUZZY_MIN         = 0.70
BRAND_BASE_BOOST        = 0.7
BRAND_STRONG_BOOST      = 1.0   # r>=0.90 أو طول قريب

# الشكل الدوائي
FORM_MATCH_BONUS        = 0.6   # strict=True
FORM_SOFT_BONUS         = 0.2   # strict=False

# إعدادات الاسترجاع من Chroma
DEFAULT_EQ_TOP_K        = 800
DEFAULT_MIN_BASE10      = 0.0

# عتبة اعتبار النتيجة "مرتبطة علميًا"
RELEVANT_MIN            = 0.70

# ======================= تطبيع وأدوات عامة =======================
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06ED]")
AR_CHARS = "ء-ي"
_NON_WORD = re.compile(rf"[^\w{AR_CHARS}]+", re.UNICODE)

def normalize_ar(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = (text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
                 .replace("ى","ي").replace("ة","ه")
                 .replace("ؤ","و").replace("ئ","ي"))
    return re.sub(r"\s+", " ", text).strip().lower()

def distance_to_score10(d: Optional[float]) -> float:
    """حوّل مسافة كوساين (0..2) إلى درجة 0..10 (نفس drug_search)."""
    if d is None:
        return 0.0
    s = (1.0 - (d / 2.0)) * 10.0
    return max(0.0, min(10.0, float(s)))

# ======================= Active / Scientific =======================
_SPLIT_ACTIVE = re.compile(r"[,\+\-/\(\)\[\]{}؛;،]|(?:\s+و\s+)|(?:\s+and\s+)", re.IGNORECASE)

def extract_active_tokens(meta: Dict[str, Any]) -> List[str]:
    """يستخرج توكنز من المواد الفعّالة + يعيد العلمي كتوكِن مستقل."""
    parts_active = []
    v1 = (meta or {}).get("active_ingredients") or (meta or {}).get("المواد الفعالة")
    if isinstance(v1, str) and v1.strip():
        parts_active.append(v1)

    tokens_active: List[str] = []
    if parts_active:
        text = " ، ".join(parts_active)
        raw_tokens = [t.strip() for t in _SPLIT_ACTIVE.split(text) if t and t.strip()]
        for t in raw_tokens:
            tn = normalize_ar(t)
            if len(tn) >= 2:
                tokens_active.append(tn)

    sci = (meta or {}).get("scientific_name") or (meta or {}).get("الاسم العلمي") or ""
    sci_norm = normalize_ar(sci) if isinstance(sci, str) else ""

    return tokens_active + ([sci_norm] if sci_norm else [])

def best_active_fuzzy_ratio(meta: Dict[str, Any], q_norm: str) -> Tuple[float, str]:
    """أفضل تطابق فزي ضمن (المواد الفعّالة + العلمي كتوكِن)."""
    toks = extract_active_tokens(meta)
    best = 0.0
    best_tok = ""
    for t in toks:
        r = difflib.SequenceMatcher(None, t, q_norm).ratio()
        if r > best:
            best = r
            best_tok = t
    return best, best_tok

def fuzzy_ratio_on_scientific_only(meta: Dict[str, Any], q_norm: str) -> float:
    sci = (meta or {}).get("scientific_name") or (meta or {}).get("الاسم العلمي") or ""
    sci_norm = normalize_ar(sci) if isinstance(sci, str) else ""
    if not sci_norm or not q_norm:
        return 0.0
    return difflib.SequenceMatcher(None, sci_norm, q_norm).ratio()

# ======================= Brand / Commercial =======================
NAME_KEYS = [
    "commercial_name", "الاسم التجاري", "الاسم_التجاري",
    "name", "اسم الدواء الأصلي",
    "scientific_name", "الاسم العلمي",
]
def best_brand_field(meta: Dict[str, Any]) -> str:
    for k in NAME_KEYS:
        val = (meta or {}).get(k)
        if isinstance(val, str) and val.strip():
            return val
    return ""

def brand_best_token_ratio(brand: str, q_norm: str) -> Tuple[float, str, float]:
    """فزي على مستوى توكن من الاسم التجاري + عقوبة طول خفيفة."""
    v = normalize_ar(brand)
    if not v or not q_norm:
        return 0.0, "", 1.0
    tokens = [t for t in _NON_WORD.split(v) if t]
    best = (0.0, "", 1.0)
    for t in tokens:
        ratio = difflib.SequenceMatcher(None, t, q_norm).ratio()
        rel_extra = max(0, len(t) - len(q_norm)) / max(len(t), 1)
        penalty = 1.0 - min(0.30, rel_extra)  # 0.7..1.0
        if t.startswith(q_norm) or t.endswith(q_norm):
            ratio = min(1.0, ratio + 0.05)
        if ratio * penalty > best[0] * best[2]:
            best = (ratio, t, penalty)
    return best

# ======================= التركيز (أرقام فقط) =======================
NUM_PATTERN = re.compile(r"\d+(?:\.\d+)?")

def grab_all_numbers(text: str) -> List[float]:
    if not text:
        return []
    return [float(x) for x in NUM_PATTERN.findall(str(text))]

def collect_concentration_texts(meta: Dict[str, Any]) -> str:
    """
    نجمع كل الحقول التي قد تحتوي أرقام تركيز:
      - concentrations/التراكيز
      - name/اسم الدواء الأصلي
      - active_ingredients/المواد الفعالة
      - dosage/الجرعة المعتادة
    """
    parts = []
    for k in ("concentrations", "التراكيز",
              "name", "اسم الدواء الأصلي",
              "active_ingredients", "المواد الفعالة",
              "dosage", "الجرعة المعتادة"):
        v = (meta or {}).get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    return " | ".join(parts)

def find_number_near_active(text: str, active_norm: str, window: int = 25) -> Optional[float]:
    """
    نحاول نلقط رقم قريب من اسم المادة المطلوبة داخل نافذة حروف.
    مثال: '... باراسيتامول (500 مجم) ...' → نلتقط 500.
    """
    if not text or not active_norm:
        return None
    t = normalize_ar(str(text))
    for m in re.finditer(re.escape(active_norm), t):
        start, end = m.span()
        l_ctx = max(0, start - window)
        r_ctx = min(len(t), end + window)
        ctx = t[l_ctx:r_ctx]
        nums = grab_all_numbers(ctx)
        if nums:
            return nums[0]
    return None

def has_exact_concentration(meta: Dict[str, Any], active_norm: str, target_mg: float) -> Tuple[bool, Optional[float]]:
    """
    true لو لقينا رقم يساوي target_mg:
      - أولًا: رقم ملاصق لاسم المادة المطلوبة (لو أمكن)
      - وإلا: أي رقم عام في النصوص المجمّعة
    """
    if target_mg <= 0:
        return False, None
    conc_text = collect_concentration_texts(meta)
    # حاول رقم ملاصق لاسم المادة
    near = find_number_near_active(conc_text, active_norm)
    if near is not None and abs(near - target_mg) < 1e-6:
        return True, near
    # وإلا: أي رقم عام يساوي الهدف
    for n in grab_all_numbers(conc_text):
        if abs(n - target_mg) < 1e-6:
            return True, n
    return False, near

# ======================= EquivalentsFinder =======================
class EquivalentsFinder:
    """
    - الاسترجاع: Active/Scientific أولًا → Scientific-only → Brand
    - الفرز النهائي: المرتبط علميًا أولًا، ثم exact concentration match، ثم final_score
    """

    def __init__(
        self,
        collection_name: str = Config.COLLECTION_NAME,
        db_dir: str = Config.DB_DIRECTORY,
        top_k: int = DEFAULT_EQ_TOP_K,
        min_base10: float = DEFAULT_MIN_BASE10,
    ) -> None:
        self.collection_name = collection_name
        self.db_dir = db_dir
        self.top_k = top_k
        self.min_base10 = min_base10

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

        self.emb = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_SETTINGS['model_name'],
            model_kwargs=Config.EMBEDDING_SETTINGS['model_kwargs'],
            encode_kwargs=Config.EMBEDDING_SETTINGS['encode_kwargs'],
        )

    # ---------- Queries بالترتيب ----------
    def _query_active_first(self, q: str) -> Dict[str, Any]:
        q_emb = self.emb.embed_query(f"active ingredients or scientific: {q}")
        return self.collection.query(
            query_embeddings=[q_emb],
            n_results=self.top_k,
            include=["distances", "metadatas", "documents"],
        )

    def _query_scientific_only(self, q: str) -> Dict[str, Any]:
        q_emb = self.emb.embed_query(f"scientific name: {q}")
        return self.collection.query(
            query_embeddings=[q_emb],
            n_results=self.top_k,
            include=["distances", "metadatas", "documents"],
        )

    def _query_brand(self, q: str) -> Dict[str, Any]:
        q_emb = self.emb.embed_query(f"brand or commercial name: {q}")
        return self.collection.query(
            query_embeddings=[q_emb],
            n_results=self.top_k,
            include=["distances", "metadatas", "documents"],
        )

    # ---------- المجرى الرئيسي ----------
    def find_equivalents(
        self,
        active_query: str,
        target_mg: float,
        tolerance_mg: float = 0.0,   # موجود للتوافق فقط
        allow_per_ml: bool = True,   # موجود للتوافق فقط
        target_form: Optional[str] = None,
        strict_form: bool = True,
        limit: int = 50,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        - يسترجع مرشحين من 3 استعلامات مرتّبة (active → scientific → brand)
        - يبني تجميعة موحّدة للمرشحين (أفضل مسافة base لكل id)
        - يحسب bonus (active/scientific/brand + form)
        - يفرز أخيرًا كالتالي:
            (1) المرتبط علميًا أولًا (active/scientific ≥ RELEVANT_MIN)
            (2) داخلهم: exact concentration match = True
            (3) ثم final_score أعلى
            (4) tie-break: base_score10 أعلى
        """
        if not active_query:
            return []

        # 1) اجمع مرشحين من الاستعلامات بالترتيب
        q_active   = self._query_active_first(active_query)
        q_science  = self._query_scientific_only(active_query)
        q_brand    = self._query_brand(active_query)

        pools = [q_active, q_science, q_brand]

        # 2) دمج المرشحين: اختر "أفضل" (أصغر مسافة ⇒ أعلى base_score10) لكل id
        merged: Dict[str, Dict[str, Any]] = {}  # id -> {dist, meta, doc}
        for res in pools:
            ids   = res.get("ids", [[]])[0]
            dists = res.get("distances", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            docs  = res.get("documents", [[]])[0]
            for _id, dist, meta, doc in zip(ids, dists, metas, docs):
                if _id not in merged or (dist is not None and dist < merged[_id]["dist"]):
                    merged[_id] = {"dist": dist, "meta": meta, "doc": doc}

        q_norm = normalize_ar(active_query)

        # 3) ابنِ الصفوف مع البونصات (بدون بونص تركيز — سنستخدمه للفرز فقط)
        rows: List[Dict[str, Any]] = []
        for _id, pack in merged.items():
            dist = pack["dist"]
            meta = pack["meta"]
            doc  = pack["doc"]

            base_score = distance_to_score10(dist)
            if base_score < self.min_base10:
                continue

            tag_parts: List[str] = []
            bonus = 0.0

            # الشكل الدوائي
            is_ok, soft_bonus = form_matches(meta, target_form, strict_form)
            if not is_ok:
                continue
            if target_form and soft_bonus:
                bonus += FORM_MATCH_BONUS if strict_form else FORM_SOFT_BONUS
                tag_parts.append("form_match")

            # المواد الفعّالة
            a_ratio, a_tok = best_active_fuzzy_ratio(meta, q_norm)
            if a_ratio >= ACTIVE_FUZZY_MIN:
                bonus += ACTIVE_BASE_BOOST * a_ratio
                if a_ratio >= 0.90:
                    bonus += ACTIVE_STRONG_BOOST
                    tag_parts.append("active_strong_fuzzy")
                else:
                    tag_parts.append(f"active_fuzzy({a_ratio:.2f})")
            else:
                tag_parts.append("active_miss")

            # العلمي فقط
            sci_ratio = fuzzy_ratio_on_scientific_only(meta, q_norm)
            if sci_ratio >= SCIENTIFIC_FUZZY_MIN:
                bonus += SCIENTIFIC_BASE_BOOST * sci_ratio
                if sci_ratio >= 0.90:
                    bonus += SCIENTIFIC_STRONG_BOOST
                    tag_parts.append("sci_strong_fuzzy")
                else:
                    tag_parts.append(f"sci_fuzzy({sci_ratio:.2f})")
            else:
                tag_parts.append("sci_miss")

            # التجاري (أقل وزن)
            brand = best_brand_field(meta)
            b_ratio, b_token, b_pen = brand_best_token_ratio(brand, q_norm)
            if b_ratio >= BRAND_FUZZY_MIN:
                bonus += BRAND_BASE_BOOST * b_ratio * b_pen
                if b_ratio >= 0.90 or (b_token and abs(len(b_token) - len(q_norm)) <= 1):
                    bonus += BRAND_STRONG_BOOST
                    tag_parts.append("brand_strong_fuzzy")
                else:
                    tag_parts.append("brand_fuzzy")
            else:
                tag_parts.append("brand_miss")

            # فحص الارتباط العلمي العام:
            is_relevant = (a_ratio >= RELEVANT_MIN) or (sci_ratio >= RELEVANT_MIN)

            # فحص التطابق العددي للتركيز (للفرز فقط، وداخل المرتبطين علميًا فقط)
            active_norm = q_norm
            conc_exact, conc_num = (False, None)
            if is_relevant:
                conc_exact, conc_num = has_exact_concentration(meta, active_norm, target_mg)

            row = {
                "id": _id,
                "query": active_query,
                "base_score10": round(base_score, 3),
                "final_score": round(base_score + bonus, 3),
                "bonus": round(bonus, 3),
                "tag": "+".join(tag_parts) if tag_parts else "vector_only",
                "brand": brand or None,
                "name": (meta or {}).get("name") or (meta or {}).get("اسم الدواء الأصلي"),
                "commercial_name": brand or (meta or {}).get("commercial_name") or (meta or {}).get("الاسم التجاري"),
                "scientific_name": (meta or {}).get("scientific_name") or (meta or {}).get("الاسم العلمي"),
                "manufacturer": (meta or {}).get("manufacturer") or (meta or {}).get("الشركة المصنعة"),
                "meta": meta,
                "doc": doc,
                # مفاتيح مساعدة للفرز النهائي:
                "_relevant": is_relevant,
                "_conc_exact": bool(conc_exact) if is_relevant else False,
                "_conc_num": conc_num,
            }
            rows.append(row)

        # 4) الفرز النهائي:
        #   - المرتبطون علميًا أولاً
        #   - داخلهم: المطابقون في التركيز
        #   - ثم: final_score أعلى
        #   - ثم: base_score10 أعلى
        rows.sort(key=lambda x: (
            0 if x.get("_relevant") else 1,
            0 if x.get("_conc_exact") else 1,
            -(x["final_score"]),
            -(x["base_score10"])
        ))

        # تنظيف المفاتيح المساعدة قبل الإرجاع + limit
        for r in rows:
            r.pop("_relevant", None)
            r.pop("_conc_exact", None)
            r.pop("_conc_num", None)

        if limit:
            rows = rows[:limit]
        return rows

# ====== الشكل الدوائي (نُعيد استخدامها هنا) ======
def form_matches(meta: Dict[str, Any], target_form: Optional[str], strict_form: bool) -> Tuple[bool, bool]:
    """يرجع (is_ok, soft_bonus_flag)"""
    if not target_form:
        return True, False
    mform = (meta or {}).get("pharma_form") or (meta or {}).get("الشكل الدوائي") or ""
    if not isinstance(mform, str):
        mform = str(mform or "")
    nf = normalize_ar(target_form)
    mf = normalize_ar(mform)
    if not nf:
        return True, False
    if mf == nf or nf in mf or mf in nf:
        return True, True
    return (False, False) if strict_form else (True, True)
