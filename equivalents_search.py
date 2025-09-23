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
from typing import List, Dict, Any, Optional, Tuple, Set
import logging

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

# لوجر مخصص للمثائل، يتبع duaya_api
logger = logging.getLogger("duaya_api.eq")

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

# ======================= بدائل (Different Actives) =======================
# أوزان بدائل تعتمد على تطابق الأمراض/الأعراض والتصنيف/القسم
ALT_DISEASE_OVERLAP_BOOST = 3.0
ALT_CLASS_MATCH_BONUS     = 0.8
ALT_SECTION_MATCH_BONUS   = 0.6

# ======================= تطبيع وأدوات عامة =======================
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06ED]")
AR_CHARS = "ء-ي"
_NON_WORD = re.compile(rf"[^\w{AR_CHARS}]+", re.UNICODE)
_LIST_SEP = re.compile(r"[،,;؛]\s*")

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

# ======================= أمراض/أعراض/تصنيف/قسم =======================
def _split_list_phrases(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    # نفصل بالقواطع الشائعة مع الحفاظ على العبارات
    parts = [p.strip() for p in _LIST_SEP.split(text) if p and p.strip()]
    # صيانة العبارات القصيرة جدًا
    return [p for p in parts if len(normalize_ar(p)) >= 2]

def extract_diseases_phrases(meta: Dict[str, Any]) -> List[str]:
    v = (meta or {}).get("diseases") or (meta or {}).get("الأمراض التي يعالجها") or ""
    return _split_list_phrases(v)

def extract_symptoms_phrases(meta: Dict[str, Any]) -> List[str]:
    v = (meta or {}).get("disease_symptoms") or (meta or {}).get("أعراض المرض") or ""
    return _split_list_phrases(v)

def normalize_set(items: List[str]) -> Set[str]:
    return {normalize_ar(x) for x in items if isinstance(x, str) and x.strip()}

def get_classification(meta: Dict[str, Any]) -> str:
    return (meta or {}).get("classification") or (meta or {}).get("التصنيف الدوائي") or ""

def get_section(meta: Dict[str, Any]) -> str:
    return (meta or {}).get("section") or (meta or {}).get("القسم") or ""

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

def _concentrations_list_to_text(val: Any) -> str:
    """حوّل قائمة [{اسم: رقم}] إلى نص موحّد مثل: 'اسم 20 / اسم 1680'."""
    try:
        if isinstance(val, list):
            parts: List[str] = []
            for item in val:
                if isinstance(item, dict) and len(item) == 1:
                    [(k, v)] = list(item.items())
                    parts.append(f"{k} {v}")
            if parts:
                return " / ".join(parts)
    except Exception:
        pass
    return ""

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
        elif k in ("concentrations", "التراكيز") and isinstance(v, list):
            rendered = _concentrations_list_to_text(v)
            if rendered:
                parts.append(rendered)
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

# ---- تحويل التراكيز إلى قائمة {مادة: رقم} في الخرج ----
_SEG_SPLIT = re.compile(r"[\/\+;,؛،]|(?:\s+و\s+)|(?:\s+and\s+)", re.IGNORECASE)
_NUM_ONLY = re.compile(r"\d+(?:\.\d+)?")
_UNITS_PATTERN = re.compile(r"\b(?:mg|mcg|ug|g|ml|iu|%|ملغ|مجم|ملغم|ميليجرام|ميليغرام|ميكروجرام|ميكروغرام|جرام|جم|مل)\b", re.IGNORECASE)
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def _to_ascii_digits(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    return text.translate(_ARABIC_DIGITS)

def _clean_material_name(raw_name: str) -> str:
    if not isinstance(raw_name, str):
        return ""
    s = _to_ascii_digits(raw_name)
    s = _UNITS_PATTERN.sub(" ", s)
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^A-Za-z\-\s\u0600-\u06FF]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_NAME_NUM_PATTERN = re.compile(r"([A-Za-z\u0600-\u06FF][A-Za-z\u0600-\u06FF\-\s\(\)]+?)\s*(\d+(?:\.\d+)?)")

def parse_concentrations_from_meta(meta: Dict[str, Any]) -> List[Dict[str, float]]:
    if not isinstance(meta, dict):
        return []
    val = meta.get("concentrations")
    if isinstance(val, list):
        try:
            ok = all(isinstance(x, dict) and len(x) == 1 for x in val)
            return val if ok else []
        except Exception:
            return []
    if not isinstance(val, str) or not val.strip():
        return []
    text = _to_ascii_digits(val)
    text_simple = _UNITS_PATTERN.sub(" ", text)
    pairs: List[Dict[str, float]] = []
    for name, num in _NAME_NUM_PATTERN.findall(text_simple):
        name_clean = _clean_material_name(name)
        if not name_clean:
            continue
        try:
            vf = float(num)
            v = int(vf) if abs(vf - int(vf)) < 1e-9 else vf
            pairs.append({name_clean: v})
        except Exception:
            continue
    if pairs:
        return pairs
    nums = [float(x) for x in _NUM_ONLY.findall(text_simple)]
    if not nums:
        return []
    ai_text = (meta.get("active_ingredients") or meta.get("المواد الفعالة") or "")
    if isinstance(ai_text, str) and ai_text.strip():
        raw_tokens = [t.strip() for t in _SEG_SPLIT.split(ai_text) if t and t.strip()]
        names = [_clean_material_name(t) for t in raw_tokens if _clean_material_name(t)]
        if len(names) == len(nums):
            out: List[Dict[str, float]] = []
            for nm, vf in zip(names, nums):
                v = int(vf) if abs(vf - int(vf)) < 1e-9 else vf
                out.append({nm: v})
            return out
    if len(nums) == 1:
        sci = meta.get("scientific_name") or meta.get("الاسم العلمي") or ai_text
        nm = _clean_material_name(str(sci or "").strip())
        if nm:
            vf = nums[0]
            v = int(vf) if abs(vf - int(vf)) < 1e-9 else vf
            return [{nm: v}]
    return []

# ======================= أدوات متعددة المواد =======================
def normalize_actives_input(actives: Any) -> List[Tuple[str, float]]:
    """
    يحوّل قائمة المواد المطلوبة (بصيغ مرنة) إلى قائمة [(name_norm, mg)].
    - يقبل شكل [{"باراسيتامول": 500}, {"سودوإيفيدرين": 30}, ...] أو [["باراسيتامول", 500], ...]
    - يزيل التكرارات حسب الاسم الموحّد.
    """
    out: List[Tuple[str, float]] = []
    try:
        if not isinstance(actives, list):
            return out
        for item in actives:
            if isinstance(item, dict) and len(item) == 1:
                [(k, v)] = list(item.items())
                name_norm = normalize_ar(_clean_material_name(str(k)))
                try:
                    mg_val = float(v)
                except Exception:
                    mg_val = 0.0
                if name_norm:
                    out.append((name_norm, mg_val))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                name_norm = normalize_ar(_clean_material_name(str(item[0])))
                try:
                    mg_val = float(item[1])
                except Exception:
                    mg_val = 0.0
                if name_norm:
                    out.append((name_norm, mg_val))
    except Exception:
        return out
    # إزالة التكرارات بنفس الاسم الموحّد
    seen: Set[str] = set()
    uniq: List[Tuple[str, float]] = []
    for nm, mg in out:
        if nm and nm not in seen:
            seen.add(nm)
            uniq.append((nm, mg))
    return uniq

def brand_best_multi_ratio(brand: str, q_norm_list: List[str]) -> Tuple[float, str, float]:
    """أفضل تطابق للماركة عبر عدة توكنات استعلام (أسماء مواد متعددة)."""
    best_ratio = 0.0
    best_tok = ""
    best_pen = 1.0
    for qn in q_norm_list:
        r, t, p = brand_best_token_ratio(brand, qn)
        if r * p > best_ratio * best_pen:
            best_ratio, best_tok, best_pen = r, t, p
    return best_ratio, best_tok, best_pen

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

    def _query_by_profile(self, diseases_text: str, symptoms_text: str, classification: str, section: str) -> Dict[str, Any]:
        # نجمع بروفايل علاجي مختصر للاستعلام
        profile_parts: List[str] = []
        if diseases_text:
            profile_parts.append(f"treats diseases: {diseases_text}")
        if symptoms_text:
            profile_parts.append(f"symptoms: {symptoms_text}")
        if classification:
            profile_parts.append(f"classification: {classification}")
        if section:
            profile_parts.append(f"section: {section}")
        profile_query = " | ".join(profile_parts) if profile_parts else ""
        q_emb = self.emb.embed_query(profile_query)
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

            # طبّق تحويل التراكيز إلى قائمة عند توفرها
            meta_out = dict(meta or {})
            try:
                parsed_conc = parse_concentrations_from_meta(meta_out)
                if parsed_conc:
                    meta_out["concentrations_raw"] = meta_out.get("concentrations")
                    meta_out["concentrations"] = parsed_conc
            except Exception:
                pass

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
                "meta": meta_out,
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

    # ---------- بدائل بمادة مختلفة لكن نفس الاستخدامات ----------
    def _collect_seed_context(
        self,
        active_query: str,
        target_form: Optional[str],
        strict_form: bool,
        seed_limit: int = 20,
    ) -> Tuple[Set[str], Set[str], str, str]:
        """يجمع سياق علاجي (أمراض/أعراض/تصنيف/قسم) من أفضل نتائج المثائل كبذور."""
        seeds = self.find_equivalents(
            active_query=active_query,
            target_mg=0.0,
            tolerance_mg=0.0,
            allow_per_ml=True,
            target_form=target_form,
            strict_form=strict_form,
            limit=seed_limit,
            debug=False,
        )
        diseases_union: Set[str] = set()
        symptoms_union: Set[str] = set()
        class_counts: Dict[str, int] = {}
        section_counts: Dict[str, int] = {}
        for r in seeds:
            meta = r.get("meta") or {}
            d = normalize_set(extract_diseases_phrases(meta))
            s = normalize_set(extract_symptoms_phrases(meta))
            diseases_union.update(d)
            symptoms_union.update(s)
            cl = normalize_ar(get_classification(meta))
            sec = normalize_ar(get_section(meta))
            if cl:
                class_counts[cl] = class_counts.get(cl, 0) + 1
            if sec:
                section_counts[sec] = section_counts.get(sec, 0) + 1

        def pick_max(counts: Dict[str, int]) -> str:
            if not counts:
                return ""
            return max(counts.items(), key=lambda kv: kv[1])[0]

        return diseases_union, symptoms_union, pick_max(class_counts), pick_max(section_counts)

    def find_alternatives(
        self,
        active_query: str,
        target_form: Optional[str] = None,
        strict_form: bool = True,
        limit: int = 50,
        exclude_ids: Optional[Set[str]] = None,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        يعثر على بدائل بمادة فعالة مختلفة لكنها تعالج نفس الأمراض/الأعراض أو تقع تحت نفس التصنيف/القسم.
        - يستخدم سياقًا مستمدًا من نتائج المثائل كبذرة للاستعلام.
        - يستبعد أي نتيجة تحتوي نفس المادة الفعالة المطلوب بديلها.
        """
        if not active_query:
            return []

        active_norm = normalize_ar(active_query)
        diseases_set, symptoms_set, seed_class, seed_section = self._collect_seed_context(
            active_query, target_form, strict_form, seed_limit=min(20, self.top_k)
        )

        # لا يوجد سياق كافٍ
        if not diseases_set and not symptoms_set and not seed_class and not seed_section:
            return []

        diseases_text = "، ".join(sorted(diseases_set))
        symptoms_text = "، ".join(sorted(symptoms_set))
        res = self._query_by_profile(diseases_text, symptoms_text, seed_class, seed_section)

        ids   = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        docs  = res.get("documents", [[]])[0]

        seeds_union = diseases_set.union(symptoms_set)

        rows: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for _id, dist, meta, doc in zip(ids, dists, metas, docs):
            if not _id or _id in seen:
                continue
            seen.add(_id)
            if exclude_ids and _id in exclude_ids:
                continue

            base_score = distance_to_score10(dist)
            if base_score < self.min_base10:
                continue

            # الشكل الدوائي
            is_ok, soft_bonus = form_matches(meta, target_form, strict_form)
            if not is_ok:
                continue

            # استبعاد نفس المادة الفعالة
            ai = normalize_ar((meta or {}).get("active_ingredients") or (meta or {}).get("المواد الفعالة") or "")
            sci = normalize_ar((meta or {}).get("scientific_name") or (meta or {}).get("الاسم العلمي") or "")
            if active_norm and (active_norm in ai or active_norm in sci):
                continue

            brand = best_brand_field(meta)

            cand_d = normalize_set(extract_diseases_phrases(meta))
            cand_s = normalize_set(extract_symptoms_phrases(meta))
            cand_union = cand_d.union(cand_s)

            overlap = 0.0
            if seeds_union:
                inter = seeds_union.intersection(cand_union)
                overlap = len(inter) / max(1, len(seeds_union))

            bonus = 0.0
            tag_parts: List[str] = []
            if soft_bonus and target_form:
                bonus += FORM_MATCH_BONUS if strict_form else FORM_SOFT_BONUS
                tag_parts.append("form_match")

            if overlap > 0:
                bonus += ALT_DISEASE_OVERLAP_BOOST * overlap
                tag_parts.append(f"alt_overlap({overlap:.2f})")
            else:
                tag_parts.append("alt_overlap(0)")

            cand_class = normalize_ar(get_classification(meta))
            cand_section = normalize_ar(get_section(meta))
            if seed_class and cand_class and (seed_class == cand_class or seed_class in cand_class or cand_class in seed_class):
                bonus += ALT_CLASS_MATCH_BONUS
                tag_parts.append("class_match")
            if seed_section and cand_section and (seed_section == cand_section or seed_section in cand_section or cand_section in seed_section):
                bonus += ALT_SECTION_MATCH_BONUS
                tag_parts.append("section_match")

            # طبّق تحويل التراكيز إلى قائمة عند توفرها
            meta_out = dict(meta or {})
            try:
                parsed_conc = parse_concentrations_from_meta(meta_out)
                if parsed_conc:
                    meta_out["concentrations_raw"] = meta_out.get("concentrations")
                    meta_out["concentrations"] = parsed_conc
            except Exception:
                pass

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
                "meta": meta_out,
                "doc": doc,
                # مفاتيح مساعدة للفرز
                "_overlap": overlap,
                "_base": base_score,
            }
            rows.append(row)

        # الفرز: أعلى تداخل ثم final_score ثم base
        rows.sort(key=lambda x: (-(x.get("_overlap", 0.0)), -(x["final_score"]), -(x.get("_base", 0.0))))
        for r in rows:
            r.pop("_overlap", None)
            r.pop("_base", None)
        if limit:
            rows = rows[:limit]
        return rows

    def find_equivalents_multi(
        self,
        actives: List[Any],
        target_form: Optional[str] = None,
        strict_form: bool = True,
        limit: int = 50,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        بحث مثائل لمجموعة مواد فعّالة مع تراكيزها.
        - يقدّم النتائج التي تحتوي كل المواد أولًا، ثم التي تحتوي اثنتين، ثم واحدة.
        - داخل كل مجموعة: يفضّل من يطابق التراكيز المقصودة، ثم final_score.
        actives: قائمة مثل [{"باراسيتامول": 500}, {"سودوإيفيدرين": 30}, {"دكستروميثورفان": 15}]
        """
        pairs = normalize_actives_input(actives)
        if debug:
            logger.debug("[multi] normalized pairs: %s", pairs)
        if not pairs:
            return []

        # 1) اجمع مرشحين من استعلامات لكل مادة + استعلامات مركبة (الكل + أزواج)
        q_norm_list = [nm for (nm, _mg) in pairs]
        if debug:
            logger.debug("[multi] q_norm_list: %s", q_norm_list)
        merged: Dict[str, Dict[str, Any]] = {}
        # استعلامات فردية
        for qn in q_norm_list:
            for res in (
                self._query_active_first(qn),
                self._query_scientific_only(qn),
                self._query_brand(qn),
            ):
                if debug:
                    ids_dbg = res.get("ids", [[]])[0]
                    logger.debug("[multi] single-query '%s' -> %d hits", qn, len(ids_dbg))
                ids   = res.get("ids", [[]])[0]
                dists = res.get("distances", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                docs  = res.get("documents", [[]])[0]
                for _id, dist, meta, doc in zip(ids, dists, metas, docs):
                    if _id not in merged or (dist is not None and dist < merged[_id]["dist"]):
                        merged[_id] = {"dist": dist, "meta": meta, "doc": doc}

        # استعلام مركّب بكل المواد
        if len(q_norm_list) >= 2:
            joined = "، ".join(q_norm_list)
            # نوسّع n_results للتركيبات
            try:
                prompts = [
                    f"active ingredients or scientific: {joined}",
                    f"scientific name: {joined}",
                    f"brand or commercial name: {joined}",
                ]
                for prompt in prompts:
                    q_emb = self.emb.embed_query(prompt)
                    res = self.collection.query(
                        query_embeddings=[q_emb],
                        n_results=min(self.top_k * 3, 5000),
                        include=["distances", "metadatas", "documents"],
                    )
                    if debug:
                        ids_dbg = res.get("ids", [[]])[0]
                        logger.debug("[multi] joined all prompt='%s' -> %d hits", prompt, len(ids_dbg))
                    ids   = res.get("ids", [[]])[0]
                    dists = res.get("distances", [[]])[0]
                    metas = res.get("metadatas", [[]])[0]
                    docs  = res.get("documents", [[]])[0]
                    for _id, dist, meta, doc in zip(ids, dists, metas, docs):
                        if _id not in merged or (dist is not None and dist < merged[_id]["dist"]):
                            merged[_id] = {"dist": dist, "meta": meta, "doc": doc}
            except Exception as e:
                if debug:
                    logger.debug("[multi] joined all query failed: %s", e)

        # استعلامات أزواج (تحسين تذكّر التركيبات)
        if len(q_norm_list) >= 3:
            for i in range(len(q_norm_list)):
                for j in range(i + 1, len(q_norm_list)):
                    pair_join = f"{q_norm_list[i]}، {q_norm_list[j]}"
                    try:
                        prompts = [
                            f"active ingredients or scientific: {pair_join}",
                            f"scientific name: {pair_join}",
                            f"brand or commercial name: {pair_join}",
                        ]
                        for prompt in prompts:
                            q_emb = self.emb.embed_query(prompt)
                            res = self.collection.query(
                                query_embeddings=[q_emb],
                                n_results=min(self.top_k * 2, 4000),
                                include=["distances", "metadatas", "documents"],
                            )
                            if debug:
                                ids_dbg = res.get("ids", [[]])[0]
                                logger.debug("[multi] pair prompt='%s' -> %d hits", prompt, len(ids_dbg))
                            ids   = res.get("ids", [[]])[0]
                            dists = res.get("distances", [[]])[0]
                            metas = res.get("metadatas", [[]])[0]
                            docs  = res.get("documents", [[]])[0]
                            for _id, dist, meta, doc in zip(ids, dists, metas, docs):
                                if _id not in merged or (dist is not None and dist < merged[_id]["dist"]):
                                    merged[_id] = {"dist": dist, "meta": meta, "doc": doc}
                    except Exception as e:
                        if debug:
                            logger.debug("[multi] pair query failed '%s': %s", pair_join, e)

        if debug:
            logger.debug("[multi] merged candidates: %d", len(merged))

        # 2) ابنِ الصفوف مع حساب المطابقات عبر كل المواد
        rows: List[Dict[str, Any]] = []
        for _id, pack in merged.items():
            dist = pack["dist"]
            meta = pack["meta"]
            doc  = pack["doc"]

            base_score = distance_to_score10(dist)
            if base_score < self.min_base10:
                continue

            # الشكل الدوائي
            is_ok, soft_bonus = form_matches(meta, target_form, strict_form)
            if not is_ok:
                continue

            bonus = 0.0
            tag_parts: List[str] = []
            if target_form and soft_bonus:
                bonus += FORM_MATCH_BONUS if strict_form else FORM_SOFT_BONUS
                tag_parts.append("form_match")

            matched_count = 0
            exact_conc_count = 0
            per_active_matches: List[Dict[str, Any]] = []

            for (act_norm, mg) in pairs:
                a_ratio, a_tok = best_active_fuzzy_ratio(meta, act_norm)
                sci_ratio = fuzzy_ratio_on_scientific_only(meta, act_norm)
                relevant = (a_ratio >= RELEVANT_MIN) or (sci_ratio >= RELEVANT_MIN)

                conc_exact = False
                conc_num: Optional[float] = None
                if relevant and mg and mg > 0:
                    conc_exact, conc_num = has_exact_concentration(meta, act_norm, mg)

                if relevant:
                    matched_count += 1
                    if conc_exact:
                        exact_conc_count += 1

                # بونصات المادة الواحدة
                if a_ratio >= ACTIVE_FUZZY_MIN:
                    bonus += ACTIVE_BASE_BOOST * a_ratio
                    if a_ratio >= 0.90:
                        bonus += ACTIVE_STRONG_BOOST
                if sci_ratio >= SCIENTIFIC_FUZZY_MIN:
                    bonus += SCIENTIFIC_BASE_BOOST * sci_ratio
                    if sci_ratio >= 0.90:
                        bonus += SCIENTIFIC_STRONG_BOOST

                per_active_matches.append({
                    "active": act_norm,
                    "mg": mg,
                    "active_ratio": round(a_ratio, 3),
                    "scientific_ratio": round(sci_ratio, 3),
                    "relevant": relevant,
                    "conc_exact": bool(conc_exact),
                    "conc_num": conc_num,
                })

            # بونص التجاري (مرة واحدة بأفضل تطابق عبر المواد)
            brand = best_brand_field(meta)
            b_ratio, b_token, b_pen = brand_best_multi_ratio(brand, q_norm_list)
            if b_ratio >= BRAND_FUZZY_MIN:
                bonus += BRAND_BASE_BOOST * b_ratio * b_pen
                if b_ratio >= 0.90 or (b_token and abs(len(b_token) - max(len(q) for q in q_norm_list)) <= 1):
                    bonus += BRAND_STRONG_BOOST
                    tag_parts.append("brand_strong_fuzzy")
                else:
                    tag_parts.append("brand_fuzzy")
            else:
                tag_parts.append("brand_miss")

            # تحويل التراكيز إن أمكن
            meta_out = dict(meta or {})
            try:
                parsed_conc = parse_concentrations_from_meta(meta_out)
                if parsed_conc:
                    meta_out["concentrations_raw"] = meta_out.get("concentrations")
                    meta_out["concentrations"] = parsed_conc
            except Exception:
                pass

            joined_query = " + ".join([nm for (nm, _m) in pairs])
            row = {
                "id": _id,
                "query": joined_query,
                "base_score10": round(base_score, 3),
                "final_score": round(base_score + bonus, 3),
                "bonus": round(bonus, 3),
                "tag": "+".join(tag_parts) if tag_parts else "vector_only",
                "brand": brand or None,
                "name": (meta or {}).get("name") or (meta or {}).get("اسم الدواء الأصلي"),
                "commercial_name": brand or (meta or {}).get("commercial_name") or (meta or {}).get("الاسم التجاري"),
                "scientific_name": (meta or {}).get("scientific_name") or (meta or {}).get("الاسم العلمي"),
                "manufacturer": (meta or {}).get("manufacturer") or (meta or {}).get("الشركة المصنعة"),
                "meta": meta_out,
                "doc": doc,
                # مفاتيح مساعدة للفرز
                "_matched_count": matched_count,
                "_exact_conc_count": exact_conc_count,
                "_actives_total": len(pairs),
                # معلومات تفصيلية اختيارية
                "matches": per_active_matches,
            }
            # استبعد من ليس لديه أي تطابق ذي صلة
            if matched_count <= 0:
                continue
            rows.append(row)
            if debug and matched_count > 0:
                try:
                    brand_dbg = best_brand_field(meta)
                    logger.debug(
                        "[multi] cand id=%s brand=%s matched=%d exact=%d base=%.3f final=%.3f",
                        _id, brand_dbg, matched_count, exact_conc_count, base_score, row["final_score"],
                    )
                except Exception:
                    pass

        # 3) الفرز النهائي: عدد المطابقات ثم عدد تراكيز مطابقة ثم final_score ثم base
        rows.sort(key=lambda x: (
            -x.get("_matched_count", 0),
            -x.get("_exact_conc_count", 0),
            -(x["final_score"]),
            -(x["base_score10"]) 
        ))

        # تنظيف المفاتيح المساعدة + limit
        for r in rows:
            r.pop("_matched_count", None)
            r.pop("_exact_conc_count", None)
            r.pop("_actives_total", None)
        if limit:
            rows = rows[:limit]
        return rows

    def _collect_seed_context_multi(
        self,
        actives: List[Any],
        target_form: Optional[str],
        strict_form: bool,
        seed_limit: int = 20,
    ) -> Tuple[Set[str], Set[str], str, str]:
        """يجمع سياق علاجي كبذور اعتمادًا على نتائج multi-equivalents."""
        seeds = self.find_equivalents_multi(
            actives=actives,
            target_form=target_form,
            strict_form=strict_form,
            limit=seed_limit,
            debug=False,
        )
        diseases_union: Set[str] = set()
        symptoms_union: Set[str] = set()
        class_counts: Dict[str, int] = {}
        section_counts: Dict[str, int] = {}
        for r in seeds:
            meta = r.get("meta") or {}
            d = normalize_set(extract_diseases_phrases(meta))
            s = normalize_set(extract_symptoms_phrases(meta))
            diseases_union.update(d)
            symptoms_union.update(s)
            cl = normalize_ar(get_classification(meta))
            sec = normalize_ar(get_section(meta))
            if cl:
                class_counts[cl] = class_counts.get(cl, 0) + 1
            if sec:
                section_counts[sec] = section_counts.get(sec, 0) + 1

        def pick_max(counts: Dict[str, int]) -> str:
            if not counts:
                return ""
            return max(counts.items(), key=lambda kv: kv[1])[0]

        return diseases_union, symptoms_union, pick_max(class_counts), pick_max(section_counts)

    def find_alternatives_multi(
        self,
        actives: List[Any],
        target_form: Optional[str] = None,
        strict_form: bool = True,
        limit: int = 50,
        exclude_ids: Optional[Set[str]] = None,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        بدائل لمجموعة مواد مختلفة لكن تعالج نفس الأمراض/الأعراض.
        - يُستبعد من لديه بالضبط نفس كل المواد المطلوبة.
        - لو كان الدواء المرشح ذو مادة واحدة وتطابق إحدى المواد الأصلية، يُستبعد.
        """
        pairs = normalize_actives_input(actives)
        if not pairs:
            return []

        orig_set: Set[str] = {nm for (nm, _mg) in pairs}
        diseases_set, symptoms_set, seed_class, seed_section = self._collect_seed_context_multi(
            actives, target_form, strict_form, seed_limit=min(20, self.top_k)
        )

        if not diseases_set and not symptoms_set and not seed_class and not seed_section:
            return []

        diseases_text = "، ".join(sorted(diseases_set))
        symptoms_text = "، ".join(sorted(symptoms_set))
        res = self._query_by_profile(diseases_text, symptoms_text, seed_class, seed_section)

        ids   = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        docs  = res.get("documents", [[]])[0]

        rows: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for _id, dist, meta, doc in zip(ids, dists, metas, docs):
            if not _id or _id in seen:
                continue
            seen.add(_id)
            if exclude_ids and _id in exclude_ids:
                continue

            base_score = distance_to_score10(dist)
            if base_score < self.min_base10:
                continue

            # الشكل الدوائي
            is_ok, soft_bonus = form_matches(meta, target_form, strict_form)
            if not is_ok:
                continue

            # تجميع توكنات المواد للمرشح
            cand_tokens = set(extract_active_tokens(meta))

            # استبعاد من يطابق كل المواد الأصلية (نفس التركيبة)
            if orig_set and orig_set.issubset(cand_tokens):
                continue

            # لو المرشح ذو مادة واحدة وهي إحدى المواد الأصلية → استبعد
            if len(cand_tokens) == 1 and any(t in orig_set for t in cand_tokens):
                continue

            brand = best_brand_field(meta)

            cand_d = normalize_set(extract_diseases_phrases(meta))
            cand_s = normalize_set(extract_symptoms_phrases(meta))
            cand_union = cand_d.union(cand_s)

            seeds_union = diseases_set.union(symptoms_set)
            overlap = 0.0
            if seeds_union:
                inter = seeds_union.intersection(cand_union)
                overlap = len(inter) / max(1, len(seeds_union))

            bonus = 0.0
            tag_parts: List[str] = []
            if soft_bonus and target_form:
                bonus += FORM_MATCH_BONUS if strict_form else FORM_SOFT_BONUS
                tag_parts.append("form_match")
            if overlap > 0:
                bonus += ALT_DISEASE_OVERLAP_BOOST * overlap
                tag_parts.append(f"alt_overlap({overlap:.2f})")
            else:
                tag_parts.append("alt_overlap(0)")

            cand_class = normalize_ar(get_classification(meta))
            cand_section = normalize_ar(get_section(meta))
            if seed_class and cand_class and (seed_class == cand_class or seed_class in cand_class or cand_class in seed_class):
                bonus += ALT_CLASS_MATCH_BONUS
                tag_parts.append("class_match")
            if seed_section and cand_section and (seed_section == cand_section or seed_section in cand_section or cand_section in seed_section):
                bonus += ALT_SECTION_MATCH_BONUS
                tag_parts.append("section_match")

            # تحويل التراكيز إن أمكن
            meta_out = dict(meta or {})
            try:
                parsed_conc = parse_concentrations_from_meta(meta_out)
                if parsed_conc:
                    meta_out["concentrations_raw"] = meta_out.get("concentrations")
                    meta_out["concentrations"] = parsed_conc
            except Exception:
                pass

            row = {
                "id": _id,
                "query": " + ".join(sorted(list(orig_set))),
                "base_score10": round(base_score, 3),
                "final_score": round(base_score + bonus, 3),
                "bonus": round(bonus, 3),
                "tag": "+".join(tag_parts) if tag_parts else "vector_only",
                "brand": brand or None,
                "name": (meta or {}).get("name") or (meta or {}).get("اسم الدواء الأصلي"),
                "commercial_name": brand or (meta or {}).get("commercial_name") or (meta or {}).get("الاسم التجاري"),
                "scientific_name": (meta or {}).get("scientific_name") or (meta or {}).get("الاسم العلمي"),
                "manufacturer": (meta or {}).get("manufacturer") or (meta or {}).get("الشركة المصنعة"),
                "meta": meta_out,
                "doc": doc,
                # مفاتيح مساعدة للفرز
                "_overlap": overlap,
                "_base": base_score,
            }
            rows.append(row)

        rows.sort(key=lambda x: (-(x.get("_overlap", 0.0)), -(x["final_score"]), -(x.get("_base", 0.0))))
        for r in rows:
            r.pop("_overlap", None)
            r.pop("_base", None)
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
