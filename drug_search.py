# -*- coding: utf-8 -*-
"""
drug_search.py
موديول عام لإيجاد الأدوية من Chroma:
- vector similarity (cosine) + fuzzy على الاسم التجاري (token-by-token) + hint دوائي عام
- يرجّع كل المعلومات (الميتاداتا كاملة) + السكورات والترتيب
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

# ====================== إعدادات افتراضية ======================
DEFAULT_TOP_K = 400
DEFAULT_MIN_SCORE10 = 5.0      # ≥5 (≈ تشابه > 0.5)
STRICT_BRAND_BOOST = 2.4       # بونص للتطابق الفزي القوي
FUZZY_BRAND_BOOST  = 1.5       # وزن الفزي على الاسم التجاري
PHARMA_HINT_BONUS  = 0.8       # بونص عام لو query قريب من الاسم العلمي/المادة الفعالة
ACTIVE_FUZZY_MIN   = 0.75      # أدنى نسبة فزي لقبول التطابق على العلمي/المادة
BRAND_FUZZY_MIN    = 0.70      # أدنى نسبة فزي لقبول التطابق على الاسم التجاري
# =============================================================

# ---- تطبيع عربي/لاتيني بسيط ----
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06ED]")
AR_CHARS = "ء-ي"
_NON_WORD = re.compile(rf"[^\w{AR_CHARS}]+", re.UNICODE)

def normalize_ar(text: str) -> str:
    if not isinstance(text, str): return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = (text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
                .replace("ى","ي").replace("ة","ه")
                .replace("ؤ","و").replace("ئ","ي"))
    return re.sub(r"\s+", " ", text).strip().lower()

# ---- كشف الإنجليزية وتوليد بدائل عربية تقريبية (ترانسلترشن مبسّط) ----
_HAS_LATIN = re.compile(r"[A-Za-z]")

def is_latin_text(text: str) -> bool:
    return bool(text) and bool(_HAS_LATIN.search(str(text)))

def latin_to_arabic_variants(text: str) -> List[str]:
    """
    يحاول توليد تهجئات عربية قريبة من الاسم الإنجليزي لرفع وزن الفزي على التجاري.
    أمثلة: "panadol" -> ["بانادول", "بنادول"]
    قواعد مبسّطة جدًا وواقعية للأسماء الشائعة؛ ليست مثالية لكنها عملية.
    """
    if not isinstance(text, str):
        return []
    s = text.strip().lower()
    if not s:
        return []

    # أولاً: أزواج حروف شائعة قبل الحرف المنفرد
    digraphs = [
        ("sh", "ش"), ("ch", "تش"), ("th", "ث"), ("ph", "ف"),
        ("gh", "غ"), ("kh", "خ"), ("oo", "و"), ("ou", "و"),
        ("ee", "ي"), ("ea", "ي"), ("ai", "اي"), ("ie", "ي"),
    ]
    tmp = s
    for en, ar in digraphs:
        tmp = tmp.replace(en, f" {ar} ")  # نحجزها مؤقتًا بمسافات

    # بعدها حروف منفردة
    char_map = {
        'a': 'ا', 'b': 'ب', 'c': 'ك', 'd': 'د', 'e': 'ي', 'f': 'ف', 'g': 'ج',
        'h': 'ه', 'i': 'ي', 'j': 'ج', 'k': 'ك', 'l': 'ل', 'm': 'م', 'n': 'ن',
        'o': 'و', 'p': 'ب', 'q': 'ق', 'r': 'ر', 's': 'س', 't': 'ت', 'u': 'و',
        'v': 'ف', 'w': 'و', 'x': 'كس', 'y': 'ي', 'z': 'ز'
    }
    out = []
    for ch in tmp:
        if ch == ' ':
            out.append(' ')
        elif 'a' <= ch <= 'z':
            out.append(char_map.get(ch, ch))
        else:
            out.append(ch)
    cand = re.sub(r"\s+", "", "".join(out))

    # تحسينات خفيفة: ضغط الحروف الممدودة الشائعة
    cand = cand.replace("يي", "ي").replace("وو", "و")

    variants = {cand}

    # متغير: إسقاط ألف بعد باء البداية لإنتاج شكل دارج (بانادول → بنادول)
    if cand.startswith("با"):
        variants.add("ب" + cand[2:])

    # متغير: معالجة ch→تش قد تكون ك→تش والعكس (ننتج بديلًا بحرف ك)
    variants.add(cand.replace("تش", "ش"))
    variants.add(cand.replace("تش", "ك"))

    # تنظيف وحصر طول معقول
    clean = [v for v in variants if 2 <= len(v) <= 64]
    # إعادة التطبيع لضمان اتساق المقارنة الفزي لاحقًا
    return list({normalize_ar(v) for v in clean})

def distance_to_score10(d: Optional[float]) -> float:
    """حوّل مسافة كوساين (0..2) إلى درجة تشابه 0..10."""
    if d is None: return 0.0
    s = (1.0 - (d / 2.0)) * 10.0
    return max(0.0, min(10.0, float(s)))

# ---- fuzzy على مستوى كلمات الاسم التجاري + عقوبة طول ----
def best_token_fuzzy_ratio(brand: str, q_norm: str) -> Tuple[float, str, float]:
    """
    يعيد:
      ratio   : أعلى تشابه (0..1)
      token   : التوكن الأفضل
      penalty : عقوبة للطول (0.7..1.0) تقل لو التوكن أطول من الاستعلام
    """
    v = normalize_ar(brand)
    if not v or not q_norm:
        return 0.0, "", 1.0
    tokens = [t for t in _NON_WORD.split(v) if t]
    best = (0.0, "", 1.0)
    for t in tokens:
        ratio = difflib.SequenceMatcher(None, t, q_norm).ratio()
        # عقوبة طول: تقلل وزن التطابق لو التوكن أطول بوضوح من الاستعلام
        rel_extra = max(0, len(t) - len(q_norm)) / max(len(t), 1)
        penalty = 1.0 - min(0.30, rel_extra)  # 0.7..1.0
        if t.startswith(q_norm) or t.endswith(q_norm):
            ratio = min(1.0, ratio + 0.05)
        if ratio * penalty > best[0] * best[2]:
            best = (ratio, t, penalty)
    return best

# ---- فزي عام على الاسم العلمي/المواد الفعالة ----
_SPLIT_ACTIVE = re.compile(r"[,\+\-/\(\)\[\]{}؛;،]|(?:\s+و\s+)|(?:\s+and\s+)", re.IGNORECASE)

def extract_active_tokens(meta: Dict[str, Any]) -> List[str]:
    parts = []
    for k in ("scientific_name", "الاسم العلمي", "active_ingredients", "المواد الفعالة"):
        val = (meta or {}).get(k)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    text = " ، ".join(parts)
    raw_tokens = [t.strip() for t in _SPLIT_ACTIVE.split(text) if t and t.strip()]
    tokens = []
    for t in raw_tokens:
        tn = normalize_ar(t)
        if len(tn) >= 3:
            tokens.append(tn)
    return tokens

def best_active_fuzzy_ratio(meta: Dict[str, Any], q_norm: str) -> Tuple[float, str]:
    tokens = extract_active_tokens(meta)
    best = 0.0
    best_tok = ""
    for t in tokens:
        r = difflib.SequenceMatcher(None, t, q_norm).ratio()
        if r > best:
            best = r
            best_tok = t
    return best, best_tok

# ---- مفاتيح الأسماء ----
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


class DrugSearch:
    """
    - يهيّئ Chroma + Embeddings مرة واحدة
    - search_one(query, ...) يرجّع قائمة دكتات بكل الميتاداتا + السكورات
    """

    def __init__(
        self,
        collection_name: str = Config.COLLECTION_NAME,
        db_dir: str = Config.DB_DIRECTORY,
        top_k: int = DEFAULT_TOP_K,
        min_score10: float = DEFAULT_MIN_SCORE10,
    ) -> None:
        self.collection_name = collection_name
        self.db_dir = db_dir
        self.top_k = top_k
        self.min_score10 = min_score10

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

        # Init Embeddings (نفس إعدادات البناء)
        self.emb = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_SETTINGS['model_name'],
            model_kwargs=Config.EMBEDDING_SETTINGS['model_kwargs'],
            encode_kwargs=Config.EMBEDDING_SETTINGS['encode_kwargs'],
        )

    def _query_once(self, query: str) -> Dict[str, Any]:
        q_emb = self.emb.embed_query(f"query: {query}")
        return self.collection.query(
            query_embeddings=[q_emb],
            n_results=self.top_k,
            include=["distances", "metadatas", "documents"],
        )

    def search_one(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score10: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        يبحث باستعلام واحد ويرجّع النتائج مرتّبة تنازليًا حسب final_score ثم base_score10.
        كل عنصر يحتوي:
          id, base_score10, final_score, bonus, tag, query, brand/name/scientific..., meta(كاملة), doc
        """
        if not query:
            return []

        tk = top_k if top_k is not None else self.top_k
        ms = min_score10 if min_score10 is not None else self.min_score10

        # اجلب نتائج المتجهات. عند إدخال إنجليزي، نجرب أيضًا بدائل عربية (ترانسلترشن) وندمج.
        primary = self._query_once(query)

        pools = [primary]
        ar_qs: List[str] = []
        if is_latin_text(query):
            ar_qs = latin_to_arabic_variants(query)
            for aq in ar_qs[:3]:  # نحد حتى 3 بدائل لأداء جيد
                try:
                    pools.append(self._query_once(aq))
                except Exception:
                    continue

        merged: Dict[str, Tuple[float, Dict[str, Any], Any]] = {}
        for res in pools:
            ids   = res.get("ids", [[]])[0]
            dists = res.get("distances", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            docs  = res.get("documents", [[]])[0]
            for _id, dist, meta, doc in zip(ids, dists, metas, docs):
                if _id is None:
                    continue
                if (_id not in merged) or (dist is not None and dist < merged[_id][0]):
                    merged[_id] = (dist, meta, doc)

        ids_list: List[str] = []
        dists_list: List[float] = []
        metas_list: List[Dict[str, Any]] = []
        docs_list: List[Any] = []
        for _id, (dist, meta, doc) in merged.items():
            ids_list.append(_id)
            dists_list.append(dist)
            metas_list.append(meta)
            docs_list.append(doc)

        q_norm = normalize_ar(query)
        rows: List[Dict[str, Any]] = []

        for _id, dist, meta, doc in zip(ids_list, dists_list, metas_list, docs_list):
            base_score = distance_to_score10(dist)
            if base_score <= ms:
                continue

            brand = best_brand_field(meta)
            bonus = 0.0
            tag   = "vector_only"

            # (1) Fuzzy على الاسم التجاري token-by-token + length penalty
            # نحسب أفضل فزي عبر الاستعلام الأصلي + بدائل عربية إن وُجدت
            q_candidates = [q_norm]
            if ar_qs:
                q_candidates.extend(ar_qs)
            ratio = 0.0
            token = ""
            penalty = 1.0
            for qv in q_candidates:
                r, t, p = best_token_fuzzy_ratio(brand or "", qv)
                if r * p > ratio * penalty:
                    ratio, token, penalty = r, t, p
            if ratio >= BRAND_FUZZY_MIN:
                fuzzy_boost = FUZZY_BRAND_BOOST * ratio * penalty
                bonus += fuzzy_boost
                tag = f"brand_fuzzy_token r={ratio:.2f} p={penalty:.2f}"

            # (2) تعزيز لو التطابق قوي جدًا/الطول قريب
            if ratio >= 0.90 or (token and abs(len(token) - len(q_norm)) <= 1):
                bonus += STRICT_BRAND_BOOST
                tag = "brand_strong_fuzzy"

            # (3) بونص دوائي عام لو الاسم العلمي/المواد الفعالة قريبة من الاستعلام
            a_ratio, a_tok = best_active_fuzzy_ratio(meta, q_norm)
            if a_ratio >= ACTIVE_FUZZY_MIN:
                bonus += PHARMA_HINT_BONUS * a_ratio
                tag = (tag + "+active_fuzzy") if tag != "vector_only" else "active_fuzzy"

            final_score = base_score + bonus

            row = {
                "id": _id,
                "query": query,
                "base_score10": round(base_score, 3),
                "final_score": round(final_score, 3),
                "bonus": round(bonus, 3),
                "tag": tag,
                "brand": brand or None,
                "name": (meta or {}).get("name") or (meta or {}).get("اسم الدواء الأصلي"),
                "commercial_name": brand or (meta or {}).get("commercial_name") or (meta or {}).get("الاسم التجاري"),
                "scientific_name": (meta or {}).get("scientific_name") or (meta or {}).get("الاسم العلمي"),
                "manufacturer": (meta or {}).get("manufacturer") or (meta or {}).get("الشركة المصنعة"),
                # كل الميتاداتا كاملة كما هي:
                "meta": meta,
                # نص الدوكيومنت لو موجود:
                "doc": doc,
            }
            rows.append(row)

        rows.sort(key=lambda x: (x["final_score"], x["base_score10"]), reverse=True)
        if limit:
            rows = rows[:limit]
        return rows
