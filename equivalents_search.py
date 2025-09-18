# -*- coding: utf-8 -*-
"""
equivalents_search.py
ابحث عن مثائل حسب: المادة الفعّالة + التركيز + (اختياري) نفس الشكل الدوائي.
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

# ====================== إعدادات عامة ======================
DEFAULT_TOP_K = 800
DEFAULT_MIN_BASE10 = 0.0
ACTIVE_FUZZY_MIN   = 0.80   # فزي للمادة
# ==========================================================

# ---- تطبيع عربي/لاتيني ----
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06ED]")
AR_CHARS = "ء-ي"
_NON_WORD = re.compile(rf"[^\w{AR_CHARS}]+", re.UNICODE)

def normalize_ar(text: str) -> str:
    if not isinstance(text, str): return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = (text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
                .replace("ى","ي").replace("ة","ه")
                .replace("ؤ","و").replace("ئ","ي")).lower()
    return re.sub(r"\s+", " ", text).strip()

def distance_to_score10(d: Optional[float]) -> float:
    if d is None: return 0.0
    s = (1.0 - (d / 2.0)) * 10.0
    return max(0.0, min(10.0, float(s)))

# ---- فزي للمادة الفعالة/الاسم العلمي ----
_SPLIT_ACTIVE = re.compile(r"[,\+\-/\(\)\[\]{}؛;،]|(?:\s+و\s+)|(?:\s+and\s+)", re.IGNORECASE)
def extract_active_tokens(meta: Dict[str, Any]) -> List[str]:
    parts = []
    for k in ("scientific_name", "الاسم العلمي", "active_ingredients", "المواد الفعالة"):
        val = (meta or {}).get(k)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    if not parts:
        return []
    text = " ، ".join(parts)
    raw_tokens = [t.strip() for t in _SPLIT_ACTIVE.split(text) if t and t.strip()]
    toks = []
    for t in raw_tokens:
        tn = normalize_ar(t)
        if len(tn) >= 3:
            toks.append(tn)
    return toks

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

# ---- استخراج تركيزات وتحويلها mg ----
_P_NUM = r"(?P<num>\d+(?:[.,]\d+)?)"
UNIT_MG = r"(?:mg|MG|ملغ|مجم|ملجم|ميلي ?جرام|ميلي ?غرام)"
UNIT_G  = r"(?:g|G|جم|جرام|غ|غرام)"
UNIT_ML = r"(?:ml|mL|ML|مل|مليلتر)"

RX_MG        = re.compile(rf"{_P_NUM}\s*{UNIT_MG}")
RX_G         = re.compile(rf"{_P_NUM}\s*{UNIT_G}")
RX_MG_PER_ML = re.compile(rf"{_P_NUM}\s*{UNIT_MG}\s*/\s*{_P_NUM}\s*{UNIT_ML}")
RX_G_PER_ML  = re.compile(rf"{_P_NUM}\s*{UNIT_G}\s*/\s*{_P_NUM}\s*{UNIT_ML}")

STRENGTH_KEYS = [
    "التراكيز", "التركيز", "strength", "dose_strength", "تركيز", "dosage_strength",
    "name", "اسم الدواء الأصلي", "commercial_name", "الاسم التجاري",
]

def _to_float(x: str) -> float:
    return float(x.replace(",", ".").strip())

def _extract_strengths_from_text(text: str) -> List[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return []
    t = text
    out: List[Dict[str, Any]] = []

    for m in RX_MG.finditer(t):
        out.append({"mg": _to_float(m.group("num")), "kind": "mg", "raw": m.group(0)})

    for m in RX_G.finditer(t):
        out.append({"mg": _to_float(m.group("num")) * 1000.0, "kind": "g", "raw": m.group(0)})

    for m in RX_MG_PER_ML.finditer(t):
        mg = _to_float(m.group("num"))
        out.append({"mg": mg, "ml": True, "kind": "mg_per_ml", "raw": m.group(0)})

    for m in RX_G_PER_ML.finditer(t):
        mg = _to_float(m.group("num")) * 1000.0
        out.append({"mg": mg, "ml": True, "kind": "g_per_ml", "raw": m.group(0)})

    return out

def extract_strengths(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    vals = []
    for k in STRENGTH_KEYS:
        v = (meta or {}).get(k)
        if isinstance(v, str) and v.strip():
            vals.append(v)
    if not vals:
        return []
    all_txt = " | ".join(vals)
    return _extract_strengths_from_text(all_txt)

def strength_matches(items: List[Dict[str, Any]], target_mg: float, allow_per_ml: bool = True, tol_mg: float = 0.0) -> Tuple[bool, Optional[Dict[str, Any]]]:
    for it in items:
        mg = it.get("mg")
        if mg is None:
            continue
        if not allow_per_ml and it.get("ml"):
            continue
        if abs(mg - float(target_mg)) <= tol_mg:
            return True, it
    return False, None

# ---- تطبيع/كشف الشكل الدوائي (عربي/إنجليزي) → كود قياسي ----
# نطوّع أشكال شائعة إلى أكواد قياسية
FORM_PATTERNS = [
    ("tablet",   [r"اقراص?", r"\btab(?:let|\.|s)?\b", r"caplet", r"chew(?:able)?", r"film[- ]?coated", r"effervescent"]),
    ("capsule",  [r"كبسول(?:ه)?", r"\bcap(?:s|\.|ule|ules)?\b", r"soft\s*gel", r"سوفت ?جيل"]),
    ("syrup",    [r"شراب", r"\bsyrup\b"]),
    ("susp",     [r"معلق", r"\bsusp(?:ension)?\b"]),
    ("drop",     [r"(قطرات|نقط)", r"\bdrops?\b"]),
    ("spray",    [r"(بخاخ|سبراي)", r"\bspray\b", r"aerosol"]),
    ("solution", [r"(محلول)(?!\s*(وريدي|للحقن))", r"\bsolution\b"]),
    ("inj",      [r"(حقن|امبول|أمبول|فيال|وريدي|للحقن)", r"\binj(?:ection)?\b", r"\bamp(?:oule)?\b", r"\bvial\b", r"\biv\b", r"infusion"]),
    ("supp",     [r"(لبوس|تحاميل)", r"\bsupp(?:ository)?\b"]),
    ("cream",    [r"كريم", r"\bcream\b"]),
    ("ointment", [r"(مرهم)", r"\bointment\b"]),
    ("gel",      [r"جل\b", r"\bgel\b"]),
    ("lotion",   [r"لوشن", r"\blotion\b"]),
    ("mouthwash",[r"(غرغره|غرغرة|غسول\s*الفم)", r"\bmouth\s*wash\b", r"\bgargle\b"]),
    ("sachet",   [r"(اكياس|أكياس|ساشيه|مسحوق)", r"\bsachet\b", r"\bpowder\b"]),
    ("shampoo",  [r"شامبو", r"\bshampoo\b"]),
    # زود أشكال تانية لو عندك (patch, foam, balm, stick, ...)
]

FORM_KEYS = ["الشكل الدوائي", "الشكل_الدوائي", "dosage_form", "form", "pharmaceutical_form",
             "name", "اسم الدواء الأصلي", "commercial_name", "الاسم التجاري"]

def canonicalize_form(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    t = normalize_ar(text)
    for code, pats in FORM_PATTERNS:
        for p in pats:
            if re.search(p, t, re.IGNORECASE):
                return code
    return None

def detect_form(meta: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """يحاول يلقط الشكل من عدة حقول ويعيد الكود القياسي + النص الخام + المصدر."""
    for k in FORM_KEYS:
        v = (meta or {}).get(k)
        if isinstance(v, str) and v.strip():
            code = canonicalize_form(v)
            if code:
                return {"canonical": code, "raw": v, "source": k}
    return {"canonical": None, "raw": None, "source": None}

# ---- كلاس البحث عن المثائل ----
class EquivalentsFinder:
    def __init__(
        self,
        collection_name: str = Config.COLLECTION_NAME,
        db_dir: str = Config.DB_DIRECTORY,
        top_k: int = DEFAULT_TOP_K,
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

    def _query_candidates(self, active_query: str, target_mg: float) -> Dict[str, Any]:
        q_text = f"active ingredient: {active_query} strength: {target_mg} mg"
        q_emb = self.emb.embed_query(q_text)
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
        target_form: Optional[str] = None,   # مثال: "أقراص" أو "Tablets" أو "كبسول"...
        strict_form: bool = True,            # True = لازم نفس الشكل؛ False = اسمح بغير معروف
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        يعيد منتجات فيها:
          - المادة ≈ active_query (فزي ≥ ACTIVE_FUZZY_MIN)
          - تركيز = target_mg (± tolerance)
          - (اختياري) نفس الشكل الدوائي = target_form
        """
        res = self._query_candidates(active_query, target_mg)
        ids      = res.get("ids", [[]])[0]
        dists    = res.get("distances", [[]])[0]
        metas    = res.get("metadatas", [[]])[0]
        docs     = res.get("documents", [[]])[0]

        q_norm = normalize_ar(active_query)
        wanted_form_code = canonicalize_form(target_form) if target_form else None

        rows: List[Dict[str, Any]] = []
        for _id, d, meta, doc in zip(ids, dists, metas, docs):
            base10 = distance_to_score10(d)
            if base10 < self.min_base10:
                continue

            # 1) المادة
            a_ratio, a_tok = best_active_fuzzy_ratio(meta, q_norm)
            if a_ratio < ACTIVE_FUZZY_MIN:
                continue

            # 2) التركيز
            strengths = extract_strengths(meta)
            ok, matched_item = strength_matches(strengths, target_mg, allow_per_ml=allow_per_ml, tol_mg=tolerance_mg)
            if not ok:
                continue

            # 3) الشكل الدوائي (لو مطلوب)
            form_info = detect_form(meta)
            form_ok = True
            form_bonus = 0.0
            if wanted_form_code:
                if form_info["canonical"]:
                    form_ok = (form_info["canonical"] == wanted_form_code)
                    form_bonus = 0.6 if form_ok else -1.0  # عقوبة لو مختلف (حتى لو غير صارم)
                else:
                    form_ok = not strict_form  # لو مفيش شكل واضح: نرفض لو strict=True
            if not form_ok:
                continue

            brand = (meta or {}).get("commercial_name") or (meta or {}).get("الاسم التجاري") \
                    or (meta or {}).get("name") or (meta or {}).get("اسم الدواء الأصلي")

            # ترتيب نهائي بسيط
            bonus = 0.0
            bonus += max(0.0, (a_ratio - ACTIVE_FUZZY_MIN)) * 2.0
            if matched_item and matched_item.get("ml"):
                bonus += 0.2
            else:
                bonus += 0.5
            bonus += form_bonus

            final_score = base10 + bonus

            rows.append({
                "id": _id,
                "final_score": round(final_score, 3),
                "base_score10": round(base10, 3),
                "bonus": round(bonus, 3),
                "commercial_name": brand,
                "scientific_name": (meta or {}).get("scientific_name") or (meta or {}).get("الاسم العلمي"),
                "match": {
                    "active_ratio": round(a_ratio, 3),
                    "active_token": a_tok,
                    "strength_hit": matched_item,   # مثال: {'mg': 500.0, 'kind': 'mg', 'raw': '500 ملغ'}
                    "form": {
                        "wanted": wanted_form_code,
                        "found_canonical": form_info["canonical"],
                        "found_raw": form_info["raw"],
                        "source": form_info["source"],
                    },
                },
                "meta": meta,
                "doc": doc,
                "query": {
                    "active": active_query,
                    "target_mg": target_mg,
                    "tolerance_mg": tolerance_mg,
                    "allow_per_ml": allow_per_ml,
                    "target_form": target_form,
                    "strict_form": strict_form,
                },
            })

        rows.sort(key=lambda x: (x["final_score"], x["base_score10"]), reverse=True)
        if limit:
            rows = rows[:limit]
        return rows


# ============== تشغيل سريع من CLI (اختياري) ==============
if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Find equivalents by active ingredient + strength + (optional) dosage form.")
    p.add_argument("-a", "--active", required=True, help="المادة الفعّالة/الاسم العلمي (عربي/إنجليزي)")
    p.add_argument("-s", "--strength", required=True, type=float, help="التركيز بالمليجرام (mg). مثال: 500")
    p.add_argument("--form", type=str, default=None, help="الشكل الدوائي المطلوب (مثال: أقراص / Tablets / كبسول / شراب / أمبول ...)")
    p.add_argument("--tol", type=float, default=0.0, help="سماحية بالمليجرام (افتراضي 0)")
    p.add_argument("--no-perml", action="store_true", help="لا تقبل صيغ mg/ml (اقبل mg فقط)")
    p.add_argument("--loose-form", action="store_true", help="لا ترفض النتائج عديمة الشكل (strict_form=False)")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--minbase", type=float, default=DEFAULT_MIN_BASE10)
    args = p.parse_args()

    finder = EquivalentsFinder(top_k=args.topk, min_base10=args.minbase)
    hits = finder.find_equivalents(
        active_query=args.active,
        target_mg=args.strength,
        tolerance_mg=args.tol,
        allow_per_ml=not args.no_perml,
        target_form=args.form,
        strict_form=not args.loose_form,
        limit=args.limit,
    )
    print(json.dumps(hits, ensure_ascii=False, indent=2))
