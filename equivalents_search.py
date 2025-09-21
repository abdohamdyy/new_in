# -*- coding: utf-8 -*-
"""
equivalents_search.py (rev: learn-from-drug_search)
- إيجاد بدائل حسب المادة الفعّالة + التركيز (+ الشكل لو مطلوب)
- مستلهم من منطق drug_search: نعتمد fuzzy matching على نصوص الحقول بدل المطابقة الحرفية
- لا نحتاج قوائم مرادفات يدوية؛ بنستخدم تطبيع + فَزّي توكِني
"""

from __future__ import annotations
import os, re, json, logging, unicodedata, difflib
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("equivalents")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[equivalents] %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

DEFAULT_EQ_TOP_K = 800
DEFAULT_EQ_MIN_BASE10 = 0.0

# ---------- Utils ----------
def distance_to_score10(d: Optional[float]) -> float:
    if d is None: return 0.0
    return max(0.0, min(10.0, (1.0 - (d/2.0)) * 10.0))

_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
def normalize_ar(text: str) -> str:
    if not isinstance(text, str): return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = (text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
                 .replace("ى","ي").replace("ة","ه")
                 .replace("ؤ","و").replace("ئ","ي"))
    return re.sub(r"\s+", " ", text).strip().lower()

def normalize_form(v: str) -> str:
    if not v: return ""
    v = v.strip().lower()
    if "قرص" in v or "اقراص" in v:        return "أقراص"
    if "كبسول" in v or "كبسوله" in v:     return "كبسول"
    if "شراب" in v or "syrup" in v:        return "شراب"
    if "قطره" in v or "نقط" in v:         return "قطرة"
    if "مرهم" in v:                        return "مرهم"
    if "جل" in v:                          return "جل"
    if "لبوس" in v or "تحميله" in v:      return "لبوس"
    if "امبول" in v or "فيال" in v or "حقن" in v: return "أمبول"
    return v.strip()

ACTIVE_KEYS = (
    "scientific_name","الاسم العلمي",
    "active_ingredients","المواد الفعالة",
)
FORM_KEYS = ("pharma_form","الشكل الدوائي","form","dosage_form","pharmaceutical_form")
STRENGTH_KEYS = (
    "concentrations","التراكيز",
    "strength","dose_strength","dosage_strength","التركيز","تركيز",
    "name","اسم الدواء الأصلي","commercial_name","الاسم التجاري",
)

def get_first_str(meta: Dict[str,Any], keys: Tuple[str,...]) -> str:
    for k in keys:
        v = (meta or {}).get(k)
        if isinstance(v, str) and v.strip(): return v
    return ""

def get_form(meta: Dict[str,Any]) -> str: return get_first_str(meta, FORM_KEYS)
def get_brand(meta: Dict[str,Any]) -> str:
    return get_first_str(meta, ("commercial_name","الاسم التجاري","الاسم_التجاري","name","اسم الدواء الأصلي"))

def blob_active_text(meta: Dict[str,Any]) -> str:
    parts=[]
    for k in ACTIVE_KEYS:
        v=meta.get(k)
        if isinstance(v,str) and v.strip(): parts.append(v)
    # fallback: أحيانًا المادة تظهر داخل الاسم
    vb = get_first_str(meta, ("name","اسم الدواء الأصلي","commercial_name","الاسم التجاري"))
    if vb: parts.append(vb)
    return " ، ".join(parts)

# ---------- Strength parsing ----------
_CONC_RE = re.compile(r"""
    (?P<drug>[^,\u061B\u060C؛\n\r]+?)\s*
    (?P<amt>\d+(?:\.\d+)?)\s*
    (?P<unit>ملغ|مجم|mg|mcg|ميكروغرام|جم|g)?\s*
    (?:/\s*(?P<per>\d+(?:\.\d+)?)\s*(?:مل|ml))?
""", re.VERBOSE | re.IGNORECASE)

def _to_mg(amount: float, unit: Optional[str]) -> float:
    if not unit: return amount
    u=unit.lower()
    if u in ("ملغ","مجم","mg"): return amount
    if u in ("mcg","ميكروغرام"): return amount/1000.0
    if u in ("جم","g"): return amount*1000.0
    return amount

def extract_strengths(meta: Dict[str,Any]) -> List[Dict[str,Any]]:
    blobs=[]
    for k in STRENGTH_KEYS:
        v=meta.get(k)
        if isinstance(v,str) and v.strip() and v.strip()!="غير متوفر":
            blobs.append(v)
    text=" ، ".join(blobs)
    out=[]
    for m in _CONC_RE.finditer(text):
        drug=(m.group("drug") or "").strip(" -\u061F\u061B\u060C،")
        if not drug: continue
        amt=float(m.group("amt")); unit=m.group("unit") or "mg"
        per=m.group("per"); mg=_to_mg(amt,unit)
        per_ml=None
        if per:
            try:
                vol=float(per)
                if vol>0: per_ml=mg/vol
            except Exception:
                pass
        out.append({"drug":drug, "mg":mg, "per_ml":per_ml})
    return out

# ---------- Fuzzy like drug_search ----------
_SPLIT_TOK = re.compile(r"[,\+\-/\(\)\[\]{}؛;،]|(?:\s+و\s+)|(?:\s+and\s+)|\s+")

def token_fuzzy_max(q: str, text: str) -> float:
    """أعلى درجة فَزّي بين q وبين أي توكن/عبارة قصيرة داخل text."""
    if not q or not text: return 0.0
    qn = normalize_ar(q)
    tn = normalize_ar(text)
    toks = [t for t in _SPLIT_TOK.split(tn) if t and len(t)>=2]
    if not toks: toks=[tn]
    best=0.0
    for t in toks:
        r = difflib.SequenceMatcher(None, qn, t).ratio()
        if r>best: best=r
    # جرّب سلاسل قصيرة من كلمتين متجاورتين (يلتقط "حمض فلاني")
    words = [w for w in tn.split(" ") if w]
    for i in range(len(words)-1):
        big = words[i]+" "+words[i+1]
        r = difflib.SequenceMatcher(None, qn, big).ratio()
        if r>best: best=r
    return best

def has_active_fuzzy(meta: Dict[str,Any], active_query: str, doc: Optional[str]) -> Tuple[bool, float, str]:
    """يرجع (hit?, score, source) حيث المصدر = meta/brand/doc."""
    blob = blob_active_text(meta)
    s1 = token_fuzzy_max(active_query, blob)
    s2 = token_fuzzy_max(active_query, get_brand(meta))
    s3 = token_fuzzy_max(active_query, doc or "")
    best = max(s1, s2, s3)
    # قبول لو:
    # - وجود substring صريح
    aqn = normalize_ar(active_query)
    if aqn and (aqn in normalize_ar(blob) or aqn in normalize_ar(get_brand(meta))):
        return True, max(best, 1.0), "substr"
    # - أو فَزّي قوي >= 0.78 (قابل للتعديل)
    if best >= 0.78:
        return True, best, "fuzzy"
    return False, best, "none"

def is_probably_drug(meta: Dict[str,Any]) -> bool:
    # لازم على الأقل عنده scientific/active_ingredients أو تصنيف دوائي
    if any(isinstance(meta.get(k), str) and meta.get(k).strip() and meta.get(k).strip()!="غير متوفر"
           for k in ("scientific_name","الاسم العلمي","active_ingredients","المواد الفعالة")):
        return True
    if any(isinstance(meta.get(k), str) and meta.get(k).strip()
           for k in ("التصنيف الدوائي","drug_class","القسم")):
        return True
    return False

def is_equivalent_item(
    meta: Dict[str,Any],
    active_query: str,
    target_mg: float,
    tolerance_mg: float=0.0,
    allow_per_ml: bool=True,
    target_form: Optional[str]=None,
    strict_form: bool=True,
    debug: bool=False,
    doc: Optional[str]=None,
) -> Optional[Dict[str,Any]]:

    brand=get_brand(meta)

    # 1) pre-check: لازم يكون شكله دواء
    if not is_probably_drug(meta):
        if debug: logger.debug(f"SKIP not a drug-like item: brand={brand}")
        return None

    # 2) الشكل الدوائي (لو المستخدم محدد form)
    form_raw=get_form(meta); form_norm=normalize_form(str(form_raw))
    if strict_form and target_form:
        if normalize_form(target_form)!=form_norm:
            if debug: logger.debug(f"SKIP form mismatch: wanted={normalize_form(target_form)} found={form_norm} brand={brand}")
            return None

    # 3) المادة الفعّالة (فَزّي)
    ok_active, active_score, src = has_active_fuzzy(meta, active_query, doc)
    if not ok_active:
        if debug: logger.debug(f"SKIP no active hit (score={active_score:.2f}): active='{active_query}' brand={brand}")
        return None

    # 4) التراكيز
    strengths=extract_strengths(meta)
    if not strengths:
        if debug:
            cf = get_first_str(meta,("concentrations","التراكيز")) or "غير متوفر"
            logger.debug(f"SKIP no concentrations parsed: brand={brand} conc_field='{cf[:80]}'")
        return None

    # اختار السطر اللي أكثر شبهًا لاسم المادة المطلوبة
    aqn = normalize_ar(active_query)
    hit=None; best=0.0
    for c in strengths:
        sc = token_fuzzy_max(aqn, c["drug"])
        if sc>best: best=sc; hit=c
    if not hit: hit=strengths[0]

    # 5) مقارنة الجرعة
    if hit["per_ml"] is not None:
        # شراب/محلول (mg/ml)
        if not allow_per_ml:
            if debug: logger.debug(f"SKIP per-ml not allowed: brand={brand}")
            return None
        # لو المستخدم محدد form مش شراب وstrict=True، ارفض
        if strict_form and target_form and normalize_form(target_form)!="شراب":
            if debug: logger.debug(f"SKIP per-ml strict reject: brand={brand}")
            return None
        mg_match=True
    else:
        mg=float(hit["mg"])
        mg_match = abs(mg - float(target_mg)) <= float(tolerance_mg)
        if not mg_match and debug:
            logger.debug(f"SKIP mg mismatch: target={target_mg}±{tolerance_mg} got={mg} brand={brand}")
    if not mg_match: return None

    return {
        "match":{
            "active_match_score": round(active_score,3),
            "active": hit["drug"],
            "mg": float(hit["mg"]),
            "per_ml": hit["per_ml"],
            "form": form_norm,
            "strength_hit": f"{hit['mg']} mg" if hit["per_ml"] is None else f"{hit['mg']} mg per ml={hit['per_ml']:.2f}",
        }
    }

class EquivalentsFinder:
    def __init__(
        self,
        collection_name: str = Config.COLLECTION_NAME,
        db_dir: str = Config.DB_DIRECTORY,
        top_k: int = DEFAULT_EQ_TOP_K,
        min_base10: float = DEFAULT_EQ_MIN_BASE10,
    ) -> None:
        self.collection_name=collection_name
        self.db_dir=db_dir
        self.top_k=top_k
        self.min_base10=min_base10

        self.client = chromadb.PersistentClient(
            path=db_dir,
            settings=Settings(persist_directory=db_dir, anonymized_telemetry=False, allow_reset=True, is_persistent=True)
        )
        self.collection = self.client.get_collection(collection_name)

        self.emb = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_SETTINGS['model_name'],
            model_kwargs=Config.EMBEDDING_SETTINGS['model_kwargs'],
            encode_kwargs=Config.EMBEDDING_SETTINGS['encode_kwargs'],
        )

    def _query_embed(self, text: str):
        return self.emb.embed_query(f"query: {text}")

    def _query(self, qemb, n_results: int):
        return self.collection.query(
            query_embeddings=[qemb],
            n_results=n_results,
            include=["distances","metadatas","documents"],
        )

    def _rank_and_filter(self, ids, dists, metas, docs, active_query, target_mg, tolerance_mg, allow_per_ml, target_form, strict_form, limit, debug):
        rows=[]; dropped={"base10":0,"not_drug":0,"no_active":0,"no_conc":0,"mg_mismatch":0,"form":0,"perml_reject":0}
        for _id, dist, meta, doc in zip(ids, dists, metas, docs):
            base=distance_to_score10(dist)
            if base < self.min_base10:
                dropped["base10"]+=1; continue

            brand=get_brand(meta)

            ok = is_equivalent_item(
                meta=meta, active_query=active_query, target_mg=float(target_mg),
                tolerance_mg=float(tolerance_mg), allow_per_ml=allow_per_ml,
                target_form=target_form, strict_form=strict_form, debug=debug, doc=doc
            )
            if not ok:
                # تخمين سبب الإسقاط (تقريبي)
                if not is_probably_drug(meta):
                    dropped["not_drug"]+=1
                else:
                    # active؟
                    hit, score, _ = has_active_fuzzy(meta, active_query, doc)
                    if not hit: dropped["no_active"]+=1
                    else:
                        strengths=extract_strengths(meta)
                        if not strengths: dropped["no_conc"]+=1
                        else:
                            if strict_form and target_form and normalize_form(target_form)!=normalize_form(get_form(meta) or ""):
                                dropped["form"]+=1
                            else:
                                any_perml = any(s["per_ml"] is not None for s in strengths)
                                if any_perml and (strict_form and target_form and normalize_form(target_form)!="شراب"):
                                    dropped["perml_reject"]+=1
                                else:
                                    dropped["mg_mismatch"]+=1
                continue

            sci = get_first_str(meta, ("scientific_name","الاسم العلمي"))
            row = {
                "id": _id,
                "query": {
                    "active": active_query, "mg": float(target_mg),
                    "form": target_form, "tolerance_mg": float(tolerance_mg),
                    "strict_form": bool(strict_form), "allow_per_ml": bool(allow_per_ml),
                },
                "base_score10": round(base,3),
                "brand": brand or None,
                "name": meta.get("name") or meta.get("اسم الدواء الأصلي"),
                "commercial_name": meta.get("commercial_name") or meta.get("الاسم التجاري"),
                "scientific_name": sci or meta.get("الاسم العلمي"),
                "manufacturer": meta.get("manufacturer") or meta.get("الشركة المصنعة"),
                "form": get_form(meta),
                "match": ok["match"],
                "meta": meta,
                "doc": doc,
                "tag": "equivalent"
            }
            rows.append(row)

        rows.sort(key=lambda x: (x["match"]["active_match_score"], x["base_score10"]), reverse=True)
        if limit: rows = rows[:limit]
        return rows, dropped

    def find_equivalents(
        self, active_query: str, target_mg: float,
        tolerance_mg: float=0.0, allow_per_ml: bool=True,
        target_form: Optional[str]=None, strict_form: bool=True,
        limit: int=50, debug: bool=False
    ) -> List[Dict[str,Any]]:
        if debug:
            logger.setLevel(logging.DEBUG)
            logger.debug(f"REQ active='{active_query}' mg={target_mg} tol={tolerance_mg} form='{target_form}' strict={strict_form} allow_per_ml={allow_per_ml} topk={self.top_k} minbase={self.min_base10}")

        # الجولة 1
        q = self._query_embed(f"active ingredient: {active_query} strength: {target_mg} mg")
        r = self._query(q, n_results=self.top_k)
        ids  = r.get("ids",[[]])[0]
        dists= r.get("distances",[[]])[0]
        metas= r.get("metadatas",[[]])[0]
        docs = r.get("documents",[[]])[0]
        if debug: logger.debug(f"CANDIDATES via vector: {len(ids)}")

        rows, dropped = self._rank_and_filter(ids,dists,metas,docs,active_query,target_mg,tolerance_mg,allow_per_ml,target_form,strict_form,limit,debug)
        if rows:
            if debug:
                logger.debug(f"RETURN {len(rows)} hits")
                logger.debug(f"DROP-STATS: {json.dumps(dropped, ensure_ascii=False)}")
            return rows

        # Rescue: استعلامات أبسط ونطاق أكبر
        if debug: logger.debug("RESCUE: expand query variants and n_results")
        variants = [
            f"{active_query} {int(target_mg)} mg",
            f"{active_query}",
            f"ماده فعاله {active_query}",
            f"الاسم العلمي {active_query}",
        ]
        bag = {}
        for vt in variants:
            qv = self._query_embed(vt)
            rv = self._query(qv, n_results=max(self.top_k, 3000))
            for i, d, m, doc in zip(rv.get("ids",[[]])[0], rv.get("distances",[[]])[0], rv.get("metadatas",[[]])[0], rv.get("documents",[[]])[0]):
                if i not in bag or d < bag[i][0]:
                    bag[i] = (d, m, doc)

        if not bag:
            if debug: logger.debug("RETURN 0 hits (rescue empty)")
            return []

        ids2, dists2, metas2, docs2 = [], [], [], []
        for i,(d,m,doc) in bag.items():
            ids2.append(i); dists2.append(d); metas2.append(m); docs2.append(doc)

        rows2, dropped2 = self._rank_and_filter(ids2,dists2,metas2,docs2,active_query,target_mg,tolerance_mg,allow_per_ml,target_form,strict_form,limit,debug)
        if debug:
            logger.debug(f"RETURN {len(rows2)} hits (after rescue)")
            logger.debug(f"DROP-STATS: {json.dumps(dropped2, ensure_ascii=False)}")
        return rows2
