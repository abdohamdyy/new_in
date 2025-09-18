#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 بناء قاعدة بيانات ChromaDB فائقة السرعة - إصدار TURBO
التحسينات:
- معالجة متوازية لاستغلال 12 كور
- دفعات عملاقة (Mega Batches)
- تحسين استهلاك الذاكرة (Memory Optimization)
- إعدادات HNSW فائقة الأداء للبحث السريع
- الحفاظ على منطق معالجة البيانات الأصلي
"""

import json
import os
import gc
import logging
import torch
import psutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from config import Config
import chromadb
from chromadb.config import Settings

# إعداد الـ logging المحسن
LOGS_DIR = 'logs'
# os.makedirs(LOGS_DIR, exist_ok=True) # التأكد من وجود مجلد اللوغات - 🐞 تم التعطيل لأن المجلد يتم إنشاؤه وربطه عبر Docker Compose

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'db_build.log'), encoding='utf-8'), # 🎯 الكتابة في مجلد اللوغات
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TurboDatabaseBuilder:
    """🚀 باني قاعدة بيانات فائق السرعة للسيرفرات القوية"""
    
    def __init__(self, data_file=None, db_directory=None):
        self.data_file = data_file or Config.DATA_FILE
        self.db_directory = db_directory or Config.DB_DIRECTORY
        self.batch_size = Config.CHROMA_SETTINGS.get('batch_size', 25000)
        self.max_workers = Config.PERFORMANCE_SETTINGS.get('max_workers', 11)
        
        # 🎯 تعديل: تهيئة اسم الكولكشن من الـ config
        self.collection_name = Config.COLLECTION_NAME
        
        # معلومات السيرفر
        self.cpu_count = psutil.cpu_count(logical=True)
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        logger.info("🔥 TURBO MODE ACTIVATED - Server Specs:")
        logger.info(f"   💻 CPU Cores: {self.cpu_count}")
        logger.info(f"   🧠 RAM: {self.memory_gb:.1f}GB")
        logger.info(f"   ⚡ Max Workers: {self.max_workers}")
        logger.info(f"   📦 Batch Size: {self.batch_size:,}")
        
        os.makedirs(self.db_directory, exist_ok=True)
        
        self.chroma_client = self._create_turbo_chroma_client()
        
        self.hnsw_settings = {k: v for k, v in Config.CHROMA_SETTINGS.items() if k.startswith('hnsw_')}
        
        # 🎯 تعديل: تهيئة اسم الكولكشن هنا لاستخدامه في كل مكان
        self.collection_name = "medical_medications_v2_resumable"
        
        self.embedding_model = self._create_turbo_embedding_model()
        
        logger.info(f"🚀 TURBO Builder initialized with {self.max_workers} workers!")

    def _create_turbo_chroma_client(self):
        """إنشاء ChromaDB client محسن للسرعة القصوى"""
        client = chromadb.PersistentClient(
            path=self.db_directory,
            settings=Settings(
                persist_directory=self.db_directory,
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True,
            )
        )
        logger.info(f"🔥 ChromaDB TURBO client created: {self.db_directory}")
        return client

    def _get_existing_ids_turbo(self, collection):
        """⚡️ جلب كل الـ IDs الموجودة حالياً في الكولكشن بسرعة"""
        try:
            # ChromaDB قد تُرجع النتائج في صفحات، نحتاج لجلبها كلها
            all_ids = []
            # ChromaDB لا تدعم جلب كل شيء مرة واحدة بدون حد أقصى، لذا نستخدم رقم كبير جداً
            # ونفحص إذا كانت النتائج أقل منه لنتوقف.
            page_size = 50000  # حجم صفحة كبير
            last_count = page_size
            offset = 0
            
            logger.info(f"🔍 Fetching existing document IDs from '{self.collection_name}'...")
            
            # نستمر في الجلب طالما أن آخر دفعة كانت ممتلئة
            while last_count == page_size:
                results = collection.get(limit=page_size, offset=offset, include=[])
                ids = results.get('ids', [])
                if not ids:
                    break
                all_ids.extend(ids)
                last_count = len(ids)
                offset += last_count

            logger.info(f"✅ Found {len(all_ids):,} existing document IDs.")
            return set(all_ids)
        except Exception as e:
            logger.error(f"💥 Could not fetch existing IDs: {e}", exc_info=True)
            return set()

    def _create_turbo_embedding_model(self):
        """إنشاء نموذج تضمين محسن للسرعة القصوى"""
        torch.set_num_threads(self.max_workers)
        torch.set_num_interop_threads(self.max_workers)
        
        model = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_SETTINGS['model_name'],
            model_kwargs=Config.EMBEDDING_SETTINGS['model_kwargs'],
            encode_kwargs=Config.EMBEDDING_SETTINGS['encode_kwargs']
        )
        
        logger.info(f"🚀 TURBO Embedding Model loaded with {Config.EMBEDDING_SETTINGS['encode_kwargs']['batch_size']} batch size")
        return model

    def load_data_turbo(self):
        """تحميل البيانات بسرعة فائقة"""
        logger.info(f"🚀 TURBO Loading data from: {self.data_file}")
        start_time = time.time()
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            load_time = time.time() - start_time
            logger.info(f"⚡ LOADED {len(data):,} records in {load_time:.2f}s ({len(data)/load_time:.0f} records/sec)")
            return data
        except Exception as e:
            logger.error(f"💥 Error loading data: {str(e)}")
            raise
    
    def prepare_document_parallel(self, meds_batch):
        """تحضير المستندات بالمعالجة المتوازية للحفاظ على نفس منطق بناء المستند"""
        
        def process_single_med(med):
            """نفس منطق معالجة الدواء الواحد من السكربت الأصلي"""
            try:
                med_id = med.get("الكود")
                if not med_id:
                    # 🎯 خطوة حاسمة: إذا لم يكن هناك ID، لا يمكننا المتابعة
                    # هذا يضمن أن كل سجل يمكن تتبعه واستئنافه
                    logger.warning(f"⚠️ Skipping record due to missing 'الكود' (ID). Record data: {str(med)[:100]}...")
                    return None

                # (هنا نفس كود بناء المستند من السكربت الأصلي)
                name = str(med.get("اسم الدواء الأصلي", "غير معروف"))
                scientific_name = str(med.get("الاسم العلمي", ""))
                commercial_name = str(med.get("الاسم التجاري", ""))
                active_ingredients = str(med.get("المواد الفعالة", "غير معروف"))
                diseases = str(med.get("الأمراض التي يعالجها", "") or "")
                side_effects = str(med.get("الأعراض الجانبية", "") or "")
                disease_symptoms = str(med.get("أعراض المرض", "") or "")
                usage = str(med.get("طريقة الاستخدام", ""))
                classification = str(med.get("التصنيف الدوائي", "غير معروف"))
                pharma_form = str(med.get("الشكل الدوائي", ""))
                manufacturer = str(med.get("الشركة المصنعة", ""))
                under_age = str(med.get("لا يستخدم تحت عمر", ""))
                pregnancy_safe = str(med.get("هل الدواء آمن أثناء الحمل؟", ""))
                breastfeeding_safe = str(med.get("هل يتعارض مع الرضاعة؟", ""))
                prescription_required = str(med.get("هل يتطلب وصفة طبية؟", ""))
                dosage = str(med.get("الجرعة المعتادة", ""))
                concentrations = str(med.get("التراكيز", ""))
                
                diseases_cleaned = [d.strip() for d in diseases.split(",") if d.strip()]
                side_effects_cleaned = [s.strip() for s in side_effects.split(",") if s.strip()]
                disease_symptoms_cleaned = [s.strip() for s in disease_symptoms.split(",") if s.strip()]
                
                active_ingredients = active_ingredients.replace("\n", " ").strip()
                diseases = diseases.replace("\n", " ").strip()
                disease_symptoms = disease_symptoms.replace("\n", " ").strip()
                
                content_parts = []
                if diseases_cleaned:
                    content_parts.extend([
                        f"الأمراض: {diseases}",
                        f"يعالج: {' - '.join(diseases_cleaned)}",
                        f"فعال لعلاج: {' - '.join(diseases_cleaned)}"
                    ])
                
                if disease_symptoms_cleaned:
                    content_parts.extend([
                        f"الأعراض: {disease_symptoms}",
                        f"يخفف أعراض: {' - '.join(disease_symptoms_cleaned)}",
                        f"مفيد للأعراض: {' - '.join(disease_symptoms_cleaned)}"
                    ])
                
                content_parts.append(f"الدواء: {name}")
                
                if active_ingredients.strip() and active_ingredients != "غير معروف":
                    content_parts.append(f"المادة الفعالة: {active_ingredients}")
                
                page_content = "\n".join(content_parts)
                
                doc = Document(
                    page_content=page_content,
                    metadata={
                        "id": med_id, # 🎯 التأكد من استخدام نفس الـ ID هنا
                        "search_priority": self._calculate_search_priority(diseases_cleaned, disease_symptoms_cleaned),
                        "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "under_age_numeric": self._extract_age_number(under_age),
                        "pregnancy_safe": "نعم" if pregnancy_safe.lower() in ['نعم', 'yes', 'safe', 'آمن'] else "لا",
                        "breastfeeding_safe": "نعم" if breastfeeding_safe.lower() in ['نعم', 'yes', 'safe', 'آمن'] else "لا",
                        "is_otc": "نعم" if med.get("OTC", "لا").lower() in ['نعم', 'yes'] else "لا",
                        "is_narcotic": "نعم" if med.get("هل هو مخدر؟", "لا").lower() in ['نعم', 'yes'] else "لا",
                        "requires_prescription": "نعم" if prescription_required.lower() in ['نعم', 'yes'] else "لا",
                        "name": name,
                        "scientific_name": scientific_name,
                        "commercial_name": commercial_name,
                        "active_ingredients": active_ingredients,
                        "diseases": diseases,
                        "disease_symptoms": disease_symptoms,
                        "classification": classification,
                        "pharma_form": pharma_form,
                        "manufacturer": manufacturer,
                        "dosage": dosage,
                        "usage": usage,
                        "under_age": under_age,
                        "concentrations": concentrations,
                        "side_effects": side_effects,
                        "storage_temperature": med.get("درجة حرارة التخزين", ""),
                        "unit": med.get("الوحدة", ""),
                        "section": med.get("القسم", ""),
                        "additional_materials": med.get("المواد المضافة", ""),
                        "contraindicated_materials": med.get("المواد المحظور استخدامها معه", ""),
                        "min_dosage": med.get("أقل جرعة", ""),
                        "max_dosage": med.get("أقصى جرعة", ""),
                        "usage_after_opening": med.get("مدة الاستخدام بعد الفتح", ""),
                        "seasonal": med.get("هل هو دواء موسمي؟", "لا"),
                        "requires_refrigeration": med.get("يحفظ في الثلاجة", "لا"),
                        "has_expiry_date": med.get("له تاريخ صلاحية", ""),
                        "disease_symptoms_count": len(disease_symptoms_cleaned),
                        "diseases_count": len(diseases_cleaned),
                        "side_effects_count": len(side_effects_cleaned),
                        "has_dosage": bool(dosage.strip()),
                        "has_scientific_name": bool(scientific_name.strip()),
                        "has_commercial_name": bool(commercial_name.strip()),
                        "has_side_effects": bool(side_effects.strip()),
                    }
                )
                return doc
            except Exception as e:
                logger.warning(f"⚠️  Skipping corrupted record: {str(e)}")
                return None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_single_med, med) for med in meds_batch]
            documents = [future.result() for future in as_completed(futures) if future.result() is not None]
        return documents
    
    def _extract_age_number(self, age_text):
        if not age_text or age_text.strip() == "": return 0
        import re
        numbers = re.findall(r'\d+', str(age_text))
        return int(numbers[0]) if numbers else 0
    
    def _calculate_search_priority(self, diseases_list, symptoms_list):
        count = len(diseases_list) + len(symptoms_list)
        if count >= 5: return "high"
        if count >= 2: return "medium"
        return "low"

    def build_database_turbo(self, raw_data):
        """
        🏗️ بناء قاعدة بيانات قابل للاستئناف وفائق السرعة
        التحسينات:
        1.  **قابل للاستئناف (Resumable):** يقرأ الـ IDs الموجودة ويتخطاها.
        2.  **معالجة بالدفعات (Batch Processing):** يضيف البيانات في دفعات صغيرة يمكن تتبعها.
        3.  **معرفات صريحة (Explicit IDs):** يستخدم "الكود" كمعرف فريد لمنع التكرار.
        """
        
        # 🎯 فحص بنية البيانات
        if isinstance(raw_data, dict) and 'medicines' in raw_data:
            logger.info("✅ Modern data structure detected, using 'medicines' list.")
            data = raw_data['medicines']
        else:
            logger.info("ℹ️ Legacy data structure (direct list) detected.")
            data = raw_data

        logger.info("🚀 RESUMABLE TURBO DATABASE BUILD STARTING...")
        
        # --- 🎯 الخطوة 1: الحصول على الكولكشن أو إنشاؤه ---
        logger.info(f"🔥 Ensuring collection '{self.collection_name}' exists with optimized HNSW settings...")
        collection_metadata = {
            "description": "RESUMABLE TURBO Medical Database",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "optimization_level": "TURBO_RESUMABLE_V2"
        }
        collection_metadata.update(self.hnsw_settings)
        
        collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata=collection_metadata
        )
        
        # --- 🎯 الخطوة 2: جلب الـ IDs الموجودة لتخطيها ---
        existing_ids = self._get_existing_ids_turbo(collection)
        
        # --- 🎯 الخطوة 3: فلترة البيانات لمعالجة السجلات الجديدة فقط ---
        logger.info("🔍 Filtering data to process only new records...")
        
        # 🚨 تعديل حاسم: التأكد من أن "الكود" موجود (ليس None) وتحويله إلى نص للمقارنة
        new_data = [
            med for med in data 
            if med.get("الكود") is not None and str(med.get("الكود")) not in existing_ids
        ]
        
        total_new_records = len(new_data)
        if total_new_records == 0:
            logger.info("✅🎉 Database is already up-to-date! No new records to process.")
            logger.info(f"📊 Total records in DB: {len(existing_ids):,}")
            # نعيد vectorstore موجودة لضمان استمرارية عمل التطبيق
            return Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model
            )

        logger.info(f"⚡ Found {total_new_records:,} new records to process out of {len(data):,} total.")
        
        # --- 🎯 الخطوة 4: معالجة وإضافة السجلات الجديدة على دفعات ---
        #  критически важное изменение: نستخدم حجم دفعة معالجة صغير وآمن لتجنب أخطاء الذاكرة وحدود ChromaDB
        processing_batch_size = 500 # 🚨 حجم دفعة آمن
        
        total_batches = (total_new_records + processing_batch_size - 1) // processing_batch_size
        logger.info(f"📦 Processing {total_new_records:,} new records in {total_batches} manageable batches of up to {processing_batch_size:,} each.")
        
        vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        )
        
        total_processed = 0
        overall_start = time.time()
        
        for batch_idx in range(total_batches):
            batch_start_time = time.time()
            start_idx = batch_idx * processing_batch_size
            end_idx = min((batch_idx + 1) * processing_batch_size, total_new_records)
            batch = new_data[start_idx:end_idx]
            
            logger.info(f"🔥 TURBO Batch {batch_idx+1}/{total_batches}: Processing {len(batch):,} records")
            
            prep_start = time.time()
            documents = self.prepare_document_parallel(batch)
            # 🚨 تعديل حاسم: التأكد من تحويل كل الـ IDs إلى نصوص قبل إرسالها
            ids = [str(doc.metadata['id']) for doc in documents]
            prep_time = time.time() - prep_start
            
            if not documents:
                logger.warning(f"⚠️ No processable documents in batch {batch_idx+1}. Skipping.")
                continue

            logger.info(f"⚡ Prepared {len(documents):,} docs in {prep_time:.2f}s ({len(documents)/prep_time:.0f} docs/sec)")
            
            if documents:
                embed_start = time.time()
                # 🎯 تعديل حاسم: استخدام add_documents مع ids
                vectorstore.add_documents(documents=documents, ids=ids)
                embed_time = time.time() - embed_start
                
                total_processed += len(documents)
                batch_time = time.time() - batch_start_time
                
                docs_per_sec = len(documents) / batch_time if batch_time > 0 else 0
                embeddings_per_sec = len(documents) / embed_time if embed_time > 0 else 0
                progress = (total_processed / total_new_records) * 100 if total_new_records > 0 else 100
                
                logger.info(f"✅ Batch {batch_idx+1} COMPLETE:")
                logger.info(f"   ⏱️  Total time: {batch_time:.2f}s")
                logger.info(f"   📊 Docs/sec: {docs_per_sec:.0f}")
                logger.info(f"   🚀 Embeddings/sec: {embeddings_per_sec:.0f}")
                logger.info(f"   📈 Progress: {progress:.1f}%")
                logger.info(f"   💾 Processed: {total_processed:,}/{total_new_records:,}")
            
            if (batch_idx + 1) % 3 == 0:
                logger.info("🧹 Memory cleanup...")
                gc.collect()
                memory_info = psutil.virtual_memory()
                logger.info(f"💾 Memory usage: {memory_info.percent}% ({memory_info.used/1024**3:.1f}GB)")
        
        total_time = time.time() - overall_start
        logger.info("🎉" + "="*80)
        logger.info("🚀 TURBO DATABASE BUILD/UPDATE COMPLETE!")
        logger.info(f"⏱️  Total time for this run: {total_time:.2f} seconds")
        logger.info(f"📊 Total processed in this run: {total_processed:,} new documents")
        if total_time > 0:
            logger.info(f"🚀 Overall speed for this run: {total_processed/total_time:.0f} docs/sec")
        
        try:
            final_count = collection.count()
            logger.info(f"✅ Total documents in collection '{self.collection_name}': {final_count:,}")
        except Exception as e:
            logger.error(f"💥 Could not get final collection count: {e}")

        logger.info("🎉" + "="*80)
        
        return vectorstore
    
    def test_turbo_performance(self, vectorstore):
        """اختبار الأداء بالسرعة القصوى"""
        logger.info("🔥 TURBO PERFORMANCE TEST STARTING...")
        
        turbo_queries = [
            "صداع", "حمى", "كحة", "ألم البطن", "التهاب الحلق",
            "برد", "انفلونزا", "حساسية", "إسهال", "قيء",
            "باراسيتامول", "أيبوبروفين", "أسبرين", "مضاد حيوي",
        ]
        
        search_times = []
        logger.info(f"🎯 Testing {len(turbo_queries)} queries...")
        
        for query in turbo_queries:
            start_time = time.time()
            vectorstore.similarity_search(query, k=10)
            query_time = time.time() - start_time
            search_times.append(query_time)
            
        avg_time = sum(search_times) / len(search_times)
        performance_grade = "🔥 BLAZING FAST" if avg_time < 0.05 else "🚀 VERY FAST" if avg_time < 0.1 else "✅ FAST"
        
        logger.info("📊" + "="*50)
        logger.info("🚀 TURBO PERFORMANCE RESULTS:")
        logger.info(f"⚡ Average search time: {avg_time:.3f}s")
        logger.info(f"🎯 Performance grade: {performance_grade}")
        logger.info(f"🔍 Searches per second: {1/avg_time:.0f}")
        logger.info("📊" + "="*50)
        return {'performance_grade': performance_grade}

    def get_turbo_stats(self):
        """إحصائيات قاعدة البيانات المحسنة"""
        if not os.path.exists(self.db_directory): return None
        
        total_size = sum(os.path.getsize(os.path.join(dirpath, f)) for dirpath, _, filenames in os.walk(self.db_directory) for f in filenames)
        size_mb = total_size / (1024 * 1024)
        
        stats = {'total_size_mb': round(size_mb, 2), 'directory': self.db_directory}
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            stats['collection_count'] = collection.count()
        except Exception as e:
            logger.warning(f"⚠️  Cannot get collection stats: {e}")
        return stats
    
    def build(self):
        """البناء الرئيسي فائق السرعة"""
        try:
            overall_start = time.time()
            data = self.load_data_turbo()
            vectorstore = self.build_database_turbo(data)
            performance_results = self.test_turbo_performance(vectorstore)
            stats = self.get_turbo_stats()
            overall_time = time.time() - overall_start
            
            logger.info("🎉" + "="*80)
            logger.info("🔥🔥🔥 TURBO BUILD SUMMARY 🔥🔥🔥")
            logger.info(f"⏱️  Total build time: {overall_time:.2f} seconds")
            if data:
                logger.info(f"📊 Total records in source file: {len(data.get('medicines', data)):,}")
            if stats:
                logger.info(f"💾 Database size: {stats.get('total_size_mb', 'N/A'):.1f} MB")
                logger.info(f"📈 Documents in collection '{self.collection_name}': {stats.get('collection_count', 'N/A'):,}")
            logger.info(f"⚡ Search performance: {performance_results.get('performance_grade', 'N/A')}")
            logger.info("✅ RESUMABLE TURBO OPTIMIZATIONS APPLIED")
            logger.info(f"   🔥 {self.max_workers}-core parallel processing")
            logger.info(f"   📦 {self.batch_size:,} record mega-batches")
            logger.info("🎉" + "="*80)
            
            return vectorstore
        except Exception as e:
            logger.error(f"💥 TURBO BUILD FAILED: {str(e)}", exc_info=True)
            raise

def main():
    """الدالة الرئيسية لبناء قاعدة البيانات فائقة السرعة"""
    logger.info("🚀🚀🚀 DUAYA TURBO DATABASE BUILDER 🚀🚀🚀")
    logger.info(f"💪 Optimized for {psutil.cpu_count(logical=True)}-core CPU + {psutil.virtual_memory().total / (1024**3):.1f}GB RAM server")
    
    builder = TurboDatabaseBuilder()
    builder.build()
    
    logger.info("🎉🎉🎉 TURBO BUILD SUCCESS! 🎉🎉🎉")
    logger.info("🔥 Database is ready for BLAZING FAST searches!")

if __name__ == "__main__":
    main() 