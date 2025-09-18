import os
import torch
from dotenv import load_dotenv

load_dotenv()

# تحسين أداء NumPy و المكتبات العلمية على مستوى النظام
os.environ.update({
    'OMP_NUM_THREADS': '12',
    'OPENBLAS_NUM_THREADS': '12', 
    'MKL_NUM_THREADS': '12',
    'NUMEXPR_MAX_THREADS': '12',
    'NUMEXPR_NUM_THREADS': '12',
    'NUMPY_NUM_THREADS': '12',
    'TOKENIZERS_PARALLELISM': 'true'
})

class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    FALLBACK_OPENAI_MODEL = os.getenv('FALLBACK_OPENAI_MODEL', 'gpt-4o')
    DATA_FILE = 'merged_data.json'  # تم تعديل المسار ليكون merged_data.json
    DB_DIRECTORY = 'chroma_db'
    COLLECTION_NAME = "medical_medications_v2_resumable" # 🚀 إضافة: اسم المجموعة المركزي
    LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    # عند True: المثيل يتطلب نفس الشكل الدوائي حرفياً. عند False: يسمح بأشكال مختلفة (مع التنويه في اللوجز)
    GENERIC_REQUIRE_SAME_FORM = os.getenv('GENERIC_REQUIRE_SAME_FORM', 'False').lower() == 'true'
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    PREFER_OPENAI = os.getenv('PREFER_OPENAI', 'False').lower() == 'true'
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # 🚀 إعدادات فائقة الأداء لـ ChromaDB (مستوحاة من v2)
    CHROMA_SETTINGS = {
        'anonymized_telemetry': False,
        'allow_reset': True,
        'is_persistent': True,
        'persist_directory': './chroma_db',
        # ⚙️ إعدادات HNSW للسرعة والدقة القصوى
        'hnsw_space': 'cosine',
        'hnsw_construction_ef': 800,  # بناء عالي الدقة
        'hnsw_search_ef': 200,        # بحث سريع ودقيق
        'hnsw_M': 48,                 # اتصالات أكثر للدقة
        # 📦 حجم دفعة عملاق لاستغلال الذاكرة
        'batch_size': 25000,
    }
    
    # 🚀 إعدادات نموذج التضمين فائقة الأداء (مستوحاة من v2)
    EMBEDDING_SETTINGS = {
        'model_name': "intfloat/multilingual-e5-large",
        'encode_kwargs': {
            'normalize_embeddings': True,
            'batch_size': 512,            # دفعة تضمين كبيرة جداً
            'convert_to_numpy': True,
            'convert_to_tensor': False,     # تحسين الذاكرة
        },
        'model_kwargs': {
            'device': 'cpu',
        }
    }
    
    # إعدادات البحث والأداء محسنة للسيرفر القوي
    SEARCH_SETTINGS = {
        'default_k': 15,  # عدد النتائج الافتراضي (زيادة للدقة)
        'max_k': 100,  # أقصى عدد نتائج (زيادة كبيرة)
        'similarity_threshold': 0.65,  # عتبة تشابه أكثر مرونة للنتائج الأفضل
        'enable_caching': True,  # تمكين التخزين المؤقت
        'cache_size': 5000,  # حجم تخزين مؤقت كبير (5 أضعاف)
        'cache_ttl': 3600,  # مدة بقاء التخزين المؤقت (ساعتين)
        # إعدادات إضافية للأداء العالي
        'enable_parallel_search': True,  # بحث متوازي
        'max_concurrent_searches': 10,  # بحث متزامن
        'use_memory_mapping': True,  # استخدام memory mapping للسرعة
    }
    
    # إعدادات جديدة للذاكرة والأداء - تحسينات كبيرة للسيرفر القوي
    PERFORMANCE_SETTINGS = {
        # إعدادات الذاكرة - استغلال كامل 46GB RAM
        'max_memory_usage': '40GB',  # استخدام 40GB من 46GB المتاحة (زيادة كبيرة)
        'gc_threshold': (1000, 15, 15),  # تحسين garbage collector للسيرفر القوي
        'enable_memory_profiling': False,  # تعطيل في الإنتاج
        
        # 🚀 إعدادات المعالجة المتوازية - استغلال كامل 12 كور (مستوحاة من v2)
        'max_workers': 11,          # ترك 1 كور للنظام
        'thread_pool_size': 11,
        'process_pool_size': 8,     # زيادة process pool
        
        # تحسينات النصوص الطويلة
        'max_text_length': 10000,  # أقصى طول نص
        'chunk_size': 3000,  # زيادة حجم القطع النصية (من 2000 إلى 3000)
        'overlap_size': 300,  # زيادة التداخل (من 200 إلى 300)
        
        # إعدادات التخزين المؤقت المتقدم
        'enable_redis_cache': True,  # استخدام Redis للتخزين المؤقت
        'redis_expire_time': 7200,  # انتهاء صلاحية Redis (ساعتين)
        'enable_disk_cache': True,  # تخزين مؤقت على القرص
        'disk_cache_size': '10GB',  # زيادة حجم التخزين المؤقت على القرص (من 5GB إلى 10GB)
        
        # إعدادات إضافية للسرعة
        'enable_parallel_processing': True,  # معالجة متوازية
        'use_multiprocessing': True,  # استخدام multiprocessing
        'optimize_for_speed': True,  # تحسين للسرعة
    }
    
    # إعدادات AI والنموذج
    AI_SETTINGS = {
        'temperature': 0.1,  # إبداع منخفض للدقة الطبية
        'max_tokens': 2000,  # أقصى عدد كلمات في الرد
        'top_p': 0.9,  # تنويع محدود
        'frequency_penalty': 0.1,  # تجنب التكرار
        'presence_penalty': 0.1,  # تشجيع المحتوى الجديد
        
        # إعدادات الاستجابة
        'response_timeout': 120,  # مهلة زمنية للاستجابة
        'retry_attempts': 3,  # عدد المحاولات عند الفشل
        'backoff_factor': 2,  # معامل التأخير بين المحاولات
        
        # تحسين استخدام الذاكرة مع النماذج الكبيرة
        'model_cache_size': 3,  # عدد النماذج المحملة في الذاكرة
        'clear_cache_interval': 3600,  # تنظيف التخزين المؤقت كل ساعة
    }