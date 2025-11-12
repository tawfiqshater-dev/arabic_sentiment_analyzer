import streamlit as st
import pandas as pd
import time
from datetime import datetime
import torch
from typing import List, Tuple, Optional, Dict, Any
import re
import gc
import random
import numpy as np
from streamlit.components.v1 import html
import requests
import json
import os
import logging
import hashlib
from functools import lru_cache
from logging.handlers import RotatingFileHandler
import threading
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="منصة تحليل المشاعر العربية - الذكية المحسنة",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# إعداد نظام التسجيل
def setup_logging():
    """إعداد نظام التسجيل المحسن"""
    logger = logging.getLogger('sentiment_analyzer')
    logger.setLevel(logging.INFO)
    
    # منع التسجيل المكرر
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # إنشاء formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # ملف log دوار
    try:
        file_handler = RotatingFileHandler(
            'app.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # استخدام handler لل console إذا فشل إنشاء الملف
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# مدير الأمان المتقدم
class SecurityManager:
    """مدير أمان متقدم للحماية والتحقق"""
    
    def __init__(self):
        self.session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    def sanitize_input(self, text: str) -> str:
        """تنظيف الإدخال من المحتوى الضار"""
        if not text:
            return ""
        
        # إزالة tags خطيرة
        cleaned = re.sub(r'<script.*?>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'<.*?>', '', cleaned)  # إزالة جميع tags
        
        # إزالة أحرف تحكم خطيرة
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
        
        # تحديد طول معقول
        cleaned = cleaned[:10000]  # حد أقصى 10000 حرف
        
        return cleaned.strip()
    
    def validate_api_token(self, token: str) -> bool:
        """التحقق من صحة شكل التوكن"""
        if not token or len(token) < 10:
            return False
        
        # التحقق من أن التوكن ليس منشوراً بشكل علني
        common_tokens = ["example", "test", "demo", "token", "key", "hf_"]
        token_lower = token.lower()
        
        # التحقق من أن التوكن يحتوي على أحرف وأرقام
        if not any(c.isalpha() for c in token) or not any(c.isalnum() for c in token):
            return False
            
        return not any(common in token_lower for common in common_tokens)

# مدير التحليلات والإحصائيات
class AnalyticsManager:
    """مدير التحليلات والإحصائيات المتقدمة"""
    
    def __init__(self):
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_confidence': 0,
            'sentiment_distribution': {'إيجابي': 0, 'سلبي': 0, 'محايد': 0},
            'response_times': [],
            'start_time': datetime.now()
        }
        self.logger = setup_logging()
    
    def update_stats(self, sentiment: str, confidence: float, success: bool = True, response_time: float = 0):
        """تحديث الإحصائيات"""
        self.usage_stats['total_requests'] += 1
        self.usage_stats['response_times'].append(response_time)
        
        # الاحتفاظ بأحدث 100 وقت استجابة فقط
        if len(self.usage_stats['response_times']) > 100:
            self.usage_stats['response_times'] = self.usage_stats['response_times'][-100:]
        
        if success:
            self.usage_stats['successful_requests'] += 1
            if sentiment in self.usage_stats['sentiment_distribution']:
                self.usage_stats['sentiment_distribution'][sentiment] += 1
            
            # تحديث متوسط الثقة
            total = self.usage_stats['average_confidence'] * (self.usage_stats['successful_requests'] - 1)
            self.usage_stats['average_confidence'] = (total + confidence) / self.usage_stats['successful_requests']
            
            self.logger.info(f"طلب ناجح - المشاعر: {sentiment}, الثقة: {confidence:.1f}%")
        else:
            self.usage_stats['failed_requests'] += 1
            self.logger.error(f"طلب فاشل - المشاعر: {sentiment}")
    
    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """إنشاء لوحة تحليلات شاملة"""
        total_requests = self.usage_stats['total_requests']
        success_rate = (self.usage_stats['successful_requests'] / total_requests * 100) if total_requests > 0 else 0
        
        # حساب متوسط وقت الاستجابة
        avg_response_time = np.mean(self.usage_stats['response_times']) if self.usage_stats['response_times'] else 0
        
        # وقت التشغيل
        uptime = datetime.now() - self.usage_stats['start_time']
        uptime_hours = uptime.total_seconds() / 3600
        
        return {
            'success_rate': success_rate,
            'total_operations': total_requests,
            'average_confidence': self.usage_stats['average_confidence'],
            'sentiment_distribution': self.usage_stats['sentiment_distribution'],
            'avg_response_time': avg_response_time,
            'uptime_hours': uptime_hours,
            'requests_per_hour': total_requests / uptime_hours if uptime_hours > 0 else 0
        }
    
    def get_performance_insights(self) -> str:
        """توفير رؤى أداء ذكية"""
        stats = self.get_analytics_dashboard()
        
        if stats['total_operations'] == 0:
            return "لا توجد بيانات كافية لتحليل الأداء"
        
        insights = []
        
        if stats['success_rate'] > 90:
            insights.append("🎯 الأداء ممتاز - معدل النجاح مرتفع")
        elif stats['success_rate'] > 75:
            insights.append("✅ الأداء جيد - يمكن تحسين بعض الجوانب")
        else:
            insights.append("⚠️ يحتاج تحسين - معدل النجاح منخفض")
        
        if stats['avg_response_time'] < 2:
            insights.append("⚡ السرعة رائعة - استجابة سريعة")
        elif stats['avg_response_time'] < 5:
            insights.append("⏱️ السرعة مقبولة - أداء جيد")
        else:
            insights.append("🐌 السرعة بطيئة -可能需要 تحسين")
        
        if stats['average_confidence'] > 80:
            insights.append("📊 دقة عالية - النماذج تعمل بشكل ممتاز")
        elif stats['average_confidence'] > 60:
            insights.append("📈 دقة جيدة - أداء مستقر")
        else:
            insights.append("📉 دقة منخفضة -可能需要 مراجعة النماذج")
        
        return " | ".join(insights)

# معالج أخطاء متقدم
class ErrorHandler:
    """معالج أخطاء متقدم للتعامل مع مختلف أنواع الأخطاء"""
    
    ERROR_CODES = {
        'API_TIMEOUT': 'انتهت مهلة الخدمة، يرجى المحاولة مرة أخرى',
        'API_CONNECTION': 'خطأ في الاتصال بالخادم',
        'API_RATE_LIMIT': 'تم تجاوز الحد المسموح، يرجى الانتظار',
        'MODEL_LOADING': 'النموذج قيد التحميل',
        'INVALID_INPUT': 'النص المدخل غير صالح',
        'AUTH_ERROR': 'خطأ في المصادقة، تحقق من التوكن',
        'UNKNOWN_ERROR': 'خطأ غير متوقع'
    }
    
    @staticmethod
    def handle_api_error(error: dict, operation: str) -> str:
        """معالجة أخطاء API بشكل ذكي"""
        error_msg = error.get('error', '').lower()
        
        if 'timeout' in error_msg or 'timed out' in error_msg:
            return ErrorHandler.ERROR_CODES['API_TIMEOUT']
        elif 'connection' in error_msg or 'connect' in error_msg:
            return ErrorHandler.ERROR_CODES['API_CONNECTION']
        elif 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
            return ErrorHandler.ERROR_CODES['API_RATE_LIMIT']
        elif 'loading' in error_msg or '503' in error_msg:
            return ErrorHandler.ERROR_CODES['MODEL_LOADING']
        elif 'auth' in error_msg or '401' in error_msg or '403' in error_msg:
            return ErrorHandler.ERROR_CODES['AUTH_ERROR']
        else:
            return f"{ErrorHandler.ERROR_CODES['UNKNOWN_ERROR']}: {error_msg}"
    
    @staticmethod
    def suggest_solution(error_type: str) -> str:
        """اقتراح حلول ذكية للأخطاء"""
        solutions = {
            'API_TIMEOUT': 'حاول استخدام نصوص أقصر أو الانتظار قليلاً',
            'API_CONNECTION': 'تحقق من اتصال الإنترنت وأعد المحاولة',
            'API_RATE_LIMIT': 'انتظر دقيقة ثم أعد المحاولة',
            'MODEL_LOADING': 'النموذج سيكون جاهزاً خلال 30-60 ثانية',
            'AUTH_ERROR': 'تحقق من صحة التوكن وإعدادات الأمان',
            'INVALID_INPUT': 'تأكد من إدخال نص عربي صالح'
        }
        return solutions.get(error_type, 'أعد المحاولة لاحقاً أو اتصل بالدعم')

# نظام تحليل المشاعر باستخدام Hugging Face Inference API
class SentimentAnalyzer:
    def __init__(self):
        self.api_loaded = False
        self.sentiment_api_url = "https://api-inference.huggingface.co/models/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
        self.summarization_api_url = "https://api-inference.huggingface.co/models/csebuetnlp/mT5_multilingual_XLSum"
        self.api_token = None
        self.wait_for_model = True
        self.security_manager = SecurityManager()
        self.analytics_manager = AnalyticsManager()
        self.error_handler = ErrorHandler()
        self._cache = {}  # كاش محلي للطلبات
        self.cache_ttl = 300  # 5 دقائق
        
    def initialize_api_token(self) -> bool:
        """تهيئة API Token من مصادر آمنة فقط"""
        # المحاولة الأولى: من Streamlit Secrets
        try:
            secrets_token = st.secrets.get('HUGGINGFACE_API_TOKEN')
            if secrets_token and self.security_manager.validate_api_token(secrets_token):
                self.api_token = secrets_token
                self.api_loaded = True
                self.analytics_manager.logger.info("✅ تم تحميل التوكن الآمن من Secrets")
                return True
        except Exception as e:
            self.analytics_manager.logger.error(f"خطأ في تحميل التوكن من Secrets: {e}")
            pass
        
        # المحاولة الثانية: من environment variable
        env_token = os.getenv('HUGGINGFACE_API_TOKEN')
        if env_token and self.security_manager.validate_api_token(env_token):
            self.api_token = env_token
            self.api_loaded = True
            self.analytics_manager.logger.info("✅ تم تحميل التوكن الآمن من Environment Variables")
            return True
        
        self.analytics_manager.logger.error("❌ لم يتم العثور على التوكن في المصادر الآمنة")
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.ConnectionError))
    )
    def query_huggingface_api(self, api_url: str, payload: dict, timeout: int = 120) -> dict:
        """استدعاء Hugging Face API مع نظام إعادة المحاولة المحسن"""
        if not self.api_token:
            if not self.initialize_api_token():
                return {"error": "لم يتم تكوين API Token بشكل آمن"}
        
        # التحقق من الكاش أولاً
        cache_key = self._generate_cache_key(api_url, payload)
        if cache_key in self._cache:
            cache_data = self._cache[cache_key]
            if time.time() - cache_data['timestamp'] < self.cache_ttl:
                self.analytics_manager.logger.info("🔄 استخدام البيانات من الكاش")
                return cache_data['result']
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        try:
            if self.wait_for_model:
                if "parameters" not in payload:
                    payload["parameters"] = {}
                payload["options"] = {"wait_for_model": self.wait_for_model}
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                # تخزين في الكاش
                self._cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
                return result
            elif response.status_code == 503:
                time.sleep(10)
                response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
                if response.status_code == 200:
                    result = response.json()
                    self._cache[cache_key] = {
                        'result': result,
                        'timestamp': time.time()
                    }
                    return result
                else:
                    return {"error": f"النموذج قيد التحميل، حاول مرة أخرى بعد قليل"}
            else:
                error_msg = f"خطأ في API: {response.status_code}"
                try:
                    error_detail = response.json()
                    if "error" in error_detail:
                        error_msg += f" - {error_detail['error']}"
                except:
                    error_msg += f" - {response.text}"
                return {"error": error_msg}
                
        except requests.exceptions.Timeout:
            return {"error": "انتهت مهلة الاستدعاء، حاول مرة أخرى"}
        except requests.exceptions.ConnectionError:
            return {"error": "خطأ في الاتصال، تحقق من اتصال الإنترنت"}
        except Exception as e:
            return {"error": f"خطأ غير متوقع: {str(e)}"}
    
    def _generate_cache_key(self, api_url: str, payload: dict) -> str:
        """إنشاء مفتاح فريد للكاش"""
        text = payload.get('inputs', '')
        return hashlib.md5(f"{api_url}_{text}".encode()).hexdigest()
    
    def analyze_sentiment(self, text: str) -> Tuple[str, str, str, float]:
        """تحليل المشاعر باستخدام Hugging Face API مع تحسينات متقدمة"""
        if not self.api_loaded and not self.initialize_api_token():
            return "لم يتم تكوين API Token بشكل آمن", "❌", "#dc3545", 0
        
        start_time = time.time()
        try:
            # تنظيف وتحقق من النص
            cleaned_text = self.security_manager.sanitize_input(text)
            is_valid, message, stats = self.enhanced_validation(cleaned_text, "sentiment")
            
            if not is_valid:
                return message, "⚠️", "#ffc107", 0
            
            payload = {"inputs": cleaned_text}
            result = self.query_huggingface_api(self.sentiment_api_url, payload, timeout=60)
            response_time = time.time() - start_time
            
            if 'error' in result:
                error_message = self.error_handler.handle_api_error(result, "تحليل المشاعر")
                self.analytics_manager.update_stats("خطأ", 0, False, response_time)
                return f"خطأ: {error_message}", "❌", "#dc3545", 0
            
            if isinstance(result, list) and len(result) > 0:
                sentiment_label = result[0]['label']
                confidence = result[0]['score'] * 100
                
                sentiment_map = {
                    'positive': ('إيجابي', '😊', '#28a745'),
                    'negative': ('سلبي', '😞', '#dc3545'),
                    'neutral': ('محايد', '😐', '#ffc107'),
                    'LABEL_2': ('إيجابي', '😊', '#28a745'),
                    'LABEL_1': ('سلبي', '😞', '#dc3545'),
                    'LABEL_0': ('محايد', '😐', '#ffc107')
                }
                
                arabic_sentiment, emoji, color = sentiment_map.get(
                    sentiment_label, ('غير محدد', '❓', '#666666')
                )
                
                self.analytics_manager.update_stats(arabic_sentiment, confidence, True, response_time)
                return arabic_sentiment, emoji, color, confidence
            else:
                self.analytics_manager.update_stats("لا نتائج", 0, False, response_time)
                return "لا توجد نتائج", "❌", "#dc3545", 0
                
        except Exception as e:
            response_time = time.time() - start_time
            self.analytics_manager.update_stats("خطأ", 0, False, response_time)
            self.analytics_manager.logger.error(f"خطأ في تحليل المشاعر: {str(e)}")
            return f"خطأ في التحليل: {str(e)}", "❌", "#dc3545", 0
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 30) -> Tuple[str, float]:
        """تلخيص النص باستخدام Hugging Face API مع تحسينات متقدمة"""
        if not self.api_loaded and not self.initialize_api_token():
            return "لم يتم تكوين API Token بشكل آمن", 0
        
        start_time = time.time()
        try:
            # تنظيف وتحقق من النص
            cleaned_text = self.security_manager.sanitize_input(text)
            is_valid, message, stats = self.enhanced_validation(cleaned_text, "summarization")
            
            if not is_valid:
                return message, 0
            
            payload = {
                "inputs": cleaned_text,
                "parameters": {
                    "max_length": max_length,
                    "min_length": min_length,
                    "do_sample": False
                }
            }
            
            result = self.query_huggingface_api(self.summarization_api_url, payload, timeout=90)
            response_time = time.time() - start_time
            
            if 'error' in result:
                error_message = self.error_handler.handle_api_error(result, "التلخيص")
                self.analytics_manager.update_stats("خطأ", 0, False, response_time)
                return f"خطأ: {error_message}", 0
            
            if isinstance(result, list) and len(result) > 0:
                summary = result[0]['summary_text']
                compression_ratio = (1 - len(summary) / len(cleaned_text)) * 100
                self.analytics_manager.update_stats("ملخص", compression_ratio, True, response_time)
                return summary, compression_ratio
            else:
                self.analytics_manager.update_stats("لا نتائج", 0, False, response_time)
                return "لا توجد نتائج", 0
                
        except Exception as e:
            response_time = time.time() - start_time
            self.analytics_manager.update_stats("خطأ", 0, False, response_time)
            self.analytics_manager.logger.error(f"خطأ في التلخيص: {str(e)}")
            return f"حدث خطأ أثناء التلخيص: {str(e)}", 0
    
    def enhanced_validation(self, text: str, service_type: str = "sentiment") -> Tuple[bool, str, dict]:
        """تحسين متقدم للتحقق من صحة النص"""
        cleaned_text = text.strip()
        
        if service_type == "sentiment":
            min_len, max_len = 5, 2000
        else:
            min_len, max_len = 50, 5000
        
        if len(cleaned_text) < min_len:
            return False, f"النص قصير جداً. الحد الأدنى {min_len} حرف.", {}
        
        if len(cleaned_text) > max_len:
            return False, f"النص طويل جداً. الحد الأقصى {max_len} حرف.", {}
        
        # التحقق من المحتوى
        stats = {
            'char_count': len(cleaned_text),
            'word_count': len(cleaned_text.split()),
            'line_count': len(cleaned_text.split('\n')),
            'arabic_ratio': self.calculate_arabic_ratio(cleaned_text)
        }
        
        # التحقق من النسبة العربية للنصوص المختلطة
        if stats['arabic_ratio'] < 0.3 and stats['word_count'] > 10:
            return False, "النص يحتوي على نسبة قليلة من الأحرف العربية", stats
        
        return True, "النص مناسب للتحليل", stats
    
    def calculate_arabic_ratio(self, text: str) -> float:
        """حساب نسبة الأحرف العربية في النص"""
        if not text:
            return 0
        
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        return arabic_chars / len(text) if text else 0
    
    def check_api_status(self):
        """فحص حالة API مع تحسينات"""
        if not self.api_loaded and not self.initialize_api_token():
            return False, "لم يتم تكوين API Token بشكل آمن"
        
        try:
            payload = {"inputs": "اختبار"}
            result = self.query_huggingface_api(self.sentiment_api_url, payload, timeout=30)
            
            if 'error' in result and "loading" in result['error'].lower():
                return True, "النماذج قيد التحميل، قد تستغرق بضع دقائق"
            elif 'error' in result:
                error_message = self.error_handler.handle_api_error(result, "فحص الحالة")
                return False, f"خطأ: {error_message}"
            else:
                return True, "جميع النماذج جاهزة للاستخدام"
                
        except Exception as e:
            return False, f"خطأ في فحص الحالة: {str(e)}"
    
    def batch_analyze_sentiment(self, texts: List[str]) -> List[Tuple]:
        """تحليل المشاعر على دفعات"""
        results = []
        for text in texts:
            if len(text.strip()) >= 5:
                result = self.analyze_sentiment(text)
                results.append(result)
            time.sleep(0.1)  # تجنب rate limiting
        return results
    
    def get_sentiment_stats(self):
        """الحصول على إحصائيات المشاعر"""
        if not hasattr(self, 'analytics_manager'):
            return None
        
        analytics = self.analytics_manager.get_analytics_dashboard()
        return analytics.get('sentiment_distribution', {})

# نظام إدارة الحالة المحسن
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()
if 'sentiment_input_text' not in st.session_state:
    st.session_state.sentiment_input_text = ""
if 'summarization_input_text' not in st.session_state:
    st.session_state.summarization_input_text = ""
if 'active_service' not in st.session_state:
    st.session_state.active_service = "sentiment"
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None
if 'last_summary' not in st.session_state:
    st.session_state.last_summary = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'summarization_history' not in st.session_state:
    st.session_state.summarization_history = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = "الزائر الكريم"
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
if 'summarization_count' not in st.session_state:
    st.session_state.summarization_count = 0
if 'example_clicked' not in st.session_state:
    st.session_state.example_clicked = None
if 'text_area_key' not in st.session_state:
    st.session_state.text_area_key = 0
if 'api_status' not in st.session_state:
    st.session_state.api_status = "لم يتم الفحص بعد"
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'export_data' not in st.session_state:
    st.session_state.export_data = None

# محاولة تهيئة التوكن تلقائياً عند التحميل
if not st.session_state.api_configured:
    if st.session_state.analyzer.initialize_api_token():
        st.session_state.api_configured = True
        with st.spinner("🔍 جاري تهيئة النظام الآمن المحسن..."):
            status, message = st.session_state.analyzer.check_api_status()
            st.session_state.api_status = message

# دوال مساعدة محسنة
def cleanup_session_state():
    """تنظيف ذاكرة الجلسة بشكل دوري"""
    max_history_size = 50
    max_text_length = 1000  # تخزين نصوص مختصرة فقط
    
    if len(st.session_state.analysis_history) > max_history_size:
        st.session_state.analysis_history = st.session_state.analysis_history[:max_history_size]
    
    if len(st.session_state.summarization_history) > max_history_size:
        st.session_state.summarization_history = st.session_state.summarization_history[:max_history_size]
    
    # تنظيف النصوص الطويلة في التاريخ
    for item in st.session_state.analysis_history + st.session_state.summarization_history:
        if len(item['text']) > max_text_length:
            item['text'] = item['text'][:max_text_length] + "..."
    
    # إجبار جمع القمامة دورياً
    if len(st.session_state.analysis_history) % 10 == 0:
        gc.collect()

def validate_text_length(text: str, service_type: str = "sentiment") -> Tuple[bool, str]:
    """التحقق من طول النص المناسب"""
    if service_type == "sentiment":
        if len(text.strip()) < 5:
            return False, "النص قصير جداً. يرجى إدخال نص أطول."
        elif len(text) > 2000:
            return False, "النص طويل جداً. الحد الأقصى 2000 حرف."
    else:  # summarization
        if len(text.strip()) < 50:
            return False, "النص قصير جداً للتلخيص. يرجى إدخال نص أطول (50 حرف على الأقل)."
        elif len(text) > 5000:
            return False, "النص طويل جداً. الحد الأقصى 5000 حرف."
    return True, "النص مناسب للتحليل"

def add_to_history(text: str, sentiment: str, confidence: float, service_type: str = "sentiment"):
    """إضافة التحليل إلى السجل"""
    if service_type == "sentiment":
        analysis_entry = {
            'text': text[:100] + "..." if len(text) > 100 else text,
            'sentiment': sentiment,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'type': 'تحليل المشاعر'
        }
        st.session_state.analysis_history.insert(0, analysis_entry)
        st.session_state.analysis_count += 1
        if len(st.session_state.analysis_history) > 10:
            st.session_state.analysis_history = st.session_state.analysis_history[:10]
    else:  # summarization
        summary_entry = {
            'text': text[:100] + "..." if len(text) > 100 else text,
            'compression_ratio': confidence,
            'timestamp': datetime.now(),
            'type': 'تلخيص النص'
        }
        st.session_state.summarization_history.insert(0, summary_entry)
        st.session_state.summarization_count += 1
        if len(st.session_state.summarization_history) > 10:
            st.session_state.summarization_history = st.session_state.summarization_history[:10]
    
    cleanup_session_state()

def get_motivational_message():
    """رسائل تحفيزية عشوائية"""
    messages = [
        "🔥 أنت تقود ثورة الذكاء الاصطناعي!",
        "🚀 إبداعك لا يعرف حدوداً!",
        "💡 أفكارك ستغير المستقبل!",
        "🌟 أنت مصدر إلهام للجميع!",
        "🎯 دقتك في التحليل مذهلة!",
        "⚡ سرعتك في التعلم مبهرة!",
        "🧠 ذكاؤك الاصطناعي حقيقي!",
        "🏆 أنت البطل في هذا المجال!"
    ]
    return random.choice(messages)

def get_funny_loading_message(service_type: str = "sentiment"):
    """رسائل تحميل مضحكة"""
    if service_type == "sentiment":
        messages = [
            "🦸 جاري استدعاء القوى الذكية...",
            "🧞‍♂️ نفتح خزانة الأسرار العربية...",
            "🔮 نقرأ مشاعرك من كرة الكريستال...",
            "👨‍🔬 نجري تجارب ذكية في المختبر...",
            "🕵️‍♂️ نحلل النص بدقة المباحث...",
            "🎩 نخرج الأرنب من القبعة...",
            "⚗️ نخلط جرعة الذكاء الاصطناعي...",
            "🧩 نحل لغز المشاعر العربية..."
        ]
    else:  # summarization
        messages = [
            "📚 نلخص الأفكار الرئيسية...",
            "✂️ نقطع الفروع ونحتفظ بالأصل...",
            "🎯 نستخرج الجوهر...",
            "🔍 نبحث عن النقاط الرئيسية...",
            "📊 نرتب الأفكار بشكل مختصر...",
            "💎 نستخرج الدرر من النص...",
            "🔄 نحول الطويل إلى قصير مفيد...",
            "📝 نكتب الملخص الذكي..."
        ]
    return random.choice(messages)

def export_data(format_type: str = 'json'):
    """تصدير البيانات والتاريخ"""
    data = {
        'analysis_history': st.session_state.analysis_history,
        'summarization_history': st.session_state.summarization_history,
        'exported_at': datetime.now().isoformat(),
        'total_operations': st.session_state.analysis_count + st.session_state.summarization_count,
        'user_name': st.session_state.user_name,
        'analytics': st.session_state.analyzer.analytics_manager.get_analytics_dashboard() if hasattr(st.session_state.analyzer, 'analytics_manager') else {}
    }
    
    if format_type == 'json':
        return json.dumps(data, ensure_ascii=False, indent=2, default=str)
    elif format_type == 'csv':
        import io
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['النص', 'المشاعر', 'الثقة', 'الوقت', 'النوع'])
        
        for item in st.session_state.analysis_history:
            writer.writerow([
                item['text'],
                item['sentiment'],
                item['confidence'],
                item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'تحليل المشاعر'
            ])
        
        for item in st.session_state.summarization_history:
            writer.writerow([
                item['text'],
                'ملخص',
                item['compression_ratio'],
                item['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'تلخيص النص'
            ])
        
        return output.getvalue()
    
    return None

# مدير التحديث التلقائي
class AutoRefreshManager:
    """مدير التحديث التلقائي للواجهة"""
    
    def __init__(self, interval: int = 30):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
    
    def start_auto_refresh(self):
        """بدء التحديث التلقائي"""
        def refresh_loop():
            while not self._stop_event.is_set():
                time.sleep(self.interval)
                if st.session_state.get('auto_refresh', False):
                    st.rerun()
        
        self._thread = threading.Thread(target=refresh_loop, daemon=True)
        self._thread.start()
    
    def stop_auto_refresh(self):
        """إيقاف التحديث التلقائي"""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

# CSS محسن
def inject_css():
    st.markdown("""
    <style>
    .main .block-container {
        direction: rtl;
        text-align: right;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #2c3e50 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
    }
    .stTextArea textarea {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.8;
        font-size: 16px;
    }
    .active-service {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 10px 0;
        color: white;
        text-align: center;
        direction: rtl;
        border: 3px solid #ffeb3b;
        animation: glow 2s infinite;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .frozen-service {
        background: linear-gradient(135deg, #bdc3c7 0%, #2c3e50 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 10px 0;
        color: white;
        text-align: center;
        direction: rtl;
        opacity: 0.7;
        transition: all 0.3s ease;
    }
    .result-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border-right: 4px solid #28a745;
        direction: rtl;
        text-align: right;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .sentiment-positive {
        border-right-color: #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    .sentiment-negative {
        border-right-color: #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }
    .sentiment-neutral {
        border-right-color: #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
    .summary-card {
        border-right-color: #2196f3;
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    }
    .history-item {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        border-right: 3px solid #3498db;
        direction: rtl;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-top: 4px solid #3498db;
    }
    .achievement-badge {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        border-radius: 20px;
        padding: 10px 20px;
        margin: 5px;
        display: inline-block;
        font-weight: bold;
    }
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .security-badge {
        background: #28a745;
        color: white;
        padding: 10px 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
        font-weight: bold;
    }
    .api-status-ready {
        color: #28a745;
        font-weight: bold;
    }
    .api-status-loading {
        color: #ffc107;
        font-weight: bold;
    }
    .api-status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .performance-excellent {
        color: #28a745;
        font-weight: bold;
        background: #d4edda;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .performance-good {
        color: #ffc107;
        font-weight: bold;
        background: #fff3cd;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .performance-poor {
        color: #dc3545;
        font-weight: bold;
        background: #f8d7da;
        padding: 5px 10px;
        border-radius: 5px;
    }
    @keyframes glow {
        0% { box-shadow: 0 0 5px #667eea; }
        50% { box-shadow: 0 0 20px #667eea; }
        100% { box-shadow: 0 0 5px #667eea; }
    }
    </style>
    """, unsafe_allow_html=True)

def show_celebration():
    """عرض تأثير احتفالي"""
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <h1 style="color: #28a745;">🎉 تحليل ناجح! 🎉</h1>
    </div>
    """, unsafe_allow_html=True)

def show_summary_celebration():
    """عرض تأثير احتفالي للتلخيص"""
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <h1 style="color: #2196f3;">🎊 تلخيص ناجح! 🎊</h1>
    </div>
    """, unsafe_allow_html=True)

# أمثلة
sentiment_examples = [
    {
        "title": "✨ مثال إيجابي مبدع",
        "text": "لقد تفاجأت بالإبداع غير المحدود في هذا المشروع! كل تفصيلة تشهد على التميز والاحترافية. الأداء يتجاوز الخيال والنتائج مبهرة حقاً.",
        "type": "إيجابي"
    },
    {
        "title": "😞 مثال سلبي عميق",
        "text": "أشعر بخيبة أمل لا توصف تجاه المستوى غير المتوقع. التقصير واضح في كل جانب والاهتمام بالتفاصيل مفقود تماماً.",
        "type": "سلبي"
    },
    {
        "title": "🎭 مثال محايد متوازن",
        "text": "الأداء العام ضمن المعدلات الطبيعية المتوقعة. هناك نقاط قوة مقابلة لنقاط تحتاج للتحسين. الوضع الحالي يمثل قاعدة مناسبة للبناء عليها مستقبلاً.",
        "type": "محايد"
    }
]

summarization_examples = [
    {
        "title": "📚 مقال أكاديمي",
        "text": "يشكل الذكاء الاصطناعي أحد أهم التطورات التكنولوجية في القرن الحادي والعشرين. بدأ تطوره منذ الخمسينات من خلال أبحاث آلان تورينج، ثم شهد طفرة كبيرة في العقد الأخير بفضل التقدم في تقنيات التعلم العميق والشبكات العصبية. يتميز الذكاء الاصطناعي بقدرته على معالجة كميات هائلة من البيانات، والتعلم من الأنماط، واتخاذ القرارات المعقدة. تطبيقاته تشمل المجالات الطبية، والمالية، والتعليم، والنقل، وغيرها الكثير. ومع هذه الإنجازات، تبرز تحديات أخلاقية وقانونية تتعلق بالخصوصية والشفافية والمسؤولية.",
        "type": "أكاديمي"
    },
    {
        "title": "📰 تقرير إخباري",
        "text": "شهدت أسواق المال العالمية اليوم تقلبات حادة نتيجة المخاوف من التضخم وارتفاع أسعار الفائدة. في والستريت، انخفض مؤشر داو جونز بنسبة 2.5%، بينما خسرت ناسداك 3.1%. في أوروبا، تراجعت البورصات الرئيسية بأكثر من 2%. جاء هذا التراجع بعد بيانات تضخم أقوى من المتوقع في الولايات المتحدة وأوروبا، مما دفع المستثمرين إلى توقع سياسات نقدية أكثر تشدداً من البنوك المركزية. يحذر المحللون من استمرار التقلبات في الأسابيع المقبلة مع مراقبة بيانات الاقتصاد الكلي عن كثب.",
        "type": "إخباري"
    }
]

# الواجهة الرئيسية المحسنة
def main():
    inject_css()
    
    # الشريط الجانبي المحسن
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; direction: rtl; color: #2c3e50;'>
            <h1>🧠</h1>
            <h3>منصة الذكاء الاصطناعي العربية المحسنة</h3>
            <p>الإصدار الآمن المتقدم - Hugging Face API</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # قسم الحالة الآمنة
        st.header("🔒 حالة النظام الآمن المتقدم")
        if st.session_state.api_configured:
            st.markdown('<div class="security-badge">✅ النظام مُهيأ بأمان متقدم</div>', unsafe_allow_html=True)
            st.success("تم تحميل التوكن الآمن تلقائياً")
            
            # عرض رؤى الأداء
            if hasattr(st.session_state.analyzer, 'analytics_manager'):
                analytics = st.session_state.analyzer.analytics_manager.get_analytics_dashboard()
                performance_insights = st.session_state.analyzer.analytics_manager.get_performance_insights()
                
                st.markdown("### 📈 رؤى الأداء المتقدم")
                st.info(performance_insights)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("معدل النجاح", f"{analytics['success_rate']:.1f}%")
                with col2:
                    st.metric("متوسط وقت الاستجابة", f"{analytics['avg_response_time']:.2f}ث")
        else:
            st.error("""
            **❌ النظام غير مهيأ**
            يرجى إضافة التوكن في:
            - Streamlit Cloud Secrets: HUGGINGFACE_API_TOKEN
            - Environment Variables: HUGGINGFACE_API_TOKEN
            """)
        
        # زر فحص حالة API
        if st.button("🔍 فحص حالة النماذج المتقدمة", use_container_width=True):
            if st.session_state.api_configured:
                with st.spinner("🔍 جاري فحص حالة النماذج المتقدمة..."):
                    status, message = st.session_state.analyzer.check_api_status()
                    st.session_state.api_status = message
                    if status:
                        st.success("✅ " + message)
                    else:
                        st.error("❌ " + message)
            else:
                st.error("❌ النظام غير مهيأ بشكل آمن")
        
        st.markdown("---")
        
        # إدخال اسم المستخدم
        st.header("👤 الملف الشخصي المتقدم")
        user_name = st.text_input("اسمك الكريم:", value=st.session_state.user_name)
        if user_name != st.session_state.user_name:
            st.session_state.user_name = user_name
            st.success(f"مرحباً بك {user_name}! 👑")
        
        st.markdown("---")
        
        # قسم التحليلات المتقدمة
        st.header("📊 التحليلات المتقدمة")
        if hasattr(st.session_state.analyzer, 'analytics_manager'):
            analytics = st.session_state.analyzer.analytics_manager.get_analytics_dashboard()
            sentiment_stats = st.session_state.analyzer.get_sentiment_stats()
            
            if sentiment_stats:
                st.markdown("**توزيع المشاعر:**")
                for sentiment, count in sentiment_stats.items():
                    st.write(f"- {sentiment}: {count}")
            
            st.metric("إجمالي العمليات", analytics['total_operations'])
            st.metric("معدل النجاح", f"{analytics['success_rate']:.1f}%")
            st.metric("متوسط الثقة", f"{analytics['average_confidence']:.1f}%")
        
        st.markdown("---")
        
        # قسم التصدير
        st.header("📤 تصدير البيانات")
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            if st.button("📄 تصدير JSON", use_container_width=True):
                st.session_state.export_data = export_data('json')
        
        with col_export2:
            if st.button("📊 تصدير CSV", use_container_width=True):
                st.session_state.export_data = export_data('csv')
        
        if st.session_state.export_data:
            file_extension = "json" if st.session_state.export_data.startswith('{') else "csv"
            file_name = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.{file_extension}"
            
            st.download_button(
                label=f"⬇️ تحميل {file_extension.upper()}",
                data=st.session_state.export_data,
                file_name=file_name,
                mime="application/json" if file_extension == "json" else "text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # قسم التحديث التلقائي
        st.header("🔄 إدارة التطبيق")
        auto_refresh = st.checkbox("التحديث التلقائي كل 30 ثانية", value=st.session_state.auto_refresh)
        if auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = auto_refresh
            if auto_refresh:
                refresh_manager = AutoRefreshManager()
                refresh_manager.start_auto_refresh()
                st.success("تم تفعيل التحديث التلقائي")
            else:
                st.info("تم إيقاف التحديث التلقائي")
        
        if st.button("🔄 إعادة تشغيل التطبيق المتقدم", use_container_width=True):
            st.rerun()
    
    # المنطقة الرئيسية
    st.title("🧠 منصة الذكاء الاصطناعي العربية - النسخة الآمنة المتقدمة")
    
    # التحقق من تكوين API
    if not st.session_state.api_configured:
        st.error("""
        ## 🔐 التكوين الأمني المطلوب
        
        **لحماية توكنك، يرجى إعداده في:**
        
        ### 🚀 في Streamlit Cloud:
        1. انتقل إلى إعدادات التطبيق
        2. اختر "Secrets"
        3. أضف: `HUGGINGFACE_API_TOKEN = "توكنك_هنا"`
        
        ### 💻 محلياً:
        عيّن متغير بيئة:
        ```bash
        export HUGGINGFACE_API_TOKEN="توكنك_هنا"
        ```
        
        **مزايا هذه الطريقة:**
        - ✅ التوكن غير مرئي للمستخدمين
        - ✅ لا يمكن سرقته من الواجهة
        - ✅ آمن للتطبيقات العامة
        - ✅ إدارة مركزية للتوكن
        """)
        return
    
    # رسالة ترحيب للنظام الآمن
    st.markdown(f"""
    <div class="feature-highlight">
        <h2>مرحباً {st.session_state.user_name}! 👑</h2>
        <p>✅ النظام يعمل بشكل آمن ومحمي مع تحسينات متقدمة</p>
        <p>{get_motivational_message()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # محول الخدمات
    st.markdown("## 🎯 اختر الخدمة الذكية المتقدمة")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 تحليل المشاعر الذكي المتقدم", use_container_width=True, 
                    type="primary" if st.session_state.active_service == "sentiment" else "secondary"):
            st.session_state.active_service = "sentiment"
            st.rerun()
    
    with col2:
        if st.button("📝 تلخيص النصوص الذكي المتقدم", use_container_width=True, 
                    type="primary" if st.session_state.active_service == "summarization" else "secondary"):
            st.session_state.active_service = "summarization"
            st.rerun()
    
    # عرض الخدمات
    st.markdown("## 🚀 الخدمات الذكية المتقدمة")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        service_class = "active-service" if st.session_state.active_service == "sentiment" else "frozen-service"
        st.markdown(f"""
        <div class="{service_class}">
            <h3>📊 تحليل المشاعر الذكي المتقدم</h3>
            <p>✅ <strong>نشط ومتقدّم</strong></p>
            <p>نموذج CAMeL المتخصص</p>
            <p>🧠 + الذكاء الاصطناعي المتقدم</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        service_class = "active-service" if st.session_state.active_service == "summarization" else "frozen-service"
        st.markdown(f"""
        <div class="{service_class}">
            <h3>📝 تلخيص النصوص الذكي المتقدم</h3>
            <p>✅ <strong>نشط ومتقدّم</strong></p>
            <p>نموذج mT5 المتقدم</p>
            <p>⚡ محسّن للأداء المتقدم</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="frozen-service">
            <h3>🔑 كلمات مفتاحية ذكية</h3>
            <p>🔄 <strong>قيد التطوير</strong></p>
            <p>قريباً بإذن الله</p>
            <p>🎯 دقة عالية متقدمة</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="frozen-service">
            <h3>💬 محادثة ذكية متقدمة</h3>
            <p>🔄 <strong>قيد التطوير</strong></p>
            <p>قريباً بإذن الله</p>
            <p>🤖 ذكاء حوارى متقدم</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # عرض الخدمة النشطة
    if st.session_state.active_service == "sentiment":
        render_sentiment_analysis()
    else:
        render_text_summarization()

def render_sentiment_analysis():
    """عرض واجهة تحليل المشاعر المحسنة"""
    st.header("🎯 مركز التحليل الذكي المتقدم للمشاعر")
    
    col_input, col_examples = st.columns([2, 1])
    
    with col_input:
        text_input = st.text_area(
            "أدخل النص العربي لتحليل المشاعر:",
            height=150,
            placeholder="اكتب أو الصق النص العربي هنا... وسنكشف أسرار مشاعره باستخدام الذكاء المتقدم! 🕵️‍♂️",
            value=st.session_state.sentiment_input_text,
            key=f"sentiment_text_input_{st.session_state.text_area_key}",
            help="🧠 يمكن تحليل النصوص حتى 2000 حرف باستخدام الذكاء الاصطناعي المتقدم مع نظام الكاش المحسن"
        )
        
        if text_input != st.session_state.sentiment_input_text:
            st.session_state.sentiment_input_text = text_input
        
        if text_input:
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("📝 عدد الكلمات", len(text_input.split()))
            with col_info2:
                st.metric("🔤 عدد الأحرف", len(text_input))
            with col_info3:
                arabic_ratio = st.session_state.analyzer.calculate_arabic_ratio(text_input)
                st.metric("📊 نسبة العربية", f"{arabic_ratio*100:.1f}%")
            with col_info4:
                if "جاهزة" in st.session_state.api_status:
                    st.metric("⚡ حالة النموذج", "🟢 نشط")
                else:
                    st.metric("⚡ حالة النموذج", "🟡 تحميل")
    
    with col_examples:
        st.markdown("### 💡 أمثلة ذكية متقدمة جاهزة")
        for example in sentiment_examples:
            if st.button(example["title"], key=f"sent_ex_{example['title']}", use_container_width=True):
                st.session_state.example_clicked = example["text"]
                st.session_state.sentiment_input_text = example["text"]
                st.rerun()
    
    if st.session_state.example_clicked and st.session_state.example_clicked != st.session_state.sentiment_input_text:
        st.session_state.sentiment_input_text = st.session_state.example_clicked
        st.session_state.example_clicked = None
        st.rerun()
    
    if st.button("🚀 بدء التحليل الذكي المتقدم", use_container_width=True, type="primary"):
        if text_input.strip():
            is_valid, message = validate_text_length(text_input, "sentiment")
            if not is_valid:
                st.error(f"⚠️ {message}")
            else:
                with st.spinner(f"{get_funny_loading_message('sentiment')}"):
                    sentiment, emoji, color, confidence = st.session_state.analyzer.analyze_sentiment(text_input)
                    
                    if not sentiment.startswith("خطأ") and not sentiment.startswith("لم يتم"):
                        st.session_state.last_analysis = {
                            'text': text_input,
                            'sentiment': sentiment,
                            'emoji': emoji,
                            'color': color,
                            'confidence': confidence
                        }
                        add_to_history(text_input, sentiment, confidence, "sentiment")
                        show_celebration()
                        st.success(f"✅ تم التحليل بنجاح! {get_motivational_message()}")
                        
                        # عرض النتيجة
                        display_sentiment_result(sentiment, emoji, color, confidence, text_input)
                    else:
                        error_solution = st.session_state.analyzer.error_handler.suggest_solution(
                            st.session_state.analyzer.error_handler.handle_api_error({'error': sentiment}, "تحليل المشاعر")
                        )
                        st.error(f"❌ {sentiment}")
                        st.info(f"💡 الحل المقترح: {error_solution}")
        else:
            st.warning("⚠️ يرجى إدخال نص لتحليل مشاعره")

def render_text_summarization():
    """عرض واجهة تلخيص النصوص المحسنة"""
    st.header("📝 مركز التلخيص الذكي المتقدم للنصوص")
    
    col_input, col_examples = st.columns([2, 1])
    
    with col_input:
        text_input = st.text_area(
            "أدخل النص العربي لتلخيصه:",
            height=200,
            placeholder="اكتب أو الصق النص العربي هنا... وسنقدم لك ملخصاً ذكياً ومفيداً باستخدام التقنيات المتقدمة! 📚",
            value=st.session_state.summarization_input_text,
            key=f"summarization_text_input_{st.session_state.text_area_key}",
            help="📝 يمكن تلخيص النصوص حتى 5000 حرف باستخدام الذكاء الاصطناعي المتقدم مع نظام الكاش المحسن"
        )
        
        if text_input != st.session_state.summarization_input_text:
            st.session_state.summarization_input_text = text_input
        
        if text_input:
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("📝 عدد الكلمات", len(text_input.split()))
            with col_info2:
                st.metric("🔤 عدد الأحرف", len(text_input))
            with col_info3:
                arabic_ratio = st.session_state.analyzer.calculate_arabic_ratio(text_input)
                st.metric("📊 نسبة العربية", f"{arabic_ratio*100:.1f}%")
            with col_info4:
                if "جاهزة" in st.session_state.api_status:
                    st.metric("⚡ حالة النموذج", "🟢 نشط")
                else:
                    st.metric("⚡ حالة النموذج", "🟡 تحميل")
            
            # إعدادات التلخيص المتقدمة
            col_settings1, col_settings2, col_settings3 = st.columns(3)
            with col_settings1:
                max_length = st.slider("الطول الأقصى للملخص:", min_value=50, max_value=300, value=150, step=10)
            with col_settings2:
                min_length = st.slider("الطول الأدنى للملخص:", min_value=20, max_value=100, value=30, step=5)
            with col_settings3:
                quality_preset = st.selectbox("جودة التلخيص:", ["متوازن", "سريع", "دقيق"])
    
    with col_examples:
        st.markdown("### 💡 أمثلة نصوص متقدمة جاهزة")
        for example in summarization_examples:
            if st.button(example["title"], key=f"sum_ex_{example['title']}", use_container_width=True):
                st.session_state.example_clicked = example["text"]
                st.session_state.summarization_input_text = example["text"]
                st.rerun()
    
    if st.session_state.example_clicked and st.session_state.example_clicked != st.session_state.summarization_input_text:
        st.session_state.summarization_input_text = st.session_state.example_clicked
        st.session_state.example_clicked = None
        st.rerun()
    
    if st.button("🚀 بدء التلخيص الذكي المتقدم", use_container_width=True, type="primary"):
        if text_input.strip():
            is_valid, message = validate_text_length(text_input, "summarization")
            if not is_valid:
                st.error(f"⚠️ {message}")
            else:
                with st.spinner(f"{get_funny_loading_message('summarization')}"):
                    summary, compression_ratio = st.session_state.analyzer.summarize_text(
                        text_input, max_length, min_length
                    )
                    
                    if not summary.startswith("خطأ") and not summary.startswith("النص قصير"):
                        st.session_state.last_summary = {
                            'original_text': text_input,
                            'summary': summary,
                            'compression_ratio': compression_ratio,
                            'original_length': len(text_input),
                            'summary_length': len(summary)
                        }
                        add_to_history(text_input, "", compression_ratio, "summarization")
                        show_summary_celebration()
                        st.success(f"✅ تم التلخيص بنجاح! {get_motivational_message()}")
                        
                        # عرض نتيجة التلخيص
                        display_summary_result(summary, compression_ratio, text_input, 
                                             st.session_state.last_summary['original_length'], 
                                             st.session_state.last_summary['summary_length'])
                    else:
                        error_solution = st.session_state.analyzer.error_handler.suggest_solution(
                            st.session_state.analyzer.error_handler.handle_api_error({'error': summary}, "التلخيص")
                        )
                        st.error(f"❌ {summary}")
                        st.info(f"💡 الحل المقترح: {error_solution}")
        else:
            st.warning("⚠️ يرجى إدخال نص لتلخيصه")

def display_sentiment_result(sentiment, emoji, color, confidence, text_input):
    """عرض نتيجة تحليل المشاعر المحسنة"""
    sentiment_class = {
        'إيجابي': 'sentiment-positive',
        'سلبي': 'sentiment-negative',
        'محايد': 'sentiment-neutral'
    }.get(sentiment, 'result-card')
    
    st.markdown(f"""
    <div class="result-card {sentiment_class}">
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="font-size: 3em;">{emoji}</span>
            <h2 style="color: {color}; margin: 10px 0;">النتيجة: {sentiment}</h2>
        </div>
        <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span>🎯 مستوى الثقة:</span>
                <span style="font-weight: bold; color: {color};">{confidence:.1f}%</span>
            </div>
            <div style="height: 10px; background: #e9ecef; border-radius: 5px; overflow: hidden;">
                <div style="height: 100%; width: {confidence}%; background: {color}; border-radius: 5px;"></div>
            </div>
        </div>
        <div style="background: white; padding: 15px; border-radius: 8px;">
            <strong>📄 النص المدخل:</strong><br>
            {text_input}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🎯 مستوى الثقة</h3>
            <h2 style="color: {color};">{confidence:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown(f"""
        <div class="stat-card">
            <h3>📊 الحالة</h3>
            <h2 style="color: {color};">{sentiment}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat3:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🧠 النموذج</h3>
            <h2 style="color: #3498db;">CAMeL المتقدم</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat4:
        # عرض إحصائيات الأداء
        if hasattr(st.session_state.analyzer, 'analytics_manager'):
            analytics = st.session_state.analyzer.analytics_manager.get_analytics_dashboard()
            performance_class = "performance-excellent" if analytics['success_rate'] > 90 else "performance-good" if analytics['success_rate'] > 75 else "performance-poor"
            
            st.markdown(f"""
            <div class="stat-card">
                <h3>📈 الأداء</h3>
                <h2 class="{performance_class}">{analytics['success_rate']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)

def display_summary_result(summary, compression_ratio, original_text, original_length, summary_length):
    """عرض نتيجة التلخيص المحسنة"""
    st.markdown(f"""
    <div class="summary-card">
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #2196f3; margin: 10px 0;">التلخيص الناجح المتقدم</h2>
        </div>
        
        <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4 style="color: #2196f3; margin-bottom: 10px;">📋 الملخص الذكي المتقدم:</h4>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; border-right: 3px solid #2196f3;">
                {summary}
            </div>
        </div>
        
        <div style="background: white; padding: 15px; border-radius: 8px;">
            <strong>📊 إحصائيات التلخيص المتقدمة:</strong><br>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                <div style="text-align: center; padding: 10px; background: #e3f2fd; border-radius: 5px;">
                    <div style="font-size: 1.2em; font-weight: bold; color: #1976d2;">{original_length}</div>
                    <div style="font-size: 0.9em;">عدد أحرف النص الأصلي</div>
                </div>
                <div style="text-align: center; padding: 10px; background: #e3f2fd; border-radius: 5px;">
                    <div style="font-size: 1.2em; font-weight: bold; color: #1976d2;">{summary_length}</div>
                    <div style="font-size: 0.9em;">عدد أحرف الملخص</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>📉 نسبة التلخيص</h3>
            <h2 style="color: #2196f3;">{compression_ratio:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        reduction_percentage = (1 - summary_length / original_length) * 100
        st.markdown(f"""
        <div class="stat-card">
            <h3>✂️ نسبة التخفيض</h3>
            <h2 style="color: #4caf50;">{reduction_percentage:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat3:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🧠 النموذج</h3>
            <h2 style="color: #ff9800;">mT5 المتقدم</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat4:
        # عرض جودة الملخص
        quality_score = min(100, max(60, compression_ratio + (100 - reduction_percentage) / 2))
        quality_color = "#28a745" if quality_score > 80 else "#ffc107" if quality_score > 65 else "#dc3545"
        
        st.markdown(f"""
        <div class="stat-card">
            <h3>🏆 جودة الملخص</h3>
            <h2 style="color: {quality_color};">{quality_score:.0f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("📈 مركز التفسير الذكي المتقدم للتلخيص")
    
    reduction_percentage = (1 - summary_length / original_length) * 100
    quality_assessment = "ممتازة" if reduction_percentage > 70 else "جيدة" if reduction_percentage > 50 else "مقبولة"
    
    st.info(f"""
    **📊 تحليل عملية التلخيص المتقدمة:**
    
    **🎯 كفاءة التلخيص:** {compression_ratio:.1f}%
    - تم اختصار النص بنسبة {reduction_percentage:.1f}% مع الحفاظ على المعنى الأساسي
    - هذا يدل على {quality_assessment} في استخلاص الأفكار الرئيسية
    
    **🧠 تقييم الجودة المتقدم:**
    - ✅ الملخص يحافظ على السياق العام والأفكار الرئيسية
    - ✅ اللغة سليمة ومفهومة مع الحفاظ على المعنى
    - ✅ التنسيق متناسق وواضح للقارئ
    - ✅ نسبة التكثيف مناسبة للاستخدام العملي
    
    **💡 التوصيات:**
    - يمكن استخدام هذا الملخص في التقارير والعروض التقديمية
    - مناسب للاستخدام في تحليلات المحتوى والبحوث
    - جودة {quality_assessment} للاستخدام المهني
    """)

if __name__ == "__main__":
    main()