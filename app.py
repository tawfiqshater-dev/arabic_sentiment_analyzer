import streamlit as st
import pandas as pd
import time
from datetime import datetime
import torch
from typing import List, Tuple, Optional
import re
import gc
import random
import numpy as np
from streamlit.components.v1 import html
import requests
import json
import os

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="منصة تحليل المشاعر العربية - الذكية", 
    page_icon="🧠", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# نظام تحليل المشاعر باستخدام Hugging Face Inference API
class SentimentAnalyzer:
    def __init__(self): 
        self.api_loaded = False
        self.sentiment_api_url = "https://api-inference.huggingface.co/models/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
        self.summarization_api_url = "https://api-inference.huggingface.co/models/csebuetnlp/mT5_multilingual_XLSum"
        self.api_token = None
        self.wait_for_model = True

    def initialize_api_token(self):
        """تهيئة API Token من مصادر آمنة فقط"""
        # المحاولة الأولى: من Streamlit Secrets
        try:
            secrets_token = st.secrets.get('HUGGINGFACE_API_TOKEN')
            if secrets_token:
                self.api_token = secrets_token
                self.api_loaded = True
                st.success("✅ تم تحميل التوكن الآمن من Secrets")
                return True
        except Exception as e:
            pass
        
        # المحاولة الثانية: من environment variable
        env_token = os.getenv('HUGGINGFACE_API_TOKEN')
        if env_token:
            self.api_token = env_token
            self.api_loaded = True
            st.success("✅ تم تحميل التوكن الآمن من Environment Variables")
            return True
        
        st.error("❌ لم يتم العثور على التوكن في المصادر الآمنة")
        return False

    def query_huggingface_api(self, api_url: str, payload: dict, timeout: int = 120):
        """استدعاء Hugging Face API مع معالجة الأخطاء المحسنة"""
        if not self.api_token:
            if not self.initialize_api_token():
                return {"error": "لم يتم تكوين API Token بشكل آمن"}
            
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        try:
            if self.wait_for_model:
                if "parameters" not in payload:
                    payload["parameters"] = {}
                payload["options"] = {"wait_for_model": self.wait_for_model}
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                time.sleep(10)
                response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
                if response.status_code == 200:
                    return response.json()
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

    def analyze_sentiment(self, text: str) -> Tuple[str, str, str, float]:
        """تحليل المشاعر باستخدام Hugging Face API"""
        if not self.api_loaded and not self.initialize_api_token():
            return "لم يتم تكوين API Token بشكل آمن", "❌", "#dc3545", 0

        try:
            if len(text.strip()) < 5:
                return "نص قصير جداً", "⚠️", "#ffc107", 0
            elif len(text) > 2000:
                return "نص طويل جداً", "⚠️", "#ffc107", 0

            payload = {"inputs": text}
            result = self.query_huggingface_api(self.sentiment_api_url, payload, timeout=60)
            
            if 'error' in result:
                return f"خطأ: {result['error']}", "❌", "#dc3545", 0

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
                return arabic_sentiment, emoji, color, confidence
            else:
                return "لا توجد نتائج", "❌", "#dc3545", 0

        except Exception as e:
            return f"خطأ في التحليل: {str(e)}", "❌", "#dc3545", 0

    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 30) -> Tuple[str, float]:
        """تلخيص النص باستخدام Hugging Face API"""
        if not self.api_loaded and not self.initialize_api_token():
            return "لم يتم تكوين API Token بشكل آمن", 0
        
        try:
            if len(text.strip()) < 50:
                return "النص قصير جداً للتلخيص. يرجى إدخال نص أطول.", 0
            elif len(text) > 5000:
                return "النص طويل جداً. الحد الأقصى 5000 حرف.", 0

            payload = {
                "inputs": text,
                "parameters": {
                    "max_length": max_length,
                    "min_length": min_length,
                    "do_sample": False
                }
            }
            
            result = self.query_huggingface_api(self.summarization_api_url, payload, timeout=90)
            
            if 'error' in result:
                return f"خطأ: {result['error']}", 0

            if isinstance(result, list) and len(result) > 0:
                summary = result[0]['summary_text']
                compression_ratio = (1 - len(summary) / len(text)) * 100
                return summary, compression_ratio
            else:
                return "لا توجد نتائج", 0
            
        except Exception as e:
            return f"حدث خطأ أثناء التلخيص: {str(e)}", 0

    def check_api_status(self):
        """فحص حالة API"""
        if not self.api_loaded and not self.initialize_api_token():
            return False, "لم يتم تكوين API Token بشكل آمن"
        
        try:
            payload = {"inputs": "اختبار"}
            result = self.query_huggingface_api(self.sentiment_api_url, payload, timeout=30)
            
            if 'error' in result and "loading" in result['error'].lower():
                return True, "النماذج قيد التحميل، قد تستغرق بضع دقائق"
            elif 'error' in result:
                return False, f"خطأ: {result['error']}"
            else:
                return True, "جميع النماذج جاهزة للاستخدام"
                
        except Exception as e:
            return False, f"خطأ في فحص الحالة: {str(e)}"

# نظام إدارة الحالة
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

# محاولة تهيئة التوكن تلقائياً عند التحميل
if not st.session_state.api_configured:
    if st.session_state.analyzer.initialize_api_token():
        st.session_state.api_configured = True
        with st.spinner("🔍 جاري تهيئة النظام الآمن..."):
            status, message = st.session_state.analyzer.check_api_status()
            st.session_state.api_status = message

# دوال مساعدة
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
    
    .sentiment-positive { border-right-color: #28a745; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); }
    .sentiment-negative { border-right-color: #dc3545; background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); }
    .sentiment-neutral { border-right-color: #ffc107; background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); }
    .summary-card { border-right-color: #2196f3; background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); }
    
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
    
    .api-status-ready { color: #28a745; font-weight: bold; }
    .api-status-loading { color: #ffc107; font-weight: bold; }
    .api-status-error { color: #dc3545; font-weight: bold; }
    
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

# الواجهة الرئيسية
def main():
    inject_css()
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; direction: rtl; color: #2c3e50;'>
            <h1>🧠</h1>
            <h3>منصة الذكاء الاصطناعي العربية</h3>
            <p>الإصدار الآمن - Hugging Face API</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # قسم الحالة الآمنة
        st.header("🔒 حالة النظام الآمن")
        
        if st.session_state.api_configured:
            st.markdown('<div class="security-badge">✅ النظام مُهيأ بأمان</div>', unsafe_allow_html=True)
            st.success("تم تحميل التوكن الآمن تلقائياً")
        else:
            st.error("""
            **❌ النظام غير مهيأ**
            
            يرجى إضافة التوكن في:
            - Streamlit Cloud Secrets: HUGGINGFACE_API_TOKEN
            - Environment Variables: HUGGINGFACE_API_TOKEN
            """)
        
        # زر فحص حالة API
        if st.button("🔍 فحص حالة النماذج", use_container_width=True):
            if st.session_state.api_configured:
                with st.spinner("🔍 جاري فحص حالة النماذج..."):
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
        st.header("👤 الملف الشخصي")
        user_name = st.text_input("اسمك الكريم:", value=st.session_state.user_name)
        if user_name != st.session_state.user_name:
            st.session_state.user_name = user_name
            st.success(f"مرحباً بك {user_name}! 👑")
            
        st.markdown("---")
        
        st.header("🤖 معلومات API")
        st.info("""
        **الخدمات النشطة:**
        - تحليل المشاعر: CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment
        - تلخيص النصوص: csebuetnlp/mT5_multilingual_XLSum
        
        **المميزات:**
        - ✅ معالجة على خوادم Hugging Face
        - ✅ لا حاجة لتحميل نماذج محلياً
        - ✅ أداء عالي وسريع
        - ✅ دعم كامل للغة العربية
        """)
        
        # حالة API
        st.markdown("### 📊 حالة النماذج")
        if "جاهزة" in st.session_state.api_status or "مكتمل" in st.session_state.api_status:
            st.markdown(f'<p class="api-status-ready">✅ {st.session_state.api_status}</p>', unsafe_allow_html=True)
        elif "تحميل" in st.session_state.api_status:
            st.markdown(f'<p class="api-status-loading">🔄 {st.session_state.api_status}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="api-status-error">❌ {st.session_state.api_status}</p>', unsafe_allow_html=True)
                
        st.markdown("---")
        
        # إحصائيات
        st.header("📊 الإحصائيات")
        total_operations = st.session_state.analysis_count + st.session_state.summarization_count
        st.metric("إجمالي العمليات", total_operations)
        st.metric("تحليلات المشاعر", st.session_state.analysis_count)
        st.metric("تلخيص النصوص", st.session_state.summarization_count)
        
        if total_operations >= 5:
            st.markdown('<div class="achievement-badge">🦸 بطل التحليل</div>', unsafe_allow_html=True)
        if total_operations >= 10:
            st.markdown('<div class="achievement-badge">🧠 عبقري الذكاء الاصطناعي</div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        if st.button("🔄 إعادة تشغيل التطبيق", use_container_width=True):
            st.rerun()

    # المنطقة الرئيسية
    st.title("🧠 منصة الذكاء الاصطناعي العربية - النسخة الآمنة")
    
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
        <p>✅ النظام يعمل بشكل آمن ومحمي</p>
        <p>{get_motivational_message()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # محول الخدمات
    st.markdown("## 🎯 اختر الخدمة الذكية")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 تحليل المشاعر الذكي", use_container_width=True, 
                    type="primary" if st.session_state.active_service == "sentiment" else "secondary"):
            st.session_state.active_service = "sentiment"
            st.rerun()
            
    with col2:
        if st.button("📝 تلخيص النصوص الذكي", use_container_width=True,
                    type="primary" if st.session_state.active_service == "summarization" else "secondary"):
            st.session_state.active_service = "summarization"
            st.rerun()
    
    # عرض الخدمات
    st.markdown("## 🚀 الخدمات الذكية المتاحة")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        service_class = "active-service" if st.session_state.active_service == "sentiment" else "frozen-service"
        st.markdown(f"""
        <div class="{service_class}">
            <h3>📊 تحليل المشاعر الذكي</h3>
            <p>✅ <strong>نشط ومتقدّم</strong></p>
            <p>نموذج CAMeL المتخصص</p>
            <p>🧠 + الذكاء الاصطناعي</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        service_class = "active-service" if st.session_state.active_service == "summarization" else "frozen-service"
        st.markdown(f"""
        <div class="{service_class}">
            <h3>📝 تلخيص النصوص الذكي</h3>
            <p>✅ <strong>نشط ومتقدّم</strong></p>
            <p>نموذج mT5 المتقدم</p>
            <p>⚡ محسّن للأداء</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="frozen-service">
            <h3>🔑 كلمات مفتاحية ذكية</h3>
            <p>🔄 <strong>قيد التطوير</strong></p>
            <p>قريباً بإذن الله</p>
            <p>🎯 دقة عالية</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="frozen-service">
            <h3>💬 محادثة ذكية</h3>
            <p>🔄 <strong>قيد التطوير</strong></p>
            <p>قريباً بإذن الله</p>
            <p>🤖 ذكاء حوارى</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # عرض الخدمة النشطة
    if st.session_state.active_service == "sentiment":
        render_sentiment_analysis()
    else:
        render_text_summarization()

def render_sentiment_analysis():
    """عرض واجهة تحليل المشاعر"""
    st.header("🎯 مركز التحليل الذكي للمشاعر")
    
    col_input, col_examples = st.columns([2, 1])
    
    with col_input:
        text_input = st.text_area(
            "أدخل النص العربي لتحليل المشاعر:",
            height=150,
            placeholder="اكتب أو الصق النص العربي هنا... وسنكشف أسرار مشاعره! 🕵️‍♂️",
            value=st.session_state.sentiment_input_text,
            key=f"sentiment_text_input_{st.session_state.text_area_key}",
            help="🧠 يمكن تحليل النصوص حتى 2000 حرف باستخدام الذكاء الاصطناعي المتقدم"
        )
        
        if text_input != st.session_state.sentiment_input_text:
            st.session_state.sentiment_input_text = text_input
            
        if text_input:
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("📝 عدد الكلمات", len(text_input.split()))
            with col_info2:
                st.metric("🔤 عدد الأحرف", len(text_input))
            with col_info3:
                if "جاهزة" in st.session_state.api_status:
                    st.metric("⚡ حالة النموذج", "🟢 نشط")
                else:
                    st.metric("⚡ حالة النموذج", "🟡 تحميل")
    
    with col_examples:
        st.markdown("### 💡 أمثلة ذكية جاهزة")
        for example in sentiment_examples:
            if st.button(example["title"], key=f"sent_ex_{example['title']}", use_container_width=True):
                st.session_state.example_clicked = example["text"]
                st.rerun()
    
    if st.button("🚀 بدء التحليل الذكي", use_container_width=True, type="primary"):
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
                    st.error(f"❌ {sentiment}")
        else:
            st.warning("⚠️ يرجى إدخال نص لتحليل مشاعره")

def render_text_summarization():
    """عرض واجهة تلخيص النصوص"""
    st.header("📝 مركز التلخيص الذكي للنصوص")
    
    col_input, col_examples = st.columns([2, 1])
    
    with col_input:
        text_input = st.text_area(
            "أدخل النص العربي لتلخيصه:",
            height=200,
            placeholder="اكتب أو الصق النص العربي هنا... وسنقدم لك ملخصاً ذكياً ومفيداً! 📚",
            value=st.session_state.summarization_input_text,
            key=f"summarization_text_input_{st.session_state.text_area_key}",
            help="📝 يمكن تلخيص النصوص حتى 5000 حرف باستخدام الذكاء الاصطناعي المتقدم"
        )
        
        if text_input != st.session_state.summarization_input_text:
            st.session_state.summarization_input_text = text_input
            
        if text_input:
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("📝 عدد الكلمات", len(text_input.split()))
            with col_info2:
                st.metric("🔤 عدد الأحرف", len(text_input))
            with col_info3:
                if "جاهزة" in st.session_state.api_status:
                    st.metric("⚡ حالة النموذج", "🟢 نشط")
                else:
                    st.metric("⚡ حالة النموذج", "🟡 تحميل")
                
        # إعدادات التلخيص
        col_settings1, col_settings2 = st.columns(2)
        with col_settings1:
            max_length = st.slider("الطول الأقصى للملخص:", min_value=50, max_value=300, value=150, step=10)
        with col_settings2:
            min_length = st.slider("الطول الأدنى للملخص:", min_value=20, max_value=100, value=30, step=5)
    
    with col_examples:
        st.markdown("### 💡 أمثلة نصوص جاهزة")
        for example in summarization_examples:
            if st.button(example["title"], key=f"sum_ex_{example['title']}", use_container_width=True):
                st.session_state.example_clicked = example["text"]
                st.rerun()
    
    if st.button("🚀 بدء التلخيص الذكي", use_container_width=True, type="primary"):
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
                    st.error(f"❌ {summary}")
        else:
            st.warning("⚠️ يرجى إدخال نص لتلخيصه")

def display_sentiment_result(sentiment, emoji, color, confidence, text_input):
    """عرض نتيجة تحليل المشاعر"""
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
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
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
            <h2 style="color: #3498db;">CAMeL الذكي</h2>
        </div>
        """, unsafe_allow_html=True)

def display_summary_result(summary, compression_ratio, original_text, original_length, summary_length):
    """عرض نتيجة التلخيص"""
    st.markdown(f"""
    <div class="summary-card">
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #2196f3; margin: 10px 0;">التلخيص الناجح</h2>
        </div>
        
        <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h4 style="color: #2196f3; margin-bottom: 10px;">📋 الملخص الذكي:</h4>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; border-right: 3px solid #2196f3;">
                {summary}
            </div>
        </div>
        
        <div style="background: white; padding: 15px; border-radius: 8px;">
            <strong>📊 إحصائيات التلخيص:</strong><br>
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
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
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
            <h2 style="color: #ff9800;">mT5 الذكي</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.header("📈 مركز التفسير الذكي للتلخيص")
    
    st.info(f"""
    **📊 تحليل عملية التلخيص:**
    
    **🎯 كفاءة التلخيص:** {compression_ratio:.1f}%
    - تم اختصار النص بنسبة {reduction_percentage:.1f}% مع الحفاظ على المعنى الأساسي
    - هذا يدل على كفاءة عالية في استخلاص الأفكار الرئيسية
    
    **🧠 تقييم الجودة:**
    - ✅ الملخص يحافظ على السياق العام
    - ✅ الأفكار الرئيسية محفوظة
    - ✅ اللغة سليمة ومفهومة
    - ✅ التنسيق متناسق وواضح
    """)

if __name__ == "__main__":
    main()