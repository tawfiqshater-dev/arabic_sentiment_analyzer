import streamlit as st
import pandas as pd
import time
from datetime import datetime
from transformers import pipeline
import torch
from typing import List, Tuple, Optional
import re
import gc
import random
import numpy as np
from streamlit.components.v1 import html

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="منصة تحليل المشاعر العربية - الذكية",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# نظام تحليل المشاعر باستخدام CAMeL
class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_name = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
    
    def load_model(self):
        """تحميل نموذج CAMeL لتحليل المشاعر"""
        if self.model_loaded:
            return True
        try:
            with st.spinner("🔄 جاري تحميل النموذج الذكي... ⚡"):
                self.model = pipeline(
                    "text-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    max_length=512,
                    truncation=True
                )
                self.model_loaded = True
                return True
        except Exception as e:
            st.error(f"❌ فشل في تحميل النموذج: {str(e)}")
            return False
    
    def analyze_sentiment(self, text: str) -> Tuple[str, str, str, float]:
        """تحليل المشاعر باستخدام نموذج CAMeL"""
        if not self.model_loaded:
            if not self.load_model():
                return "خطأ في التحميل", "❌", "#dc3545", 0
        
        try:
            # التحقق من طول النص
            if len(text.strip()) < 5:
                return "نص قصير جداً", "⚠️", "#ffc107", 0
            elif len(text) > 2000:
                return "نص طويل جداً", "⚠️", "#ffc107", 0
            
            # تحليل المشاعر
            result = self.model(text)
            sentiment_label = result[0]['label']
            confidence = result[0]['score'] * 100
            
            # ترميز النتائج للعربية
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
            
        except Exception as e:
            st.error(f"❌ خطأ في تحليل المشاعر: {str(e)}")
            return "خطأ في التحليل", "❌", "#dc3545", 0

# نظام إدارة الحالة
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()
if 'sentiment_input_text' not in st.session_state:
    st.session_state.sentiment_input_text = ""
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'show_exit_modal' not in st.session_state:
    st.session_state.show_exit_modal = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = "الزائر الكريم"
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
if 'example_clicked' not in st.session_state:
    st.session_state.example_clicked = None
if 'text_area_key' not in st.session_state:
    st.session_state.text_area_key = 0

# دوال مساعدة
def validate_text_length(text: str) -> Tuple[bool, str]:
    """التحقق من طول النص المناسب"""
    if len(text.strip()) < 5:
        return False, "النص قصير جداً. يرجى إدخال نص أطول."
    elif len(text) > 2000:
        return False, "النص طويل جداً. الحد الأقصى 2000 حرف."
    return True, "النص مناسب للتحليل"

def add_to_history(text: str, sentiment: str, confidence: float):
    """إضافة التحليل إلى السجل"""
    analysis_entry = {
        'text': text[:100] + "..." if len(text) > 100 else text,
        'sentiment': sentiment,
        'confidence': confidence,
        'timestamp': datetime.now()
    }
    st.session_state.analysis_history.insert(0, analysis_entry)
    st.session_state.analysis_count += 1
    # الحفاظ على آخر 10 تحليلات فقط
    if len(st.session_state.analysis_history) > 10:
        st.session_state.analysis_history = st.session_state.analysis_history[:10]

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

def get_funny_loading_message():
    """رسائل تحميل مضحكة"""
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
    return random.choice(messages)

# CSS محسن مع إصلاحات للون النص في الشريط الجانبي
def inject_css():
    st.markdown("""
    <style>
    .main .block-container {
        direction: rtl;
        text-align: right;
    }
    
    /* إصلاح ألوان النص في الشريط الجانبي */
    .css-1d391kg, .css-1lcbmhc, .css-1outwn7 {
        color: #2c3e50 !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #2c3e50 !important;
    }
    
    .sidebar .sidebar-content * {
        color: #2c3e50 !important;
    }
    
    .sidebar .sidebar-content .stMarkdown, 
    .sidebar .sidebar-content .stTextInput,
    .sidebar .sidebar-content .stButton button,
    .sidebar .sidebar-content .stInfo,
    .sidebar .sidebar-content .stSuccess,
    .sidebar .sidebar-content .stWarning {
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
    
    /* تأثيرات الأنيميشن */
    @keyframes glow {
        0% { box-shadow: 0 0 5px #667eea; }
        50% { box-shadow: 0 0 20px #667eea, 0 0 30px #764ba2; }
        100% { box-shadow: 0 0 5px #667eea; }
    }
    
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes celebrate {
        0% { transform: scale(1) rotate(0deg); }
        25% { transform: scale(1.2) rotate(5deg); }
        50% { transform: scale(1.3) rotate(0deg); }
        75% { transform: scale(1.2) rotate(-5deg); }
        100% { transform: scale(1) rotate(0deg); }
    }
    
    @keyframes typewriter {
        from { width: 0; }
        to { width: 100%; }
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
        animation: glow 2s infinite, float 3s ease-in-out infinite;
        transition: all 0.3s ease;
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
        animation: pulse 2s ease-in-out;
    }
    
    .sentiment-positive {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-right: 4px solid #28a745;
        animation: pulse 2s ease-in-out;
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-right: 4px solid #dc3545;
        animation: pulse 2s ease-in-out;
    }
    
    .sentiment-neutral {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-right: 4px solid #ffc107;
        animation: pulse 2s ease-in-out;
    }
    
    .history-item {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        border-right: 3px solid #3498db;
        direction: rtl;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    
    .history-item:hover {
        transform: translateX(-5px);
    }
    
    .confidence-bar {
        height: 10px;
        background: #e9ecef;
        border-radius: 5px;
        margin: 5px 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    
    .example-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-right: 3px solid #3498db;
        direction: rtl;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .example-card:hover {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        transform: translateY(-2px);
        animation: pulse 0.5s ease-in-out;
    }
    
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-top: 4px solid #3498db;
        animation: float 3s ease-in-out infinite;
    }
    
    .achievement-badge {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        border-radius: 20px;
        padding: 10px 20px;
        margin: 5px;
        display: inline-block;
        animation: glow 1.5s infinite;
        font-weight: bold;
    }
    
    .exit-modal {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        z-index: 1000;
        text-align: center;
        direction: rtl;
        animation: pulse 0.5s ease-in-out;
    }
    
    .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.5);
        z-index: 999;
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        animation: glow 2s infinite;
    }
    
    .celebration-effect {
        animation: celebrate 1s ease-in-out;
        display: inline-block;
    }
    
    .typewriter {
        overflow: hidden;
        border-right: .15em solid orange;
        white-space: nowrap;
        margin: 0 auto;
        letter-spacing: .15em;
        animation: typewriter 3.5s steps(40, end);
    }
    
    .success-glow {
        animation: glow 1s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)

def show_exit_modal():
    """عرض نافذة الخروج"""
    st.markdown("""
    <div class="modal-overlay"></div>
    <div class="exit-modal">
        <h2>🎯 شكراً لك على استخدام التطبيق!</h2>
        <p>لقد قمت بتحليل <strong>{}</strong> نص بنجاح</p>
        <p>{} 👑</p>
        <div style="margin: 20px 0;">
            <div class="achievement-badge">بطل الذكاء الاصطناعي</div>
        </div>
        <p>نتمنى لك يوماً مليئاً بالإبداع والتميز! 🚀</p>
        <div style="margin-top: 20px;">
            <button onclick="window.close();" style="
                background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 25px;
                font-size: 16px;
                cursor: pointer;
                margin: 5px;
            ">إغلاق النافذة</button>
        </div>
    </div>
    """.format(st.session_state.analysis_count, get_motivational_message()), unsafe_allow_html=True)

def show_celebration():
    """عرض تأثير احتفالي بديل عن البالونات"""
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <div class="celebration-effect">
            <h1 style="color: #28a745;">🎉 تحليل ناجح! 🎉</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)

# أمثلة محسنة بشكل إبداعي
examples = [
    {
        "title": "✨ مثال إيجابي مبدع",
        "text": "لقد تفاجأت بالإبداع غير المحدود في هذا المشروع! كل تفصيلة تشهد على التميز والاحترافية. الأداء يتجاوز الخيال والنتائج مبهرة حقاً. هذا إنجاز يستحق الدراسة والاحتذاء به.",
        "type": "إيجابي"
    },
    {
        "title": "😞 مثال سلبي عميق", 
        "text": "أشعر بخيبة أمل لا توصف تجاه المستوى غير المتوقع. التقصير واضح في كل جانب والاهتمام بالتفاصيل مفقود تماماً. إنه أمر محبط ويحتاج لمراجعة شاملة وجذرية.",
        "type": "سلبي"
    },
    {
        "title": "🎭 مثال محايد متوازن",
        "text": "الأداء العام ضمن المعدلات الطبيعية المتوقعة. هناك نقاط قوة مقابلة لنقاط تحتاج للتحسين. الوضع الحالي يمثل قاعدة مناسبة للبناء عليها مستقبلاً.",
        "type": "محايد"
    },
    {
        "title": "📱 مراجعة منتج شاملة",
        "text": "الجهاز الجديد يجمع بين أناقة التصميم ودقة الأداء. الشاشة مبهرة والألوان زاهية، لكن البطارية تحتاج للتحسين. الكاميرا رائعة في النهار وتحتاج لدعم في الليل. السعر معقول مقارنة بالإمكانيات.",
        "type": "مراجعة"
    }
]

# الواجهة الرئيسية
def main():
    inject_css()
    
    # نافذة الخروج
    if st.session_state.show_exit_modal:
        show_exit_modal()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏃 البقاء في التطبيق", use_container_width=True):
                st.session_state.show_exit_modal = False
                st.rerun()
        with col2:
            if st.button("🔄 إعادة التشغيل", use_container_width=True):
                st.session_state.show_exit_modal = False
                st.rerun()
        with col3:
            if st.button("🚪 إغلاق التطبيق", type="primary", use_container_width=True):
                st.success("شكراً لك! نتمنى لك التوفيق 👑")
                time.sleep(2)
                st.stop()
        return
    
    # معالجة النقر على الأمثلة أولاً
    if st.session_state.get('example_clicked') is not None:
        example_text = st.session_state.example_clicked
        st.session_state.sentiment_input_text = example_text
        st.session_state.example_clicked = None
        # زيادة المفتاح لإجبار إعادة التحميل
        st.session_state.text_area_key += 1
        st.rerun()
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; direction: rtl; color: #2c3e50;'>
            <h1>🧠</h1>
            <h3>منصة الذكاء الاصطناعي العربية</h3>
            <p>الإصدار المميز - محسّن للأجهزة المتوسطة</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # إدخال اسم المستخدم
        st.header("👤 الملف الشخصي")
        user_name = st.text_input("اسمك الكريم:", value=st.session_state.user_name)
        if user_name != st.session_state.user_name:
            st.session_state.user_name = user_name
            st.success(f"مرحباً بك {user_name}! 👑")
        
        st.markdown("---")
        st.header("🤖 معلومات النموذج")
        
        st.info(f"""
        **النموذج النشط:** CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment
        
        **المميزات:**
        - مخصص للغة العربية
        - دقة عالية في تحليل المشاعر
        - مدرب على بيانات عربية متنوعة
        - محسّن للأجهزة المتوسطة
        """)
        
        # حالة تحميل النموذج
        if st.session_state.analyzer.model_loaded:
            st.success("✅ النموذج محمل وجاهز للاستخدام")
        else:
            st.warning("🔄 النموذج جاهز للتحميل عند الطلب")
        
        st.markdown("---")
        st.header("📊 سجل التحليلات")
        
        if st.session_state.analysis_history:
            for i, analysis in enumerate(st.session_state.analysis_history[:5]):
                sentiment_color = {
                    'إيجابي': '#28a745',
                    'سلبي': '#dc3545',
                    'محايد': '#ffc107'
                }.get(analysis['sentiment'], '#666666')
                
                st.markdown(f"""
                <div class="history-item">
                    <div style="font-size: 0.9em; color: #666;">{analysis['timestamp'].strftime('%H:%M')}</div>
                    <div><strong>{analysis['text']}</strong></div>
                    <div style="color: {sentiment_color}; font-weight: bold;">{analysis['sentiment']} ({analysis['confidence']:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("لا توجد تحليلات سابقة")
        
        if st.button("🗑️ مسح السجل", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()
        
        st.markdown("---")
        
        # إحصائيات المستخدم
        st.header("🏆 إنجازاتك")
        st.markdown(f"""
        <div style="text-align: center;">
            <h3>عدد التحليلات: {st.session_state.analysis_count}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.analysis_count >= 5:
            st.markdown('<div class="achievement-badge">🦸 بطل التحليل</div>', unsafe_allow_html=True)
        if st.session_state.analysis_count >= 10:
            st.markdown('<div class="achievement-badge">🧠 عبقري المشاعر</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # زر الخروج
        if st.button("🚪 خروج آمن", use_container_width=True, type="secondary"):
            st.session_state.show_exit_modal = True
            st.rerun()
        
        st.info("""
        **💡 معلومات الإصدار:**
        - الخدمة النشطة: تحليل المشاعر المتقدم
        - النموذج: CAMeL المتخصص للعربية
        - محسّن للأجهزة المتوسطة
        - إصدار المبدعين والمبتكرين
        """)

    # المنطقة الرئيسية
    st.title("🧠 منصة الذكاء الاصطناعي العربية - الإصدار المميز")
    
    # رسالة ترحيب مخصصة
    st.markdown(f"""
    <div class="feature-highlight">
        <h2>مرحباً {st.session_state.user_name}! 👑</h2>
        <p>{get_motivational_message()}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # عرض الخدمات
    st.markdown("## 🎯 الخدمات الذكية المتاحة")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="active-service">
            <h3>📊 تحليل المشاعر الذكي</h3>
            <p>✅ <strong>نشط ومتقدّم</strong></p>
            <p>نموذج CAMeL المتخصص</p>
            <p>🧠 + الذكاء الاصطناعي</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="frozen-service">
            <h3>📝 تلخيص النصوص الذكي</h3>
            <p>🔄 <strong>قيد التطوير</strong></p>
            <p>قريباً بإذن الله</p>
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
    
    # قسم تحليل المشاعر
    st.header("🎯 مركز التحليل الذكي للمشاعر")
    
    col_input, col_examples = st.columns([2, 1])
    
    with col_input:
        # استخدام key ديناميكي لمربع النص
        text_input = st.text_area(
            "أدخل النص العربي لتحليل المشاعر:",
            height=150,
            placeholder="اكتب أو الصق النص العربي هنا... وسنكشف أسرار مشاعره! 🕵️‍♂️",
            value=st.session_state.sentiment_input_text,
            key=f"main_text_input_{st.session_state.text_area_key}",
            help="🧠 يمكن تحليل النصوص حتى 2000 حرف باستخدام الذكاء الاصطناعي المتقدم"
        )
        
        # تحديث حالة الجلسة مباشرة
        if text_input != st.session_state.sentiment_input_text:
            st.session_state.sentiment_input_text = text_input
        
        if text_input:
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("📝 عدد الكلمات", len(text_input.split()))
            with col_info2:
                st.metric("🔤 عدد الأحرف", len(text_input))
            with col_info3:
                st.metric("⚡ جاهزية النموذج", "🟢 نشط" if st.session_state.analyzer.model_loaded else "🟡 جاهز")
    
    with col_examples:
        st.markdown("### 💡 أمثلة ذكية جاهزة")
        
        for example in examples:
            # استخدام callback function للتعامل مع النقر على الأمثلة
            if st.button(example["title"], key=f"ex_{example['title']}", use_container_width=True):
                st.session_state.example_clicked = example["text"]
                st.rerun()
    
    # زر التحليل مع تأثير خاص
    if st.button("🚀 بدء التحليل الذكي", use_container_width=True, type="primary"):
        if text_input.strip():
            # التحقق من طول النص
            is_valid, message = validate_text_length(text_input)
            if not is_valid:
                st.error(f"⚠️ {message}")
            else:
                # تحليل المشاعر مع رسالة تحميل مميزة
                with st.spinner(f"{get_funny_loading_message()}"):
                    time.sleep(1)  # تأثير درامي بسيط
                    sentiment, emoji, color, confidence = st.session_state.analyzer.analyze_sentiment(text_input)
                
                # حفظ النتائج
                st.session_state.last_analysis = {
                    'text': text_input,
                    'sentiment': sentiment,
                    'emoji': emoji,
                    'color': color,
                    'confidence': confidence
                }
                
                # إضافة إلى السجل
                add_to_history(text_input, sentiment, confidence)
                
                # عرض تأثير احتفالي بديل عن البالونات
                show_celebration()
                st.success(f"✅ تم التحليل بنجاح! {get_motivational_message()}")
                
                # تحديد فئة النتيجة للتنسيق
                sentiment_class = {
                    'إيجابي': 'sentiment-positive',
                    'سلبي': 'sentiment-negative', 
                    'محايد': 'sentiment-neutral'
                }.get(sentiment, 'result-card')
                
                # عرض النتيجة الرئيسية
                st.markdown(f"""
                <div class="result-card {sentiment_class}">
                    <div style="text-align: center; margin-bottom: 20px;">
                        <span style="font-size: 3em;" class="celebration-effect">{emoji}</span>
                        <h2 style="color: {color}; margin: 10px 0;" class="typewriter">النتيجة: {sentiment}</h2>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>🎯 مستوى الثقة:</span>
                            <span style="font-weight: bold; color: {color};">{confidence:.1f}%</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {confidence}%; background: {color};"></div>
                        </div>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px;">
                        <strong>📄 النص المدخل:</strong><br>
                        {text_input}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # إحصائيات إضافية
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    st.markdown(f"""
                    <div class="stat-card success-glow">
                        <h3>🎯 مستوى الثقة</h3>
                        <h2 style="color: {color};">{confidence:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_stat2:
                    st.markdown(f"""
                    <div class="stat-card success-glow">
                        <h3>📊 الحالة</h3>
                        <h2 style="color: {color};">{sentiment}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_stat3:
                    st.markdown(f"""
                    <div class="stat-card success-glow">
                        <h3>🧠 النموذج</h3>
                        <h2 style="color: #3498db;">CAMeL الذكي</h2>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ يرجى إدخال نص لتحليل مشاعره")
    
    st.markdown("---")
    
    # قسم تفسير النتائج
    if st.session_state.last_analysis:
        st.header("📈 مركز التفسير الذكي")
        analysis = st.session_state.last_analysis
        sentiment = analysis['sentiment']
        confidence = analysis['confidence']
        
        if sentiment == "إيجابي":
            st.info(f"""
            **🎉 النتيجة إيجابية!** (ثقة: {confidence:.1f}%)
            
            **🧠 التفسير الذكي:** 
            النص يعبر عن مشاعر إيجابية قوية تشير إلى الرضا والسعادة والإعجاب. 
            هذا يدل على تجربة ناجحة أو انطباع ممتاز.
            
            **💫 المؤشرات:**
            - كلمات إيجابية ومتفائلة
            - تراكيب تعبيرية مشجعة
            - تقييمات إيجابية واضحة
            """)
        elif sentiment == "سلبي":
            st.error(f"""
            **😔 النتيجة سلبية** (ثقة: {confidence:.1f}%)
            
            **🧠 التفسير الذكي:**
            النص يعبر عن مشاعر سلبية تشير إلى الاستياء أو خيبة الأمل.
            هذا يدل على تجربة غير مرضية تحتاج للتحسين.
            
            **💫 المؤشرات:**
            - كلمات سلبية وناقدة
            - تراكيب تعبيرية محبطة
            - شكاوى وملاحظات سلبية
            """)
        else:  # محايد
            st.warning(f"""
            **😐 النتيجة محايدة** (ثقة: {confidence:.1f}%)
            
            **🧠 التفسير الذكي:**
            النص يعبر عن موقف متوازن دون مشاعر قوية.
            هذا يدل على تقييم موضوعي أو وصف واقعي.
            
            **💫 المؤشرات:**
            - لغة وصفية محايدة
            - تقييمات متوازنة
            - معلومات واقعية
            """)
    
    st.markdown("---")
    
    # قسم الإنجازات
    st.header("🏆 لوحة الإنجازات الذكية")
    
    col_ach1, col_ach2, col_ach3 = st.columns(3)
    
    with col_ach1:
        st.markdown(f"""
        <div class="stat-card">
            <h3>📈 إجمالي التحليلات</h3>
            <h1 style="color: #3498db;">{st.session_state.analysis_count}</h1>
            <p>تحليل حتى الآن</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ach2:
        efficiency = min(st.session_state.analysis_count * 10, 100)
        st.markdown(f"""
        <div class="stat-card">
            <h3>⚡ كفاءة المستخدم</h3>
            <h1 style="color: #e74c3c;">{efficiency}%</h1>
            <p>مستوى متقدم</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_ach3:
        st.markdown(f"""
        <div class="stat-card">
            <h3>🎯 الدقة المتوقعة</h3>
            <h1 style="color: #27ae60;">95%</h1>
            <p>دقة النموذج</p>
        </div>
        """, unsafe_allow_html=True)
    
    # شارات الإنجازات
    st.subheader("🎖️ شارات إنجازاتك")
    achievement_cols = st.columns(4)
    
    with achievement_cols[0]:
        if st.session_state.analysis_count >= 1:
            st.markdown('<div class="achievement-badge">🎯 مبتدئ</div>', unsafe_allow_html=True)
    
    with achievement_cols[1]:
        if st.session_state.analysis_count >= 3:
            st.markdown('<div class="achievement-badge">🚀 محترف</div>', unsafe_allow_html=True)
    
    with achievement_cols[2]:
        if st.session_state.analysis_count >= 5:
            st.markdown('<div class="achievement-badge">🧠 خبير</div>', unsafe_allow_html=True)
    
    with achievement_cols[3]:
        if st.session_state.analysis_count >= 10:
            st.markdown('<div class="achievement-badge">🏆 أسطورة</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # أزرار إضافية للتحكم
    st.header("⚙️ أدوات التحكم")
    col_control1, col_control2, col_control3 = st.columns(3)
    
    with col_control1:
        if st.button("🔄 تحديث الصفحة", use_container_width=True):
            st.rerun()
    
    with col_control2:
        if st.button("🧹 مسح النص", use_container_width=True):
            st.session_state.sentiment_input_text = ""
            st.session_state.text_area_key += 1
            st.rerun()
    
    with col_control3:
        if st.button("🚪 قائمة الخروج", use_container_width=True, type="primary"):
            st.session_state.show_exit_modal = True
            st.rerun()
    
    # تذييل الصفحة
    st.markdown("""
    <div style='text-align: center; color: #666; direction: rtl; padding: 20px;'>
        <h3>🧠 منصة الذكاء الاصطناعي العربية - الإصدار المميز</h3>
        <p>✅ <strong>الخدمة النشطة:</strong> تحليل المشاعر الذكي باستخدام CAMeL المتقدم</p>
        <p>🚀 <strong>محسّن للأجهزة المتوسطة</strong> - أداء ممتاز مع ذاكرة 8GB</p>
        <p>🎯 <strong>صمم خصيصاً للمبدعين والمبتكرين</strong></p>
        <p>✨ <strong>فريق الذكاء الاصطناعي - كاك بنك</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()