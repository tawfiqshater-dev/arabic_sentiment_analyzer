import streamlit as st
import pandas as pd
import time
from datetime import datetime
import requests
import json
from typing import List, Tuple, Optional
import re
import random
import numpy as np

# إعداد صفحة Streamlit
st.set_page_config(
    page_title="منصة الذكاء الاصطناعي العربية - السحابية",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# نظام Hugging Face API
class HuggingFaceAPI:
    def __init__(self):
        self.api_token = None
        self.api_urls = {
            'sentiment': "https://api-inference.huggingface.co/models/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
            'summarization': "https://api-inference.huggingface.co/models/csebuetnlp/mT5_multilingual_XLSum",
            'keywords': "https://api-inference.huggingface.co/models/yanekyuk/bert-keyword-extractor",
            'chat': "https://api-inference.huggingface.co/models/UBC-NLP/AraT5-base"
        }
        
    def set_api_token(self, token):
        """تعيين توكن Hugging Face API"""
        self.api_token = token
        
    def query_api(self, model_type, inputs, parameters=None):
        """استدعاء Hugging Face API"""
        if not self.api_token:
            return None, "❌ لم يتم تعيين توكن Hugging Face API"
            
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {"inputs": inputs}
        if parameters:
            payload["parameters"] = parameters
            
        try:
            response = requests.post(
                self.api_urls[model_type],
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json(), None
            elif response.status_code == 503:
                return None, "⏳ النموذج جاري التحميل، يرجى المحاولة مرة أخرى بعد بضع ثوان"
            else:
                return None, f"❌ خطأ في API: {response.status_code} - {response.text}"
                
        except requests.exceptions.Timeout:
            return None, "⏰ انتهت مهلة الطلب، يرجى المحاولة مرة أخرى"
        except Exception as e:
            return None, f"❌ خطأ في الاتصال: {str(e)}"

# نظام إدارة الحالة
if 'hf_api' not in st.session_state:
    st.session_state.hf_api = HuggingFaceAPI()
if 'active_service' not in st.session_state:
    st.session_state.active_service = "sentiment"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = "الزائر الكريم"
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
if 'api_token' not in st.session_state:
    st.session_state.api_token = ""

# دوال مساعدة
def validate_text_length(text: str, min_len=5, max_len=2000) -> Tuple[bool, str]:
    """التحقق من طول النص المناسب"""
    if len(text.strip()) < min_len:
        return False, f"النص قصير جداً. يرجى إدخال نص أطول من {min_len} حروف."
    elif len(text) > max_len:
        return False, f"النص طويل جداً. الحد الأقصى {max_len} حرف."
    return True, "النص مناسب للتحليل"

def get_motivational_message():
    """رسائل تحفيزية عشوائية"""
    messages = [
        "🔥 أنت تقود ثورة الذكاء الاصطناعي!",
        "🚀 إبداعك لا يعرف حدوداً!",
        "💡 أفكارك ستغير المستقبل!",
        "🌟 أنت مصدر إلهام للجميع!",
        "🎯 دقتك في التحليل مذهلة!"
    ]
    return random.choice(messages)

def simple_arabic_summarizer(text, max_sentences=3):
    """تلخيص بسيط للنصوص العربية بدون API"""
    sentences = re.split(r'[.!؟]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if len(sentences) <= max_sentences:
        return text
    
    important_sentences = sorted(sentences, key=len, reverse=True)[:max_sentences]
    return ' '.join(important_sentences)

def simple_keyword_extractor(text, num_keywords=5):
    """استخراج كلمات مفتاحية بسيط للنصوص العربية"""
    stop_words = {'في', 'من', 'إلى', 'على', 'أن', 'ما', 'هذا', 'هذه', 'كان', 'يكون'}
    
    words = re.findall(r'\b\w+\b', text)
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [(word, freq/len(words)) for word, freq in sorted_words[:num_keywords]]

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
    
    .stTextArea textarea {
        direction: rtl;
        text-align: right;
    }
    
    .active-service {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        text-align: center;
        border: 3px solid #ffeb3b;
    }
    
    .inactive-service {
        background: linear-gradient(135deg, #bdc3c7 0%, #2c3e50 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        text-align: center;
        opacity: 0.8;
    }
    
    .result-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border-right: 4px solid #28a745;
        direction: rtl;
        text-align: right;
    }
    
    .chat-message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 15px 15px 0 15px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
    }
    
    .chat-message-bot {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #2c3e50;
        padding: 15px;
        border-radius: 15px 15px 15px 0;
        margin: 10px 0;
        border-right: 4px solid #3498db;
        direction: rtl;
        text-align: right;
    }
    
    .keyword-badge {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        color: #2c3e50;
        padding: 8px 15px;
        border-radius: 20px;
        margin: 5px;
        display: inline-block;
        font-weight: bold;
    }
    
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-top: 4px solid #3498db;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# الواجهة الرئيسية
def main():
    inject_css()
    
    # 🔐 تحميل التوكن تلقائياً من Secrets إذا كان متوفراً
    if not st.session_state.api_token:
        st.session_state.api_token = st.secrets.get("HF_TOKEN", "")
        if st.session_state.api_token:
            st.session_state.hf_api.set_api_token(st.session_state.api_token)
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; direction: rtl; color: #2c3e50;'>
            <h1>🧠</h1>
            <h3>منصة الذكاء الاصطناعي العربية</h3>
            <p>الإصدار السحابي - باستخدام Hugging Face API</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # إدخال API Token (اختياري إذا كان موجوداً في Secrets)
        st.header("🔑 إعدادات API")
        
        # عرض حالة التوكن
        if st.session_state.api_token:
            st.success("✅ تم تحميل التوكن تلقائياً من الإعدادات")
            if st.checkbox("🔄 تغيير التوكن"):
                api_token = st.text_input(
                    "Hugging Face API Token الجديد:",
                    type="password",
                    help="احصل على التوكن من https://huggingface.co/settings/tokens"
                )
                if api_token and api_token != st.session_state.api_token:
                    st.session_state.api_token = api_token
                    st.session_state.hf_api.set_api_token(api_token)
                    st.success("✅ تم تحديث التوكن بنجاح!")
        else:
            api_token = st.text_input(
                "Hugging Face API Token:",
                type="password",
                help="احصل على التوكن من https://huggingface.co/settings/tokens"
            )
            if api_token and api_token != st.session_state.api_token:
                st.session_state.api_token = api_token
                st.session_state.hf_api.set_api_token(api_token)
                st.success("✅ تم تعيين التوكن بنجاح!")
        
        st.markdown("---")
        st.header("👤 الملف الشخصي")
        user_name = st.text_input("اسمك الكريم:", value=st.session_state.user_name)
        if user_name != st.session_state.user_name:
            st.session_state.user_name = user_name
        
        st.markdown("---")
        st.header("🎯 اختر الخدمة")
        
        service_options = {
            "تحليل المشاعر": "sentiment",
            "تلخيص النصوص": "summarization", 
            "كلمات مفتاحية": "keywords",
            "محادثة ذكية": "chat"
        }
        
        selected_service = st.radio(
            "الخدمات:",
            list(service_options.keys()),
            index=list(service_options.values()).index(st.session_state.active_service)
        )
        
        st.session_state.active_service = service_options[selected_service]
        
        st.markdown("---")
        st.header("📊 الإحصائيات")
        st.metric("عدد التحليلات", st.session_state.analysis_count)
        st.metric("الخدمة النشطة", selected_service)
        
        if st.session_state.api_token:
            st.success("✅ API متصل وجاهز")
        else:
            st.warning("⚠️ يرجى إدخال API Token")

    # المنطقة الرئيسية
    st.title("🧠 منصة الذكاء الاصطناعي العربية - الإصدار السحابي")
    
    # رسالة ترحيب
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2>مرحباً {st.session_state.user_name}! 👑</h2>
        <p>{get_motivational_message()}</p>
        <p><small>🌐 جميع الخدمات تعمل عبر Hugging Face API - لا حاجة لتحميل النماذج محلياً</small></p>
        {"<p><small>🔐 التوكن محمل تلقائياً من الإعدادات الآمنة</small></p>" if st.session_state.api_token else ""}
    </div>
    """, unsafe_allow_html=True)
    # عرض الخدمات
    st.markdown("## 🎯 الخدمات الذكية المتاحة")
    cols = st.columns(4)
    
    services = [
        {"name": "تحليل المشاعر", "icon": "📊", "active": st.session_state.active_service == "sentiment"},
        {"name": "تلخيص النصوص", "icon": "📝", "active": st.session_state.active_service == "summarization"},
        {"name": "كلمات مفتاحية", "icon": "🔑", "active": st.session_state.active_service == "keywords"},
        {"name": "محادثة ذكية", "icon": "💬", "active": st.session_state.active_service == "chat"}
    ]
    
    for i, service in enumerate(services):
        with cols[i]:
            css_class = "active-service" if service["active"] else "inactive-service"
            st.markdown(f"""
            <div class="{css_class}">
                <h3>{service['icon']} {service['name']}</h3>
                <p>{"✅ نشط ومتقدّم" if service["active"] else "⚡ انقر لتفعيل"}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # منطقة الخدمات النشطة
    active_service = st.session_state.active_service
    
    if active_service == "sentiment":
        render_sentiment_analysis()
    elif active_service == "summarization":
        render_text_summarization()
    elif active_service == "keywords":
        render_keyword_extraction()
    elif active_service == "chat":
        render_chat_interface()

def render_sentiment_analysis():
    """واجهة تحليل المشاعر باستخدام API"""
    st.header("📊 تحليل المشاعر العربي عبر API")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "أدخل النص العربي لتحليل المشاعر:",
            height=120,
            placeholder="اكتب أو الصق النص العربي هنا...",
            help="يمكن تحليل النصوص حتى 2000 حرف"
        )
        
        if text_input:
            st.metric("عدد الكلمات", len(text_input.split()))
            st.metric("عدد الأحرف", len(text_input))
    
    with col2:
        st.markdown("### 💡 أمثلة سريعة")
        examples = [
            "لقد تفاجأت بالإبداع غير المحدود في هذا المشروع! كل تفصيلة تشهد على التميز والاحترافية.",
            "أشعر بخيبة أمل لا توصف تجاه المستوى غير المتوقع. التقصير واضح في كل جانب.",
            "الأداء العام ضمن المعدلات الطبيعية المتوقعة. هناك نقاط قوة مقابلة لنقاط تحتاج للتحسين."
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"مثال {i+1}", key=f"sent_ex_{i}", use_container_width=True):
                st.rerun()
    
    if st.button("🚀 تحليل المشاعر عبر API", type="primary", use_container_width=True):
        if not st.session_state.api_token:
            st.error("❌ يرجى إدخال Hugging Face API Token في الشريط الجانبي")
            return
            
        if text_input.strip():
            is_valid, message = validate_text_length(text_input)
            if not is_valid:
                st.error(f"⚠️ {message}")
            else:
                with st.spinner("🔄 جاري تحليل المشاعر عبر API..."):
                    result, error = st.session_state.hf_api.query_api('sentiment', text_input)
                    
                    if error:
                        st.error(error)
                    else:
                        try:
                            sentiment_data = result[0]
                            sentiment_label = sentiment_data['label']
                            confidence = sentiment_data['score'] * 100
                            
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
                            
                            st.session_state.analysis_count += 1
                            
                            st.success(f"✅ تم التحليل بنجاح عبر API! الثقة: {confidence:.1f}%")
                            
                            st.markdown(f"""
                            <div class="result-card">
                                <div style="text-align: center;">
                                    <span style="font-size: 3em;">{emoji}</span>
                                    <h2 style="color: {color};">{arabic_sentiment}</h2>
                                    <p style="font-size: 1.2em; color: {color};">مستوى الثقة: {confidence:.1f}%</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ خطأ في معالجة النتيجة: {str(e)}")

def render_text_summarization():
    """واجهة تلخيص النصوص باستخدام API"""
    st.header("📝 تلخيص النصوص العربي عبر API")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "أدخل النص العربي لتلخيصه:",
            height=150,
            placeholder="الصق النص الطويل هنا...",
            help="يمكن تلخيص النصوص حتى 2000 حرف"
        )
        
        if text_input:
            st.metric("عدد الكلمات قبل التلخيص", len(text_input.split()))
    
    with col2:
        st.markdown("### ⚙️ إعدادات التلخيص")
        summary_length = st.slider("طول الملخص:", 50, 300, 150)
        st.info("الطول الأمثل: 150 كلمة")
    
    if st.button("🎯 توليد الملخص عبر API", type="primary", use_container_width=True):
        if not st.session_state.api_token:
            st.error("❌ يرجى إدخال Hugging Face API Token في الشريط الجانبي")
            return
            
        if text_input.strip():
            is_valid, message = validate_text_length(text_input, min_len=100)
            if not is_valid:
                st.error(f"⚠️ {message}")
            else:
                with st.spinner("🔄 جاري تلخيص النص عبر API..."):
                    parameters = {
                        "max_length": summary_length,
                        "min_length": 40,
                        "do_sample": False
                    }
                    
                    result, error = st.session_state.hf_api.query_api('summarization', text_input, parameters)
                    
                    if error:
                        st.error(f"{error} - جاري استخدام التلخيص البسيط...")
                        summary = simple_arabic_summarizer(text_input)
                        st.info("ℹ️ استخدام التلخيص البسيط (API غير متوفر)")
                    else:
                        try:
                            summary = result[0]['summary_text']
                        except:
                            summary = simple_arabic_summarizer(text_input)
                            st.info("ℹ️ استخدام التلخيص البسيط (استجابة API غير متوقعة)")
                    
                    st.session_state.analysis_count += 1
                    st.success("✅ تم التلخيص بنجاح!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📄 النص الأصلي")
                        st.info(f"الطول: {len(text_input.split())} كلمة")
                        st.text_area("", text_input, height=200, key="original_text", label_visibility="collapsed")
                    
                    with col2:
                        st.subheader("📝 الملخص المولد")
                        st.success(f"الطول: {len(summary.split())} كلمة")
                        st.text_area("", summary, height=200, key="summary_text", label_visibility="collapsed")

def render_keyword_extraction():
    """واجهة استخراج الكلمات المفتاحية باستخدام API"""
    st.header("🔑 استخراج الكلمات المفتاحية عبر API")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "أدخل النص العربي لاستخراج الكلمات المفتاحية:",
            height=120,
            placeholder="أدخل النص هنا...",
            help="يمكن معالجة النصوص حتى 2000 حرف"
        )
        
        if text_input:
            st.metric("عدد الكلمات", len(text_input.split()))
    
    with col2:
        st.markdown("### ⚙️ الإعدادات")
        num_keywords = st.slider("عدد الكلمات المفتاحية:", 3, 10, 5)
    
    if st.button("🎯 استخراج الكلمات عبر API", type="primary", use_container_width=True):
        if not st.session_state.api_token:
            st.error("❌ يرجى إدخال Hugging Face API Token في الشريط الجانبي")
            return
            
        if text_input.strip():
            is_valid, message = validate_text_length(text_input)
            if not is_valid:
                st.error(f"⚠️ {message}")
            else:
                with st.spinner("🔄 جاري استخراج الكلمات المفتاحية عبر API..."):
                    # محاولة استخدام API أولاً
                    result, error = st.session_state.hf_api.query_api('keywords', text_input)
                    
                    if error or not result:
                        st.info("ℹ️ استخدام الاستخراج البسيط (API غير متوفر)")
                        keywords = simple_keyword_extractor(text_input, num_keywords)
                    else:
                        try:
                            # محاولة تفسير استجابة API
                            if isinstance(result, list) and len(result) > 0:
                                keywords = [(item.get('word', ''),
                                           item.get('score', 0.5)) 
                                          for item in result[:num_keywords]]
                            else:
                                keywords = simple_keyword_extractor(text_input, num_keywords)
                                st.info("ℹ️ استخدام الاستخراج البسيط (استجابة API غير متوقعة)")
                        except:
                            keywords = simple_keyword_extractor(text_input, num_keywords)
                            st.info("ℹ️ استخدام الاستخراج البسيط (خطأ في معالجة API)")
                    
                    st.session_state.analysis_count += 1
                    st.success("✅ تم الاستخراج بنجاح!")
                    
                    st.subheader("🏷️ الكلمات المفتاحية المستخرجة")
                    
                    for keyword, score in keywords:
                        st.markdown(f'<div class="keyword-badge">{keyword} (ثقة: {score:.2f})</div>', 
                                  unsafe_allow_html=True)

def render_chat_interface():
    """واجهة المحادثة الذكية باستخدام API"""
    st.header("💬 محادثة ذكية عربية عبر API")
    
    # عرض سجل المحادثة
    st.subheader("📝 سجل المحادثة")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history[-8:]:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message-user">
                    <strong>👤 أنت:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message-bot">
                    <strong>🤖 المساعد الذكي:</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # إدخال الرسالة
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "اكتب رسالتك هنا:",
            placeholder="اسألني عن أي شيء باللغة العربية...",
            key="chat_input"
        )
    
    with col2:
        st.markdown("")
        st.markdown("")
        send_button = st.button("🚀 إرسال", use_container_width=True)
    
    if send_button and user_input.strip():
        if not st.session_state.api_token:
            st.error("❌ يرجى إدخال Hugging Face API Token في الشريط الجانبي")
            return
        
        # إضافة رسالة المستخدم للسجل
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # توليد الرد باستخدام API
        with st.spinner("🔄 جاري توليد الرد عبر API..."):
            try:
                # استخدام نموذج المحادثة عبر API
                prompt = f"المستخدم: {user_input}\nالمساعد:"
                result, error = st.session_state.hf_api.query_api('chat', prompt)
                
                if error:
                    # استخدام ردود مبرمجة إذا فشل API
                    arabic_responses = [
                        "مرحباً بك! أنا مساعد ذكي متخصص في اللغة العربية. كيف يمكنني مساعدتك؟",
                        "شكراً لسؤالك! أنا هنا لمساعدتك في أي استفسار باللغة العربية.",
                        "أهلاً وسهلاً! يمكنني الإجابة على أسئلتك وتحليل النصوص العربية.",
                        "سعيد بتواصلك معي! ما الذي تريد أن تعرفه عن الذكاء الاصطناعي واللغة العربية؟",
                        "أهلاً! أنا جاهز للإجابة على استفساراتك باللغة العربية."
                    ]
                    assistant_response = random.choice(arabic_responses)
                    st.info("ℹ️ استخدام الردود المبرمجة (API غير متوفر)")
                else:
                    try:
                        assistant_response = result[0]['generated_text']
                        # تنظيف الرد إذا لزم الأمر
                        if "المساعد:" in assistant_response:
                            assistant_response = assistant_response.split("المساعد:")[-1].strip()
                    except:
                        arabic_responses = [
                            "أفهم سؤالك! هل يمكنك توضيح المزيد؟",
                            "هذا موضوع مثير للاهتمام! هل لديك أسئلة أخرى؟",
                            "شكراً على سؤالك! هل تريد معرفة المزيد عن هذا الموضوع؟"
                        ]
                        assistant_response = random.choice(arabic_responses)
                        st.info("ℹ️ استخدام الردود المبرمجة (استجابة API غير متوقعة)")
                
                # إضافة رد المساعد للسجل
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': assistant_response,
                    'timestamp': datetime.now()
                })
                
                st.session_state.analysis_count += 1
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ خطأ في المحادثة: {str(e)}")
    
    # أزرار تحكم إضافية
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ مسح المحادثة", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("💡 اقتراح سؤال", use_container_width=True):
            suggestions = [
                "ما هو الذكاء الاصطناعي؟",
                "كيف يمكنني تحسين مهاراتي؟",
                "ما هي أحدث التقنيات في 2024؟",
                "تكلم عن أهمية التعليم",
                "ما هو مستقبل العمل عن بعد؟"
            ]
            st.session_state.chat_input = random.choice(suggestions)
            st.rerun()

if __name__ == "__main__":
    main()