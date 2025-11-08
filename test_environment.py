import streamlit as st
from transformers import pipeline
import torch
import sys

def test_environment():
    st.title("🧪 اختبار بيئة المشروع")
    
    st.write("### التحقق من المكونات:")
    
    # اختبار Python
    st.write(f"**Python version:** {sys.version.split()[0]}")
    
    # اختبار PyTorch
    st.write(f"**PyTorch version:** {torch.__version__}")
    
    # اختبار CUDA
    st.write(f"**CUDA available:** {torch.cuda.is_available()}")
    
    # اختبار Streamlit
    try:
        st.success("✅ Streamlit يعمل بشكل صحيح")
    except Exception as e:
        st.error(f"❌ خطأ في Streamlit: {e}")
    
    # اختبار تحميل نموذج بسيط
    try:
        with st.spinner("جاري اختبار تحميل النماذج..."):
            classifier = pipeline("sentiment-analysis")
            test_result = classifier("I love this!")
            st.success(f"✅ النماذج تعمل: {test_result}")
    except Exception as e:
        st.warning(f"⚠️ تحميل النماذج به مشكلة: {e}")

if __name__ == "__main__":
    test_environment()
