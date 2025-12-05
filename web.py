import streamlit as st
import re
import string
from pyvi import ViTokenizer
import pickle
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import time

# --- 1. CẤU HÌNH TRANG & CSS ---
st.set_page_config(
    page_title="Phân loại Cảm xúc",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        with open('MODEL/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        with open('MODEL/machine_deep_cai_thien.pkl', 'rb') as f:
            model_ml = pickle.load(f)

        model_dl = load_model('MODEL/best_model.h5')
        return vectorizer, model_ml, model_dl

    except:
            st.error(f"Lỗi tải file")
            return None, None, None

vectorizer, model_ml, model_deep = load_resources()

# --- 3. TIỀN XỬ LÝ ---
emoji_pattern = re.compile("[" 
                u"\U0001F600-\U0001F64F" 
                u"\U0001F300-\U0001F5FF" 
                u"\U0001F680-\U0001F6FF" 
                u"\U0001F1E0-\U0001F1FF" 
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.lower()
    text = re.sub(emoji_pattern, " ", text)
    text = re.sub(r'([a-zA-Zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ])\1+', r'\1', text)
    text = text.strip()
    while text.endswith(tuple(string.punctuation + string.whitespace)):
        text = text[:-1]
    while text.startswith(tuple(string.punctuation + string.whitespace)):
        text = text[1:]
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text)
    text = ViTokenizer.tokenize(text)
    return text

def ma_hoa(text):
    text = clean_text(text)
    text = vectorizer.transform([text])
    return text

# --- 4. HÀM DỰ ĐOÁN & HIỂN THỊ ---

def hien_thi_card(is_positive, time_val, confidence=None):
    """
    Hàm hiển thị kết quả.
    is_positive: True (Tích cực), False (Tiêu cực)
    """
    if is_positive: 
        base_text = "😊 TÍCH CỰC"
        css_class = "positive"
    else: 
        base_text = "☹️ TIÊU CỰC"
        css_class = "negative"
    
    display_html = base_text

    html_content = f"""
    <div class="result-card {css_class}">
        <h3>{display_html}</h3>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # Format 2 số sau dấu phẩy cho thời gian
        st.metric("⏱️ Thời gian", f"{time_val:.2f}s")
    with col2:
        if confidence is not None:
            # Format 2 số sau dấu phẩy cho độ tin cậy
            st.metric("🎯 Độ tin cậy", f"{confidence:.2f}%")
        else:
            st.metric("Mô hình", "ML")

def ML(text_vectorized):
    st.subheader("🤖 Machine Learning")
    
    time_bd = time.perf_counter()
    kq = model_ml.predict(text_vectorized)
    time_kt = time.perf_counter()
    time_ml = time_kt - time_bd
    
    is_positive = (kq[0] == 1)
    
    hien_thi_card(is_positive, time_ml)

def DL(text_vectorized):
    st.subheader("🧠 Deep Learning")
    

    time_bd = time.perf_counter()
    pred = model_deep.predict(text_vectorized)
    time_kt = time.perf_counter()
    time_dl = time_kt - time_bd
    
    # Logic mới theo yêu cầu:
    # pred[0][0] > 0.5 là Tích cực
    val = pred[0][0]
    if val > 0.5:
        is_positive = True
        confidence = val * 100
    else:
        is_positive = False
        confidence = (1 - val) * 100
    
    hien_thi_card(is_positive, time_dl, confidence)


# --- 5. GIAO DIỆN CHÍNH ---

st.title("💡 Phân loại Đánh giá Khách hàng")
st.write("---")

nhap = st.text_area("📝 Nhập văn bản đánh giá vào đây:", height=100)

if st.button("🚀 Phân loại ngay", type="primary", use_container_width=True):
    if nhap.strip() == "":
        st.warning("Vui lòng nhập nội dung!")
    else:
        with st.spinner("Đang phân tích..."):
            text_processed = ma_hoa(nhap)
            
            col1, col2 = st.columns(2)
            
            with col1:
                DL(text_processed)
            
            with col2:
                ML(text_processed)