# ========================================
# ĐỒ ÁN: TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT
# PhoBERT + Dictionary + Rule phủ định + Teencode + SQLite + Testcase
# ========================================

import streamlit as st
from transformers import pipeline
import sqlite3
from datetime import datetime
import pandas as pd
from underthesea import word_tokenize
from utils.teencode_dict import normalize_teencode, remove_accents
from utils.test_case import test_cases  # 10 case chuẩn thầy

# ========================================
# 1. Load PhoBERT pipeline
# ========================================
@st.cache_resource
def load_classifier():
    model_name = "wonrax/phobert-base-vietnamese-sentiment"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

classifier = load_classifier()

# ========================================
# 2. Dictionary 25 từ
# ========================================
sentiment_dict = {
    "vui": "POSITIVE", "cảm ơn": "POSITIVE", "tuyệt": "POSITIVE",
    "hay": "POSITIVE", "đỉnh": "POSITIVE", "thích": "POSITIVE",
    "yêu": "POSITIVE", "hạnh phúc": "POSITIVE", "vui vẻ": "POSITIVE", "thuận": "POSITIVE",
    "ok": "NEUTRAL", "ổn": "NEUTRAL", "ổn định": "NEUTRAL",
    "bình thường": "NEUTRAL", "cũng được": "NEUTRAL",
    "buồn": "NEGATIVE", "chán": "NEGATIVE", "ghét": "NEGATIVE",
    "tồi": "NEGATIVE", "dở": "NEGATIVE", "thất vọng": "NEGATIVE",
    "khó chịu": "NEGATIVE", "tệ": "NEGATIVE", "khủng khiếp": "NEGATIVE",
    "bực mình": "NEGATIVE", "mệt mỏi": "NEGATIVE"
}

# ========================================
# 3. Preprocess text
# ========================================
def preprocess(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if len(text) < 5 or len(text) > 50:
        return None

    # 1. Chuẩn hóa teencode
    text_norm = normalize_teencode(text.lower())

    # 2. Tokenize bằng underthesea (giữ dấu)
    try:
        tokens = word_tokenize(text_norm)
    except:
        tokens = text_norm.split()

    # 3. Nếu token ít hơn 2 hoặc quá nhiều → loại
    if len(tokens) < 2 or len(tokens) > 20:
        return None

    # 4. Nếu toàn không dấu → giữ bản gốc có dấu nếu có trong dictionary
    has_diacritics = any(ord(c) > 127 for c in text_norm)
    if not has_diacritics:
        # dò từ điển sentiment_dict để phục hồi dấu
        for key in sentiment_dict.keys():
            key_no = remove_accents(key)
            if key_no in text_norm:
                text_norm = text_norm.replace(key_no, key)
    
    return " ".join(tokens)
# ========================================
# 4. Rule phủ định
# ========================================
def negation_rule(text):
    text_low = text.lower()
    no_acc = remove_accents(text_low)
    if "không " in text_low or "khong " in no_acc:
        positive_words = ["vui", "vui ve", "tuyet", "thich", "yeu", "hanh phuc", "hay", "dinh", "cam on"]
        negative_words = ["buon", "chan", "ghet", "toi", "do", "met", "te"]
        for w in positive_words:
            if f"khong {w}" in no_acc or f"không {w.replace('_',' ')}" in text_low:
                return "NEGATIVE"
        for w in negative_words:
            if f"khong {w}" in no_acc or f"không {w.replace('_',' ')}" in text_low:
                return "NEUTRAL"
    return None

# ========================================
# 5. Dictionary match
# ========================================
def dict_match(text):
    if not text:
        return None
    t = text.lower()
    t_no = remove_accents(t)
    # multi-word first
    for key, label in sentiment_dict.items():
        key_norm = key.lower()
        key_no = remove_accents(key_norm)
        if " " in key_norm and (key_norm in t or key_no in t_no):
            return label
    # single-word
    tokens = t.split()
    tokens_no = t_no.split()
    for key, label in sentiment_dict.items():
        if " " not in key and (key.lower() in tokens or remove_accents(key.lower()) in tokens_no):
            return label
    return None

# ========================================
# 6. Normalize label
# ========================================
def normalize_label(label):
    mapping = {"POS": "POSITIVE", "NEG": "NEGATIVE", "NEU": "NEUTRAL",
               "POSITIVE":"POSITIVE","NEGATIVE":"NEGATIVE","NEUTRAL":"NEUTRAL"}
    return mapping.get(label.upper(), label.upper())

# ========================================
# 7. Classify sentiment
# ========================================
def classify_sentiment(text, threshold=0.5):
    clean = preprocess(text)
    if clean is None:
        return None, 0.0

    # Rule phủ định
    neg_label = negation_rule(clean)
    if neg_label:
        return normalize_label(neg_label), 0.98

    # Dictionary ưu tiên
    dic_label = dict_match(clean)
    if dic_label:
        return normalize_label(dic_label), 0.99

    # PhoBERT fine-tuned
    result = classifier(clean)[0]
    label = normalize_label(result['label'])
    confidence = result['score']

    # Confidence thấp nhưng câu dài → giữ label PhoBERT
    if len(clean.split()) <= 5 and confidence < threshold:
        label = "NEUTRAL"

    return label, confidence

# ========================================
# 8. SQLite
# ========================================
def init_db():
    conn = sqlite3.connect('history.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS sentiments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT,
                        sentiment TEXT,
                        timestamp TEXT
                    )''')
    conn.commit()
    conn.close()

def save_result(text, sentiment):
    conn = sqlite3.connect('history.db')
    timestamp = datetime.now().isoformat()
    conn.execute("INSERT INTO sentiments (text,sentiment,timestamp) VALUES (?,?,?)",
                 (text, sentiment, timestamp))
    conn.commit()
    conn.close()

init_db()

# ========================================
# 9. Streamlit UI
# ========================================
st.title("Trợ lý Phân loại Cảm xúc Tiếng Việt")
st.markdown("Dùng PhoBERT để phân tích cảm xúc từ văn bản tiếng Việt.")

text_input = st.text_area("Nhập câu tiếng Việt:", height=120)

if st.button("Phân loại cảm xúc"):
    if not text_input.strip():
        st.error("Câu quá ngắn hoặc không hợp lệ!")
    else:
        with st.spinner("Đang phân tích..."):
            sentiment, score = classify_sentiment(text_input)
            if sentiment is None:
                st.error("Câu không hợp lệ!")
            else:
                st.success(f"**Kết quả: {sentiment}** (Độ tin cậy: {score:.2%})")
                save_result(text_input, sentiment)

# Lịch sử (LIMIT 50)
if st.checkbox("Xem lịch sử"):
    df = pd.read_sql_query("SELECT id,text,sentiment,timestamp FROM sentiments ORDER BY timestamp DESC LIMIT 50",
                           sqlite3.connect("history.db"))
    st.dataframe(df)

# ========================================
# 10. Testcase
# ========================================
st.sidebar.header("Test Độ Chính Xác")
if st.sidebar.button("Chạy 10 test case"):
    correct = 0
    results = []
    for case in test_cases:
        text_case = case.get("text")
        expected = case.get("expected", "NEUTRAL")
        pred, conf = classify_sentiment(text_case)
        pred = pred if pred else "NEUTRAL"
        ok = pred.upper() == expected.upper()
        if ok:
            correct += 1
        results.append({
            "Câu": text_case,
            "Dự đoán": pred.upper(),
            "Mong đợi": expected.upper(),
            "Độ tin cậy": f"{conf:.2%}",
            "Kết quả": "✔️ Đúng" if ok else "❌ Sai"
        })
    acc = correct / len(test_cases) * 100
    st.sidebar.success(f"🎯 Độ chính xác: {acc:.1f}% ({correct}/{len(test_cases)})")
    st.sidebar.dataframe(pd.DataFrame(results))