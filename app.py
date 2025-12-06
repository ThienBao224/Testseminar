# ========================================
# ĐỒ ÁN: TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT
# Theo hướng dẫn thầy: PhoBERT + Dictionary + Threshold + SQLite LIMIT 50
# (Phiên bản: dùng utils.teencode_dict + utils.test_case)
# ========================================

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sqlite3
from datetime import datetime
import pandas as pd
from underthesea import word_tokenize
from utils.teencode_dict import normalize_teencode, remove_accents
from utils.test_case import test_cases  # test_cases: list 10 mẫu
import re

# ========================================
# 1. Tải mô hình (PhoBERT theo thầy)
# ========================================
@st.cache_resource
def load_model():
    model_name = "wonrax/phobert-base-vietnamese-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ========================================
# 2. Dictionary 25 từ sentiment
# ========================================
sentiment_dict = {
    "vui": "POSITIVE", "tuyệt": "POSITIVE", "hay": "POSITIVE", "đỉnh": "POSITIVE", "thích": "POSITIVE",
    "yêu": "POSITIVE", "ok": "NEUTRAL", "ổn": "NEUTRAL", "bình thường": "NEUTRAL", "cũng được": "NEUTRAL",
    "buồn": "NEGATIVE", "chán": "NEGATIVE", "ghét": "NEGATIVE", "tồi": "NEGATIVE", "dở": "NEGATIVE",
    "thất vọng": "NEGATIVE", "khó chịu": "NEGATIVE", "tệ": "NEGATIVE", "khủng khiếp": "NEGATIVE",
    "hạnh phúc": "POSITIVE", "vui vẻ": "POSITIVE", "rất vui": "POSITIVE", "không thích": "NEGATIVE",
    "bực mình": "NEGATIVE", "mệt mỏi": "NEGATIVE"
}

# ========================================
# 3. Preprocessing
# ========================================
def preprocess_text(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    if len(text) < 5 or len(text) > 50:
        return None
    # chuẩn hóa teencode
    text_norm = normalize_teencode(text.lower())
    # tokenize underthesea
    try:
        tokens = word_tokenize(text_norm)
    except Exception:
        tokens = text_norm.split()
    if len(tokens) < 2 or len(tokens) > 20:
        return None
    return " ".join(tokens)

# ========================================
# 4. Rule phủ định
# ========================================
def negation_rule(text):
    text_low = text.lower()
    no_acc = remove_accents(text_low)
    if "không " in text_low or "khong " in no_acc:
        positive_words = ["vui", "vui ve", "vui_ve", "tuyet", "tuyệt", "thich", "thích",
                          "yeu", "yêu", "hanh phuc", "hạnh phúc", "hay", "dinh", "đỉnh", "cam on", "cảm ơn"]
        negative_words = ["buon", "buồn", "chan", "chán", "ghet", "ghét", "toi", "tệ", "te", "mệt", "met"]
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
# 6. Model predict
# ========================================
def model_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_id = torch.max(probs, dim=-1)
        confidence = confidence.item()
        predicted_id = predicted_id.item()
    label_map = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}
    label = label_map.get(predicted_id, "NEUTRAL")
    return label, confidence

# ========================================
# 7. Classify sentiment (theo mẫu thầy)
# ========================================
def classify_sentiment(text, threshold=0.5):
    pre = preprocess_text(text)
    if pre is None:
        return None, 0.0
    neg = negation_rule(pre)
    if neg:
        return neg, 0.98
    dic = dict_match(pre)
    if dic:
        return dic, 0.99
    label, confidence = model_predict(pre)
    if confidence < threshold:
        return "NEUTRAL", confidence
    return label, confidence

# ========================================
# 8. SQLite DB
# ========================================
def init_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_result(text, sentiment):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute('INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)',
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
                st.error("Câu không hợp lệ (quá ngắn/dài hoặc không có cảm xúc)!")
            else:
                st.success(f"**Kết quả: {sentiment}** (Độ tin cậy: {score:.2%})")
                save_result(text_input, sentiment)

# Lịch sử (LIMIT 50)
if st.checkbox("Xem lịch sử"):
    conn = sqlite3.connect('history.db')
    df = pd.read_sql_query("SELECT id, text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT 50", conn)
    conn.close()
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("Chưa có dữ liệu.")

# ========================================
# 10. Test 10 case từ utils.test_case
# ========================================
st.sidebar.header("Test Độ Chính Xác")

if st.sidebar.button("Chạy 10 test case"):
    correct = 0
    results = []
    for case in test_cases:
        pred, conf = classify_sentiment(case["text"])
        pred = pred if pred else "NEUTRAL"
        ok = pred.upper() == case["true"].upper()
        if ok:
            correct += 1
        results.append({
            "Câu": case["text"],
            "Dự đoán": pred.upper(),
            "Mong đợi": case["true"].upper(),
            "Kết quả": "✔️ Đúng" if ok else "❌ Sai"
        })
    acc = (correct / len(test_cases)) * 100
    st.sidebar.success(f"Độ chính xác: {acc:.1f}% ({correct}/{len(test_cases)})")
    st.sidebar.dataframe(pd.DataFrame(results))
