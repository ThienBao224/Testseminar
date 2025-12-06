

# ========================================
# ĐỒ ÁN: TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT
# Theo hướng dẫn thầy: PhoBERT + Dictionary + Threshold + SQLite LIMIT 50
# ========================================

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sqlite3
from datetime import datetime
import pandas as pd
from underthesea import word_tokenize  # Theo thầy: underthesea
import threading  # Để tránh treo UI

# === 1. Tải mô hình (PhoBERT theo thầy) ===
@st.cache_resource
def load_model():
    model_name = "wonrax/phobert-base-vietnamese-sentiment"  # PhoBERT fine-tune VN
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# === 2. Dictionary 25 từ sentiment (theo thầy) ===
sentiment_dict = {
    "vui": "POSITIVE", "tuyệt": "POSITIVE", "hay": "POSITIVE", "đỉnh": "POSITIVE", "thích": "POSITIVE",
    "yêu": "POSITIVE", "ok": "NEUTRAL", "ổn": "NEUTRAL", "bình thường": "NEUTRAL", "cũng được": "NEUTRAL",
    "buồn": "NEGATIVE", "chán": "NEGATIVE", "ghét": "NEGATIVE", "tồi": "NEGATIVE", "dở": "NEGATIVE",
    "thất vọng": "NEGATIVE", "khó chịu": "NEGATIVE", "tệ": "NEGATIVE", "khủng khiếp": "NEGATIVE",
    "hạnh phúc": "POSITIVE", "vui vẻ": "POSITIVE", "rất vui": "POSITIVE", "không thích": "NEGATIVE",
    "bực mình": "NEGATIVE", "mệt mỏi": "NEGATIVE"  # Đủ 25
}

# === 3. Preprocessing (theo thầy) ===
def preprocess_text(text):
    text = text.lower()
    if len(text) < 5 or len(text) > 50:  # Giới hạn ký tự
        return None  # Không hợp lệ
    words = word_tokenize(text)  # underthesea
    if len(words) < 2 or len(words) > 20:  # Giới hạn từ
        return None
    return ' '.join(words)

# === 4. Phân loại (threshold 0.5 → NEUTRAL) ===
def classify_sentiment(text):
    preprocessed = preprocess_text(text)
    if preprocessed is None:
        return None, 0

    # Kiểm tra dictionary trước
    for word, label in sentiment_dict.items():
        if word in preprocessed:
            return label, 0.99  # High confidence

    # Dùng model
    inputs = tokenizer(preprocessed, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence = torch.max(probs).item()
        predicted_id = torch.argmax(probs).item()
    
    if confidence < 0.5:  # Theo thầy
        return "NEUTRAL", confidence
    
    label_map = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}
    return label_map[predicted_id], confidence

# === 5. Khởi tạo DB ===
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

init_db()

# === 6. Lưu (parameterized, chống injection) ===
def save_result(text, sentiment):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()  # ISO format theo thầy
    c.execute('INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)',
              (text, sentiment, timestamp))
    conn.commit()
    conn.close()

# === 7. Giao diện ===
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

# === 8. Lịch sử (LIMIT 50) ===
if st.checkbox("Xem lịch sử"):
    conn = sqlite3.connect('history.db')
    df = pd.read_sql_query("SELECT id, text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT 50", conn)
    conn.close()
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("Chưa có dữ liệu.")

# === 9. Test tự động (10 case của thầy) ===
st.sidebar.header("Test Độ Chính Xác")
test_cases = [
    {"text": "Hôm nay tôi rất vui", "true": "POSITIVE"},
    {"text": "Món ăn nay dở quá", "true": "NEGATIVE"},
    {"text": "Thời tiết bình thường", "true": "NEUTRAL"},
    {"text": "Rat vui hom nay", "true": "POSITIVE"},
    {"text": "Công việc ổn định", "true": "NEUTRAL"},
    {"text": "Phim nay hay lâm", "true": "POSITIVE"},
    {"text": "Tồi buồn vi thất bại", "true": "NEGATIVE"},
    {"text": "Ngây mai di học", "true": "NEUTRAL"},
    {"text": "Cam on ban rat nhieu", "true": "POSITIVE"},
    {"text": "Mệt mỏi quá hôm nay", "true": "NEGATIVE"}
]

if st.sidebar.button("Chạy 10 test case"):
    correct = 0
    for case in test_cases:
        sentiment, _ = classify_sentiment(case["text"])
        if sentiment == case["true"]:
            correct += 1
    accuracy = (correct / 10) * 100
    st.sidebar.success(f"Độ chính xác: {accuracy:.1f}% ({correct}/10)")

