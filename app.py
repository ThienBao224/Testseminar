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
from utils.teencode_dict import normalize_teencode  # Giữ lại theo yêu cầu
from utils.test_case import test_cases  # Test case từ file riêng

# === 1. Tải mô hình PhoBERT (theo thầy) ===
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
    "bực mình": "NEGATIVE", "mệt mỏi": "NEGATIVE"
}

# === 3. Preprocessing (theo thầy + normalize teencode) ===
def preprocess_text(text):
    text = normalize_teencode(text)  # Giữ nguyên theo yêu cầu bạn
    text = text.lower()

    # Giới hạn ký tự (theo thầy)
    if len(text) < 5 or len(text) > 50:
        return None

    words = word_tokenize(text)

    # Giới hạn số từ (theo thầy)
    if len(words) < 2 or len(words) > 20:
        return None

    return ' '.join(words)

# === 4. Phân loại (dictionary → model → threshold 0.5) ===
def classify_sentiment(text):
    preprocessed = preprocess_text(text)
    if preprocessed is None:
        return None, 0

    # Kiểm tra dictionary trước
    for word, label in sentiment_dict.items():
        if word in preprocessed:
            return label, 0.99  # High confidence dictionary

    # Dùng model (PhoBERT)
    inputs = tokenizer(preprocessed, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence = torch.max(probs).item()
        predicted_id = torch.argmax(probs).item()

    # Threshold (theo thầy)
    if confidence < 0.5:
        return "NEUTRAL", confidence

    label_map = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}
    return label_map[predicted_id], confidence

# === 5. Khởi tạo DB SQLite ===
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

# === 6. Lưu kết quả (parameterized query) ===
def save_result(text, sentiment):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute(
        'INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)',
        (text, sentiment, timestamp)
    )
    conn.commit()
    conn.close()

# === 7. Giao diện Streamlit ===
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

# === 8. Hiển thị lịch sử 50 dòng mới nhất ===
if st.checkbox("Xem lịch sử"):
    conn = sqlite3.connect('history.db')
    df = pd.read_sql_query(
        "SELECT id, text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT 50",
        conn
    )
    conn.close()

    if not df.empty:
        st.dataframe(df)
    else:
        st.info("Chưa có dữ liệu.")

# === 9. Sidebar – Test 10 test case ===
st.sidebar.header("Kiểm thử mô hình")

# Chuẩn hóa label cho an toàn (POSITIVE/positive/Positive → POSITIVE)
def normalize_label(label: str):
    return label.strip().upper()

if st.sidebar.button("Chạy kiểm thử"):
    correct = 0
    results = []

    for case in test_cases:
        pred, conf = classify_sentiment(case["text"])
        pred_norm = normalize_label(pred)
        expected_norm = normalize_label(case["expected"])
        ok = (pred_norm == expected_norm)

        if ok:
            correct += 1

        results.append({
            "Câu": case["text"],
            "Dự đoán": pred_norm,
            "Độ tin cậy": f"{conf*100:.1f}%",
            "Mong đợi": expected_norm,
            "Kết quả": "✔️ Đúng" if ok else "❌ Sai"
        })

    acc = correct / len(test_cases) * 100
    st.sidebar.success(f"🎉 Kết quả: {correct}/{len(test_cases)} = {acc:.1f}%")

    # Hiển thị chi tiết ở main area để không bị chật sidebar
    st.subheader("📊 Kết quả chi tiết từng test")
    st.dataframe(pd.DataFrame(results))
