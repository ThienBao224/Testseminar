# ========================================
# ĐỒ ÁN: TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT
# PhoBERT + Dictionary + Threshold + SQLite LIMIT 50
# ========================================

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sqlite3
from datetime import datetime
import pandas as pd
from underthesea import word_tokenize
from utils.teencode_dict import normalize_teencode
from utils.test_case import test_cases

# === 1. Load model PhoBERT ===
@st.cache_resource
def load_model():
    model_name = "wonrax/phobert-base-vietnamese-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# === 2. Dictionary sentiment ===
sentiment_dict = {
    "vui": "POSITIVE", "tuyệt": "POSITIVE", "hay": "POSITIVE", "đỉnh": "POSITIVE", "thích": "POSITIVE",
    "yêu": "POSITIVE", "ok": "NEUTRAL", "ổn": "NEUTRAL", "bình thường": "NEUTRAL", "cũng được": "NEUTRAL",
    "buồn": "NEGATIVE", "chán": "NEGATIVE", "ghét": "NEGATIVE", "tồi": "NEGATIVE", "dở": "NEGATIVE",
    "thất vọng": "NEGATIVE", "khó chịu": "NEGATIVE", "tệ": "NEGATIVE", "khủng khiếp": "NEGATIVE",
    "hạnh phúc": "POSITIVE", "vui vẻ": "POSITIVE", "rất vui": "POSITIVE", "không thích": "NEGATIVE",
    "bực mình": "NEGATIVE", "mệt mỏi": "NEGATIVE"
}

# === 3. Preprocessing ===
def preprocess_text(text):
    text = normalize_teencode(text)      # DÙNG TEENCODE MỚI
    text = text.lower().strip()

    if len(text) < 5 or len(text) > 50:
        return None

    words = word_tokenize(text)

    if len(words) < 2 or len(words) > 20:
        return None

    return " ".join(words)

# === 4. Classify sentiment ===
def classify_sentiment(text):
    pre = preprocess_text(text)
    if pre is None:
        return None, 0

    # Dictionary trước
    for word, label in sentiment_dict.items():
        if word in pre:
            return label, 0.99

    # Model PhoBERT
    inputs = tokenizer(pre, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf = torch.max(probs).item()
        pred_id = torch.argmax(probs).item()

    if conf < 0.5:
        return "NEUTRAL", conf

    label_map = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}
    return label_map[pred_id], conf

# === 5. Init DB ===
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

# === 6. Save result ===
def save_result(text, label):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    t = datetime.now().isoformat()
    c.execute("INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)", 
              (text, label, t))
    conn.commit()
    conn.close()

# === 7. Streamlit UI ===
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

# === 8. Lịch sử ===
if st.checkbox("Xem lịch sử"):
    conn = sqlite3.connect("history.db")
    df = pd.read_sql_query(
        "SELECT id, text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT 50",
        conn
    )
    conn.close()

    if df.empty:
        st.info("Chưa có dữ liệu.")
    else:
        st.dataframe(df)

# === 9. Test case ===

test_container = st.sidebar.container()   # Tạo khung cố định trong sidebar

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

    # ==== TẤT CẢ KẾT QUẢ ĐẦU RA NẰM TRONG SIDEBAR ====
    with test_container:
        st.success(f"🎉 Kết quả: {correct}/{len(test_cases)} = {acc:.1f}%")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
