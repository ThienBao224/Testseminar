# =======================================================
# TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT
# PhoBERT fine-tuned + Dictionary + Rule phủ định + SQLite + Testcases
# =======================================================

import streamlit as st
import torch
from transformers import pipeline
import sqlite3
from datetime import datetime
import pandas as pd
import unicodedata

# =======================================================
# 1. HÀM BỎ DẤU
# =======================================================
def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

# =======================================================
# 2. XỬ LÝ VIẾT TẮT
# =======================================================
abbrev_map = {
    "ko": "không", "k": "không", "khong": "không", "hok": "không",
    "dc": "được", "dk": "được",
    "cx": "cũng", "vs": "với", "ms": "mới",
    "mik": "mình", "mk": "mình", "bn": "bạn",
    "vl": "rất", "vcl": "rất",
    "okela": "ok", "oki": "ok",
    "bùn": "buồn", "zui": "vui", "dui": "vui", "hihi": "vui", "rầu": "chán", "gét": "ghét"
}

def normalize_abbrev(text):
    tokens = text.split()
    out = []
    for w in tokens:
        w_no = remove_accents(w)
        if w in abbrev_map:
            out.append(abbrev_map[w])
        elif w_no in abbrev_map:
            out.append(abbrev_map[w_no])
        else:
            out.append(w)
    return " ".join(out)

# =======================================================
# 3. TIỀN XỬ LÝ
# =======================================================
def preprocess(text):
    text = text.lower().strip()
    if len(text) < 2 or len(text) > 120:
        return None
    return normalize_abbrev(text)

# =======================================================
# 4. LOAD PHOBERT FINE-TUNED
# =======================================================
@st.cache_resource
def load_pipeline():
    model_name = "wonrax/phobert-base-vietnamese-sentiment"
    return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

classifier = load_pipeline()

# =======================================================
# 5. DICTIONARY 25 TỪ
# =======================================================
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

# =======================================================
# 6. MATCH DICTIONARY
# =======================================================
def dict_match(text):
    t = text.lower().strip()
    t_no = remove_accents(t)
    tokens = t.split()
    tokens_no = t_no.split()

    # Cụm từ 2-3 từ
    for key, label in sentiment_dict.items():
        key_norm = key.lower()
        key_no = remove_accents(key_norm)
        if " " in key_norm:
            if key_norm in t or key_no in t_no:
                return label

    # Từ đơn
    for key, label in sentiment_dict.items():
        key_norm = key.lower()
        key_no = remove_accents(key_norm)
        if " " not in key_norm:
            if key_norm in tokens or key_no in tokens_no:
                return label
    return None

# =======================================================
# 7. RULE PHỦ ĐỊNH
# =======================================================
def negation_rule(text):
    text_low = text.lower()
    no_acc = remove_accents(text_low)
    if "khong " in no_acc or "không " in text_low:
        positive_words = ["vui", "vui vẻ", "tuyệt", "thích",
                          "yêu", "hạnh phúc", "hay", "đỉnh", "cảm ơn"]
        negative_words = ["buồn", "chán", "ghét", "tồi", "dở",
                          "thất vọng", "khó chịu", "tệ", "mệt", "mệt mỏi"]
        for w in positive_words:
            if f"khong {remove_accents(w)}" in no_acc:
                return "NEGATIVE"
        for w in negative_words:
            if f"khong {remove_accents(w)}" in no_acc:
                return "NEUTRAL"
    return None

# =======================================================
# 8. CHUẨN HÓA NHÃN
# =======================================================
def normalize_label(label):
    label_map = {
        "POS": "POSITIVE",
        "NEG": "NEGATIVE",
        "NEU": "NEUTRAL",
        "POSITIVE": "POSITIVE",
        "NEGATIVE": "NEGATIVE",
        "NEUTRAL": "NEUTRAL"
    }
    return label_map.get(label.upper(), label.upper())

# =======================================================
# 9. PHÂN LOẠI SENTIMENT
# =======================================================
def classify_sentiment(text, threshold=0.7):
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
    label = normalize_label(result['label'])   # chuẩn hóa
    confidence = result['score']

    # Câu ngắn + confidence thấp → NEUTRAL
    if len(clean.split()) <= 5 and confidence < threshold:
        label = "NEUTRAL"

    return label, confidence

# =======================================================
# 10. SQLITE
# =======================================================
def init_db():
    conn = sqlite3.connect("history.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            sentiment TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_result(text, sentiment):
    conn = sqlite3.connect("history.db")
    timestamp = datetime.now().isoformat()
    conn.execute("INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
                 (text, sentiment, timestamp))
    conn.commit()
    conn.close()

init_db()

# =======================================================
# 11. STREAMLIT UI
# =======================================================
st.title("Trợ lý phân loại cảm xúc tiếng Việt")

text = st.text_area("Nhập câu văn:", height=100)

if st.button("Phân tích cảm xúc"):
    sent, conf = classify_sentiment(text)
    if sent is None:
        st.error("Câu quá ngắn hoặc không hợp lệ!")
    else:
        st.success(f"Kết quả: **{sent}** (Độ tin cậy: {conf*100:.1f}%)")
        save_result(text, sent)

# Lịch sử
if st.checkbox("Xem lịch sử (50 gần nhất)"):
    df = pd.read_sql_query(
        "SELECT id, text, sentiment, timestamp FROM sentiments ORDER BY id DESC LIMIT 50",
        sqlite3.connect("history.db")
    )
    st.dataframe(df)

# =======================================================
# 12. TESTCASE
# =======================================================
test_cases = [
    {"text": "Hôm nay tôi rất vui", "expected": "POSITIVE"},
    {"text": "Món ăn này dở quá", "expected": "NEGATIVE"},
    {"text": "Thời tiết bình thường", "expected": "NEUTRAL"},
    {"text": "Rat vui hom nay", "expected": "POSITIVE"},
    {"text": "Công việc ổn định", "expected": "NEUTRAL"},
    {"text": "Phim này hay lắm", "expected": "POSITIVE"},
    {"text": "Tôi buồn vì thất bại", "expected": "NEGATIVE"},
    {"text": "Ngày mai đi học", "expected": "NEUTRAL"},
    {"text": "Cảm ơn bạn rất nhiều", "expected": "POSITIVE"},
    {"text": "Mệt mỏi quá hôm nay", "expected": "NEGATIVE"},
    {"text": "Hom nay toi rat vui", "expected": "POSITIVE"},
    {"text": "Mon an nay do qua", "expected": "NEGATIVE"},
    {"text": "Thoi tiet binh thuong", "expected": "NEUTRAL"},
    {"text": "Rat vui hom nay", "expected": "POSITIVE"},
    {"text": "Cong viec on dinh", "expected": "NEUTRAL"},
    {"text": "Phim nay hay lam", "expected": "POSITIVE"},
    {"text": "Toi buon vi that bai", "expected": "NEGATIVE"},
    {"text": "Ngay mai di hoc", "expected": "NEUTRAL"},
    {"text": "Cam on ban rat nhieu", "expected": "POSITIVE"},
    {"text": "Met moi qua hom nay", "expected": "NEGATIVE"},
]

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
    st.sidebar.dataframe(pd.DataFrame(results))
