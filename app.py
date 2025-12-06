# ========================================
# ĐỒ ÁN: TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT - FINAL
# Kết hợp: PhoBERT (AutoModel) + Dictionary + Rule phủ định +
#          Normalize teencode + underthesea + SQLite + Testcases
# ========================================

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sqlite3
from datetime import datetime
import pandas as pd
from underthesea import word_tokenize
import unicodedata
from utils.test_case import test_cases           # file testcases tách riêng
from utils.teencode_dict import normalize_teencode  # hàm bạn đã chuẩn hóa

# ---------------------------
# 1. Load PhoBERT (AutoModel) - cached
# ---------------------------
@st.cache_resource
def load_model():
    model_name = "wonrax/phobert-base-vietnamese-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------
# 2. Dictionary 25 từ
# ---------------------------
sentiment_dict = {
    "vui": "POSITIVE", "tuyệt": "POSITIVE", "hay": "POSITIVE", "đỉnh": "POSITIVE", "thích": "POSITIVE",
    "yêu": "POSITIVE", "ok": "NEUTRAL", "ổn": "NEUTRAL", "bình thường": "NEUTRAL", "cũng được": "NEUTRAL",
    "buồn": "NEGATIVE", "chán": "NEGATIVE", "ghét": "NEGATIVE", "tồi": "NEGATIVE", "dở": "NEGATIVE",
    "thất vọng": "NEGATIVE", "khó chịu": "NEGATIVE", "tệ": "NEGATIVE", "khủng khiếp": "NEGATIVE",
    "hạnh phúc": "POSITIVE", "vui vẻ": "POSITIVE", "rất vui": "POSITIVE", "không thích": "NEGATIVE",
    "bực mình": "NEGATIVE", "mệt mỏi": "NEGATIVE"
}

# ---------------------------
# Helpers: remove accents, normalize label
# ---------------------------
def remove_accents(text: str) -> str:
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

def normalize_label(label):
    """Chuẩn hoá nhãn đầu ra/expected thành POSITIVE/NEGATIVE/NEUTRAL"""
    if label is None:
        return "NONE"
    s = str(label).strip().upper()
    if s.startswith("POS"):
        return "POSITIVE"
    if s.startswith("NEG"):
        return "NEGATIVE"
    if s.startswith("NEU") or s == "NONE":
        return "NEUTRAL" if s.startswith("NEU") else "NONE"
    # nếu là số id
    if s == "0":
        return "NEGATIVE"
    if s == "1":
        return "POSITIVE"
    if s == "2":
        return "NEUTRAL"
    return s

# ---------------------------
# Match dictionary (cụm từ trước, từ đơn sau) - hỗ trợ no-accent
# ---------------------------
def dict_match(text: str):
    t = text.lower().strip()
    t_no = remove_accents(t)
    tokens = t.split()
    tokens_no = t_no.split()

    # cụm nhiều từ trước
    for key, label in sentiment_dict.items():
        key_norm = key.lower()
        key_no = remove_accents(key_norm)
        if " " in key_norm:
            if key_norm in t or key_no in t_no:
                return label

    # từ đơn
    for key, label in sentiment_dict.items():
        key_norm = key.lower()
        key_no = remove_accents(key_norm)
        if " " not in key_norm:
            if key_norm in tokens or key_no in tokens_no:
                return label
    return None

# ---------------------------
# Rule phủ định (ví dụ: "không vui" -> NEGATIVE)
# ---------------------------
def negation_rule(text: str):
    text_low = text.lower()
    no_acc = remove_accents(text_low)
    # tìm "khong " không dấu hoặc "không " có dấu
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

# ---------------------------
# 3. Preprocessing: normalize teencode -> lowercase -> tokenization -> length checks
# ---------------------------
def preprocess_text(text: str):
    if text is None:
        return None

    # 1) normalize teencode (từ utils)
    try:
        text = normalize_teencode(text)
    except Exception:
        # fallback: nếu import normalize_teencode có vấn đề, dùng nguyên input
        pass

    text = text.lower().strip()

    # giới hạn ký tự (theo thầy)
    if len(text) < 5 or len(text) > 120:  # hơi nới hạn tối đa để cover teencode dài
        return None

    # tokenize bằng underthesea
    try:
        words = word_tokenize(text)
    except Exception:
        # fallback: split nếu underthesea lỗi
        words = text.split()

    if len(words) < 2 or len(words) > 40:  # giới hạn từ (nới chút để cover)
        return None

    # trả về chuỗi đã token (join) để dễ match dictionary
    return " ".join(words)

# ---------------------------
# 4. Classification: dictionary -> negation -> model -> threshold
# ---------------------------
def classify_sentiment(text: str, threshold=0.5):
    """Trả về (label, confidence). Nếu không hợp lệ trả về (None, 0.0)."""
    clean = preprocess_text(text)
    if clean is None:
        return None, 0.0

    # 1) rule phủ định (dùng trên bản tokenized text)
    neg_label = negation_rule(clean)
    if neg_label:
        return normalize_label(neg_label), 0.98

    # 2) dictionary ưu tiên
    dic_label = dict_match(clean)
    if dic_label:
        return normalize_label(dic_label), 0.99

    # 3) model
    try:
        inputs = tokenizer(clean, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidence = float(torch.max(probs).item())
            pred_id = int(torch.argmax(probs).item())
    except Exception as e:
        # nếu model lỗi, trả NEUTRAL an toàn
        return "NEUTRAL", 0.0

    # threshold quy ước
    if confidence < threshold:
        return "NEUTRAL", confidence

    id2label = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}
    return id2label.get(pred_id, "NEUTRAL"), confidence

# ---------------------------
# 5. SQLite init & save
# ---------------------------
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

def save_result(text: str, sentiment: str):
    conn = sqlite3.connect("history.db")
    timestamp = datetime.now().isoformat()
    conn.execute("INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
                 (text, sentiment, timestamp))
    conn.commit()
    conn.close()

init_db()

# ---------------------------
# 6. Streamlit UI
# ---------------------------
st.set_page_config(page_title="Trợ lý Phân loại Cảm xúc Tiếng Việt", layout="centered")
st.title("Trợ lý Phân loại Cảm xúc Tiếng Việt")
st.markdown("Dùng **PhoBERT** + dictionary + rule phủ định + teencode normalization để phân tích cảm xúc tiếng Việt.")

text_input = st.text_area("Nhập câu tiếng Việt:", height=140, placeholder="Gõ câu vào đây...")

col1, col2 = st.columns([2, 1])
with col2:
    if st.button("Phân loại cảm xúc"):
        if not text_input or not text_input.strip():
            st.error("Câu quá ngắn hoặc không hợp lệ!")
        else:
            with st.spinner("Đang phân tích..."):
                label, score = classify_sentiment(text_input)
                if label is None:
                    st.error("Câu không hợp lệ (quá ngắn/dài hoặc số từ không đúng).")
                else:
                    st.success(f"**Kết quả:** {label}  •  (Độ tin cậy: {score*100:.1f}%)")
                    save_result(text_input, label)

# history
if st.checkbox("Xem lịch sử (50 gần nhất)"):
    conn = sqlite3.connect("history.db")
    df = pd.read_sql_query("SELECT id, text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT 50", conn)
    conn.close()
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("Chưa có dữ liệu.")

# ---------------------------
# Sidebar: kiểm thử
# ---------------------------
st.sidebar.header("Kiểm thử mô hình (Testcases)")

def run_tests_and_show():
    correct = 0
    results = []
    for case in test_cases:
        pred, conf = classify_sentiment(case["text"])
        pred_norm = normalize_label(pred)
        expected_norm = normalize_label(case.get("expected") or case.get("true"))
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
    acc = correct / len(test_cases) * 100 if test_cases else 0.0
    return correct, acc, results

if st.sidebar.button("Chạy kiểm thử"):
    correct, acc, results = run_tests_and_show()
    st.sidebar.success(f"🎉 Tổng: {correct}/{len(test_cases)} = {acc:.1f}%")
    st.subheader("📊 Kết quả chi tiết từng test")
    st.dataframe(pd.DataFrame(results))

