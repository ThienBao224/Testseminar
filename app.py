# ========================================
# ĐỒ ÁN: TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT
# Theo hướng dẫn thầy: PhoBERT + Dictionary + Threshold + SQLite LIMIT 50
# (Phiên bản: hợp nhất code của Thầy + các cải tiến của Thiên Bảo)
# - Giữ cấu trúc mẫu thầy (số phần, tên hàm)
# - Thêm: bỏ dấu, chuẩn hoá viết tắt (teencode), rule phủ định, dict ưu tiên
# ========================================

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sqlite3
from datetime import datetime
import pandas as pd
from underthesea import word_tokenize  # theo thầy
import unicodedata
import re

# === LƯU Ý:
# - Model được dùng: "wonrax/phobert-base-vietnamese-sentiment"
# - Nhãn map tuỳ model: ở đây giả định mapping 0:NEGATIVE,1:POSITIVE,2:NEUTRAL
# - Threshold theo thầy: 0.5 -> nếu confidence < 0.5 trả NEUTRAL

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
# 2. Dictionary 25 từ sentiment (theo thầy, có bổ sung 1-2 từ từ code của em)
# ========================================
sentiment_dict = {
    "vui": "POSITIVE", "tuyệt": "POSITIVE", "hay": "POSITIVE", "đỉnh": "POSITIVE", "thích": "POSITIVE",
    "yêu": "POSITIVE", "ok": "NEUTRAL", "ổn": "NEUTRAL", "bình thường": "NEUTRAL", "cũng được": "NEUTRAL",
    "buồn": "NEGATIVE", "chán": "NEGATIVE", "ghét": "NEGATIVE", "tồi": "NEGATIVE", "dở": "NEGATIVE",
    "thất vọng": "NEGATIVE", "khó chịu": "NEGATIVE", "tệ": "NEGATIVE", "khủng khiếp": "NEGATIVE",
    "hạnh phúc": "POSITIVE", "vui vẻ": "POSITIVE", "rất vui": "POSITIVE", "không thích": "NEGATIVE",
    "bực mình": "NEGATIVE", "mệt mỏi": "NEGATIVE"
}
# đảm bảo đúng 25 từ (nếu em muốn đổi, sửa ở đây)

# ========================================
# 3. Preprocessing (theo thầy) + mở rộng của em
#    - chuyển thường
#    - giới hạn ký tự và số từ theo mẫu thầy (tối giản)
#    - dùng underthesea để tokenize
#    - normalize teencode & bỏ dấu cho một số bước kiểm tra
# ========================================

# 3.1 Bỏ dấu (dùng để so sánh cụm không dấu)
def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return str(text)

# 3.2 Map viết tắt / teencode (bản cơ bản của em)
abbrev_map = {
    "ko": "không", "k": "không", "khong": "không", "hok": "không",
    "dc": "được", "dk": "được",
    "cx": "cũng", "vs": "với", "ms": "mới",
    "mik": "mình", "mk": "mình", "bn": "bạn",
    "vl": "rất", "vcl": "rất",
    "okela": "ok", "oki": "ok",
    "bun": "buồn", "bùn": "buồn", "zui": "vui", "dui": "vui", "hihi": "vui", "ra u": "chán",
    "rau": "chán", "gét": "ghét", "get": "ghét"
}

def normalize_abbrev(text):
    # giữ nguyên dấu ở đây, nhưng kiểm tra cả dạng không dấu
    tokens = re.split(r'(\s+)', text)  # preserve spaces
    out = []
    for t in tokens:
        if t.strip() == "":
            out.append(t)
            continue
        key = t.lower()
        key_no = remove_accents(key)
        if key in abbrev_map:
            out.append(abbrev_map[key])
        elif key_no in abbrev_map:
            out.append(abbrev_map[key_no])
        else:
            out.append(t)
    return "".join(out)

# 3.3 Tiền xử lý chính
def preprocess_text(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text_low = text.lower()
    # Giới hạn theo thầy: nếu quá ngắn hoặc quá dài -> None
    if len(text_low) < 5 or len(text_low) > 200:
        return None
    # Chuẩn hóa teencode
    text_norm = normalize_abbrev(text_low)
    # Tokenize bằng underthesea để chuẩn hoá từ
    try:
        tokens = word_tokenize(text_norm)
    except Exception:
        # nếu underthesea có lỗi, fallback: simple split
        tokens = text_norm.split()
    # Giới hạn số từ theo thầy (tùy chỉnh hợp lý)
    if len(tokens) < 2 or len(tokens) > 120:
        return None
    return ' '.join(tokens)

# ========================================
# 4. Phân loại (threshold 0.5 → NEUTRAL) (giữ cơ chế model inference theo mẫu thầy)
# ========================================
def model_predict(text):
    # trả về label (str) và confidence (float)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_id = torch.max(probs, dim=-1)
        confidence = confidence.item()
        predicted_id = predicted_id.item()
    # label map theo thầy/mô hình fine-tune (nếu model dùng nhãn POS/NEG/NEU thì cần chuẩn hoá)
    label_map = {0: "NEGATIVE", 1: "POSITIVE", 2: "NEUTRAL"}
    label = label_map.get(predicted_id, "NEUTRAL")
    return label, confidence

# ========================================
# 5. Rule phủ định (dùng code em đã viết, tích hợp vào flow)
#    Nếu detect "không + positive_word" => NEGATIVE
#    Nếu detect "không + negative_word" => NEUTRAL (theo heuristics của em)
# ========================================
def negation_rule(text):
    text_low = text.lower()
    no_acc = remove_accents(text_low)
    # detect "không <word>" (cả dạng không dấu và có dấu)
    if "không " in text_low or "khong " in no_acc:
        positive_words = ["vui", "vui ve", "vui_vẻ", "tuyet", "tuyệt", "thich", "thích",
                          "yeu", "yêu", "hanh phuc", "hạnh phúc", "hay", "dinh", "đỉnh", "cam on", "cảm ơn"]
        negative_words = ["buon", "buồn", "chan", "chán", "ghet", "ghét", "toi", "tệ", "te"]
        for w in positive_words:
            if f"khong {w}" in no_acc or f"không {w.replace('_',' ')}" in text_low:
                return "NEGATIVE"
        for w in negative_words:
            if f"khong {w}" in no_acc or f"không {w.replace('_',' ')}" in text_low:
                return "NEUTRAL"
    return None

# ========================================
# 6. Dictionary match (ưu tiên, theo mẫu thầy: nếu chứa key -> trả label, confidence cao)
#    - so sánh cả dạng có dấu/không dấu, cụm từ 2-3 từ cũng xét
# ========================================
def dict_match(text):
    if not text:
        return None
    t = text.lower()
    t_no = remove_accents(t)
    # check multi-word keys first
    for key, label in sentiment_dict.items():
        key_norm = key.lower()
        key_no = remove_accents(key_norm)
        if " " in key_norm:
            if key_norm in t or key_no in t_no:
                return label
    # single-word keys
    tokens = t.split()
    tokens_no = t_no.split()
    for key, label in sentiment_dict.items():
        if " " in key:
            continue
        key_norm = key.lower()
        key_no = remove_accents(key_norm)
        if key_norm in tokens or key_no in tokens_no:
            return label
    return None

# ========================================
# 7. Hàm classify chính (kết hợp: preprocess -> negation -> dict -> model -> threshold 0.5)
# ========================================
def classify_sentiment(text, threshold=0.5):
    pre = preprocess_text(text)
    if pre is None:
        return None, 0.0

    # 1) Rule phủ định ưu tiên
    neg = negation_rule(pre)
    if neg:
        return neg, 0.98

    # 2) Dictionary ưu tiên
    dic = dict_match(pre)
    if dic:
        return dic, 0.99

    # 3) Dùng model PhoBERT
    label, confidence = model_predict(pre)

    # 4) Áp threshold theo thầy: nếu confidence < 0.5 => NEUTRAL
    if confidence < threshold:
        return "NEUTRAL", confidence

    return label, confidence

# ========================================
# 8. Khởi tạo DB (SQLite) (theo mẫu thầy)
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

init_db()

# ========================================
# 9. Lưu (parameterized, chống injection)
# ========================================
def save_result(text, sentiment):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute('INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)',
              (text, sentiment, timestamp))
    conn.commit()
    conn.close()

# ========================================
# 10. Giao diện Streamlit (giữ đúng style mẫu thầy)
# ========================================
st.title("Trợ lý Phân loại Cảm xúc Tiếng Việt")
st.markdown("Dùng PhoBERT để phân tích cảm xúc từ văn bản tiếng Việt. (Kết hợp dictionary và rule phủ định)")

text_input = st.text_area("Nhập câu tiếng Việt:", height=120)

if st.button("Phân loại cảm xúc"):
    if not text_input or not text_input.strip():
        st.error("Câu quá ngắn hoặc không hợp lệ!")
    else:
        with st.spinner("Đang phân tích..."):
            sentiment, score = classify_sentiment(text_input)
            if sentiment is None:
                st.error("Câu không hợp lệ (quá ngắn/dài hoặc không có cảm xúc)!")
            else:
                st.success(f"**Kết quả: {sentiment}** (Độ tin cậy: {score:.2%})")
                save_result(text_input, sentiment)

# ========================================
# 11. Lịch sử (LIMIT 50) (theo mẫu thầy)
# ========================================
if st.checkbox("Xem lịch sử"):
    conn = sqlite3.connect('history.db')
    df = pd.read_sql_query("SELECT id, text, sentiment, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT 50", conn)
    conn.close()
    if not df.empty:
        st.dataframe(df)
    else:
        st.info("Chưa có dữ liệu.")

# ========================================
# 12. Test tự động (10 case của thầy)
# ========================================
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
    results = []
    for case in test_cases:
        sentiment, _ = classify_sentiment(case["text"])
        # chuẩn hoá
        pred = sentiment if sentiment is not None else "NEUTRAL"
        pred_norm = pred.upper()
        true_norm = case["true"].upper()
        ok = (pred_norm == true_norm)
        if ok:
            correct += 1
        results.append({"Câu": case["text"], "Dự đoán": pred_norm, "Mong đợi": true_norm, "Kết quả": "ĐÚNG" if ok else "SAI"})
    accuracy = (correct / len(test_cases)) * 100
    st.sidebar.success(f"Độ chính xác: {accuracy:.1f}% ({correct}/{len(test_cases)})")
    st.sidebar.dataframe(pd.DataFrame(results))

# ========================================
# 13. Ghi chú (không bắt buộc hiển thị)
# - File này giữ cấu trúc mẫu thầy (số bước, LIMIT 50, threshold = 0.5),
#   nhưng bổ sung các bước tiền xử lý và rule của Thiên Bảo.
# - Nếu muốn thay đổi threshold về 0.7 (như phiên bản em trước),
#   chỉnh tham số default của classify_sentiment(...)
# - Nếu muốn dùng pipeline() thay vì tokenizers + model, có thể thay đổi ở phần load_model() và model_predict().
# ========================================
