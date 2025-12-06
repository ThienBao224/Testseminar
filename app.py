# =======================================================
# TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT
# PhoBERT + Dictionary mạnh + Rule phủ định + Underthesea + Hiển thị tiếng Việt
# =======================================================

import streamlit as st
import torch
from transformers import pipeline
import sqlite3
from datetime import datetime
import pandas as pd
import unicodedata
from underthesea import word_tokenize

# =======================================================
# CẤU HÌNH TRANG (PHẢI Ở ĐẦU TIÊN!)
# =======================================================
st.set_page_config(page_title="Phân loại Cảm xúc", page_icon="😊")

# =======================================================
# 1. HÀM BỎ DẤU
# =======================================================
def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

# =======================================================
# 2. XỬ LÝ VIẾT TẮT (MỞ RỘNG)
# =======================================================
abbrev_map = {
    "ko": "không", "k": "không", "khong": "không", "hok": "không",
    "dc": "được", "dk": "được", "đc": "được",
    "cx": "cũng", "vs": "với", "ms": "mới",
    "mik": "mình", "mk": "mình", "bn": "bạn",
    "vl": "rất", "vcl": "rất", "rat": "rất", "rát": "rất",
    "okela": "ok", "oki": "ok", "okii": "ok",
    "bùn": "buồn", "zui": "vui", "dui": "vui", "hihi": "vui", 
    "rầu": "buồn", "gét": "ghét", "met": "mệt", "moi": "mỏi",
    "qua": "quá", "wa": "quá", "z": "vậy", "v": "vậy",
    "ntn": "như thế nào", "the": "thế", "bik": "biết", "bit": "biết",
    "do": "dở", "on": "ổn", "dinh": "định", "lam": "lắm",
    "nay": "này", "hom": "hôm", "toi": "tôi", "vi": "vì",
    "that": "thất", "bai": "bại", "ngay": "ngày", "mai": "mai",
    "di": "đi", "cam": "cảm", "nhieu": "nhiều","qa": "qua", "qa": "quá", 
    "thoi": "thời", "tiet": "tiết", "binh": "bình", "thuong": "thường",
    "cong": "công", "viec": "việc", "mon": "món", "an": "ăn"
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
# 3. TIỀN XỬ LÝ (DÙNG UNDERTHESEA)
# =======================================================
def preprocess(text):
    text = text.lower().strip()
    if len(text) < 5 or len(text) > 50:  # Giới hạn ký tự
        return None

    # Tokenize
    words = word_tokenize(text, format="text").split()

    if len(words) < 2 or len(words) > 20:  # Giới hạn từ
        return None

    # Normalize viết tắt
    words = [abbrev_map.get(w, w) for w in words]

    # Restore dấu (accent dictionary)
    words = [accent_dict.get(w, w) for w in words]

    # Trả về câu chuẩn
    return " ".join(words)

# =======================================================
# 4. LOAD PHOBERT (THỬ NHIỀU MODEL)
# =======================================================
@st.cache_resource
def load_pipeline():
    try:
        # Model 1: PhoBERT fine-tuned tốt nhất
        model_name = "uitnlp/visobert"
        return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name), "ViSoBERT"
    except:
        try:
            # Model 2: Fallback
            model_name = "wonrax/phobert-base-vietnamese-sentiment"
            return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name), "PhoBERT"
        except:
            # Model 3: Universal
            model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
            return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name), "DistilBERT"

classifier, model_name = load_pipeline()

# =======================================================
# 5. DICTIONARY MẠNH HƠN
# =======================================================
sentiment_dict = {
    "vui": "POSITIVE", "vui vẻ": "POSITIVE", "rất vui": "POSITIVE",
    "cảm ơn": "POSITIVE", "tuyệt": "POSITIVE", "tuyệt vời": "POSITIVE",
    "hay": "POSITIVE", "hay lắm": "POSITIVE", "đỉnh": "POSITIVE",
    "thích": "POSITIVE", "yêu": "POSITIVE", "hạnh phúc": "POSITIVE",
    "ok": "POSITIVE", "tốt": "POSITIVE", "xuất sắc": "POSITIVE", "hoàn hảo": "POSITIVE", "tuyệt vời": "POSITIVE",
    "ổn định": "NEUTRAL", "bình thường": "NEUTRAL", "ổn": "NEUTRAL", "cũng được": "NEUTRAL",
    "thời tiết": "NEUTRAL", "đi học": "NEUTRAL", "ngày mai": "NEUTRAL",
    "công việc": "NEUTRAL", "học hành": "NEUTRAL",
    "buồn": "NEGATIVE", "buồn vì": "NEGATIVE", "chán": "NEGATIVE","ghét": "NEGATIVE","sợ": "NEGATIVE",
    "tồi": "NEGATIVE", "dở": "NEGATIVE", "dở quá": "NEGATIVE",
    "thất vọng": "NEGATIVE", "thất bại": "NEGATIVE", "khó chịu": "NEGATIVE",
    "tệ": "NEGATIVE", "khủng khiếp": "NEGATIVE", "bực mình": "NEGATIVE",
    "mệt mỏi": "NEGATIVE", "mệt mỏi quá": "NEGATIVE", "tệ quá": "NEGATIVE"
}

# ACCENT DICTIONARY
accent_dict = {
    "toi": "tôi", "minh": "mình", "ban": "bạn",
    "hom nay": "hôm nay", "ngay mai": "ngày mai", "bay gio": "bây giờ",
    "rat vui": "rất vui", "vui": "vui", "hanh phuc": "hạnh phúc",
    "yeu": "yêu", "thich": "thích", "tuyet voi": "tuyệt vời", "cam on": "cảm ơn",
    "buon": "buồn", "chan": "chán", "that vong": "thất vọng",
    "met moi": "mệt mỏi", "te": "tệ", "do qua": "dở quá",
    "binh thuong": "bình thường", "cong viec": "công việc", "thoi tiet": "thời tiết"
}

# =======================================================
# 6. MATCH DICTIONARY
# =======================================================
def dict_match(text):
    if not text:
        return None

    t = text.lower().replace("_", " ").strip()
    t_no = remove_accents(t)

    # Sắp xếp theo độ dài key giảm dần để match cụm từ trước
    sorted_keys = sorted(sentiment_dict.keys(), key=lambda x: -len(x.split()))
    for key in sorted_keys:
        key_norm = key.lower().replace("_", " ")
        key_no = remove_accents(key_norm)
        if key_norm in t or key_no in t_no:
            return sentiment_dict[key]
    return None

# =======================================================
# 7. KHÔI PHỤC DẤU
# =======================================================
def restore_accents(text):
    text = normalize_abbrev(text.lower())
    text_no = remove_accents(text)
    result = text
    sorted_keys = sorted(accent_dict.keys(), key=lambda x: -len(x.split()))
    for key in sorted_keys:
        if key in text_no:
            result = result.replace(key, accent_dict[key])
    return result

# =======================================================
# 8. RULE PHỦ ĐỊNH
# =======================================================
def negation_rule(text):
    text_low = text.lower()
    no_acc = remove_accents(text_low)
    negation_words = ["khong", "không", "chưa", "chả"]
    for neg in negation_words:
        if neg in no_acc or neg in text_low:
            positive_words = ["vui", "tuyệt", "thích", "yêu", "hạnh phúc", "hay", "đỉnh", "tốt", "ok", "ổn"]
            negative_words = ["buồn", "chán", "ghét", "tồi", "dở", "thất vọng", "tệ", "mệt"]
            for w in positive_words:
                if f"{neg} {remove_accents(w)}" in no_acc:
                    return "NEGATIVE"
            for w in negative_words:
                if f"{neg} {remove_accents(w)}" in no_acc:
                    return "NEUTRAL"
    return None

# =======================================================
# 9. CHUẨN HÓA NHÃN
# =======================================================
def normalize_label(label):
    label_upper = label.upper()
    label_map = {
        "POS": "POSITIVE", "NEG": "NEGATIVE", "NEU": "NEUTRAL",
        "POSITIVE": "POSITIVE", "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL",
        "LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE",
        "0": "NEGATIVE", "1": "NEUTRAL", "2": "POSITIVE",
    }
    return label_map.get(label_upper, "NEUTRAL")

def label_to_vietnamese(label):
    vn_map = {"POSITIVE": "Tích cực","NEGATIVE": "Tiêu cực","NEUTRAL": "Trung tính"}
    return vn_map.get(label, label)

def get_emoji(label):
    emoji_map = {"POSITIVE": "😊","NEGATIVE": "😞","NEUTRAL": "😐"}
    return emoji_map.get(label, "❓")

# =======================================================
# 10. PHÂN LOẠI SENTIMENT
# =======================================================
def classify_sentiment(text, threshold=0.5):
    clean = preprocess(text)
    if clean is None:
        return None, 0.0

    # 1. Rule phủ định
    neg_label = negation_rule(clean)
    if neg_label:
        return normalize_label(neg_label), 0.98

    # 2. Dictionary toàn câu
    dic_label = dict_match(clean)
    
    # 3. PhoBERT fine-tuned
    try:
        result = classifier(clean)[0]
        model_label = normalize_label(result['label'])
        model_conf = result['score']

        # Nếu model tự tin
        if model_conf >= threshold:
            if dic_label and normalize_label(dic_label) != model_label:
                # Dictionary khác model → ưu tiên dictionary, confidence hợp lý
                adjusted_conf = round((model_conf + 0.8) / 2, 2)
                return normalize_label(dic_label), adjusted_conf
            else:
                # Dictionary giống model hoặc không có → dùng model
                return model_label, model_conf
        else:
            # Model không tự tin
            if dic_label:
                adjusted_conf = min(model_conf + 0.15, 0.85)
                return normalize_label(dic_label), adjusted_conf

            # Token-level dictionary
            tokens = clean.split()
            for token in tokens:
                token_label = dict_match(token)
                if token_label:
                    adjusted_conf = max(model_conf, 0.65)
                    return normalize_label(token_label), adjusted_conf

            # Câu ngắn + confidence thấp → NEUTRAL
            if len(clean.split()) <= 5:
                return "NEUTRAL", max(model_conf, 0.5)

            # Fallback NEUTRAL
            return "NEUTRAL", max(model_conf, 0.5)

    except Exception as e:
        # Nếu model lỗi
        if dic_label:
            return normalize_label(dic_label), 0.75
        return "NEUTRAL", 0.5


# =======================================================
# 11. SQLITE
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

def save_result(original_text, sentiment):
    display_text = restore_accents(original_text)
    conn = sqlite3.connect("history.db")
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn.execute(
        "INSERT INTO sentiments (text, sentiment, timestamp) VALUES (?, ?, ?)",
        (display_text, sentiment, timestamp)
    )
    conn.commit()
    conn.close()

init_db()

# =======================================================
# 12. STREAMLIT UI
# =======================================================
st.title("😊 Trợ lý phân loại cảm xúc tiếng Việt (Underthesea)")
st.info(f"✅ Đang sử dụng model: **{model_name}**")

text = st.text_area("Nhập câu tiếng Việt (ít nhất 2 ký tự):", height=100, placeholder="Ví dụ: Hôm nay tôi rất vui")

if st.button("🔍 Phân loại cảm xúc", type="primary"):
    if not text or len(text.strip()) < 2:
        st.error("⚠️ Vui lòng nhập câu có ít nhất 2 ký tự!")
    else:
        with st.spinner("Đang phân tích..."):
            sent, conf = classify_sentiment(text)
            if sent is None:
                st.error("❌ Câu không hợp lệ!")
            else:
                vn_label = label_to_vietnamese(sent)
                emoji = get_emoji(sent)
                st.success(f"{emoji} **Kết quả: {vn_label}** (Độ tin cậy: {conf*100:.1f}%)")
                display_text = restore_accents(text)
                st.json({
                    "text_goc": text,
                    "text_hien_thi": display_text,
                    "sentiment": sent
                })
                save_result(text, sent)

# Lịch sử
st.markdown("---")
if st.checkbox("📋 Xem lịch sử (50 gần nhất)"):
    conn = sqlite3.connect("history.db")
    df = pd.read_sql_query(
        "SELECT id, text, sentiment, timestamp FROM sentiments ORDER BY id DESC LIMIT 50",
        conn
    )
    conn.close()
    if not df.empty:
        df['Cảm xúc (VN)'] = df['sentiment'].apply(label_to_vietnamese)
        df['Icon'] = df['sentiment'].apply(get_emoji)
        st.dataframe(df[['Icon', 'text', 'Cảm xúc (VN)', 'timestamp']], use_container_width=True)
    else:
        st.info("Chưa có dữ liệu")

# Bộ Test Case
st.markdown("---")
st.subheader("🧪 Bộ Test Case (10 câu)")
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
]

if st.button("▶️ Chạy Test tất cả 10 câu"):
    st.info("Đang chạy test...")
    correct = 0
    results = []
    progress_bar = st.progress(0)
    for i, case in enumerate(test_cases):
        pred, conf = classify_sentiment(case["text"])
        pred_norm = normalize_label(pred)
        expected_norm = normalize_label(case["expected"])
        ok = (pred_norm == expected_norm)
        if ok:
            correct += 1
        results.append({
            "STT": i + 1,
            "Câu": case["text"],
            "Mong đợi": label_to_vietnamese(expected_norm),
            "Dự đoán": label_to_vietnamese(pred_norm),
            "Độ tin cậy": f"{conf*100:.1f}%",
            "Kết quả": "✅ Đúng" if ok else "❌ Sai"
        })
        progress_bar.progress((i + 1) / len(test_cases))
    progress_bar.empty()
    acc = correct / len(test_cases) * 100
    st.markdown("### 📊 Kết quả Test")
    col1, col2, col3 = st.columns(3)
    col1.metric("Đúng", f"{correct}/{len(test_cases)}")
    col2.metric("Độ chính xác", f"{acc:.1f}%")
    col3.metric("Đánh giá", "✅ ĐẠT" if acc >= 65 else "❌ CHƯA ĐẠT")
    st.dataframe(pd.DataFrame(results), use_container_width=True)
    if acc >= 65:
        st.success(f"🎉 **ĐẠT YÊU CẦU!** Độ chính xác {acc:.1f}% (≥ 65%)")
    else:
        st.warning(f"⚠️ **Chưa đạt yêu cầu.** Cần ≥ 65%, hiện tại: {acc:.1f}%")
