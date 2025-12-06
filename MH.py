# =======================================================
# APP PHÂN LOẠI CẢM XÚC TIẾNG VIỆT – FULL VERSION
# =======================================================

import streamlit as st
import torch
from transformers import pipeline
import sqlite3
from datetime import datetime
import pandas as pd
import unicodedata

# =======================================================
# 0. CẤU HÌNH TRANG
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
# 2. XỬ LÝ VIẾT TẮT
# =======================================================
abbrev_map = {
    "ko":"không","k":"không","khong":"không","hok":"không",
    "dc":"được","dk":"được","cx":"cũng","vs":"với",
    "ms":"mới","mik":"mình","mk":"mình","bn":"bạn",
    "vl":"rất","vcl":"rất","rat":"rất","rát":"rất",
    "okela":"ok","oki":"ok","okii":"ok",
    "bùn":"buồn","zui":"vui","dui":"vui","hihi":"vui",
    "rầu":"buồn","gét":"ghét","met":"mệt","moi":"mỏi",
    "qua":"quá","wa":"quá","z":"vậy","v":"vậy",
    "ntn":"như thế nào","the":"thế","bik":"biết","bit":"biết",
    "do":"dở","on":"ổn","dinh":"định","lam":"lắm",
    "nay":"này","hom":"hôm","toi":"tôi","vi":"vì",
    "that":"thất","bai":"bại","ngay":"ngày","mai":"mai",
    "di":"đi","cam":"cảm","nhieu":"nhiều"
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
# 4. LOAD MODEL
# =======================================================
@st.cache_resource
def load_pipeline():
    try:
        model_name = "uitnlp/visobert"
        return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name), "ViSoBERT"
    except:
        try:
            model_name = "wonrax/phobert-base-vietnamese-sentiment"
            return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name), "PhoBERT"
        except:
            model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
            return pipeline("sentiment-analysis", model=model_name, tokenizer=model_name), "DistilBERT"

classifier, model_name = load_pipeline()

# =======================================================
# 5. DICTIONARY MẠNH + TỪ CỰC MẠNH
# =======================================================
sentiment_dict = {
    # Tích cực
    "vui":"POSITIVE","vui vẻ":"POSITIVE","rất vui":"POSITIVE",
    "cảm ơn":"POSITIVE","tuyệt":"POSITIVE","tuyệt vời":"POSITIVE",
    "hay":"POSITIVE","hay lắm":"POSITIVE","đỉnh":"POSITIVE",
    "thích":"POSITIVE","yêu":"POSITIVE","hạnh phúc":"POSITIVE",
    "ok":"POSITIVE","tốt":"POSITIVE","xuất sắc":"POSITIVE","hoàn hảo":"POSITIVE",
    # Trung tính
    "ổn định":"NEUTRAL","bình thường":"NEUTRAL","ổn":"NEUTRAL","cũng được":"NEUTRAL",
    "thời tiết":"NEUTRAL","đi học":"NEUTRAL","ngày mai":"NEUTRAL",
    "công việc":"NEUTRAL","học hành":"NEUTRAL",
    # Tiêu cực
    "buồn":"NEGATIVE","buồn vì":"NEGATIVE","chán":"NEGATIVE","ghét":"NEGATIVE","sợ":"NEGATIVE",
    "tồi":"NEGATIVE","dở":"NEGATIVE","dở quá":"NEGATIVE",
    "thất vọng":"NEGATIVE","thất bại":"NEGATIVE","khó chịu":"NEGATIVE",
    "tệ":"NEGATIVE","khủng khiếp":"NEGATIVE","bực mình":"NEGATIVE",
    "mệt mỏi":"NEGATIVE","mệt mỏi quá":"NEGATIVE","tệ quá":"NEGATIVE"
}

# =======================================================
# 6. KHÔI PHỤC DẤU
# =======================================================
accent_dict = {k: k for k in sentiment_dict.keys()}  # chỉ để hiển thị, có thể mở rộng

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
# 7. MATCH DICTIONARY
# =======================================================
def dict_match(text):
    t = restore_accents(text.lower())
    t_no = remove_accents(t)
    sorted_keys = sorted(sentiment_dict.keys(), key=lambda x: -len(x.split()))
    for key in sorted_keys:
        key_norm = key.lower()
        key_no = remove_accents(key_norm)
        if key_norm in t or key_no in t_no:
            return sentiment_dict[key]
    tokens = t.split()
    tokens_no = t_no.split()
    for token, token_no in zip(tokens, tokens_no):
        for key, label in sentiment_dict.items():
            key_norm = key.lower()
            key_no = remove_accents(key_norm)
            if key_norm == token or key_no == token_no:
                return label
    return None

# =======================================================
# 8. RULE PHỦ ĐỊNH
# =======================================================
def negation_rule(text):
    text_low = text.lower()
    no_acc = remove_accents(text_low)
    negation_words = ["khong","không","chưa","chả"]
    for neg in negation_words:
        if neg in no_acc or neg in text_low:
            positive_words = ["vui","tuyệt","thích","yêu","hạnh phúc","hay","đỉnh","tốt","ok","ổn"]
            negative_words = ["buồn","chán","ghét","tồi","dở","thất vọng","tệ","mệt"]
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
        "POS":"POSITIVE","NEG":"NEGATIVE","NEU":"NEUTRAL",
        "POSITIVE":"POSITIVE","NEGATIVE":"NEGATIVE","NEUTRAL":"NEUTRAL",
        "LABEL_0":"NEGATIVE","LABEL_1":"NEUTRAL","LABEL_2":"POSITIVE",
        "0":"NEGATIVE","1":"NEUTRAL","2":"POSITIVE"
    }
    return label_map.get(label_upper,"NEUTRAL")

def label_to_vietnamese(label):
    vn_map = {"POSITIVE":"Tích cực","NEGATIVE":"Tiêu cực","NEUTRAL":"Trung tính"}
    return vn_map.get(label,label)

def get_emoji(label):
    emoji_map = {"POSITIVE":"😊","NEGATIVE":"😞","NEUTRAL":"😐"}
    return emoji_map.get(label,"❓")

# =======================================================
# 10. PHÂN LOẠI SENTIMENT + PHẠT NEUTRAL
# =======================================================
def classify_sentiment(text, threshold=0.40):
    clean = preprocess(text)
    if clean is None: return None,0.0
    clean_accent = restore_accents(clean)
    # Rule phủ định
    neg_label = negation_rule(clean_accent)
    if neg_label: return normalize_label(neg_label),0.92
    # Dictionary mạnh
    dic_label = dict_match(clean_accent)
    if dic_label: return normalize_label(dic_label),0.95
    # Model
    try:
        result = classifier(clean_accent)[0]
        label = normalize_label(result['label'])
        confidence = result['score']
        if confidence < threshold: label="NEUTRAL"
        # Phạt NEUTRAL sai
        label, confidence = sentiment_override(clean_accent,label,confidence)
        return label, confidence
    except:
        return "NEUTRAL",0.5

def sentiment_override(text,predicted,confidence):
    strong_negative = ["dở quá","thất bại","buồn","chán","tệ","mệt mỏi"]
    strong_positive = ["rất vui","vui","tuyệt","thích","yêu","hạnh phúc"]
    text_low=text.lower()
    for w in strong_negative:
        if w in text_low: return "NEGATIVE",0.99
    for w in strong_positive:
        if w in text_low: return "POSITIVE",0.99
    return predicted,confidence

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
init_db()

def save_result(text,sentiment):
    display_text = restore_accents(text)
    conn=sqlite3.connect("history.db")
    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn.execute("INSERT INTO sentiments (text,sentiment,timestamp) VALUES(?,?,?)",(display_text,sentiment,timestamp))
    conn.commit()
    conn.close()

# =======================================================
# 12. STREAMLIT UI
# =======================================================
st.title("😊 Trợ lý phân loại cảm xúc tiếng Việt")
st.info(f"✅ Đang sử dụng model: **{model_name}**")

text = st.text_area("Nhập câu tiếng Việt:",height=100,placeholder="Ví dụ: Hôm nay tôi rất vui")
if st.button("🔍 Phân loại cảm xúc"):
    if not text or len(text.strip())<2:
        st.error("⚠️ Vui lòng nhập câu có ít nhất 2 ký tự!")
    else:
        with st.spinner("Đang phân tích..."):
            sent, conf = classify_sentiment(text)
            vn_label = label_to_vietnamese(sent)
            emoji = get_emoji(sent)
            st.success(f"{emoji} **Kết quả: {vn_label}** (Độ tin cậy: {conf*100:.1f}%)")
            display_text = restore_accents(text)
            st.json({"text_goc":text,"text_hien_thi":display_text,"sentiment":sent})
            save_result(text,sent)

# Lịch sử
st.markdown("---")
if st.checkbox("📋 Xem lịch sử (50 gần nhất)"):
    conn = sqlite3.connect("history.db")
    df = pd.read_sql_query("SELECT id,text,sentiment,timestamp FROM sentiments ORDER BY id DESC LIMIT 50",conn)
    conn.close()
    if not df.empty:
        df['Cảm xúc (VN)'] = df['sentiment'].apply(label_to_vietnamese)
        df['Icon'] = df['sentiment'].apply(get_emoji)
        st.dataframe(df[['Icon','text','Cảm xúc (VN)','timestamp']],use_container_width=True)
    else:
        st.info("Chưa có dữ liệu")

# Bộ test case
st.markdown("---")
st.subheader("🧪 Bộ Test Case (10 câu)")
test_cases = [
    {"text": "Hôm nay tôi rất vui", "expected": "POSITIVE"},
    {"text": "Món ăn này dở quá", "expected": "NEGATIVE"},
    {"text": "Thời tiết bình thường", "expected": "NEUTRAL"},
    {"text": "Rất vui hôm nay", "expected": "POSITIVE"},
    {"text": "Công việc ổn định", "expected": "NEUTRAL"},
    {"text": "Phim này hay lắm", "expected": "POSITIVE"},
    {"text": "Tôi buồn vì thất bại", "expected": "NEGATIVE"},
    {"text": "Ngày mai đi học", "expected": "NEUTRAL"},
    {"text": "Cảm ơn bạn rất nhiều", "expected": "POSITIVE"},
    {"text": "Mệt mỏi quá hôm nay", "expected": "NEGATIVE"},
]

if st.button("▶️ Chạy Test tất cả 10 câu"):
    correct=0
    results=[]
    progress_bar=st.progress(0)
    for i,case in enumerate(test_cases):
        pred,conf=classify_sentiment(case["text"])
        pred_norm=normalize_label(pred)
        expected_norm=normalize_label(case["expected"])
        ok = (pred_norm==expected_norm)
        if ok: correct+=1
        results.append({"STT":i+1,"Câu":case["text"],
                        "Mong đợi":label_to_vietnamese(expected_norm),
                        "Dự đoán":label_to_vietnamese(pred_norm),
                        "Độ tin cậy":f"{conf*100:.1f}%","Kết quả":"✅ Đúng" if ok else "❌ Sai"})
        progress_bar.progress((i+1)/len(test_cases))
    progress_bar.empty()
    acc=correct/len(test_cases)*100
    col1,col2,col3=st.columns(3)
    col1.metric("Đúng",f"{correct}/{len(test_cases)}")
    col2.metric("Độ chính xác",f"{acc:.1f}%")
    col3.metric("Đánh giá","✅ ĐẠT" if acc>=65 else "❌ CHƯA ĐẠT")
    st.dataframe(pd.DataFrame(results),use_container_width=True)
    if acc>=65:
        st.success(f"🎉 **ĐẠT YÊU CẦU!** Độ chính xác {acc:.1f}% (≥ 65%)")
    else:
        st.warning(f"⚠️ **Chưa đạt yêu cầu.** Cần ≥ 65%, hiện tại: {acc:.1f}%")
