import unicodedata

# =======================================================
# BẢN ĐỒ VIẾT TẮT / TEENCODE
# =======================================================
abbrev_map = {
    "ko": "không", "k": "không", "khong": "không", "hok": "không",
    "dc": "được", "dk": "được",
    "cx": "cũng", "vs": "với", "ms": "mới",
    "mik": "mình", "mk": "mình", "bn": "bạn",
    "vl": "rất", "vcl": "rất",
    "okela": "ok", "oki": "ok",
    "bùn": "buồn", "zui": "vui", "dui": "vui",
    "hihi": "vui", "rầu": "chán", "gét": "ghét"
}

def remove_accents(text):
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')
    return text

# =======================================================
# HÀM CHUẨN HÓA VIẾT TẮT
# =======================================================
def normalize_teencode(text):
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
