# =======================================================
# TEST CASELIST ĐƯỢC TÁCH RIÊNG
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

    # Không dấu
    {"text": "Hom nay toi rat vui", "expected": "POSITIVE"},
    {"text": "Mon an nay do qua", "expected": "NEGATIVE"},
    {"text": "Thoi tiet binh thuong", "expected": "NEUTRAL"},
    {"text": "Cong viec on dinh", "expected": "NEUTRAL"},
    {"text": "Phim nay hay lam", "expected": "POSITIVE"},
    {"text": "Toi buon vi that bai", "expected": "NEGATIVE"},
    {"text": "Ngay mai di hoc", "expected": "NEUTRAL"},
    {"text": "Cam on ban rat nhieu", "expected": "POSITIVE"},
    {"text": "Met moi qua hom nay", "expected": "NEGATIVE"},
]
