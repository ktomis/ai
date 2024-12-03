from src2.data_loader import tai_du_lieu_va_kham_pha
from src2.data_preprocessing import xu_ly_du_lieu
from src2.model_training import huan_luyen_va_danh_gia_mo_hinh

# Buoc 1: Tai du lieu va kham pha
data = tai_du_lieu_va_kham_pha(r'data/processed/Clean_Dataset.csv')

# Buoc 2: Xu ly du lieu
X, y, label_encoders, scaler = xu_ly_du_lieu(data)

# Buoc 3: Huan luyen va danh gia mo hinh
huan_luyen_va_danh_gia_mo_hinh(X, y)
