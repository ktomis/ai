import os

# Định nghĩa cấu trúc thư mục
folders = [
    "data/raw",          # Dữ liệu thô
    "data/processed",    # Dữ liệu đã xử lý
    "notebooks",         # Jupyter notebooks
    "src",               # Mã nguồn chính
    "models",            # Lưu mô hình đã huấn luyện
    "tests"              # Kiểm thử
]

# Tạo các thư mục
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Tạo các tệp chính
files = [
    "src/data_preprocessing.py",
    "src/feature_engineering.py",
    "src/model_training.py",
    "src/model_evaluation.py",
    "src/utils.py",
    "app.py",  # Tệp triển khai ứng dụng
    "requirements.txt",
    "README.md",
    ".gitignore"
]

for file in files:
    with open(file, "w") as f:
        pass  # Tạo file trống

print("Cấu trúc dự án đã được tạo!")
