import streamlit as st
import pandas as pd
import joblib

# Tải mô hình và công cụ xử lý
preprocessing_data = joblib.load('../models/random_forest_datathay.pkl')
model = preprocessing_data['model']
label_encoders = preprocessing_data['label_encoders']
scaler = preprocessing_data['scaler']
feature_columns = preprocessing_data['feature_columns']  # Lấy tên cột đã lưu khi huấn luyện

# Tiêu đề ứng dụng
st.title("Dự đoán giá vé máy bay")

# Tạo giao diện nhập liệu
user_input = {}
st.subheader("Nhập thông tin chuyến bay:")
for col, le in label_encoders.items():
    st.write(f"Chọn giá trị cho '{col}':")
    options = list(le.classes_)
    options.append('unknown')  # Thêm giá trị 'unknown' để xử lý trường hợp không xác định
    user_input[col] = st.selectbox(f"{col}:", options)

# Khi người dùng nhấn nút "Dự đoán"
if st.button("Dự đoán"):
    # Tạo DataFrame với các cột phù hợp
    user_input_df = pd.DataFrame([user_input], columns=feature_columns)
    user_input_df.fillna('unknown', inplace=True)  # Đảm bảo không có giá trị NaN

    # Mã hóa dữ liệu đầu vào
    for col, le in label_encoders.items():
        if col in user_input_df.columns:
            user_input_df[col] = user_input_df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1  # Xử lý giá trị không hợp lệ
            )

    # Kiểm tra cột không tồn tại và thêm với giá trị mặc định
    for missing_col in set(feature_columns) - set(user_input_df.columns):
        user_input_df[missing_col] = 0  # Giá trị mặc định là 0

    # Chuẩn hóa dữ liệu
    user_input_scaled = scaler.transform(user_input_df)

    # Dự đoán giá vé
    prediction = model.predict(user_input_scaled)

    # Hiển thị kết quả
    st.write(f"### Giá vé dự đoán: {prediction[0]:,.2f}₹ INR")
