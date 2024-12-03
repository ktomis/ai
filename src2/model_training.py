from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib
import numpy as np


def huan_luyen_va_danh_gia_mo_hinh(X_train, y_train, X_test, y_test, label_encoders, scaler):
    # Huấn luyện mô hình
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Mã hóa và chuẩn hóa X_test
    for col, le in label_encoders.items():
        if col in X_test.columns:
            # Kiểm tra giá trị chưa thấy
            unseen_labels = set(X_test[col]) - set(le.classes_)
            if unseen_labels:
                print(f"Thêm giá trị mới cho cột {col}: {unseen_labels}")
                le.classes_ = np.append(le.classes_, list(unseen_labels))  # Phương pháp 1

            # Mã hóa dữ liệu
            X_test[col] = le.transform(X_test[col])

    X_test_scaled = scaler.transform(X_test)

    # Dự đoán
    y_pred = model.predict(X_test_scaled)

    # Kiểm tra và đánh giá
    if y_test is not None:
        r2_score = metrics.r2_score(y_test, y_pred)
        print(f"R2 Score trên tập kiểm tra: {r2_score:.4f}")
    else:
        print("Không có cột 'Price' trong dữ liệu kiểm tra. Chỉ thực hiện dự đoán.")

    # Lưu mô hình và các công cụ xử lý
    joblib.dump({
    'model': model,
    'label_encoders': label_encoders,
    'scaler': scaler,
    'feature_columns': X_train.columns.tolist()  # Lưu danh sách cột
    }, '../models/random_forest_datathay.pkl')

    print("Mô hình đã được lưu thành công!")

if __name__ == '__main__':
    from data_loader import tai_du_lieu_va_kham_pha
    from data_preprocessing import xu_ly_du_lieu

    # Đường dẫn dữ liệu
    duong_dan_train = r'C:\MyProJect_Study04\AirfarePrediction_AI\data\processed\cleaned_train_data.csv'
    duong_dan_test = r'C:\MyProJect_Study04\AirfarePrediction_AI\data\processed\cleaned_test_data.csv'

    # Tải và xử lý dữ liệu huấn luyện
    train_data = tai_du_lieu_va_kham_pha(duong_dan_train)
    X_train, y_train, label_encoders, scaler = xu_ly_du_lieu(train_data)

    # Tải và xử lý dữ liệu kiểm tra
    test_data = tai_du_lieu_va_kham_pha(duong_dan_test)
    if 'Price' in test_data.columns:
        y_test = test_data['Price']
        X_test = test_data.drop(columns=['Price'])
    else:
        y_test = None
        X_test = test_data

    # Huấn luyện và đánh giá mô hình
    huan_luyen_va_danh_gia_mo_hinh(X_train, y_train, X_test, y_test, label_encoders, scaler)
