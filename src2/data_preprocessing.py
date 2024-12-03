from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd

def xu_ly_du_lieu(data):
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)
    if 'flight' in data.columns:
        data.drop(columns=['flight'], inplace=True)
    if 'class' in data.columns:
        data.rename(columns={'class': 'flight_class'}, inplace=True)

    # Kiểm tra và tách cột mục tiêu
    y = None
    if 'Price' in data.columns:
        y = data['Price']
        X = data.drop(columns=['Price'], errors='ignore')
    else:
        X = data

    # Mã hóa dữ liệu dạng chuỗi
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y, label_encoders, scaler

if __name__ == '__main__':
    from data_loader import tai_du_lieu_va_kham_pha

    duong_dan_train = r'C:\MyProJect_Study04\AirfarePrediction_AI\data\processed\cleaned_train_data.csv'
    duong_dan_test = r'C:\MyProJect_Study04\AirfarePrediction_AI\data\processed\cleaned_test_data.csv'

    train_data = tai_du_lieu_va_kham_pha(duong_dan_train)
    test_data = tai_du_lieu_va_kham_pha(duong_dan_test)

    X_train, y_train, label_encoders, scaler = xu_ly_du_lieu(train_data)
    print("\nX_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    X_test, _, _, _ = xu_ly_du_lieu(test_data)
    print("\nX_test shape:", X_test.shape)
