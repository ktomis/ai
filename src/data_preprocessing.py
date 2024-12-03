from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd

def xu_ly_du_lieu(data):
    # Loai bo cot khong can thiet
    if 'Unnamed: 0' in data.columns:
        data.drop(columns=['Unnamed: 0'], inplace=True)
    if 'flight' in data.columns:
        data.drop(columns=['flight'], inplace=True)

    # Doi ten cot 'class' thanh 'flight_class'
    if 'class' in data.columns:
        data.rename(columns={'class': 'flight_class'}, inplace=True)

    # Tach du lieu X va y
    y = data['price']
    X = data.drop(columns=['price'], errors='ignore')

    # Ma hoa cot chuoi
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Chuan hoa du lieu
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y, label_encoders, scaler

# if __name__ == '__main__':
#     import pandas as pd
#     from data_loader import tai_du_lieu_va_kham_pha

#     # Tai du lieu truoc
#     duong_dan = r'C:\MyProJect_Study04\AirfarePrediction_AI\data\processed\Clean_Dataset.csv'
#     data = tai_du_lieu_va_kham_pha(duong_dan)

#     # Xu ly du lieu
#     X, y, label_encoders, scaler = xu_ly_du_lieu(data)
#     print("\nXu ly du lieu thanh cong!")
#     print("X shape:", X.shape)
#     print("y shape:", y.shape)
