import pandas as pd
import joblib

def du_doan_gia_ve(model, scaler, label_encoders, du_lieu_dau_vao, columns_model):
    # Chuyen du lieu dau vao thanh DataFrame
    sample_df = pd.DataFrame([du_lieu_dau_vao])

    # Them cac cot thieu voi gia tri mac dinh (neu can)
    for col in columns_model:
        if col not in sample_df.columns:
            sample_df[col] = 0  # Gia tri mac dinh cho cac cot thieu

    # Sap xep cot theo thu tu cua columns_model
    sample_df = sample_df[columns_model]

    # Ma hoa du lieu
    for col, le in label_encoders.items():
        if col in sample_df.columns:
            sample_df[col] = le.transform(sample_df[col])

    # Chuan hoa du lieu
    sample_scaled = scaler.transform(sample_df)

    # Du doan gia ve
    gia_ve_du_doan = model.predict(sample_scaled)
    return gia_ve_du_doan[0]


# if __name__ == '__main__':
#     # Tai mo hinh va cac cong cu xu ly
#     model = joblib.load('../models/random_forest.pkl')

#     # Tai cac cong cu xu ly du lieu
#     from data_preprocessing import xu_ly_du_lieu
#     from data_loader import tai_du_lieu_va_kham_pha

#     duong_dan = r'C:\MyProJect_Study04\AirfarePrediction_AI\data\processed\Clean_Dataset.csv'
#     data = tai_du_lieu_va_kham_pha(duong_dan)
#     _, _, label_encoders, scaler = xu_ly_du_lieu(data)

#     # Danh sach cot mo hinh
#     columns_model = ['airline', 'source_city', 'departure_time', 'stops', 
#                     'arrival_time', 'destination_city', 'flight_class', 
#                     'duration', 'days_left']

#     # Du lieu dau vao
#     input_data = {
#         'airline': 'SpiceJet',
#         'source_city': 'Delhi',
#         'departure_time': 'Evening',
#         'stops': 'zero',
#         'arrival_time': 'Night',
#         'destination_city': 'Mumbai',
#         'flight_class': 'Economy',
#         'duration': 2.17,
#         'days_left': 5
#     }

#     # Du doan
#     gia_ve = du_doan_gia_ve(model, scaler, label_encoders, input_data, columns_model)
#     print("Gia ve du doan:", gia_ve)