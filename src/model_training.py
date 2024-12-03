from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
import joblib


def huan_luyen_va_danh_gia_mo_hinh(X, y):
    # Chia du lieu thanh tap huan luyen va kiem tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dinh nghia cac mo hinh
    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    # Huan luyen va danh gia
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_score = metrics.r2_score(y_test, y_pred)
        results[model_name] = r2_score
        print(colored(f"{model_name} R2 Score: {r2_score:.4f}", 'green'))

    # Truc quan hoa ket qua
    # result_df = pd.DataFrame(list(results.items()), columns=['Model', 'R2_Score'])
    # sns.barplot(x='Model', y='R2_Score', data=result_df)
    # plt.title('Hieu suat mo hinh')
    # plt.ylabel('R2 Score')
    # plt.show()

    # Luu mo hinh Random Forest
    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
    best_model.fit(X, y)
    joblib.dump(best_model, '../models/random_forest.pkl')
    print("Mo hinh RandomForest da duoc luu thanh cong!")
    # print("Danh sach cac cot su dung trong mo hinh:", X.columns.tolist())


# if __name__ == '__main__':
#     from data_loader import tai_du_lieu_va_kham_pha
#     from data_preprocessing import xu_ly_du_lieu

#     # Tai va xu ly du lieu
#     duong_dan = r'C:\MyProJect_Study04\AirfarePrediction_AI\data\processed\Clean_Dataset.csv'
#     data = tai_du_lieu_va_kham_pha(duong_dan)
#     X, y, label_encoders, scaler = xu_ly_du_lieu(data)

#     # Huan luyen va danh gia mo hinh
#     huan_luyen_va_danh_gia_mo_hinh(X, y)
    
