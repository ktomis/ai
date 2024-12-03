import pandas as pd
from termcolor import colored

def tai_du_lieu_va_kham_pha(duong_dan):
    try:
        data = pd.read_csv(duong_dan)
        print(colored('Tải dữ liệu thành công!', 'green'))
    except FileNotFoundError:
        print(colored('Lỗi: Không tìm thấy dữ liệu. Kiểm tra lại đường dẫn!', 'red'))
        raise

    print("\nDữ liệu mẫu:")
    print(data.head())
    print("\nThông tin dữ liệu:")
    data.info()
    return data

if __name__ == '__main__':
    duong_dan_train = r'C:\MyProJect_Study04\AirfarePrediction_AI\data\processed\cleaned_train_data.csv'
    duong_dan_test = r'C:\MyProJect_Study04\AirfarePrediction_AI\data\processed\cleaned_test_data.csv'

    train_data = tai_du_lieu_va_kham_pha(duong_dan_train)
    test_data = tai_du_lieu_va_kham_pha(duong_dan_test)
