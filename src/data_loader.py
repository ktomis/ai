import pandas as pd
from termcolor import colored

def tai_du_lieu_va_kham_pha(duong_dan):
    # Tai du lieu
    try:
        data = pd.read_csv(duong_dan)
        print(colored('Tai du lieu thanh cong!', 'green'))
    except FileNotFoundError:
        print(colored('Loi: Khong tim thay du lieu. Kiem tra lai duong dan!', 'red'))
        raise

    # Kham pha du lieu
    print("\nDu lieu mau:")
    print(data.head())
    print("\nThong tin du lieu:")
    data.info()

    return data

if __name__ == '__main__':
    duong_dan = r'C:\MyProJect_Study04\AirfarePrediction_AI\data\processed\Clean_Dataset.csv'
    data = tai_du_lieu_va_kham_pha(duong_dan)
    print("\nDu lieu da tai thanh cong!")
    print("\n5 dong dau tien cua du lieu:")
    print(data.head())
